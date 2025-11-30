import argparse
import json
import os
import sys
import logging
from dataclasses import dataclass
# from typing import Optional
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoModelForImageTextToText, get_linear_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# 作为脚本运行时，确保能导入到上级 utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_utils import CaptionIterableDataset, postprocess_zh, ImageTSVIndex, CaptionIndex
from tqdm import tqdm

# 基本日志配置
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger("sft_train")

# 设置 HuggingFace/网络相关环境变量与缓存目录
cache_dir = "/mnt/d/HuggingFaceModels/"
os.environ['TORCH_HOME'] = cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


@dataclass
class TrainArgs:
    train_jsonl: str
    images_tsv: str
    output_dir: str
    config: str


def build_model_and_proc(cfg):
    # 优先使用更贴近图文到文本任务的接口
    # 低精度与设备映射，避免整模型上卡导致 OOM
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    torch_dtype = dtype_map.get(str(cfg.get('torch_dtype', 'bfloat16')).lower())
    device_map = cfg.get('device_map', 'auto')

    common_kwargs = dict(
        trust_remote_code=cfg.get("trust_remote_code", True),
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if torch_dtype is not None:
        common_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        common_kwargs["device_map"] = device_map
    try:
        log.info(f"Loading model AutoModelForImageTextToText: {cfg['model_name']}")
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_name"], **common_kwargs
        )
    except Exception:
        try:
            log.warning("AutoModelForImageTextToText failed, fallback to AutoModelForVision2Seq")
            model = AutoModelForVision2Seq.from_pretrained(
                cfg["model_name"], **common_kwargs
            )
        except Exception:
            log.warning("AutoModelForVision2Seq failed, fallback to AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["model_name"], **common_kwargs
            )
    log.info("Loading processor")
    processor = AutoProcessor.from_pretrained(
        cfg["model_name"], trust_remote_code=cfg.get("trust_remote_code", True), cache_dir=cache_dir,
        local_files_only=True, use_fast=False
    )
    if cfg.get("use_lora", True):
        lcfg = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=cfg.get("lora_dropout", 0.05),
                          target_modules=cfg.get("target_modules", None))
        log.info(f"Applying LoRA: r={lcfg.r}, alpha={lcfg.lora_alpha}, dropout={lcfg.lora_dropout}")
        model = get_peft_model(model, lcfg)
    return model, processor


from PIL import Image

def collate_fn(batch, processor, image_size: int = 448):
    images = [b.image for b in batch]
    prompts = ["请为商品图片生成精准中文描述："] * len(batch)
    targets = [postprocess_zh(b.caption or "") for b in batch]

    proc_images = []
    texts = []
    for img, prompt, tgt in zip(images, prompts, targets):
        try:
            img = img.convert("RGB")
        except Exception:
            pass
        img = img.resize((image_size, image_size), Image.BILINEAR)
        proc_images.append(img)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": tgt}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    inputs = processor(text=texts, images=proc_images, return_tensors="pt", padding=True)
    log.debug({
        "batch_size": len(proc_images),
        "image_size": image_size,
        "input_ids_len": [len(ids) for ids in inputs["input_ids"]],
    })

    # 显式设置有效的图像网格，避免极端情况下为 0
    try:
        patch_size = int(getattr(getattr(getattr(processor, "image_processor", object()), "patch_size", 14), "__int__", lambda: 14)())
    except Exception:
        patch_size = 14
    h = (image_size // patch_size) or 1
    w = (image_size // patch_size) or 1
    inputs["image_grid_thw"] = torch.tensor([[1, h, w]] * len(proc_images), dtype=torch.long)
    log.debug({
        "patch_size": patch_size,
        "grid_thw": inputs["image_grid_thw"].tolist(),
    })

    # 保留处理器生成的视觉网格信息，交由模型正确使用

    # 构造与 input_ids 等长的 labels，只监督 assistant 段落
    batch_input_ids = inputs["input_ids"].tolist()
    tgt_token_lists = [processor(text=t)["input_ids"][0] for t in targets]

    def find_subseq(seq, subseq):
        n, m = len(seq), len(subseq)
        if m == 0 or n < m:
            return -1
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return -1

    labels = []
    for inp_ids, tgt_ids in zip(batch_input_ids, tgt_token_lists):
        lbl = [-100] * len(inp_ids)
        pos = find_subseq(inp_ids, tgt_ids)
        if pos == -1:
            # 若匹配失败，保守地监督末尾 len(tgt_ids) 个 token
            pos = max(0, len(inp_ids) - len(tgt_ids))
        end = min(len(inp_ids), pos + len(tgt_ids))
        lbl[pos:end] = inp_ids[pos:end]
        labels.append(lbl)
    log.debug({
        "labels_span": [
            {
                "first_supervised_idx": next((i for i,x in enumerate(lbl) if x != -100), -1),
                "last_supervised_idx": len(lbl)-1 - next((i for i,x in enumerate(reversed(lbl)) if x != -100), -1),
            } for lbl in labels
        ]
    })

    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--images_tsv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "../configs/sft.json"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 关闭自动 device_placement，避免 prepare() 将整模型迁移到单卡导致 OOM
    accelerator = Accelerator(
        device_placement=False,
        mixed_precision=cfg.get("mixed_precision", None)
    )
    model, processor = build_model_and_proc(cfg)
    log.info(f"Training config: batch_size={cfg.get('batch_size', 1)}, epochs={cfg.get('epochs', 1)}, lr={cfg.get('learning_rate', 2e-5)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 2e-5), weight_decay=cfg.get("weight_decay", 0.0))
    # 估算训练步数（保守估计）
    num_update_steps = cfg.get("epochs", 1) * 1000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.get("warmup_steps", 100), num_training_steps=num_update_steps)

    # 先准备核心对象，DataLoader 按轮次再单独 prepare
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.train()

    pics_per_round = int(cfg.get("pics_for_one_trn_round", 10000) or 10000)
    num_rounds = int(cfg.get("num_trn_rounds", 1) or 1)
    log.info(f"Round training enabled: pics_per_round={pics_per_round}, num_rounds={num_rounds}")

    global_step = 0
    # 仅为本轮需要的图片行数从 images.tsv 中收集 image_id（图片优先），避免先走 captions
    def collect_round_image_ids_from_tsv(images_tsv: str, start_index: int, max_samples: int):
        ids = []
        skipped = 0
        yielded = 0
        with open(images_tsv, "r", encoding="utf-8") as f:
            for line in f:
                if skipped < start_index:
                    skipped += 1
                    if skipped % 200000 == 0:
                        log.info(f"Skipping images.tsv lines: skipped={skipped}/{start_index}")
                    continue
                if yielded >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    image_id, _ = line.split("\t", 1)
                    ids.append(image_id)
                    yielded += 1
                except ValueError:
                    continue
        log.info(f"Collected round image_ids from TSV: start_index={start_index}, max_samples={max_samples}, collected={len(ids)}")
        return set(ids)
    for round_idx in range(num_rounds):
        start_index = round_idx * pics_per_round
        # 按图片驱动：先从 images.tsv 取本轮的 image_id，再为这些 ID 构建偏移索引与 captions 索引
        round_ids = collect_round_image_ids_from_tsv(args.images_tsv, start_index, pics_per_round)
        resolver = ImageTSVIndex(args.images_tsv, target_ids=round_ids, log_progress=True)
        cap_index = CaptionIndex(args.train_jsonl, target_ids=round_ids, log_progress=True)
        ds = CaptionIterableDataset(
            args.train_jsonl,
            args.images_tsv,
            image_size=cfg.get("image_size", 448),
            max_samples=pics_per_round,
            start_index=0,
            resolver=resolver,
            target_image_ids=list(round_ids),
            caption_index=cap_index,
        )
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        dl = DataLoader(ds, batch_size=cfg.get("batch_size", 1), collate_fn=lambda b: collate_fn(b, processor, cfg.get("image_size", 448)))
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        dl = accelerator.prepare(dl)
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        log.info(f"Starting round {round_idx+1}/{num_rounds} with start_index={start_index}")
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for epoch in range(cfg.get("epochs", 1)):
            for batch in tqdm(dl, desc=f"SFT Round {round_idx+1} Epoch {epoch}"):
                batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                log.debug({
                    "device": str(accelerator.device),
                    "pixel_values_shape": list(batch.get("pixel_values", torch.empty(0)).shape),
                    "grid_thw": batch.get("image_grid_thw", torch.empty(0)).tolist() if isinstance(batch.get("image_grid_thw"), torch.Tensor) else None,
                    "input_ids_len": [len(ids) for ids in batch.get("input_ids", [])],
                })
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if accelerator.is_main_process and (global_step % cfg.get("log_steps", 50) == 0):
                    log.info(f"round={round_idx+1} step={global_step} loss={loss.item():.4f}")
                if accelerator.is_main_process and (global_step % cfg.get("save_steps", 500) == 0):
                    os.makedirs(args.output_dir, exist_ok=True)
                    model.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    log.info(f"Checkpoint saved to {args.output_dir}")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        log.info(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
