import argparse
import json
import os
import sys
import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from accelerate import Accelerator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_utils import CaptionIterableDataset, postprocess_zh, ImageTSVIndex, CaptionIndex
sys.path.append(os.path.join(os.path.dirname(__file__), "../eval"))
from cider_d import CiderD
from tqdm import tqdm

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger("scst_train")
from PIL import Image

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
class Args:
    train_jsonl: str
    images_tsv: str
    sft_model: str
    output_dir: str
    config: str


def sample_baseline(model, processor, images, prompts, beam_size=4, max_len=64, image_size=448):
    proc_images = [img.convert("RGB").resize((image_size, image_size), Image.BILINEAR) for img in images]
    inputs = processor(images=proc_images, text=prompts, return_tensors="pt", padding=True)
    # 保留 image_grid_thw 以匹配 Qwen2.5-VL 的期望
    log.debug({
        "beam_size": beam_size,
        "max_len": max_len,
        "pixel_values_shape": list(inputs.get("pixel_values", torch.empty(0)).shape),
        "grid_thw": inputs.get("image_grid_thw", torch.empty(0)).tolist() if isinstance(inputs.get("image_grid_thw"), torch.Tensor) else None,
    })
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_ids = model.generate(**inputs, max_new_tokens=max_len, num_beams=beam_size)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
    return [postprocess_zh(t) for t in texts]


def sample_candidate(model, processor, images, prompts, max_len=64, image_size=448):
    proc_images = [img.convert("RGB").resize((image_size, image_size), Image.BILINEAR) for img in images]
    inputs = processor(images=proc_images, text=prompts, return_tensors="pt", padding=True)
    # 保留 image_grid_thw 以匹配 Qwen2.5-VL 的期望
    log.debug({
        "max_len": max_len,
        "pixel_values_shape": list(inputs.get("pixel_values", torch.empty(0)).shape),
        "grid_thw": inputs.get("image_grid_thw", torch.empty(0)).tolist() if isinstance(inputs.get("image_grid_thw"), torch.Tensor) else None,
    })
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_ids = model.generate(**inputs, max_new_tokens=max_len, do_sample=True, temperature=1.0, top_p=0.9)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
    return [postprocess_zh(t) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--images_tsv", required=True)
    ap.add_argument("--sft_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "../configs/scst.json"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 关闭自动 device_placement，避免 prepare() 将整模型迁移到单卡导致 OOM
    accelerator = Accelerator(
        device_placement=False,
        mixed_precision=cfg.get("mixed_precision", None)
    )
    log.info(f"Loading base model: {cfg['model_name']}")
    # 半精度 + 设备映射，降低显存占用
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
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_name"], **common_kwargs
    )
    log.info("Loading processor")
    processor = AutoProcessor.from_pretrained(
        cfg["model_name"], trust_remote_code=cfg.get("trust_remote_code", True), cache_dir=cache_dir,
        local_files_only=True, use_fast=False
    )

    # 加载 SFT 权重（若在 peft 下，需同时加载适配器）
    try:
        log.info(f"Loading SFT adapter from {args.sft_model}")
        model.load_adapter(args.sft_model)
    except Exception:
        log.warning("load_adapter failed, fallback to directly loading fine-tuned model dir")
        model = AutoModelForVision2Seq.from_pretrained(
            args.sft_model, trust_remote_code=cfg.get("trust_remote_code", True), cache_dir=cache_dir, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            args.sft_model, trust_remote_code=cfg.get("trust_remote_code", True), cache_dir=cache_dir,
            local_files_only=True, use_fast=False
        )

    ds = CaptionIterableDataset(args.train_jsonl, args.images_tsv, image_size=cfg.get("image_size", 448), max_samples=cfg.get("max_samples"))
    pics_per_round = int(cfg.get("pics_for_one_trn_round", 10000) or 10000)
    num_rounds = int(cfg.get("num_trn_rounds", 1) or 1)
    log.info(f"Round training enabled (SCST): pics_per_round={pics_per_round}, num_rounds={num_rounds}")

    model = accelerator.prepare(model)
    model.train()

    # 构建 refs 以计算 CIDEr-D 奖励（图片优先：先取一定量图片，再匹配对应 captions）
    refs_map = {}
    def collect_image_ids_from_tsv(images_tsv: str, start_index: int, max_samples: int):
        ids = []
        skipped = 0
        yielded = 0
        with open(images_tsv, "r", encoding="utf-8") as f:
            for line in f:
                if skipped < start_index:
                    skipped += 1
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
        log.info(f"Warm-up collected image_ids from TSV: collected={len(ids)}")
        return set(ids)
    warm_ids = collect_image_ids_from_tsv(args.images_tsv, start_index=0, max_samples=1000)
    warm_resolver = ImageTSVIndex(args.images_tsv, target_ids=warm_ids, log_progress=True)
    warm_cap_index = CaptionIndex(args.train_jsonl, target_ids=warm_ids, log_progress=True)
    warm_ds = CaptionIterableDataset(
        args.train_jsonl, args.images_tsv,
        image_size=cfg.get("image_size", 448), max_samples=1000, start_index=0,
        resolver=warm_resolver, target_image_ids=list(warm_ids), caption_index=warm_cap_index
    )
    for sample in tqdm(warm_ds, desc="Build CIDEr IDF"):
        img_id = sample.image_id
        cap = sample.caption
        if img_id and cap:
            refs_map.setdefault(img_id, []).append(cap)
    log.info(f"Built refs for CIDEr-D: {len(refs_map)} images")
    cider = CiderD(refs_map)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-5))
    log.info(f"SCST config: batch_size={cfg.get('batch_size', 1)}, epochs={cfg.get('epochs', 1)}, lr={cfg.get('learning_rate', 1e-5)}")

    for round_idx in range(num_rounds):
        start_index = round_idx * pics_per_round
        # 图片优先：先从 images.tsv 采样本轮 image_id，再构建偏移与 captions 索引
        round_ids = collect_image_ids_from_tsv(args.images_tsv, start_index=start_index, max_samples=pics_per_round)
        resolver = ImageTSVIndex(args.images_tsv, target_ids=round_ids, log_progress=True)
        cap_index = CaptionIndex(args.train_jsonl, target_ids=round_ids, log_progress=True)
        ds = CaptionIterableDataset(
            args.train_jsonl, args.images_tsv,
            image_size=cfg.get("image_size", 448), max_samples=pics_per_round, start_index=0,
            resolver=resolver, target_image_ids=list(round_ids), caption_index=cap_index
        )
        dl = DataLoader(ds, batch_size=cfg.get("batch_size", 1))
        dl = accelerator.prepare(dl)
        log.info(f"Starting round {round_idx+1}/{num_rounds} with start_index={start_index}")
        for epoch in range(cfg.get("epochs", 1)):
            for batch in tqdm(dl, desc=f"SCST Round {round_idx+1} Epoch {epoch}"):
                # DataLoader 默认返回列表，包含多个 Sample
                samples = batch if isinstance(batch, list) else [batch]
                images = [s.image for s in samples]
                image_ids = [s.image_id for s in samples]
                prompts = ["请为商品图片生成精准中文描述："] * len(images)

                with torch.no_grad():
                    baseline_txts = sample_baseline(
                        model, processor, images, prompts,
                        beam_size=cfg.get("beam_size", 4),
                        max_len=cfg.get("max_gen_len", 64),
                        image_size=cfg.get("image_size", 448)
                    )
                    candidate_txts = sample_candidate(
                        model, processor, images, prompts,
                        max_len=cfg.get("max_gen_len", 64),
                        image_size=cfg.get("image_size", 448)
                    )

                # 奖励：CIDEr-D(baseline) 与 CIDEr-D(candidate)
                rewards = []
                base_rewards = []
                for img_id, btxt, ctxt in zip(image_ids, baseline_txts, candidate_txts):
                    rlist = refs_map.get(img_id, [])
                    rb = cider.score_one(btxt, rlist)
                    rc = cider.score_one(ctxt, rlist)
                    rewards.append(rc - rb)  # advantage
                    base_rewards.append(rb)
                log.debug({
                    "avg_base_cider": sum(base_rewards)/max(1, len(base_rewards)),
                    "avg_advantage": sum(rewards)/max(1, len(rewards)),
                })

                # 简化：用 logit 对齐奖励做损失近似（演示用；真实 SCST 需策略梯度）
                # 这里用 teacher-forcing 的 token 负对数似然与奖励权重相乘，作为近似
                proc_images = [
                    img.convert("RGB").resize((cfg.get("image_size", 448), cfg.get("image_size", 448)), Image.BILINEAR)
                    for img in images
                ]
                inputs = processor(images=proc_images, text=prompts, return_tensors="pt", padding=True)
                # 保留 image_grid_thw 以匹配 Qwen2.5-VL 的期望
                log.debug({
                    "pixel_values_shape": list(inputs.get("pixel_values", torch.empty(0)).shape),
                    "grid_thw": inputs.get("image_grid_thw", torch.empty(0)).tolist() if isinstance(inputs.get("image_grid_thw"), torch.Tensor) else None,
                })
                labels_enc = processor(text=candidate_txts, return_tensors="pt", padding=True)
                inputs["labels"] = labels_enc["input_ids"].to(model.device)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = outputs.loss * torch.tensor(rewards, device=model.device).mean()
                log.info(f"loss={loss.item():.4f}")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        log.info(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
