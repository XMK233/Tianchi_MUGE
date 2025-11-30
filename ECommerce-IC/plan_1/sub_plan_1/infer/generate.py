import argparse
import json
import re
import os
import sys
import logging
import torch

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_utils import postprocess_zh, ImageTSVIndex, CaptionIndex
from tqdm import tqdm

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger("infer")

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


def _strip_prompt_artifacts(text: str) -> str:
    """Remove chat template prompt artifacts from decoded text.
    Covers role tokens and the specific Chinese instruction string.
    """
    if not isinstance(text, str):
        return text
    artifacts = [
        "systemYouareahelpfulasistant.user请为商品图片生成精准中文描述:asistant",
        "systemYou are a helpful assistant.user请为商品图片生成精准中文描述：assistant",
        "You are a helpful assistant",
        "请为商品图片生成精准中文描述：",
        "请为商品图片生成精准中文描述:",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "system",
        "user",
        "assistant",
        "asistant",
    ]
    for a in artifacts:
        text = text.replace(a, "")
    # Normalize leftover colons/spaces
    text = text.strip().lstrip(":：\n\r\t ")
    return text


def _strip_edge_punct(text: str) -> str:
    """Trim leading and trailing punctuation and whitespace from text.
    Keeps inner punctuation intact.
    """
    if not isinstance(text, str):
        return text
    # Define a broad set of punctuation symbols (English + Chinese)
    edge_pattern = r'^[\s\.,!\?;:\-—–，。！？、：；“”"\'`~·•()\[\]{}《》<>|\/…]+|[\s\.,!\?;:\-—–，。！？、：；“”"\'`~·•()\[\]{}《》<>|\/…]+$'
    # Remove repeatedly from both ends until stable
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'^[\s\.,!\?;:\-—–，。！？、：；“”"\'`~·•()\[\]{}《》<>|\/…]+', '', text)
        text = re.sub(r'[\s\.,!\?;:\-—–，。！？、：；“”"\'`~·•()\[\]{}《》<>|\/…]+$', '', text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--images_tsv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--max_gen_len", type=int, default=64)
    ap.add_argument("--pics_for_one_pred_round", type=int, default=10000)
    ap.add_argument("--num_pred_rounds", type=int, default=1)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--image_size", type=int, default=448)
    ap.add_argument("--torch_dtype", type=str, default=None, help="Override dtype: float32|float16|bfloat16")
    ap.add_argument("--device_map", type=str, default=None, help="Override device map, e.g., 'cpu' or 'auto'")
    args = ap.parse_args()

    # 确保输出目录存在（避免因目录不存在导致写文件失败）
    try:
        out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            log.info(f"Created output directory: {out_dir}")
    except Exception as e:
        log.error(f"Failed to ensure output directory: {e}")
        raise

    log.info(f"Loading processor from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, cache_dir=cache_dir,
        local_files_only=True, use_fast=False
    )

    # 适配 SFT 保存：若存在 LoRA 适配器（adapter_config.json），则从其配置的 base_model 加载并挂载适配器
    adapter_cfg_path = os.path.join(args.model_dir, "adapter_config.json")
    if args.device is not None:
        device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 与训练对齐：尽量以低精度+自动设备映射加载，降低显存占用
    common_kwargs = dict(trust_remote_code=True, cache_dir=cache_dir, local_files_only=True)
    # dtype & device_map override if provided
    if args.torch_dtype:
        dt = args.torch_dtype.lower()
        if dt in ("float32", "fp32"):
            common_kwargs["torch_dtype"] = torch.float32
        elif dt in ("float16", "fp16"):
            common_kwargs["torch_dtype"] = torch.float16
        elif dt in ("bfloat16", "bf16"):
            common_kwargs["torch_dtype"] = torch.bfloat16
    if args.device_map:
        common_kwargs["device_map"] = args.device_map
    # Default policy when not overridden
    if "torch_dtype" not in common_kwargs and device == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability()
            # Ampere(8.x) 及以上优先 bfloat16，否则用 float16
            torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
        except Exception:
            torch_dtype = torch.float16
        common_kwargs.update({"torch_dtype": torch_dtype, "device_map": "auto"})
    if "torch_dtype" not in common_kwargs and device == "cpu":
        # CPU 上保持 float32，避免半精度在 CPU 上的不支持/溢出问题
        common_kwargs.update({"torch_dtype": torch.float32})
    if os.path.exists(adapter_cfg_path):
        try:
            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base = adapter_cfg.get("base_model_name_or_path")
            if not base:
                raise ValueError("adapter_config.json missing base_model_name_or_path")
            log.info(f"Detected LoRA adapter; loading base model: {base}")
            try:
                model = AutoModelForImageTextToText.from_pretrained(base, **common_kwargs)
            except Exception:
                try:
                    model = AutoModelForVision2Seq.from_pretrained(base, **common_kwargs)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(base, **common_kwargs)
            model = PeftModel.from_pretrained(model, args.model_dir)
        except Exception as e:
            log.warning(f"Failed to load LoRA adapter, fallback to direct load from model_dir: {e}")
            try:
                model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **common_kwargs)
            except Exception:
                try:
                    model = AutoModelForVision2Seq.from_pretrained(args.model_dir, **common_kwargs)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **common_kwargs)
    else:
        # 非 LoRA 情况：直接按目录加载完整权重
        try:
            model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **common_kwargs)
        except Exception:
            try:
                model = AutoModelForVision2Seq.from_pretrained(args.model_dir, **common_kwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(args.model_dir, **common_kwargs)
    model.eval()
    # 若使用了 device_map（如 auto），避免再整体 .to() 破坏分布式放置
    if "device_map" not in common_kwargs:
        try:
            model.to(device)
            log.info(f"Model moved to device={device}")
        except Exception as e:
            log.warning(f"Failed to move model to {device}: {e}")
    # 以模型实际参数设备为准，统一迁移输入张量
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)
    log.info(f"Model working device: {model_device}")

    # 图片优先：先从 images.tsv 采样固定数量的图片 ID，再匹配 test_jsonl 中的条目
    def collect_image_ids_from_tsv(images_tsv: str, start_index: int, max_samples: int):
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
        log.info(f"Collected image_ids from TSV: start_index={start_index}, max_samples={max_samples}, collected={len(ids)}")
        return set(ids)

    outs = []
    pics_per_round = int(args.pics_for_one_pred_round or 10000)
    num_rounds = int(args.num_pred_rounds or 1)
    log.info(f"Round inference enabled: pics_per_round={pics_per_round}, num_rounds={num_rounds}")
    for round_idx in range(num_rounds):
        start_index = round_idx * pics_per_round
        round_ids = collect_image_ids_from_tsv(args.images_tsv, start_index, pics_per_round)
        # 仅为本轮目标 ID 构建偏移索引与 test 索引（无 caption 也记录存在性）
        resolver = ImageTSVIndex(args.images_tsv, target_ids=round_ids, log_progress=True)
        cap_index = CaptionIndex(args.test_jsonl, target_ids=round_ids, log_progress=True)
        selected_ids = [iid for iid in round_ids if iid in cap_index.map]
        log.info(f"Round {round_idx+1}: matched test items={len(selected_ids)}/{len(round_ids)}")
        loaded = 0
        # 记录显存信息，辅助定位 OOM
        if device == "cuda":
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                log.info(f"CUDA memory before round {round_idx+1}: free={free_mem/1e9:.2f}GB total={total_mem/1e9:.2f}GB")
            except Exception:
                pass
        for image_id in selected_ids:
            if loaded >= pics_per_round:
                break
            img = resolver.get_image(image_id)
            if img is None:
                continue
            img = img.convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
            # 与训练一致的聊天模板（仅提供 user 段，生成时添加 assistant 提示）
            messages = [
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "请为商品图片生成精准中文描述："}]},
            ]
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], return_tensors="pt")
            except Exception:
                # 若不支持 chat_template，退化为简单文本+图像输入
                inputs = processor(images=[img], text=["请为商品图片生成精准中文描述："], return_tensors="pt")
            try:
                inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            except Exception as e:
                log.warning(f"Failed to move inputs to model device {model_device}: {e}")
            log.debug({
                "pixel_values_shape": list(inputs.get("pixel_values", None).shape) if hasattr(inputs.get("pixel_values", None), 'shape') else None,
                "grid_thw": inputs.get("image_grid_thw", None).tolist() if hasattr(inputs.get("image_grid_thw", None), 'tolist') else None,
            })
            # 降低生成占用：禁用缓存可减少 KV-cache 内存（代价是速度）
            try:
                model.generation_config.use_cache = False
            except Exception:
                pass
            with torch.inference_mode():
                try:
                    gen_ids = model.generate(**inputs, num_beams=args.beam_size, max_new_tokens=args.max_gen_len)
                except torch.cuda.OutOfMemoryError:
                    log.warning("CUDA OOM during generate; retry with num_beams=1 and CPU fallback.")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        model.to("cpu")
                        inputs = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                    except Exception as e:
                        log.warning(f"Failed to move to CPU fallback: {e}")
                    gen_ids = model.generate(**inputs, num_beams=1, max_new_tokens=args.max_gen_len)
            # 释放中间对象，减少显存/内存占用
            try:
                del inputs
            except Exception:
                pass
            # 成功生成后，如在 CUDA 上，清理缓存减少碎片化
            if device == "cuda":
                try:
                    del img
                    torch.cuda.empty_cache()
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    log.debug(f"CUDA memory after one sample: free={free_mem/1e9:.2f}GB total={total_mem/1e9:.2f}GB")
                except Exception:
                    pass
            decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            cleaned = _strip_prompt_artifacts(decoded)
            final_text = _strip_edge_punct(postprocess_zh(cleaned)).strip()
            outs.append({"image_id": image_id, "text": final_text})
            loaded += 1
        log.info(f"Finished round {round_idx+1}/{num_rounds}, generated {loaded} captions")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for o in outs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(outs)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
