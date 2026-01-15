import os
import json
import random
import logging
import warnings
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)

# HF & Torch（直接导入，不使用异常捕获）
import torch
import transformers
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
import deepspeed
from preview_utils import save_warmup_preview
from transformers.utils import is_flash_attn_2_available

# 同目录下的数据加载器（直接通过路径导入，避免异常分支）
import sys
sys.path.append(os.path.dirname(__file__))
from data_loader import (
    load_ic_batch,
    ICTsvJsonlDataset,
)
import tqdm
import gc
import inspect

def free_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        # branch_log()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                log.info(f"[GPU{i}] after cleanup used={mem.used/1024**2:.2f} MB / total={mem.total/1024**2:.2f} MB")
        except Exception:
            except_log("pynvml memory log failed")


log = logging.getLogger("pipeline_template")

# 引入评测指标：CIDEr（近似实现，见 metrics.py）—直接路径导入
from metrics import compute_cider

def branch_log():
    """在分支内记录当前调用行号。"""
    try:
        ln = inspect.currentframe().f_back.f_lineno
        log.info(f"[Branch] Entered at line {ln}")
    except Exception:
        pass


def except_log(msg: str | None = None):
    """在异常分支记录当前调用行号，提示 try 代码存在问题。"""
    try:
        ln = inspect.currentframe().f_back.f_lineno
        base = f"[Except] Entered at line {ln}; try-block had an issue"
        log.info(base if msg is None else base + f" | {msg}")
    except Exception:
        pass


# ---------------------------
# 本地模型加载：Qwen2.5-VL-3B-Instruct
# ---------------------------
def _has_config(path: str) -> bool:
    """判断给定目录下是否存在 `config.json` 文件。"""
    return os.path.isfile(os.path.join(path, "config.json"))


def _resolve_model_dir(local_dir: str) -> str:
    """将 HF 缓存根目录（models--Org--Repo）解析为实际 snapshot 路径。

    优先规则：
    1) 若 local_dir 本身含有 config.json，则直接使用；
    2) 若存在 snapshots 子目录，选择其中包含 config.json 的最新修改时间的子目录；
    3) 回退：在 local_dir 下递归搜索第一个包含 config.json 的目录；
    4) 若仍未找到，返回原始 local_dir（后续会触发错误并提示）。
    """
    if _has_config(local_dir):
        branch_log()
        return local_dir

    snap_root = os.path.join(local_dir, "snapshots")
    if os.path.isdir(snap_root):
        # branch_log()
        candidates = []
        for name in os.listdir(snap_root):
            p = os.path.join(snap_root, name)
            if os.path.isdir(p) and _has_config(p):
                # branch_log()
                candidates.append(p)
        if candidates:
            # branch_log()
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    # 递归搜索 config.json
    for dirpath, dirnames, filenames in os.walk(local_dir):
        if "config.json" in filenames:
            branch_log()
            return dirpath

    return local_dir

def load_qwen_vl(local_dir: str, for_training: bool = False):
    """从本地路径加载 Qwen2.5-VL-3B-Instruct，多模态推理用。

    要求：
    - local_dir 指向本地缓存目录，例如：
      /mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct
    - 禁止联网，直接走本地文件。
    """
    # 屏蔽 FutureWarning（pynvml 相关），避免噪音
    warnings.filterwarnings("ignore", category=FutureWarning)

    if 'TRANSFORMERS_OFFLINE' not in os.environ:
        # branch_log()
        # 优先离线模式，避免任何网络请求
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 允许 TF32 提速（Ampere+ GPU 上对矩阵乘更快，保持数值稳定）
    if torch.cuda.is_available():
        # branch_log()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    resolved_dir = _resolve_model_dir(local_dir)

    processor = AutoProcessor.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    # 自动选择 dtype 与设备映射，尽量利用 GPU；无 GPU 则落到 CPU
    # 优先尝试专用类（trust_remote_code 可能注册到 transformers 命名空间）
    specific_classes = [
        getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None),
        getattr(transformers, "Qwen2_5_VLForCausalLM", None),
        getattr(transformers, "Qwen2VLForConditionalGeneration", None),
        getattr(transformers, "Qwen2VLForCausalLM", None),
    ]
    auto_causal_lm = getattr(transformers, "AutoModelForCausalLM", None)
    auto_model_cg = getattr(transformers, "AutoModelForConditionalGeneration", None)
    auto_model = getattr(transformers, "AutoModel", None)
    # 确定性选择第一个可用的模型类
    model_cls_order = [*specific_classes, auto_causal_lm, auto_model_cg, auto_model]
    model_cls = next((c for c in model_cls_order if c is not None), None)
    if model_cls is None:
        raise RuntimeError("No suitable model class available for Qwen VL.")
    # 训练时禁用 device_map=auto，避免模型被切到多设备导致 DeepSpeed 管理复杂和显存碎片
    if for_training:
        # branch_log()
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            dtype='auto',
            device_map=None,
            local_files_only=True,
        )
        # 训练侧关闭缓存并开启梯度检查点以降低显存（移除冗余异常分支）
        if hasattr(model, 'config'):
            # branch_log()
            model.config.use_cache = False
        if hasattr(model, 'gradient_checkpointing_enable'):
            # branch_log()
            model.gradient_checkpointing_enable()
    else:
        # branch_log()
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            dtype='auto',
            device_map=None,  # 避免自动切到 CPU，统一由我们手动放到 CUDA
            local_files_only=True,
        )
        # 推理场景尽量将整模移动到 CUDA，以确保真正使用 GPU 进行计算
        if torch.cuda.is_available():
            # branch_log()
            model.to(torch.device('cuda'))
            dev = next(model.parameters()).device
            log.info(f"[Infer] model moved to device: {dev}")
        # 推理侧启用缓存与更快的注意力实现（如可用则使用 FlashAttention2，否则回退到 SDPA）
        if hasattr(model, 'config'):
            # branch_log()
            model.config.use_cache = True
            if is_flash_attn_2_available(): ## 实际上并没有flash_attn_2。
                # branch_log()
                setattr(model.config, 'attn_implementation', 'flash_attention_2')
            else:
                # branch_log()
                setattr(model.config, 'attn_implementation', 'sdpa')
    model.eval()
    return model, processor

def caption_batch(
    samples,
    model,
    processor,
    prompt: str = "请用中文简洁描述这张图片。",
    max_new_tokens: int = 64,
    infer_bs: int = 2,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    postprocess: bool = False,
):
    ## 备选prompt：请为商品图片生成精准中文描述：
    """批量生成图片描述，支持一次性批量推理以提速。

    参数：
    - samples：由数据加载器返回的样本列表，每个样本包含 `image` 与可选元信息。
    - model：已加载的多模态模型实例。
    - processor：与模型配套的处理器。
    - prompt：文本提示。
    返回：
    - 与输入样本一一对应的描述字符串列表。
    """
    if not samples:
        branch_log()
        return []
    # 若模型支持批量 generate，则一次性处理整批（显著减少 Python 循环与 I/O 开销）
    if hasattr(model, 'generate'):
        # branch_log()
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        # 加入生成前缀，避免模型回显输入
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        results = []
        # 推理设备选择：若可用则强制使用 CUDA，以避免 CPU 生成导致 0% GPU 利用率
        if torch.cuda.is_available():
            # branch_log()
            dev = torch.device('cuda')
        else:
            # 回退：尝试读取模型参数设备，否则使用 CPU
            try:
                dev = next(model.parameters()).device
            except Exception:
                except_log("infer device fallback to cpu")
                dev = torch.device('cpu')
        for start in tqdm.tqdm(range(0, len(samples), max(1, infer_bs)), desc="Captioning images, with ``generate``"):
            chunk = samples[start:start+max(1, infer_bs)]
            texts = [chat_text] * len(chunk)
            images = [s["image"] for s in chunk]
            inputs = processor(text=texts, images=images, return_tensors="pt")
            # 确保输入移动到推理设备，以避免在 CPU 上生成导致慢速与 0% GPU 利用率
            if dev is not None:
                # branch_log()
                inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            with torch.inference_mode():
                # 统一生成参数，加入终止标记与去重约束
                tok = getattr(processor, 'tokenizer', None)
                gen_kwargs = {
                    'max_new_tokens': int(max_new_tokens),
                    'do_sample': False,
                    # 'no_repeat_ngram_size': 3,
                    # 'repetition_penalty': 1.1,
                    # "num_beams": 2, 
                }
                # 仅在使用 beam search 时才启用 early_stopping，避免无效参数告警
                if int(gen_kwargs.get('num_beams', 1)) > 1:
                    branch_log()
                    gen_kwargs['early_stopping'] = True
                if tok is not None:
                    # branch_log()
                    if getattr(tok, 'eos_token_id', None) is not None:
                        # branch_log()
                        gen_kwargs['eos_token_id'] = tok.eos_token_id
                    if getattr(tok, 'pad_token_id', None) is not None:
                        # branch_log()
                        gen_kwargs['pad_token_id'] = tok.pad_token_id
                # 推理侧可使用 autocast 降显存
                # try:
                if use_amp and torch.cuda.is_available():
                    # branch_log()
                    if amp_dtype.lower() == "bf16":
                        # branch_log()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model.generate(**inputs, **gen_kwargs)
                    else:
                        # branch_log()
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model.generate(**inputs, **gen_kwargs)
                else:
                    # branch_log()
                    out = model.generate(**inputs, **gen_kwargs)
            # 仅解码生成的新 token，避免包含原始对话文本
            in_ids = inputs.get("input_ids")
            for bi in range(len(chunk)):
                gen_ids = out[bi][in_ids[bi].shape[0]:].detach().cpu()
                t = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
                results.append(str(t).strip())
        return results

    # 最终兜底：返回占位文案
    return ["这是一张电商商品图片。"] * len(samples)