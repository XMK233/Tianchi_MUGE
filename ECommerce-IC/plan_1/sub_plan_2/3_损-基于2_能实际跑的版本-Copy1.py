"""
训练/验证/测试流水线模板（pipeline_template）

功能概述：
- 使用同目录下的 `data_loader.py` 加载 TSV + JSONL 数据（无表头），支持分轮次/分批次，保证各轮不重叠。
- 训练部分：按轮次加载固定数量的训练样本，留出模型训练与保存的位置（占位，不写具体训练）。
- 验证部分：按批次加载验证集，生成占位中文描述（随机），与标准答案计算 CIDEr 分数（内置简化版 CIDEr）。
- 测试部分：按批次加载测试集，生成占位中文描述，保存 JSONL 结果，格式类似 `example_pred.jsonl`。

使用说明（示例）：
>>> python -m ECommerce-IC.plan_1.sub_plan_2.pipeline_template 
或在代码中调用下述函数。
"""

import os
import json
import random
import logging
import warnings
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

def free_torch_memory():
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            log.info(f"[GPU{i}] after cleanup used={mem.used/1024**2:.2f} MB / total={mem.total/1024**2:.2f} MB")
    except Exception:
        pass


log = logging.getLogger("pipeline_template")

# 引入评测指标：CIDEr（近似实现，见 metrics.py）—直接路径导入
from metrics import compute_cider


# ---------------------------
# 文本生成占位：随机中文字符串
# ---------------------------
def random_chinese_text(min_len: int = 8, max_len: int = 20) -> str:
    """生成一段随机中文文本。

    参数：
    - min_len：最短字符数。
    - max_len：最长字符数。
    返回：
    - 随机中文字符串。
    """
    length = random.randint(min_len, max_len)
    chars = []
    for _ in range(length):
        # 常用 CJK 统一表意文字范围（不严格），随机采样
        code = random.randint(0x4E00, 0x9FA5)
        chars.append(chr(code))
    return "".join(chars)


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
        return local_dir

    snap_root = os.path.join(local_dir, "snapshots")
    if os.path.isdir(snap_root):
        candidates = []
        for name in os.listdir(snap_root):
            p = os.path.join(snap_root, name)
            if os.path.isdir(p) and _has_config(p):
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]

    # 递归搜索 config.json
    for dirpath, dirnames, filenames in os.walk(local_dir):
        if "config.json" in filenames:
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
        # 优先离线模式，避免任何网络请求
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 允许 TF32 提速（Ampere+ GPU 上对矩阵乘更快，保持数值稳定）
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

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
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            dtype='auto',
            device_map=None,
            local_files_only=True,
        )
        # 训练侧关闭缓存并开启梯度检查点以降低显存
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = False
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"[Warn] enabling gradient checkpointing/use_cache failed: {e}")
    else:
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            dtype='auto',
            device_map=None,  # 避免自动切到 CPU，统一由我们手动放到 CUDA
            local_files_only=True,
        )
        # 推理场景尽量将整模移动到 CUDA，以确保真正使用 GPU 进行计算
        try:
            if torch.cuda.is_available():
                model.to(torch.device('cuda'))
                # 记录主参数设备，便于诊断 GPU 未被利用的问题
                try:
                    dev = next(model.parameters()).device
                    log.info(f"[Infer] model moved to device: {dev}")
                except Exception:
                    pass
        except Exception as e:
            log.info(f"[Infer] model.to(cuda) skipped: {e}")
        # 推理侧启用缓存与更快的注意力实现（如可用则使用 FlashAttention2，否则回退到 SDPA）
        try:
            if hasattr(model, 'config'):
                model.config.use_cache = True
                if is_flash_attn_2_available():
                    setattr(model.config, 'attn_implementation', 'flash_attention_2')
                else:
                    setattr(model.config, 'attn_implementation', 'sdpa')
        except Exception as e:
            print(f"[Warn] setting attn implementation/use_cache failed: {e}")
    model.eval()
    return model, processor


def _caption_one_image(image, model, processor, prompt: str = "请用中文简洁描述这张图片。") -> str:
    """对单张图片生成中文描述。

    参数：
    - image：PIL.Image 或 numpy 数组形式的图像。
    - model：已加载的多模态模型实例。
    - processor：与模型配套的处理器，用于构造输入与解码输出。
    - prompt：文本提示，指导生成的风格与内容。
    返回：
    - 生成的中文描述字符串；必要时返回兜底占位文案。
    """
    # 1) 使用官方聊天模板，确保在文本中插入图像占位符（如 <|image_1|>）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # 不直接 tokenize，这里让 processor 统一打包 text 与 images
    # 加入生成前缀，避免模型回显输入
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat_text, images=[image], return_tensors="pt")
    dev = getattr(model, 'device', None)
    if dev is not None:
        inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    if hasattr(model, 'generate'):
        with torch.inference_mode():
            tok = getattr(processor, 'tokenizer', None)
            gen_kwargs = {
                'max_new_tokens': 64,
                'do_sample': False,
                'no_repeat_ngram_size': 3,
                'repetition_penalty': 1.1,
                'early_stopping': True,
            }
            if tok is not None:
                if getattr(tok, 'eos_token_id', None) is not None:
                    gen_kwargs['eos_token_id'] = tok.eos_token_id
                if getattr(tok, 'pad_token_id', None) is not None:
                    gen_kwargs['pad_token_id'] = tok.pad_token_id
            out = model.generate(**inputs, **gen_kwargs)
        # 仅解码生成的新 token，避免包含原始对话文本
        in_ids = inputs.get("input_ids")
        if in_ids is not None:
            gen_ids = out[0][in_ids[0].shape[0]:].detach().cpu()
            text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
            text = _sanitize_text(text, tok, 64)
            return str(text).strip()
        else:
            decoded = processor.batch_decode(out, skip_special_tokens=True)
            t = _sanitize_text(decoded[0], tok, 64)
            return str(t).strip()

    # 2) 回退：兼容不同 chat 签名
    if hasattr(model, 'chat'):
        tok = getattr(processor, 'tokenizer', None)
        if tok is not None:
            resp = model.chat(tok, query=prompt, images=[image], history=None)
            if isinstance(resp, (tuple, list)):
                return str(resp[0]).strip()
            return str(resp).strip()
        else:
            resp = model.chat(query=prompt, images=[image])
            if isinstance(resp, (tuple, list)):
                return str(resp[0]).strip()
            return str(resp).strip()

    # 3) 最终兜底占位，避免崩溃
    return "这是一张电商商品图片。"


def _sanitize_text(text: str, tokenizer=None, max_new_tokens: int | None = None) -> str:
    """轻量后处理：去除首尾多余标点/空白、简单去重片段、长度兜底约束。

    - 去除首尾中文/英文常见标点和空白；
    - 按中文逗号分段，移除连续重复片段（保留顺序）；
    - 若提供 tokenizer 与 max_new_tokens，则确保不超过生成 token 上限（再decode）。
    """
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    # 去掉首尾标点
    strip_chars = "，。！？；：,.!?:;、~··…—-\u3000\t\r\n"
    s = s.strip(strip_chars)
    # 去重片段（按中文逗号拆分）
    parts = [p.strip(strip_chars) for p in s.split("，")]
    dedup = []
    seen = set()
    for p in parts:
        if not p:
            continue
        key = p
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    s = "，".join(dedup) if dedup else text.strip(strip_chars)
    # 兜底长度约束：若实际生成超过 max_new_tokens（以 tokenizer 为准），强制截断
    if tokenizer is not None and isinstance(max_new_tokens, int) and max_new_tokens > 0:
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if len(toks) > max_new_tokens:
                toks = toks[:max_new_tokens]
                s = tokenizer.decode(toks, skip_special_tokens=True).strip(strip_chars)
        except Exception:
            pass
    return s


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
        return []
    # # 逐张生成，避免不同模型对批量多图的接口差异
    # rst = []
    # for s in tqdm.tqdm(samples, desc="Captioning images"):
    #     rst.append(_caption_one_image(s["image"], model, processor, prompt=prompt))
    # return rst
    # 若模型支持批量 generate，则一次性处理整批（显著减少 Python 循环与 I/O 开销）
    if hasattr(model, 'generate'):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        # 加入生成前缀，避免模型回显输入
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        results = []
        # 推理设备选择：若可用则强制使用 CUDA，以避免 CPU 生成导致 0% GPU 利用率
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            # 回退：尝试读取模型参数设备，否则使用 CPU
            try:
                dev = next(model.parameters()).device
            except Exception:
                dev = torch.device('cpu')
        for start in tqdm.tqdm(range(0, len(samples), max(1, infer_bs)), desc="Captioning images, with ``generate``"):
            chunk = samples[start:start+max(1, infer_bs)]
            texts = [chat_text] * len(chunk)
            images = [s["image"] for s in chunk]
            inputs = processor(text=texts, images=images, return_tensors="pt")
            # 确保输入移动到推理设备，以避免在 CPU 上生成导致慢速与 0% GPU 利用率
            if dev is not None:
                inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            with torch.inference_mode():
                # 统一生成参数，加入终止标记与去重约束
                tok = getattr(processor, 'tokenizer', None)
                gen_kwargs = {
                    'max_new_tokens': int(max_new_tokens),
                    'do_sample': False,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 1.1,
                }
                # 仅在使用 beam search 时才启用 early_stopping，避免无效参数告警
                if int(gen_kwargs.get('num_beams', 1)) > 1:
                    gen_kwargs['early_stopping'] = True
                if tok is not None:
                    if getattr(tok, 'eos_token_id', None) is not None:
                        gen_kwargs['eos_token_id'] = tok.eos_token_id
                    if getattr(tok, 'pad_token_id', None) is not None:
                        gen_kwargs['pad_token_id'] = tok.pad_token_id
                # 推理侧可使用 autocast 降显存
                try:
                    if use_amp and torch.cuda.is_available():
                        if amp_dtype.lower() == "bf16":
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = model.generate(**inputs, **gen_kwargs)
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = model.generate(**inputs, **gen_kwargs)
                    else:
                        out = model.generate(**inputs, **gen_kwargs)
                except RuntimeError as e:
                    # OOM 兜底：减小批次或生成长度，逐个生成
                    log.warning(f"[Infer] batch generate failed: {e}. Falling back to per-sample generation with shorter length.")
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    out = []
                    fb_kwargs = dict(gen_kwargs)
                    fb_kwargs['max_new_tokens'] = max(8, min(32, int(gen_kwargs.get('max_new_tokens', 64))))
                    for bi in range(len(chunk)):
                        one_inputs = {k: (v[bi:bi+1] if hasattr(v, 'shape') else v) for k, v in inputs.items()}
                        if use_amp and torch.cuda.is_available():
                            if amp_dtype.lower() == "bf16":
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    o = model.generate(**one_inputs, **fb_kwargs)
                            else:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    o = model.generate(**one_inputs, **fb_kwargs)
                        else:
                            o = model.generate(**one_inputs, **fb_kwargs)
                        out.append(o[0])
            # 仅解码生成的新 token，避免包含原始对话文本
            in_ids = inputs.get("input_ids")
            if in_ids is not None:
                for bi in range(len(chunk)):
                    gen_ids = out[bi][in_ids[bi].shape[0]:].detach().cpu()
                    t = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
                    if postprocess:
                        t = _sanitize_text(t, getattr(processor, 'tokenizer', None), max_new_tokens)
                    results.append(str(t).strip())
            else:
                decoded = processor.batch_decode(out, skip_special_tokens=True)
                for t in decoded:
                    if postprocess:
                        t = _sanitize_text(t, getattr(processor, 'tokenizer', None), max_new_tokens)
                    results.append(str(t).strip())
        return results

    # 回退：逐张调用 chat 接口（某些实现不支持批量 generate）
    if hasattr(model, 'chat'):
        results = []
        tok = getattr(processor, 'tokenizer', None)
        for s in tqdm.tqdm(samples, desc="Captioning images, with ``chat``"):
            img = s["image"]
            if tok is not None:
                resp = model.chat(tok, query=prompt, images=[img], history=None)
            else:
                resp = model.chat(query=prompt, images=[img])
            text = resp[0] if isinstance(resp, (tuple, list)) else resp
            results.append(str(text).strip())
        return results

    # 最终兜底：返回占位文案
    return ["这是一张电商商品图片。"] * len(samples)

# ---------------------------
# 训练专用：展开每图的多条描述为独立样本
# ---------------------------
def _expand_multi_text_samples(samples: list[dict]) -> list[dict]:
    """将每张图片的多条文本描述展开为多个样本。

    输入样本形如：{"image_id": str, "image": PIL.Image, "text": Union[str, list[str], None]}
    输出样本形如：{"image_id": str, "image": PIL.Image, "text": str}

    - 当 text 为 list[str] 时：为同一张图的每条文本生成一个独立样本。
    - 当 text 为 str 时：保持为单一样本。
    - 当 text 缺失或为空：丢弃该图（训练阶段需要监督信号）。
    """
    expanded = []
    for s in samples:
        img_id = s.get("image_id")
        img = s.get("image")
        text = s.get("text")
        if isinstance(text, list):
            for t in text:
                if isinstance(t, str) and len(t.strip()) > 0:
                    expanded.append({"image_id": img_id, "image": img, "text": t.strip()})
        elif isinstance(text, str) and len(text.strip()) > 0:
            expanded.append({"image_id": img_id, "image": img, "text": text.strip()})
        else:
            # 无文本，训练阶段跳过
            continue
    return expanded

# ---------------------------
# 训练模板：分轮加载、占位训练与保存
# ---------------------------
def run_training_rounds(
    train_tsv: str,
    train_jsonl: str,
    rounds: int,
    per_round_lines: int,
    image_size: int = 448,
    show_progress: bool = True,
    save_dir=None,
    local_model_dir: str = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    train_bs: int = 2,
    lr: float = 5e-5,
    epochs: int = 1,
    use_amp: bool = True,
    amp_dtype: str = "bf16",  # 可选: 'bf16' 或 'fp16'
    use_deepspeed: bool = True,
    deepspeed_stage: int = 2,
    gradient_accumulation_steps: int = 1,
):
    """用 LoRA 做真实 SFT 训练并保存适配器。

    参数：
    - train_tsv：训练集 TSV 文件路径（图片 base64）。
    - train_jsonl：训练集 JSONL 文件路径（文本）。
    - rounds：训练轮数（按行分块加载）。
    - per_round_lines：每轮加载的样本行数。
    - image_size：图像缩放尺寸。
    - show_progress：是否显示加载进度条。
    - save_dir：LoRA 适配器保存目录（会创建）。
    - local_model_dir：本地模型目录（离线加载）。
    - lora_r/alpha/dropout：LoRA 超参数。
    - train_bs：微批大小。
    - lr：学习率。
    - epochs：每轮的 epoch 数。
    返回：
    - 无（在 save_dir 写出 LoRA 适配器）。
    """
    assert isinstance(train_bs, int) and train_bs > 0
    os.makedirs(save_dir or "./outputs", exist_ok=True)

    # 1) 加载基础 VL 模型与处理器
    base_model, processor = load_qwen_vl(local_model_dir, for_training=True)

    # 2) 注入 LoRA 适配器（覆盖常见的注意力/MLP投影层）
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lconf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lconf)
    model.train()

    # 统计 LoRA 可训练参数与内存占用
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    trainable_mb = trainable_bytes / (1024 ** 2)
    # 粗略估算训练期显存（权重+梯度+Adam状态），约 4× 参数大小（不同 dtype/实现会有差异）
    est_train_mem_mb = trainable_mb * 4.0
    log.info(
        f"[LoRA] 可训练参数: {trainable_params:,} | 参数内存约: {trainable_mb:.2f} MB | 训练期显存估算: ~{est_train_mem_mb:.2f} MB"
    )
    # 当前 GPU 显存占用（如可用）
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            alloc_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            reserv_mb = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            log.info(f"[GPU{idx}] allocated={alloc_mb:.2f} MB | reserved={reserv_mb:.2f} MB")

    # 3) 优化器与（可选）DeepSpeed 初始化
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if use_deepspeed:
        ds_config = {
            "train_batch_size": max(1, train_bs * gradient_accumulation_steps),
            "train_micro_batch_size_per_gpu": max(1, train_bs),
            "gradient_accumulation_steps": max(1, gradient_accumulation_steps),
            "zero_optimization": {
                "stage": int(deepspeed_stage),
                "offload_param": {"device": "none"},
                "offload_optimizer": {"device": "none"},
            },
            "bf16": {"enabled": amp_dtype.lower() == "bf16"},
            "fp16": {"enabled": amp_dtype.lower() == "fp16", "initial_scale_power": 16},
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
        # 打印 DeepSpeed 初始化状态
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        log.info(f"[DS] initialized with stage={deepspeed_stage}, world_size={world_size}, micro_bs={train_bs}, grad_acc={gradient_accumulation_steps}")
    else:
        model_engine = None

    # 4) 训练循环（分轮按需加载，过滤无文本样本）
    device = (model_engine.device if model_engine is not None else next(model.parameters()).device)
    prompt = "请为商品图片生成精准中文描述："
    global_step = 0

    for r in range(rounds):
        start = r * per_round_lines
        log.info(f"[Train] Round {r+1}/{rounds} | lines: {per_round_lines} @ start {start}")

        # 加载为列表以避免对 Dataset 进行切片（其不支持 slice）并自动跳过无效图片
        samples = load_ic_batch(
            train_tsv,
            train_jsonl,
            start_line=start,
            num_lines=per_round_lines,
            image_size=image_size,
            show_progress=show_progress,
        )
        # 展开每图的多条描述为独立训练样本，并记录该轮最终样本量
        expanded_samples = _expand_multi_text_samples(samples)
        round_sample_count = len(expanded_samples)
        log.info(f"[Train] Round {r+1}: loaded {len(samples)} images -> expanded to {round_sample_count} samples")
        if round_sample_count == 0:
            log.warning(f"[Train] Round {r+1}: no usable samples after expansion")
            continue
        # # 过滤缺失文本的样本
        # samples = [s for s in samples if isinstance(s.get("text"), str) and len(s.get("text", "")) > 0]
        # if not samples:
        #     log.info(f"[Train] Round {r+1}: no usable samples (missing text)")
        #     continue

        for epoch in range(epochs):
            log.info(f"[Train] Epoch {epoch+1}/{epochs} on {round_sample_count} samples")
            for i in tqdm.tqdm(range(0, round_sample_count, train_bs), desc=f"Epoch {epoch+1}/{epochs}"):
                batch = expanded_samples[i:i+train_bs]
                if not batch:
                    continue
                images = [s["image"] for s in batch]
                targets = [s.get("text") or "" for s in batch]

                texts = []
                for tgt in targets:
                    messages = [
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                        {"role": "assistant", "content": [{"type": "text", "text": tgt}]},
                    ]
                    t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    texts.append(t)

                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()
                # 仅监督 assistant 段落：计算用户段长度并屏蔽其之前的 label
                for j, img in enumerate(images):
                    msg_user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                    txt_user = processor.apply_chat_template(msg_user, tokenize=False, add_generation_prompt=False)
                    ids_user = processor(text=txt_user, images=[img], return_tensors="pt")["input_ids"][0].to(device)
                    user_len = ids_user.shape[0]
                    seq_len = labels[j].shape[0]
                    # 屏蔽用户段与 padding
                    labels[j, :min(user_len, seq_len)] = -100
                if "attention_mask" in inputs:
                    labels[inputs["attention_mask"] == 0] = -100
                inputs["labels"] = labels

                if model_engine is None:
                    optimizer.zero_grad(set_to_none=True)
                    if use_amp and device.type == "cuda":
                        if amp_dtype.lower() == "bf16":
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                out = model(**inputs)
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = model(**inputs)
                    else:
                        out = model(**inputs)
                    loss = out.loss
                    loss.backward()
                    optimizer.step()
                else:
                    # DeepSpeed 引擎前向/反向/步进
                    # DeepSpeed 路径按照 DS 配置控制混合精度，不再套 torch.autocast，避免双重设置
                    out = model_engine(**inputs)
                    loss = out.loss
                    model_engine.backward(loss)
                    model_engine.step()
                global_step += 1
                # if global_step % 10 == 0:
            log.info(f"[Train] step={global_step}, loss={loss.item():.4f}")

        # 该轮结束可做一次小样例推理检查
        warmup_n = min(2, len(samples))
        infer_model = (model_engine.module if model_engine is not None else model)
        # 切到 eval 模式，避免训练态下的随机性与回显
        infer_model.eval()
        # Warmup 推理：显式限制生成长度、开启轻量后处理去重与去噪
        preds = caption_batch(
            samples[:warmup_n],
            infer_model,
            processor,
            prompt="请为商品图片生成精准中文描述：",
            max_new_tokens=64,
            infer_bs=2,
            use_amp=True,
            amp_dtype="bf16",
            postprocess=True,
        )
        log.info(f"[Train] Warmup inference (LoRA): {preds}")

        # 保存样例图片并生成简易 HTML 预览（提取到 preview_utils.save_warmup_preview）
        html_path = save_warmup_preview(samples[:warmup_n], preds, save_dir, round_index=r+1)
        if html_path is None:
            log.warning("[Train] Warmup preview failed to build.")
        # 每轮结束时输出最终样本总量
        log.info(f"[Train] Round {r+1} final sample count: {round_sample_count}")

        # 每轮结束：尽可能清理缓存与临时对象，恢复训练模式
        try:
            # 重置为训练模式（避免上一轮 warmup 将模型留在 eval 态）
            if model_engine is not None:
                model_engine.train()
            else:
                model.train()
            # 清理推理侧的 KV 缓存（部分实现提供）
            infer_mod = (model_engine.module if model_engine is not None else model)
            if hasattr(infer_mod, "clear_kv_cache"):
                try:
                    infer_mod.clear_kv_cache()
                except Exception:
                    pass
        except Exception:
            pass
        # 删除本轮的临时变量引用，利于 GC 释放显存
        for name in [
            'expanded_samples', 'samples', 'preds', 'inputs', 'labels', 'out', 'images', 'targets', 'texts'
        ]:
            try:
                del locals()[name]
            except Exception:
                pass
        # 强制进行一次 CUDA 同步与显存清理
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        free_torch_memory()

    # 5) 保存 LoRA 适配器
    assert save_dir is not None and len(str(save_dir)) > 0
    save_target = (model_engine.module if model_engine is not None else model)
    save_target.save_pretrained(save_dir)
    log.info(f"[Train] LoRA adapter saved to: {save_dir}")

    # 训练资源释放：删除大型对象并清理显存，防止与验证/测试相互影响
    try:
        del samples
    except Exception:
        pass
    try:
        del model_engine
    except Exception:
        pass
    try:
        del model
        del base_model
        del processor
    except Exception:
        pass
    free_torch_memory()


# ---------------------------
# 验证模板：分批加载、占位生成文本、计算简化 CIDEr
# ---------------------------
def run_validation(
    valid_tsv: str,
    valid_jsonl: str,
    rounds: int,
    per_round_lines: int,
    image_size: int = 224,
    show_progress: bool = True,
    local_model_dir: str = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    lora_dir: str | None = None,

    infer_bs: int = 2,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    max_new_tokens: int = 64,
    model: torch.nn.Module | None = None,
    processor: AutoProcessor | None = None,
) -> float:
    """验证阶段模板：生成中文描述并计算简化版 CIDEr。

    参数：
    - valid_tsv：验证集 TSV 文件路径。
    - valid_jsonl：验证集 JSONL 文件路径。
    - rounds：验证轮数（按行分块加载）。
    - per_round_lines：每轮加载的样本行数。
    - image_size：图像缩放尺寸。
    - show_progress：是否显示加载进度条。
    - local_model_dir：本地模型目录（离线加载）。
    返回：
    - 简化版 CIDEr 分数（float）。
    """
    # 要求外部传入已加载的模型/处理器以复用（避免重复加载）
    if model is None or processor is None:
        raise ValueError("run_validation 需要传入已加载的 model 和 processor；请在外部完成一次性加载并复用。")
    # 显式将模型放到 CUDA（如果可用），避免 CPU 上的推理导致 0% GPU 利用率
    try:
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
            model.eval()
            try:
                dev = next(model.parameters()).device
                log.info(f"[Valid] Model device after to(cuda): {dev}")
            except Exception:
                pass
    except Exception as e:
        log.info(f"[Valid] model.to(cuda) skipped: {e}")
    all_preds = []
    all_refs = []
    for r in range(rounds):
        start = r * per_round_lines
        log.info(f"[Valid] Round {r+1}/{rounds} | lines: {per_round_lines} @ start {start}")
        batch = load_ic_batch(
            valid_tsv,
            valid_jsonl,
            start_line=start,
            num_lines=per_round_lines,
            image_size=image_size,
            show_progress=show_progress,
        )
        if not batch:
            log.info(f"[Valid] Round {r+1}: empty batch, stopping early")
            break
        preds = caption_batch(batch, model, processor, max_new_tokens=max_new_tokens, infer_bs=infer_bs, use_amp=use_amp, amp_dtype=amp_dtype)
        refs = [s.get("text") for s in batch]
        all_preds.extend(preds)
        all_refs.extend(refs)
        log.info(f"[Valid] processed {len(all_preds)} samples so far")

        # 每轮验证结束后清理缓存与显存
        try:
            infer_mod = getattr(model, 'module', model)
            if hasattr(infer_mod, 'clear_kv_cache'):
                try:
                    infer_mod.clear_kv_cache()
                except Exception:
                    pass
        except Exception:
            pass
        for name in ['batch', 'preds', 'refs']:
            try:
                del locals()[name]
            except Exception:
                pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        free_torch_memory()

    cider = compute_cider(all_preds, all_refs)
    log.info(f"[Valid] Simplified CIDEr: {cider:.4f}")

    # 不释放外部传入的模型/处理器，仅清理缓存
    free_torch_memory()
    return cider


# ---------------------------
# 测试模板：分批加载、占位生成文本、输出 JSONL（example_pred.jsonl）
# ---------------------------
def run_test(
    test_tsv: str,
    test_jsonl: str,
    rounds: int,
    per_round_lines: int,
    output_jsonl: str,
    image_size: int = 224,
    show_progress: bool = True,
    local_model_dir: str = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    lora_dir: str | None = None,

    infer_bs: int = 2,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    max_new_tokens: int = 64,
    model: torch.nn.Module | None = None,
    processor: AutoProcessor | None = None,
):
    """测试阶段模板：生成中文描述并输出为 JSONL 文件。

    参数：
    - test_tsv：测试集 TSV 文件路径。
    - test_jsonl：测试集 JSONL 文件路径。
    - rounds：测试轮数（按行分块加载）。
    - per_round_lines：每轮加载的样本行数。
    - output_jsonl：预测结果输出 JSONL 路径。
    - image_size：图像缩放尺寸。
    - show_progress：是否显示加载进度条。
    - local_model_dir：本地模型目录（离线加载）。
    返回：
    - 无（在指定路径写入预测结果，每行包含 `img_id` 与 `text`）。
    """
    # 确保输出目录存在
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    os.makedirs(out_dir or ".", exist_ok=True)
    print(f"[Test] output_jsonl: {output_jsonl}")

    # 要求外部传入已加载的模型/处理器以复用（避免重复加载）
    if model is None or processor is None:
        raise ValueError("run_test 需要传入已加载的 model 和 processor；请在外部完成一次性加载并复用。")
    # 显式将模型放到 CUDA（如果可用），避免 CPU 上的推理导致 0% GPU 利用率
    try:
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
            model.eval()
            try:
                dev = next(model.parameters()).device
                log.info(f"[Test] Model device after to(cuda): {dev}")
            except Exception:
                pass
    except Exception as e:
        log.info(f"[Test] model.to(cuda) skipped: {e}")
    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in range(rounds):
            start = r * per_round_lines
            log.info(f"[Test] Round {r+1}/{rounds} | lines: {per_round_lines} @ start {start}")
            batch = load_ic_batch(
                test_tsv,
                test_jsonl,
                start_line=start,
                num_lines=per_round_lines,
                image_size=image_size,
                show_progress=show_progress,
            )
            if not batch:
                log.info(f"[Test] Round {r+1}: empty batch, stopping early")
                break
            preds = caption_batch(batch, model, processor, max_new_tokens=max_new_tokens, infer_bs=infer_bs, use_amp=use_amp, amp_dtype=amp_dtype)
            for s, pred_text in zip(batch, preds):
                obj = {"img_id": s.get("image_id"), "text": pred_text}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
            log.info(f"[Test] wrote {count} predictions so far")

            # 每轮测试结束后清理缓存与显存
            try:
                infer_mod = getattr(model, 'module', model)
                if hasattr(infer_mod, 'clear_kv_cache'):
                    try:
                        infer_mod.clear_kv_cache()
                    except Exception:
                        pass
            except Exception:
                pass
            for name in ['batch', 'preds']:
                try:
                    del locals()[name]
                except Exception:
                    pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            free_torch_memory()

    log.info(f"[Test] saved predictions to {output_jsonl} (total {count})")

    # 不释放外部传入的模型/处理器，仅进行一次显存清理
    free_torch_memory()


if __name__ == "__main__":
    version_symb = "v3"

    logging.basicConfig(level=logging.INFO)

    # 这里给出一个模板式的调用示例。实际路径请根据你的数据位置替换。
    base_dir = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-IC/"
    train_tsv = os.path.join(base_dir, "IC_train.tsv")
    train_jsonl = os.path.join(base_dir, "IC_train.jsonl")
    valid_tsv = os.path.join(base_dir, "IC_valid.tsv")
    valid_jsonl = os.path.join(base_dir, "IC_valid.jsonl")
    test_tsv = os.path.join(base_dir, "IC_test.tsv")
    test_jsonl = os.path.join(base_dir, "IC_test.jsonl")

    IMAGE_SIZE = 224
    # ROUNDS = 1

    # 设置 LoRA 适配器保存地址（用于后续验证/测试加载）
    lora_save_dir = os.path.join(base_dir, f"outputs_{version_symb}_lora")

    # 1) 训练：按轮次加载且不重叠，使用 LoRA 做真实 SFT
    run_training_rounds(
        train_tsv=train_tsv,
        train_jsonl=train_jsonl,
        rounds=1,
        per_round_lines=100,
        image_size=IMAGE_SIZE,
        show_progress=True,
        save_dir=lora_save_dir,
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
        
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        train_bs=2,
        lr=5e-5,
        epochs=1,
    )
    print("\n\n")

    # 2) 验证：按轮次/每轮行数加载，计算简化版 CIDEr
    # 提前加载并复用模型与处理器，避免重复初始化耗时
    # 一次性加载并优化推理模型与处理器，后续验证/测试复用以避免重复加载
    base_model, base_processor = load_qwen_vl(
        "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
        for_training=False,
    )
    infer_model = PeftModel.from_pretrained(base_model, lora_save_dir) if os.path.isdir(lora_save_dir) else base_model
    try:
        if torch.cuda.is_available():
            infer_model.to(torch.device('cuda'))
            infer_model.eval()
            try:
                dev = next(infer_model.parameters()).device
                log.info(f"[Entry] Inference model device: {dev}")
            except Exception:
                pass
    except Exception as e:
        log.info(f"[Entry] infer_model.to(cuda) skipped: {e}")

    # 清理训练阶段显存，避免与验证共享导致崩溃
    free_torch_memory()

    cider = run_validation(
        valid_tsv=valid_tsv,
        valid_jsonl=valid_jsonl,
        rounds=1,
        per_round_lines=100,
        image_size=IMAGE_SIZE,
        show_progress=True,
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
        lora_dir=lora_save_dir,
        # 推理加速参数（可按显存情况调整）
        infer_bs=4,
        use_amp=True,
        amp_dtype="bf16",
        max_new_tokens=48,
        # 复用已加载模型与处理器，避免重复加载
        model=infer_model,
        processor=base_processor,
    )
    log.info(f"Validation (simplified) CIDEr: {cider:.4f}")
    print("\n\n")

    # 3) 测试：分批生成占位文本，存为 example_pred.jsonl 风格
    output_jsonl = os.path.join(base_dir, f"{version_symb}.jsonl")
    # 清理验证阶段显存，切换到测试
    free_torch_memory()

    run_test(
        test_tsv=test_tsv,
        test_jsonl=test_jsonl,
        rounds=1,
        per_round_lines=100,
        output_jsonl=output_jsonl,
        image_size=IMAGE_SIZE,
        show_progress=True,
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
        lora_dir=lora_save_dir,
        # 推理加速参数（与验证保持一致）
        infer_bs=4,
        use_amp=True,
        amp_dtype="bf16",
        max_new_tokens=48,
        # 复用已加载模型与处理器，避免重复加载
        model=infer_model,
        processor=base_processor,
    )
    print("\n\n")
