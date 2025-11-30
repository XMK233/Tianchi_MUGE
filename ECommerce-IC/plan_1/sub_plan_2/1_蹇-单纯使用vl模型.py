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

# 同目录下的数据加载器（直接通过路径导入，避免异常分支）
import sys
sys.path.append(os.path.dirname(__file__))
from data_loader import (
    load_ic_batch,
    ICTsvJsonlDataset,
)
import tqdm


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


def load_qwen_vl(local_dir: str):
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
    model = model_cls.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
        dtype='auto',
        device_map='auto',
        local_files_only=True,
    )
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
    chat_text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=chat_text, images=[image], return_tensors="pt")
    dev = getattr(model, 'device', None)
    if dev is not None:
        inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    if hasattr(model, 'generate'):
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        return str(decoded[0]).strip()

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


def caption_batch(samples, model, processor, prompt: str = "请用中文简洁描述这张图片。"):
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
        print("可批量。。。")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        chat_text = processor.apply_chat_template(messages, tokenize=False)
        texts = [chat_text] * len(samples)
        images = [s["image"] for s in samples]
        inputs = processor(text=texts, images=images, return_tensors="pt")
        dev = getattr(model, 'device', None)
        if dev is not None:
            inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        return [str(t).strip() for t in decoded]

    # 回退：逐张调用 chat 接口（某些实现不支持批量 generate）
    if hasattr(model, 'chat'):
        results = []
        tok = getattr(processor, 'tokenizer', None)
        for s in tqdm.tqdm(samples, desc="Captioning images"):
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
):
    """训练阶段模板（基线）：仅加载模型并进行暖启动推理，不进行训练。

    参数：
    - train_tsv：训练集 TSV 文件路径（图片 base64）。
    - train_jsonl：训练集 JSONL 文件路径（文本）。
    - rounds：训练轮数（按行分块加载）。
    - per_round_lines：每轮加载的样本行数。
    - image_size：图像缩放尺寸。
    - show_progress：是否显示加载进度条。
    - save_dir：占位的保存目录（不实际训练，仅确保存在）。
    - local_model_dir：本地模型目录（离线加载）。
    返回：
    - 无（通过日志输出暖启动推理示例）。
    """
    os.makedirs(save_dir or "./outputs", exist_ok=True)
    # 按用户要求：不做任何 SFT 或再次训练，仅作为 baseline。
    # 这里仅加载模型以确保后续验证/测试复用；训练阶段不进行参数更新。
    model, processor = load_qwen_vl(local_model_dir)
    for r in range(rounds):
        start = r * per_round_lines
        log.info(f"[Train] Round {r+1}/{rounds} | lines: {per_round_lines} @ start {start}")
        ds = ICTsvJsonlDataset(
            train_tsv,
            train_jsonl,
            start_line=start,
            num_lines=per_round_lines,
            image_size=image_size,
            show_progress=show_progress,
        )

        # 不训练：仅占位日志，验证模型可正常前向（选取少量样本运行推理）。
        warmup_n = min(4, len(ds))
        if warmup_n > 0:
            batch = [ds[i] for i in range(warmup_n)]
            preds = caption_batch(batch, model, processor)
            log.info(f"[Train] Warmup inference examples: {preds[:2]}")
        log.info(f"[Train] Baseline: no training performed on {len(ds)} samples")


# ---------------------------
# 验证模板：分批加载、占位生成文本、计算简化 CIDEr
# ---------------------------
def run_validation(
    valid_tsv: str,
    valid_jsonl: str,
    rounds: int,
    per_round_lines: int,
    image_size: int = 448,
    show_progress: bool = True,
    local_model_dir: str = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
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
    model, processor = load_qwen_vl(local_model_dir)
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
        preds = caption_batch(batch, model, processor)
        refs = [s.get("text") for s in batch]
        all_preds.extend(preds)
        all_refs.extend(refs)
        log.info(f"[Valid] processed {len(all_preds)} samples so far")

    cider = compute_cider(all_preds, all_refs)
    log.info(f"[Valid] Simplified CIDEr: {cider:.4f}")
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
    image_size: int = 448,
    show_progress: bool = True,
    local_model_dir: str = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
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

    model, processor = load_qwen_vl(local_model_dir)
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
            preds = caption_batch(batch, model, processor)
            for s, pred_text in zip(batch, preds):
                obj = {"img_id": s.get("image_id"), "text": pred_text}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
            log.info(f"[Test] wrote {count} predictions so far")

    log.info(f"[Test] saved predictions to {output_jsonl} (total {count})")


if __name__ == "__main__":
    version_symb = "v1"

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

    # 1) 训练：按轮次加载且不重叠（Baseline：不进行任何训练，仅加载与暖启动推理）
    run_training_rounds(
        train_tsv=train_tsv,
        train_jsonl=train_jsonl,
        rounds=1,
        per_round_lines=10,
        image_size=IMAGE_SIZE,
        show_progress=True,
        save_dir=os.path.join(base_dir, "outputs"),
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    )
    print("\n\n")

    # 2) 验证：按轮次/每轮行数加载，计算简化版 CIDEr
    cider = run_validation(
        valid_tsv=valid_tsv,
        valid_jsonl=valid_jsonl,
        rounds=1,
        per_round_lines=10,
        image_size=IMAGE_SIZE,
        show_progress=True,
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    )
    log.info(f"Validation (simplified) CIDEr: {cider:.4f}")
    print("\n\n")

    # 3) 测试：分批生成占位文本，存为 example_pred.jsonl 风格
    output_jsonl = os.path.join(base_dir, f"{version_symb}.jsonl")
    run_test(
        test_tsv=test_tsv,
        test_jsonl=test_jsonl,
        rounds=1,
        per_round_lines=10,
        output_jsonl=output_jsonl,
        image_size=IMAGE_SIZE,
        show_progress=True,
        local_model_dir="/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct",
    )
    print("\n\n")
