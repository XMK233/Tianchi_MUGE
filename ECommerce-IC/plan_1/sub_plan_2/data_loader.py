"""
通用数据加载器：按起始行和行数从 TSV 加载图片，并在 JSONL 中匹配对应文本。

支持以下文件约定（无表头）：
- IC_train.tsv：训练集对应的图片（base64编码）
- IC_train.jsonl：训练集对应的caption/text
- IC_valid.tsv：验证集对应的图片（base64编码）
- IC_valid.jsonl：验证集对应的caption/text
- IC_test.tsv：测试集对应的图片（base64编码）
- IC_test.jsonl：需预测的文件，每条JSON仅给出 img_id，选手需要补充 "text" 字段

功能特性：
- 可指定从哪一行开始加载（start_line）以及加载多少行（num_lines）
- 先按指定范围加载 TSV 中的图片，再在 JSONL 中查找对应的文本（text/caption）
- 返回的数据易于传入大模型进行 LoRA SFT 或 infer

示例：
>>> from data_loader import load_ic_batch
>>> samples = load_ic_batch(
...     tsv_path="/path/to/IC_train.tsv",
...     jsonl_path="/path/to/IC_train.jsonl",
...     start_line=0,
...     num_lines=1024,
...     image_size=448,
... )
>>> samples[0]
{'image_id': 'xxx', 'image': <PIL.Image.Image ...>, 'text': '中文描述...'}

如需与 HF Processor 对接：
>>> from transformers import AutoProcessor
>>> processor = AutoProcessor.from_pretrained("/path/to/model", trust_remote_code=True)
>>> batch_inputs = default_collate(samples[:8], processor=processor)

"""

import base64
import io
import json
import logging
import os
from tqdm import tqdm

from PIL import Image

log = logging.getLogger("data_loader")

def _decode_base64_image(b64_str):
    try:
        raw = base64.b64decode(b64_str, validate=False)
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        return img
    except Exception as e:
        log.info("[DataLoader] try-block failed in _decode_base64_image")
        log.warning(f"Failed to decode base64 image: {e}")
        return None


def _resize_image(img, size):
    try:
        return img.resize((size, size), Image.BILINEAR)
    except Exception:
        log.info("[DataLoader] try-block failed in _resize_image")
        return img


def iter_tsv_images(tsv_path, start_line=0, num_lines=None, show_progress=True):
    """按行遍历 TSV（无表头）。

    TSV 每行格式：`img_id\t<base64>`
    仅遍历从 `start_line` 开始的 `num_lines` 行（若 None 则直到文件结束）。
    返回 (img_id, base64_str) 二元组迭代器。
    """
    assert start_line >= 0, "start_line 必须为非负整数"
    yielded = 0
    desc = f"TSV {os.path.basename(tsv_path)}"
    total = num_lines if isinstance(num_lines, int) else None
    pbar = tqdm(total=total, desc=desc, unit="img", disable=(not show_progress))
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if num_lines is not None and yielded >= num_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    img_id, b64 = line.split("\t", 1)
                    yield img_id, b64
                    yielded += 1
                    if show_progress:
                        pbar.update(1)
                except ValueError:
                    # 非法行，跳过
                    log.warning(f"Invalid line {i}: {line}")
                    continue
    finally:
        if show_progress:
            pbar.close()


def _build_caption_index(jsonl_path, target_ids=None, show_progress=True):
    """构建 img_id -> text/caption 的索引。
    若 target_ids 提供，则仅索引这些 id；否则索引全部。
    当某条 JSON 缺失文本（例如 test 集），映射值为 None。
    支持字段：`text` 优先，其次尝试 `caption`。
    """
    index = {}
    desc = f"JSONL {os.path.basename(jsonl_path)}"
    total = len(target_ids) if isinstance(target_ids, set) else None
    found = 0
    pbar = tqdm(total=total, desc=desc, unit="match", disable=(not show_progress))
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    log.info("[DataLoader] try-block failed in _build_caption_index: parse json line")
                    log.warning(f"Failed to parse JSON line: {line}")
                    continue
                img_id = obj.get("img_id") or obj.get("image_id")
                if not img_id:
                    continue
                if target_ids is not None and img_id not in target_ids:
                    continue
                text = obj.get("text")
                if text is None:
                    text = obj.get("caption")
                if img_id not in index:
                    index[img_id] = text
                    found += 1
                    if show_progress:
                        pbar.update(1)
                    if total is not None and found >= total:
                        break
    finally:
        if show_progress:
            pbar.close()
    return index


def load_ic_batch(tsv_path, jsonl_path, start_line=0, num_lines=None, image_size=448, show_progress=True):
    """加载指定范围的 TSV 图片，并匹配 JSONL 文本。

    返回列表，每个元素为：{"image_id": str, "image": PIL.Image, "text": Optional[str]}
    注意：对于 test 集，text 可能为 None。
    """
    # 先从 TSV 取指定范围的图片 id 与 base64
    pairs = list(iter_tsv_images(tsv_path, start_line=start_line, num_lines=num_lines, show_progress=show_progress))
    image_ids = {pid for pid, _ in pairs}

    # 仅针对选定的 id 构建文本索引，避免不必要内存占用
    cap_index = _build_caption_index(jsonl_path, target_ids=image_ids, show_progress=show_progress)

    samples = []
    for img_id, b64 in tqdm(pairs, desc="Loading images..."):
        img = _decode_base64_image(b64)
        if img is None:
            continue
        img = _resize_image(img, image_size)
        text = cap_index.get(img_id)
        samples.append({"image_id": img_id, "image": img, "text": text})
    return samples


def default_collate(samples, processor=None):
    """将 samples 打包为批次，支持直接与 HF Processor 对接。

    - 当提供 processor 时：返回 processor 的张量字典，例如包含 pixel_values、input_ids 等。
    - 当未提供 processor：返回简单的 lists 字典，方便自定义处理。
    """
    if not samples:
        return {}
    images = [s["image"] for s in samples]
    texts = [s.get("text") for s in samples]
    ids = [s.get("image_id") for s in samples]
    if processor is not None:
        # 若部分 text 为空（例如 test 集），以空串替代，避免 tokenizer 抛错；
        # 具体任务中也可以改为统一的指令模板。
        safe_texts = [t if isinstance(t, str) else "" for t in texts]
        try:
            inputs = processor(text=safe_texts, images=images, return_tensors="pt")
            inputs["image_ids"] = ids
            return inputs
        except Exception as e:
            log.info("[DataLoader] try-block failed in default_collate: processor(text, images)")
            log.warning(f"Processor collate failed, falling back to raw lists: {e}")
    return {"image_ids": ids, "images": images, "texts": texts}


class ICTsvJsonlDataset:
    """简易 Dataset 封装，便于与 PyTorch/DataLoader 结合。

    示例：
    >>> ds = ICTsvJsonlDataset(tsv_path, jsonl_path, start_line=0, num_lines=2048, image_size=336)
    >>> item = ds[0]
    {"image_id": "...", "image": <PIL.Image>, "text": "..."}
    """

    def __init__(self, tsv_path, jsonl_path, start_line=0, num_lines=None, image_size=448, show_progress=True):
        self.tsv_path = tsv_path
        self.jsonl_path = jsonl_path
        self.start_line = start_line
        self.num_lines = num_lines
        self.image_size = image_size
        self._pairs = list(iter_tsv_images(tsv_path, start_line=start_line, num_lines=num_lines, show_progress=show_progress))
        self._cap_index = _build_caption_index(jsonl_path, target_ids={pid for pid, _ in self._pairs}, show_progress=show_progress)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        img_id, b64 = self._pairs[idx]
        img = _decode_base64_image(b64)
        if img is None:
            raise IndexError(f"Invalid image at index {idx}")
        img = _resize_image(img, self.image_size)
        text = self._cap_index.get(img_id)
        return {"image_id": img_id, "image": img, "text": text}
