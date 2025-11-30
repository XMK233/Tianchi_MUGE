import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Iterator, Dict, Optional, List, Set

from PIL import Image
from torch.utils.data import IterableDataset
import time

log = logging.getLogger("data_utils")


def decode_image_b64(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


@dataclass
class Sample:
    image_id: str
    caption: Optional[str]
    image: Image.Image


class CaptionIterableDataset(IterableDataset):
    def __init__(
        self,
        captions_jsonl: str,
        images_tsv: str,
        image_size: int = 448,
        max_samples: Optional[int] = None,
        start_index: int = 0,
        resolver: Optional["ImageTSVIndex"] = None,
        target_image_ids: Optional[List[str]] = None,
        caption_index: Optional["CaptionIndex"] = None,
    ):
        self.captions_jsonl = captions_jsonl
        self.images_tsv = images_tsv
        self.image_size = image_size
        self.max_samples = max_samples
        self.start_index = max(0, int(start_index))
        self._resolver = resolver or ImageTSVIndex(images_tsv)
        self._target_image_ids = target_image_ids
        self._caption_index = caption_index

    def __iter__(self) -> Iterator[Sample]:
        count = 0
        yielded = 0
        captioned = 0
        # 若传入 target_image_ids，则改为按图片驱动迭代，再查找对应 caption
        if self._target_image_ids is not None:
            total_targets = len(self._target_image_ids)
            log.info(f"Dataset iter start (images-first): targets={total_targets}, image_size={self.image_size}")
            for idx, image_id in enumerate(self._target_image_ids):
                if self.max_samples is not None and yielded >= self.max_samples:
                    break
                img = self._resolver.get_image(image_id)
                if img is None:
                    continue
                caption = None
                if self._caption_index is not None:
                    caption = self._caption_index.get_caption(image_id)
                yielded += 1
                if caption is not None:
                    captioned += 1
                yield Sample(image_id=image_id, caption=caption, image=img)
            log.info(f"Dataset iter end (images-first): images_loaded={yielded}, captions_non_null={captioned}, targets={total_targets}")
            return
        # 否则保持原有基于 caption 的按序迭代
        log.info(f"Dataset iter start: start_index={self.start_index}, max_samples={self.max_samples}, image_size={self.image_size}")
        with open(self.captions_jsonl, "r", encoding="utf-8") as f:
            for raw in f:
                # 跳过前 start_index 条样本
                if count < self.start_index:
                    count += 1
                    continue
                if self.max_samples is not None and yielded >= self.max_samples:
                    break
                item = json.loads(raw)
                image_id = str(item.get("image_id"))
                # 兼容两种标注字段：优先使用 caption，其次使用 text[0]
                caption = item.get("caption")
                if caption is None:
                    texts = item.get("text")
                    if isinstance(texts, list) and len(texts) > 0:
                        caption = texts[0]
                img = self._resolver.get_image(image_id)
                if img is None:
                    continue
                count += 1
                yielded += 1
                if caption is not None:
                    captioned += 1
                yield Sample(image_id=image_id, caption=caption, image=img)
        log.info(f"Dataset iter end: images_loaded={yielded}, captions_non_null={captioned}, attempted_from_index={self.start_index}")

class ImageTSVIndex:
    """为 images.tsv 建偏移索引（image_id -> 文件偏移），避免将全部 base64 加载进内存。
    支持按需构建：若提供 target_ids，仅为这些 ID 建索引并在全部命中后提前停止扫描。
    行格式：image_id\t<base64>
    """
    def __init__(self, tsv_path: str, target_ids: Optional[set] = None, log_progress: bool = True):
        self.tsv_path = tsv_path
        self.offsets: Dict[str, int] = {}
        self._build_index(target_ids=target_ids, log_progress=log_progress)

    def _build_index(self, target_ids: Optional[set] = None, log_progress: bool = True):
        n_lines = 0
        # print("HHD1" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        found = 0
        # print("HHD2" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        target_total = len(target_ids) if target_ids is not None else None
        # print("训练样本id", target_ids)
        # print("HHD3" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            # print("HHD4" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            while True:
                pos = f.tell()
                line = f.readline()
                # print("HHD5" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                if not line:
                    break
                # print("HHD6" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                n_lines += 1
                if not line.strip():
                    continue
                # print("HHD7" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                try:
                    # print("HHD8" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    image_id, _ = line.strip().split("\t", 1)
                    # 若未指定 target_ids，则为所有条目建索引；否则只记录目标集
                    if target_ids is None or image_id in target_ids:
                        if image_id not in self.offsets:
                            self.offsets[image_id] = pos
                            if target_ids is not None:
                                found += 1
                    if log_progress and (n_lines % 2000 == 0):
                        if target_ids is None:
                            log.info(f"Indexing images.tsv: processed_lines={n_lines}, indexed_entries={len(self.offsets)}")
                        else:
                            log.info(f"Indexing images.tsv (partial): processed_lines={n_lines}, found_targets={found}/{target_total}")
                    # 若仅构建目标集并已全部命中，提前停止
                    if target_ids is not None and found >= target_total:
                        break
                except ValueError:
                    continue
        if log_progress:
            if target_ids is None:
                log.info(f"Built images.tsv index: total_lines={n_lines}, indexed_entries={len(self.offsets)}")
            else:
                log.info(f"Built partial images.tsv index: scanned_lines={n_lines}, found_targets={found}/{target_total}")

    def get_image(self, image_id: str) -> Optional[Image.Image]:
        pos = self.offsets.get(image_id)
        if pos is None:
            return None
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            f.seek(pos)
            line = f.readline()
        try:
            _, b64 = line.strip().split("\t", 1)
            return decode_image_b64(b64)
        except Exception:
            return None


class CaptionIndex:
    """为 captions.jsonl 建立部分索引（image_id -> caption），仅针对目标图片ID。
    若提供 target_ids，则扫描文件并在全部命中后提前停止；否则会建立全量索引。
    """
    def __init__(self, captions_jsonl: str, target_ids: Optional[Set[str]] = None, log_progress: bool = True):
        self.captions_jsonl = captions_jsonl
        self.map: Dict[str, Optional[str]] = {}
        self._build_index(target_ids=target_ids, log_progress=log_progress)

    def _build_index(self, target_ids: Optional[Set[str]] = None, log_progress: bool = True):
        n_lines = 0
        found = 0
        target_total = len(target_ids) if target_ids is not None else None
        with open(self.captions_jsonl, "r", encoding="utf-8") as f:
            for raw in f:
                n_lines += 1
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                image_id = str(item.get("image_id"))
                if not image_id:
                    continue
                if target_ids is None or image_id in target_ids:
                    # 兼容两种标注字段：优先使用 caption，其次使用 text[0]
                    caption = item.get("caption")
                    if caption is None:
                        texts = item.get("text")
                        if isinstance(texts, list) and len(texts) > 0:
                            caption = texts[0]
                    if image_id not in self.map:
                        self.map[image_id] = caption
                        if target_ids is not None:
                            found += 1
                if log_progress and (n_lines % 200000 == 0):
                    if target_ids is None:
                        log.info(f"Indexing captions.jsonl: processed_lines={n_lines}, indexed_entries={len(self.map)}")
                    else:
                        log.info(f"Indexing captions.jsonl (partial): processed_lines={n_lines}, found_targets={found}/{target_total}")
                if target_ids is not None and found >= target_total:
                    break
        if log_progress:
            if target_ids is None:
                log.info(f"Built captions index: total_lines={n_lines}, indexed_entries={len(self.map)}")
            else:
                log.info(f"Built partial captions index: scanned_lines={n_lines}, found_targets={found}/{target_total}")

    def get_caption(self, image_id: str) -> Optional[str]:
        return self.map.get(image_id)


def postprocess_zh(text: str) -> str:
    # 去除特殊 token 和多余空格，规范数字/单位，去冗余重复片段
    if not text:
        return ""
    text = text.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
    text = text.replace("##", "")
    text = "".join(text.split())  # 移除所有空白
    # 合并连续重复，如 "舒适舒适舒适" -> "舒适"
    out = []
    for ch in text:
        if not out or out[-1] != ch:
            out.append(ch)
    text = "".join(out)
    # 简单单位规范：中文全角转半角，常见单位统一小写
    trans = str.maketrans({
        "，": ",", "。": ".", "：": ":", "；": ";",
        "（": "(", "）": ")", "％": "%",
    })
    text = text.translate(trans)
    return text
