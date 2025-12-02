"""
预览工具：将样例图片与预测文本保存为本地 HTML 页面。

功能：
- 将输入的样例（含 `image`/`image_id`）与预测文本 `preds` 进行配对；
- 稳健转换并保存图片（支持 PIL/NumPy/Torch）；
- 生成简易 HTML 预览页面，便于快速查看效果。
"""

import os
import logging
from html import escape

from PIL import Image

log = logging.getLogger("preview_utils")


def _to_pil_rgb(obj):
    """将输入转换为 RGB PIL.Image；支持 PIL、numpy、torch.Tensor。失败返回 None。"""
    try:
        # PIL Image
        if isinstance(obj, Image.Image):
            try:
                obj.load()  # 强制加载，避免懒加载导致保存空图
            except Exception:
                pass
            if obj.mode != "RGB":
                try:
                    obj = obj.convert("RGB")
                except Exception:
                    pass
            return obj
    except Exception:
        log.info("[Preview] try-block failed in _to_pil_rgb: PIL branch")
        pass
    # Torch Tensor
    try:
        import torch
        import numpy as np
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
            # CHW → HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            # 归一化到 uint8
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        log.info("[Preview] try-block failed in _to_pil_rgb: torch branch")
        pass
    # NumPy 数组或带 numpy() 方法对象
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            arr = obj
        elif hasattr(obj, "numpy"):
            arr = obj.numpy()
        else:
            arr = None
        if arr is not None:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                # 可能是 CHW
                import numpy as _np
                arr = _np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        log.info("[Preview] try-block failed in _to_pil_rgb: numpy branch")
        pass
    # 最后尝试直接 fromarray
    try:
        return Image.fromarray(obj).convert("RGB")
    except Exception:
        log.info("[Preview] try-block failed in _to_pil_rgb: fromarray fallback")
        return None


def save_warmup_preview(samples, preds, save_dir: str | None, round_index: int) -> str | None:
    """保存样例图片并生成简易 HTML 预览。

    参数：
    - samples：样例列表，每个元素需包含 `image` 与 `image_id`。
    - preds：对应的预测文本列表。
    - save_dir：主保存目录（可为 None，默认当前目录）。
    - round_index：当前轮次索引（用于目录命名）。

    返回：
    - 生成的 HTML 路径（失败时返回 None）。
    """
    try:
        preview_dir = os.path.join(save_dir or ".", f"warmup_round_{round_index}")
        os.makedirs(preview_dir, exist_ok=True)
        items = []
        for idx, (s, pred) in enumerate(zip(samples, preds)):
            img = s.get("image")
            img_id = s.get("image_id") or f"sample_{idx}"
            img_path = os.path.join(preview_dir, f"{idx}_{img_id}.jpg")
            saved_ok = False
            try:
                pil_img = _to_pil_rgb(img)
                if pil_img is not None:
                    pil_img.save(img_path, format="JPEG")
                    saved_ok = True
                else:
                    log.warning(f"[Train] Warmup preview: cannot convert image {idx} to PIL RGB.")
            except Exception as e:
                log.warning(f"[Train] Failed to save preview image {idx}: {e}")
            items.append((img_path if saved_ok else None, img_id, pred))

        html_path = os.path.join(preview_dir, "index.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Warmup Preview</title>"
                    "<style>body{font-family:sans-serif} .card{margin:12px;padding:8px;border:1px solid #ddd;display:inline-block}"
                    "img{max-width:320px;display:block;margin-bottom:6px}</style></head><body>")
            f.write(f"<h2>Round {round_index} Warmup Preview</h2>")
            for (img_path, img_id, pred) in items:
                f.write("<div class='card'>")
                if img_path and os.path.exists(img_path):
                    rel = os.path.basename(img_path)
                    f.write(f"<img src='{escape(rel)}' alt='{escape(str(img_id))}'>")
                else:
                    f.write("<div style='width:320px;height:200px;background:#eee;display:flex;align-items:center;justify-content:center;'>图片保存失败</div>")
                f.write(f"<div><b>image_id:</b> {escape(str(img_id))}</div>")
                f.write(f"<div><b>pred:</b> {escape(str(pred))}</div>")
                f.write("</div>")
            f.write("</body></html>")
        log.info(f"[Train] Warmup preview saved: {html_path}")
        return html_path
    except Exception as e:
        log.info("[Preview] try-block failed in save_warmup_preview: building HTML and assets")
        log.warning(f"[Train] Failed to build warmup preview: {e}")
        return None
