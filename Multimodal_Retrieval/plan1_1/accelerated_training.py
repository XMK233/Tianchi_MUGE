import gc
from typing import List, Tuple

import torch
from torch.cuda.amp import autocast, GradScaler


def set_perf_knobs():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def info_nce_from_projected(text_proj: torch.Tensor, img_proj: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (text_proj @ img_proj.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t = torch.nn.functional.cross_entropy(logits, labels)
    loss_i = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2


@torch.no_grad()
def precompute_base_features(model, pairs: List[Tuple[str, str, object]], device: torch.device, use_amp: bool = True):
    texts = [t for (t, _img_id, _img) in pairs]
    images = [img for (_t, _img_id, img) in pairs]
    if use_amp and device.type == "cuda":
        # Use CUDA AMP autocast compatible with older PyTorch (no device_type arg)
        with autocast(enabled=True, dtype=torch.float16):
            t_feats = model.text_extractor.extract_features(texts)
            i_feats = model.image_extractor.extract_features(images)
    else:
        t_feats = model.text_extractor.extract_features(texts)
        i_feats = model.image_extractor.extract_features(images)
    text_base = t_feats.detach().to(device).to(torch.float16)
    image_base = i_feats.detach().to(device).to(torch.float16)
    return text_base, image_base


def run_accelerated_training(
    model,
    loader,
    device,
    optim: torch.optim.Optimizer,
    temperature: float = 0.07,
    train_image_batch_size: int = 20000,
    max_train_batches: int = 10,
    epochs_per_batch: int = 3,
    step_batch_size: int = 512,
    use_amp: bool = True,
    compile_fusion: bool = False,
    log_fn=print,
):
    set_perf_knobs()

    text_proj = model.fusion.text_projector
    img_proj = model.fusion.image_projector
    if compile_fusion and hasattr(torch, "compile"):
        try:
            text_proj = torch.compile(text_proj)
            img_proj = torch.compile(img_proj)
        except Exception:
            pass

    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    for b_idx, img_batch in enumerate(
        loader.load_images_batch(split="train", batch_size=train_image_batch_size, max_batches=max_train_batches)
    ):
        img_map = {item["img_id"]: item["image"] for item in img_batch if item["image"] is not None}
        if not img_map:
            continue

        train_df = loader.cached_train_df if hasattr(loader, "cached_train_df") else loader.load_queries(split="train")
        pairs: List[Tuple[str, str, object]] = []
        for _, row in train_df.iterrows():
            q = row.get("query_text", None)
            ids = row.get("item_ids", [])
            if not q or not ids:
                continue
            for iid in ids:
                sid = str(iid)
                if sid in img_map and img_map[sid] is not None:
                    pairs.append((q, sid, img_map[sid]))
                    break

        if len(pairs) == 0:
            continue

        log_fn(f"Batch {b_idx+1}: images={len(img_map)}, usable_pairs={len(pairs)}")

        text_base, image_base = precompute_base_features(model, pairs, device, use_amp=use_amp)
        N = text_base.size(0)

        for e in range(1, epochs_per_batch + 1):
            epoch_loss = 0.0
            steps = 0
            s = 0
            while s < N:
                bs = min(step_batch_size, N - s)
                tb = text_base[s : s + bs]
                ib = image_base[s : s + bs]

                # Use CUDA AMP autocast compatible with older PyTorch (no device_type arg)
                with autocast(enabled=use_amp and device.type == "cuda", dtype=torch.float16):
                    t_proj = text_proj(tb)
                    i_proj = img_proj(ib)
                    if model.normalize_features:
                        t_proj = torch.nn.functional.normalize(t_proj, p=2, dim=1)
                        i_proj = torch.nn.functional.normalize(i_proj, p=2, dim=1)
                    loss = info_nce_from_projected(t_proj, i_proj, temperature)

                optim.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

                epoch_loss += loss.item()
                steps += 1
                s += bs

            avg_loss = epoch_loss / max(steps, 1)
            log_fn(f"Batch {b_idx+1} - Epoch {e}/{epochs_per_batch}: avg loss={avg_loss:.4f}")

        del text_base, image_base, img_map, pairs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()