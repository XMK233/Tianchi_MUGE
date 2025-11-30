import argparse
import json
import os
import logging
import sys
from collections import defaultdict

# 允许作为脚本直接运行时找到同目录下的 cider_d
sys.path.append(os.path.dirname(__file__))
from cider_d import CiderD


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger("evaluate_cider")

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

def load_refs(path):
    refs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            refs[str(item["image_id"])].append(item["caption"])
    return refs


def load_preds(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            preds[str(item["image_id"]) ] = item["caption"]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs_jsonl", required=True)
    ap.add_argument("--preds_jsonl", required=True)
    ap.add_argument("--images_tsv")
    ap.add_argument("--pics_for_one_eval_round", type=int, default=0, help="If >0, filter refs/preds to a subset of images-first sampled IDs")
    ap.add_argument("--num_eval_rounds", type=int, default=1)
    args = ap.parse_args()

    refs = load_refs(args.refs_jsonl)
    preds = load_preds(args.preds_jsonl)
    # 可选：图片优先过滤，仅评估采样到的图片集合
    if args.images_tsv and args.pics_for_one_eval_round and args.pics_for_one_eval_round > 0:
        allowed = set()
        pics_per_round = int(args.pics_for_one_eval_round)
        num_rounds = int(args.num_eval_rounds or 1)
        log.info(f"Eval filtering enabled: pics_per_round={pics_per_round}, num_rounds={num_rounds}")
        for round_idx in range(num_rounds):
            start_index = round_idx * pics_per_round
            round_ids = collect_image_ids_from_tsv(args.images_tsv, start_index, pics_per_round)
            allowed.update(round_ids)
        # 过滤 refs 和 preds 到 allowed 集合交集
        refs = {k: v for k, v in refs.items() if k in allowed}
        preds = {k: v for k, v in preds.items() if k in allowed}
        log.info(f"Filtered refs={len(refs)}, preds={len(preds)} by allowed IDs={len(allowed)}")
    cider = CiderD(refs)

    scores = []
    for img_id, cand in preds.items():
        r = refs.get(img_id, [])
        if not r:
            continue
        s = cider.score_one(cand, r)
        scores.append(s)
    mean_score = sum(scores) / max(1, len(scores))
    print(json.dumps({"metric": "CIDEr-D", "score": mean_score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
