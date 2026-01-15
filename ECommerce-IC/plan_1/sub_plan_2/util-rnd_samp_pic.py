import os
import random
import re
from tqdm import tqdm

# CODE_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-IC/plan_1/sub_plan_2/4.1_益-基于4_加多轮数_加多训练标注.py"

def parse_paths():
    # with open(CODE_PATH, "r", encoding="utf-8") as f:
    #     text = f.read()
    # m = re.search(r"base_dir\s*=\s*[\"'](.+?)[\"']", text)
    # if m:
    #     base_dir = m.group(1)
    # else:
    base_dir = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-IC/"
    train_tsv = os.path.join(base_dir, "IC_train.tsv")
    valid_tsv = os.path.join(base_dir, "IC_valid.tsv")
    out_train = os.path.join(base_dir, "IC_train_rnd_3w.tsv")
    out_valid = os.path.join(base_dir, "IC_valid_rnd_2k.tsv")
    return train_tsv, valid_tsv, out_train, out_valid

def count_lines(path):
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in tqdm(f, desc=f"Counting {os.path.basename(path)}", unit="line"):
            c += 1
    return c

def sample_chunked(tsv_path, out_csv_path, sample_size, chunk_size=200000, seed=2025):
    random.seed(seed)
    total = count_lines(tsv_path)
    target = min(sample_size, total)
    remaining_lines = total
    remaining = target
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)) or ".", exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as fout:
        with open(tsv_path, "r", encoding="utf-8") as fin:
            pbar = tqdm(total=total, desc=f"Processing {os.path.basename(tsv_path)}", unit="line")
            chunk = []
            for line in fin:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    img_id, b64 = line.split("\t", 1)
                except ValueError:
                    pbar.update(1)
                    continue
                chunk.append((img_id, b64))
                if len(chunk) >= chunk_size:
                    n = min(len(chunk), remaining if remaining_lines <= len(chunk) else int(round(remaining * len(chunk) / remaining_lines)))
                    if n > 0:
                        for r in random.sample(chunk, n):
                            fout.write(f"{r[0]}\t{r[1]}\n")
                        remaining -= n
                    remaining_lines -= len(chunk)
                    chunk = []
                pbar.update(1)
            if chunk:
                n = min(len(chunk), remaining)
                if n > 0:
                    for r in random.sample(chunk, n):
                        fout.write(f"{r[0]}\t{r[1]}\n")
                    remaining -= n
                remaining_lines -= len(chunk)
            pbar.close()
    return target

def main():
    train_tsv, valid_tsv, out_train, out_valid = parse_paths()
    sample_chunked(train_tsv, out_train, 30000, chunk_size=1000, seed=2025)
    sample_chunked(valid_tsv, out_valid, 2000, chunk_size=1000, seed=2025)
    print(out_train)
    print(out_valid)

if __name__ == "__main__":
    main()
