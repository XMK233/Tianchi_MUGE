import os
import pandas as pd
import numpy as np

SRC_CSV = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/1-利用文本进行分类/classification_results_coarse.csv"
IMG_TSV = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
VAL_OUT = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.cls1.tsv"
TRAIN_OUT = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.cls1.tsv"

def build_id_splits():
    df = pd.read_csv(SRC_CSV, sep='\t')
    groups = df.groupby('coarse_label')
    np.random.seed(42)
    val_ids = []
    train_ids = []
    for name, g in groups:
        idx = np.arange(len(g))
        np.random.shuffle(idx)
        n_val = max(1, int(len(g) / 3))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_ids.extend(g.iloc[val_idx]['img_id'].tolist())
        train_ids.extend(g.iloc[train_idx]['img_id'].tolist())
    return set(val_ids), set(train_ids)

def write_split_files(val_ids, train_ids):
    with open(IMG_TSV, 'r', encoding='utf-8') as fin, \
         open(VAL_OUT, 'w', encoding='utf-8') as fval, \
         open(TRAIN_OUT, 'w', encoding='utf-8') as ftrain:
        for line in fin:
            if not line.strip():
                continue
            img_id = line.split('\t', 1)[0]
            if img_id in val_ids:
                fval.write(line)
            elif img_id in train_ids:
                ftrain.write(line)

def main():
    val_ids, train_ids = build_id_splits()
    write_split_files(val_ids, train_ids)

if __name__ == "__main__":
    main()

