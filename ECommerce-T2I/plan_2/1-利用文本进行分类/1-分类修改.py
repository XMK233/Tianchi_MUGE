import re
import json
import os
import pandas as pd

def parse_to_dict(s):
    res = {}
    for cat, items in re.findall(r"\*\s+\*\*(.*?)\*\*：(.+)", s):
        labels = re.findall(r"`([^`]+)`", items)
        for label in labels:
            res[label] = cat
    return res

DEFAULT_INPUT = """
*   **上衣类**：`外套_上衣_长袖`、`女装_印花_外套`、`上衣_外套_长袖`
*   **裤装类**：`裤子_牛仔裤_直筒`
*   **裙装类**：`连衣裙_裙子_女装`
*   **童装类**：`女童_童装_男童`
*   **鞋类**：`女鞋_单鞋_真皮`
*   **包袋**：`单肩_真皮_背包`
*   **饰品**：`耳环_项链_耳钉`、`韩国_帽子_头饰`
*   **家居**：`北欧_餐厅_中式`、`摆件_装饰_卡通`
*   **器皿**：`陶瓷_茶杯_茶壶`
*   **数码**：`手机_硅胶_保护套`
*   **美妆**：`补水_精华_毛孔`
*   **食品**：`零食_普洱茶_特产`
"""

dic = parse_to_dict(DEFAULT_INPUT)

CSV_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/1-利用文本进行分类/classification_results.csv"
CSV_OUT = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/1-利用文本进行分类/classification_results_coarse.csv"
JSON_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/1-利用文本进行分类/classification_summary.json"
JSON_OUT = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/1-利用文本进行分类/classification_summary_coarse.json"

def replace_csv_coarse():
    if not os.path.exists(CSV_PATH):
        return False
    df = pd.read_csv(CSV_PATH, sep='\t')
    if 'predicted_label' not in df.columns:
        return False
    df['coarse_label'] = df['predicted_label'].map(lambda x: dic.get(str(x), str(x)))
    df.to_csv(CSV_OUT, index=False, sep='\t')
    return True

def replace_json_coarse():
    if not os.path.exists(JSON_PATH):
        return False
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        name = item.get('label_name', '')
        item['coarse_label'] = dic.get(str(name), item.get('coarse_label', str(name)))
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True

if __name__ == "__main__":
    ok_csv = replace_csv_coarse()
    ok_json = replace_json_coarse()
    print(json.dumps(dic, ensure_ascii=False, indent=2))
    print(f"csv_updated={ok_csv}, json_updated={ok_json}")
