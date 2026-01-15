import os
import base64
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# 定义文件路径
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.cls1.tsv"
CSV_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/1-利用文本进行分类/classification_results_coarse.csv"
TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
OUTPUT_DIR = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_3/0-看图片"

# 定义图像预处理变换
img_sz = (256, 256)
tf = transforms.Compose([
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 从文本文件加载文本映射 (img_id -> text)
text_map = {}
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            text_map[parts[0]] = parts[1]

# 从CSV文件加载粗分类标签映射 (img_id -> coarse_label)
meta = pd.read_csv(CSV_PATH, sep="\t")
if "coarse_label" not in meta.columns:
    meta = pd.read_csv(CSV_PATH, sep=",")
coarse_map = {}
if "img_id" in meta.columns and "coarse_label" in meta.columns:
    for _, row in meta.iterrows():
        coarse_map[str(row["img_id"])] = str(row["coarse_label"])

# 从TSV文件加载图片数据并处理
def sanitize_filename(filename):
    """清理文件名，移除或替换非法字符"""
    illegal_chars = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
    for char in illegal_chars:
        filename = filename.replace(char, "_")
    # 限制文件名长度
    return filename[:200]  # 限制200个字符

count = 0
with open(TSV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        
        img_id = parts[0]
        img_b64 = parts[1]
        
        # 获取对应的文本和粗分类标签
        text = text_map.get(img_id, f"image_{img_id}")
        coarse_label = coarse_map.get(img_id, "unknown")
        
        try:
            # 解码base64图片
            img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            
            # 应用预处理变换
            img_tensor = tf(img)
            
            # 将张量转换回PIL图像
            to_pil = transforms.ToPILImage()
            processed_img = to_pil(img_tensor)
            
            # 创建粗分类标签文件夹
            label_dir = os.path.join(OUTPUT_DIR, coarse_label)
            os.makedirs(label_dir, exist_ok=True)
            
            # 生成合法的文件名
            filename = sanitize_filename(text) + ".png"
            file_path = os.path.join(label_dir, filename)
            
            # 保存图片
            processed_img.save(file_path, format="PNG")
            
            count += 1
            if count % 100 == 0:
                print(f"已处理并保存 {count} 张图片")
                
        except Exception as e:
            print(f"处理图片 {img_id} 时出错: {e}")
            continue

print(f"处理完成，共保存 {count} 张图片")
