import sys
import os
import shutil
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

# 从 4.11_大过-基于9-去掉ema.py 复制的 load_vlm 函数
def load_vlm(path, device):
    base_path = path
    config_path = os.path.join(base_path, "config.json")
    if not os.path.exists(config_path):
        snapshots_dir = os.path.join(base_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                cand = os.path.join(snapshots_dir, name)
                if os.path.exists(os.path.join(cand, "config.json")):
                    base_path = cand
                    break
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    # dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Use bfloat16 if available to avoid CUDA errors (NaN/Inf)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 # Fallback to fp32 for stability if bf16 not available
        
    model = AutoModelForVision2Seq.from_pretrained(base_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return tokenizer, model

def main():
    source_root = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/Q版大明衣冠图志"
    target_root = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1"
    
    model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Source: {source_root}")
    print(f"Target: {target_root}")

    # 1. 复制和筛选文件
    print("Step 1: Filtering and copying files...")
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"Created target directory: {target_root}")

    image_paths = []

    if not os.path.exists(source_root):
        print(f"Error: Source directory {source_root} does not exist.")
        return

    # 遍历源目录
    # 只需要第一层子目录
    try:
        subdirs = sorted(os.listdir(source_root))
    except Exception as e:
        print(f"Error listing source directory: {e}")
        return

    for d in subdirs:
        dir_path = os.path.join(source_root, d)
        if not os.path.isdir(dir_path):
            continue
        
        # 检查目录名是否以数字开头，且在 2-22 之间
        match = re.match(r'^(\d+)', d)
        if match:
            try:
                num = int(match.group(1))
                if 2 <= num <= 22:
                    # 符合条件，处理该目录
                    target_subdir = os.path.join(target_root, d)
                    if not os.path.exists(target_subdir):
                        os.makedirs(target_subdir)
                    
                    files = sorted(os.listdir(dir_path))
                    for f in files:
                        # 筛选 .jpg 且不以 ._ 开头
                        if f.lower().endswith('.jpg') and not f.startswith('._'):
                            src_file = os.path.join(dir_path, f)
                            dst_file = os.path.join(target_subdir, f)
                            
                            # 复制文件
                            shutil.copy2(src_file, dst_file)
                            image_paths.append(dst_file)
            except ValueError:
                continue
    
    print(f"Copied {len(image_paths)} images to {target_root}")

    if len(image_paths) == 0:
        print("No images found matching criteria. Exiting.")
        return

    # 2. 加载 VLM 提取文字
    print("Step 2: Loading VLM and extracting text...")

    blacklist_file = os.path.join(target_root, "blacklist.txt")
    blacklist = set()
    if os.path.exists(blacklist_file):
        with open(blacklist_file, 'r', encoding='utf-8') as f:
            for line in f:
                blacklist.add(line.strip())
    
    try:
        # 使用 load_vlm 加载 tokenizer 和 model
        # 注意：load_vlm 会处理 path 查找逻辑，所以我们应该使用它返回的 model 对象
        # 但为了加载 processor，我们需要确定的 base_path
        # 这里为了简单，我们重新利用 load_vlm 内部的路径查找逻辑，或者直接信任 load_vlm 返回的 model.config._name_or_path (如果可用)
        # 或者直接用 load_vlm 加载 model，然后尝试独立加载 Processor
        
        tokenizer, model = load_vlm(model_path, device)
        print("Model loaded.")

        # 加载 Processor
        # 由于 load_vlm 内部可能会改变 base_path (如果有 snapshots)，我们需要确保 Processor 加载的是同一个路径
        # 简单的办法是再次运行路径查找逻辑，或者直接用 model_path (假设 AutoProcessor 足够智能或路径结构标准)
        # 为了稳妥，我们把 load_vlm 里的路径查找逻辑也拿出来用一下，或者直接假设 model_path 正确
        # 实际上 load_vlm 已经处理了 snapshot，我们可以通过 model.name_or_path 获取实际加载的路径吗？
        # 通常可以。
        
        real_model_path = model_path
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
             snapshots_dir = os.path.join(model_path, "snapshots")
             if os.path.isdir(snapshots_dir):
                for name in os.listdir(snapshots_dir):
                    cand = os.path.join(snapshots_dir, name)
                    if os.path.exists(os.path.join(cand, "config.json")):
                        real_model_path = cand
                        break
        
        print(f"Loading processor from {real_model_path}...")
        processor = AutoProcessor.from_pretrained(real_model_path, trust_remote_code=True)
        
    except Exception as e:
        print(f"Failed to load VLM or Processor: {e}")
        return

    results_file = os.path.join(target_root, "extracted_texts.txt")
    print(f"Extracting text to {results_file}...")

    processed_files = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        if len(parts) > 1 and "CUDA error" in parts[1]:
                            continue
                        processed_files.add(parts[0])
        except Exception as e:
            print(f"Error reading existing results: {e}")

    # Manually skip known bad file causing CUDA assert
    bad_file = "15 士庶妻冠服/09圆领褙子.jpg"
    processed_files.add(bad_file)

    print(f"Resuming... {len(processed_files)} images already processed. {len(blacklist)} in blacklist.")

    with open(results_file, 'a', encoding='utf-8') as out_f:
        for img_path in tqdm(image_paths):
            rel_path = os.path.relpath(img_path, target_root)
            if rel_path in processed_files:
                continue
            if rel_path in blacklist:
                continue

            try:
                # 构造 Prompt
                prompt_text = "请提取并输出图片中包含的所有文字。如果图片中没有文字，请回答“无文字”。只输出识别到的文字内容，不要包含其他分析。"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                # 准备输入
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs = [Image.open(img_path).convert("RGB")]
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 生成
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                # 解码
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # 写入结果 (格式: 相对路径 \t 识别文字)
                rel_path = os.path.relpath(img_path, target_root)
                # 去除换行符以便存为单行
                clean_text = output_text.replace('\n', ' ').replace('\r', ' ').strip()
                out_f.write(f"{rel_path}\t{clean_text}\n")
                out_f.flush()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # 即使出错也记录一下
                rel_path = os.path.relpath(img_path, target_root)
                out_f.write(f"{rel_path}\tERROR: {str(e)}\n")

    print("All done!")

if __name__ == "__main__":
    main()
