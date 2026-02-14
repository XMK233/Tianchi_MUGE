import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

# Method from plan_4_1/4.11_大过-基于9-去掉ema.py#L204-218
# Modified AutoModel -> AutoModelForVision2Seq to support generate()
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
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if device == 'cpu':
        dtype = torch.bfloat16 # Try bfloat16 to save memory, or float16
        
    model = AutoModelForVision2Seq.from_pretrained(base_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return tokenizer, model

def main():
    target_dirs = [
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/2 皇帝冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/3 皇后冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/4 皇太子妃亲王妃郡王妃冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/5 妃嫔冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/6 皇子公主冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/7 文武官冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/8 命妇冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/9 宫人舍人使官服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/10 军校冠服",

        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/11 军士巾服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/12 状元进士生员冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/13 吏员皂隶巾服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/14 士庶巾服",

        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/15 士庶妻冠服",

        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/16 僧道冠服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/17 仆役巾服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/18 杂流巾服",
        # "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/19 乐舞巾服",
        "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/20 婚礼冠服",
        "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/21 丧礼冠服",
        "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/22 祭服",
    ]    
    output_file = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/extracted_texts.txt"
    model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
    
    # Check for CPU override
    if len(sys.argv) > 1 and sys.argv[1] == '--cpu':
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    try:
        # We need the base path resolved for the processor as well
        # Reuse the logic inside load_vlm by calling it, then getting the path from the model
        tokenizer, model = load_vlm(model_path, device)
        print("Model loaded.")
        
        # Use the model's path for processor to ensure consistency
        real_model_path = model.config._name_or_path
        # If _name_or_path is not a full path, we might need to rely on the logic again.
        # But load_vlm doesn't return the path. 
        # Let's just re-run the path logic for processor.
        
        base_path = model_path
        config_path = os.path.join(base_path, "config.json")
        if not os.path.exists(config_path):
            snapshots_dir = os.path.join(base_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                for name in os.listdir(snapshots_dir):
                    cand = os.path.join(snapshots_dir, name)
                    if os.path.exists(os.path.join(cand, "config.json")):
                        base_path = cand
                        break
        
        print(f"Loading processor from {base_path}...")
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        
    except Exception as e:
        print(f"Failed to load VLM: {e}")
        return

    # Load processed files to avoid duplicates
    processed_files = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    # If previously failed with error, treat as not processed to retry
                    if len(parts) > 1 and ("ERROR" in parts[1] or "CUDA error" in parts[1]):
                        continue
                    processed_files.add(parts[0])

    print(f"Already processed {len(processed_files)} files.")

    # Collect images
    image_paths = []
    root_base = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1"
    
    for d in target_dirs:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            continue
            
        files = sorted(os.listdir(d))
        for f in files:
            if f.lower().endswith('.jpg') and not f.startswith('._'):
                full_path = os.path.join(d, f)
                image_paths.append(full_path)

    print(f"Found {len(image_paths)} images to process.")

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for img_path in tqdm(image_paths):
            rel_path = os.path.relpath(img_path, root_base)
            
            if rel_path in processed_files:
                continue

            try:
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

                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                clean_text = output_text.replace('\n', ' ').replace('\r', ' ').strip()
                out_f.write(f"{rel_path}\t{clean_text}\n")
                out_f.flush()
                
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                out_f.write(f"{rel_path}\tERROR: {str(e)}\n")
                out_f.flush()

    print("Done.")

if __name__ == "__main__":
    main()
