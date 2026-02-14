import sys
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

# Set environment variable for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    
    print(f"Loading model from {base_path} to {device}...")
    # Load processor first
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    
    # Load model
    dtype = torch.float32 # Use float32 for CPU debugging safety
    model = AutoModelForVision2Seq.from_pretrained(base_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return processor, model

def main():
    target_root = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1"
    model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
    
    # Use CPU for precise error stack trace
    device = "cpu" 
    print(f"Running on {device} for debugging...")

    try:
        processor, model = load_vlm(model_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    test_images = [
        "15 士庶妻冠服/20681192_977275.jpg",
        "2 皇帝冠服/30b6f3d3eab230cfa8ec9a7a.jpg"
    ]

    for rel_path in test_images:
        img_path = os.path.join(target_root, rel_path)
        print(f"\nProcessing {img_path}...")
        
        if not os.path.exists(img_path):
            print("File not found.")
            continue

        try:
            # Construct Prompt
            prompt_text = "请提取并输出图片中包含的所有文字。"
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
            
            print("Input shapes:")
            for k, v in inputs.items():
                print(f"  {k}: {v.shape}")

            # Generate
            print("Generating...")
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Success! Output: {output_text[:100]}...")

        except Exception as e:
            print(f"ERROR processing {rel_path}:")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
