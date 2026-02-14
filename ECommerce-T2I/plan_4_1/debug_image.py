import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

def debug_image(img_rel_path):
    target_root = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1"
    img_path = os.path.join(target_root, img_rel_path)
    
    print(f"Checking image: {img_path}")
    if not os.path.exists(img_path):
        print("File not found!")
        return

    try:
        img = Image.open(img_path)
        print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        img = img.convert("RGB")
        print(f"Converted to RGB. Size: {img.size}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
    
    # Resolve path logic from x-处理汉服图片.py
    if not os.path.exists(os.path.join(model_path, "config.json")):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                cand = os.path.join(snapshots_dir, name)
                if os.path.exists(os.path.join(cand, "config.json")):
                    model_path = cand
                    break
    
    print(f"Using model path: {model_path}")
    
    print("Loading model on CPU (float32)...")
    try:
        # Load directly to CPU
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print("Model loaded on CPU.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [img]
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Verify inputs
        print("Input keys:", inputs.keys())
        for k, v in inputs.items():
            if hasattr(v, 'shape'):
                print(f"{k} shape: {v.shape}")
        
        print("Generating...")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"Output: {output_text}")
        print("Success on CPU!")

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the first failing image
    debug_image("15 士庶妻冠服/09圆领褙子.jpg")
