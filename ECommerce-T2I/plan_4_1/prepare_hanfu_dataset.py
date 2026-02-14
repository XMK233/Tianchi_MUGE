import os
import random
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm

DATASET_FILE = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服2/dataset_processed.txt"
IMAGE_ROOT = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服2"
OUTPUT_DIR = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I"

def process_image_to_base64(img_path):
    try:
        with Image.open(img_path) as img:
            # Convert to RGB to avoid issues with RGBA/Palette
            img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    # Read dataset
    data = []
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file not found: {DATASET_FILE}")
        return

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                data.append((parts[0], parts[1]))
    
    # Group by category
    groups = {}
    for item in data:
        rel_path, text = item
        # Category is the folder name (first part of path)
        folder = os.path.dirname(rel_path)
        if folder not in groups:
            groups[folder] = []
        groups[folder].append(item)
    
    train_set = []
    val_set = []
    
    print(f"Found {len(groups)} categories.")
    
    for folder, items in groups.items():
        # Shuffle items for random selection
        random.shuffle(items)
        
        # Split
        if len(items) >= 3:
            val_items = items[:3]
            train_items = items[3:]
        else:
            # If less than 3, take all for validation (safe fallback)
            val_items = items
            train_items = []
        
        val_set.extend(val_items)
        train_set.extend(train_items)
        
    print(f"Total Train: {len(train_set)}, Total Val: {len(val_set)}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process and Save
    for name, dataset in [("hanfu_train", train_set), ("hanfu_val", val_set)]:
        img_tsv_path = os.path.join(OUTPUT_DIR, f"{name}.img.tsv")
        text_tsv_path = os.path.join(OUTPUT_DIR, f"{name}.text.tsv")
        
        print(f"Processing {name}...")
        
        with open(img_tsv_path, 'w', encoding='utf-8') as f_img, \
             open(text_tsv_path, 'w', encoding='utf-8') as f_text:
            
            for rel_path, text in tqdm(dataset):
                full_path = os.path.join(IMAGE_ROOT, rel_path)
                if not os.path.exists(full_path):
                    print(f"Missing file: {full_path}")
                    continue
                
                b64_str = process_image_to_base64(full_path)
                if b64_str:
                    # Use rel_path as ID
                    img_id = rel_path
                    
                    # Write TSV lines
                    f_img.write(f"{img_id}\t{b64_str}\n")
                    f_text.write(f"{img_id}\t{text}\n")
                    
    print("Done.")

if __name__ == "__main__":
    main()
