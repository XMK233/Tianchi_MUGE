import os
import re

input_file = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/extracted_texts.txt"
output_file = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/汉服1/dataset_processed.txt"

def extract_key_info(folder_name, text):
    # Remove number prefix from folder name to get category (e.g. "2 皇帝冠服" -> "皇帝冠服")
    category = re.sub(r'^\d+\s*', '', folder_name).strip()
    
    # Pattern 1: Category followed by "之" (with optional spaces) and then content
    # e.g. "皇帝冠服之祙褂", "皇帝冠服 之常服"
    pattern1 = r'({}\s*之\s*\S+)'.format(re.escape(category))
    match1 = re.search(pattern1, text)
    if match1:
        return match1.group(1)
        
    # Pattern 2: Category followed by space and then content
    # e.g. "皇帝冠服 衮冕"
    pattern2 = r'({}\s+\S+)'.format(re.escape(category))
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1)

    # Fallback: If category not found, try to return the first meaningful phrase
    # Split by space and take the first part
    parts = text.strip().split()
    if parts:
        # If the first part is "大明衣冠" or "明", maybe take the second part?
        if parts[0] in ["大明衣冠", "明"] and len(parts) > 1:
            # If the second part is the category (which we missed above for some reason?), retry logic?
            # But regex search above scans the whole string, so we wouldn't miss it if it existed.
            # So if we are here, category is NOT in text.
            # Just return the second part which is likely the item.
            return parts[1]
        return parts[0]
        
    return text # Absolute fallback

processed_count = 0
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 2:
            continue
            
        id_col = parts[0]
        text_col = parts[1]
        
        # Parse folder from id (id format: "folder/filename")
        if '/' in id_col:
            folder = id_col.split('/')[0]
        else:
            folder = "" # Should not happen based on format
            
        if folder:
            extracted = extract_key_info(folder, text_col)
        else:
            extracted = text_col
            
        # Clean up extraction (remove leading/trailing whitespace)
        extracted = extracted.strip()
        
        fout.write(f"{id_col}\t{extracted}\n")
        processed_count += 1

print(f"Processed {processed_count} lines. Output saved to {output_file}")
