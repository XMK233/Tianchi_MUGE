#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商商品标签生成方案

功能：
1. 读取ECommerce-IC数据集的各个.jsonl文件
2. 使用Qwen-VL模型分析文本描述
3. 为每个商品生成电商标签（如女装、男装、家具等）
4. 输出标签对应列表
"""

import os
import json
import logging
import torch
import warnings
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 屏蔽警告
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 离线模式

# 数据集路径
DATASET_DIR = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-IC"

# 模型路径
# 模型配置
MODEL_DIR = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
USE_CPU = True  # 设置为True使用CPU进行推理，False使用GPU

# 输出路径
OUTPUT_DIR = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-IC/plan_2/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _resolve_model_dir(local_dir: str) -> str:
    """解析模型目录，确保找到config.json"""
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        return local_dir
    
    # 递归搜索 config.json
    for dirpath, dirnames, filenames in os.walk(local_dir):
        if "config.json" in filenames:
            return dirpath
    
    return local_dir


def load_qwen_vl(local_dir: str, for_training: bool = False):
    """从本地路径加载 Qwen2.5-VL-3B-Instruct"""
    import transformers
    from transformers import AutoProcessor
    
    # 允许 TF32 提速
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    resolved_dir = _resolve_model_dir(local_dir)

    processor = AutoProcessor.from_pretrained(
        resolved_dir,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    
    # 自动选择 dtype 与设备映射
    specific_classes = [
        getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None),
        getattr(transformers, "Qwen2_5_VLForCausalLM", None),
        getattr(transformers, "Qwen2VLForConditionalGeneration", None),
        getattr(transformers, "Qwen2VLForCausalLM", None),
    ]
    auto_causal_lm = getattr(transformers, "AutoModelForCausalLM", None)
    auto_model_cg = getattr(transformers, "AutoModelForConditionalGeneration", None)
    auto_model = getattr(transformers, "AutoModel", None)
    
    model_cls_order = [*specific_classes, auto_causal_lm, auto_model_cg, auto_model]
    model_cls = next((c for c in model_cls_order if c is not None), None)
    
    if model_cls is None:
        raise RuntimeError("No suitable model class available for Qwen VL.")
    
    # 训练时禁用 device_map=auto，避免模型被切到多设备导致 DeepSpeed 管理复杂和显存碎片
    if for_training:
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() and not USE_CPU else torch.float32,
        )
    else:
        # 推理时使用 device_map=auto 自动分配
        model = model_cls.from_pretrained(
            resolved_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() and not USE_CPU else torch.float32,
            device_map="auto" if torch.cuda.is_available() and not USE_CPU else None,
            low_cpu_mem_usage=True,
        )
    
    return model, processor


def load_jsonl_data(jsonl_path, chunk_size=1000):
    """加载JSONL文件数据，支持分片加载
    
    Args:
        jsonl_path: JSONL文件路径
        chunk_size: 每个分片的大小
        
    Yields:
        数据分片列表
    """
    logger.info(f"Loading data from {jsonl_path} in chunks of {chunk_size}")
    
    current_chunk = []
    total_samples = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading JSONL chunks"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                current_chunk.append(obj)
                total_samples += 1
                
                # 当达到分片大小时，返回当前分片
                if len(current_chunk) >= chunk_size:
                    logger.info(f"Loaded chunk of {len(current_chunk)} samples")
                    yield current_chunk
                    current_chunk = []  # 清空当前分片
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {e}")
                continue
    
    # 返回最后一个不完整的分片
    if current_chunk:
        logger.info(f"Loaded final chunk of {len(current_chunk)} samples")
        yield current_chunk
    
    logger.info(f"Loaded total {total_samples} samples from {jsonl_path}")


def generate_product_label(texts, model, processor):
    """为商品文本描述生成标签"""
    # 合并多条文本描述
    if isinstance(texts, list):
        merged_text = " ".join(texts)
    else:
        merged_text = texts
    
    # 构建提示
    prompt = f"请分析以下商品描述，为其生成一个主要的电商商品标签（如女装、男装、家具、数码、美妆等）。只返回标签，不返回任何其他内容。\n商品描述：{merged_text}"
    
    # 构建对话模板
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成标签
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=processor.eos_token_id
            )
        
        # 解码输出
        label = processor.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
        return label
        
    except Exception as e:
        logger.error(f"Error generating label: {e}")
        return "未知"


def process_all_data():
    """处理所有数据文件"""
    # 加载模型
    logger.info("Loading Qwen-VL model...")
    model, processor = load_qwen_vl(MODEL_DIR, for_training=False)
    model.eval()
    logger.info("Model loaded successfully!")
    
    # 数据文件列表
    jsonl_files = [
        "IC_train.jsonl",
        "IC_valid.jsonl",
        "IC_test.jsonl"
    ]
    
    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(DATASET_DIR, jsonl_file)
        
        if not os.path.exists(jsonl_path):
            logger.warning(f"File {jsonl_path} does not exist, skipping...")
            continue
        
        # 输出文件路径
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(jsonl_file)[0]}_labels.jsonl")
        
        # 初始化统计信息
        chunk_count = 0
        total_processed = 0
        
        # 处理每个数据分片
        logger.info(f"Processing {jsonl_file} in chunks...")
        
        for data_chunk in load_jsonl_data(jsonl_path, chunk_size=500):
            chunk_count += 1
            chunk_processed = 0
            
            logger.info(f"Processing chunk {chunk_count} (size: {len(data_chunk)})...")
            
            # 打开输出文件（追加模式）
            with open(output_path, "a", encoding="utf-8") as f:
                for item in tqdm(data_chunk, desc=f"Processing chunk {chunk_count}"):
                    image_id = item.get("img_id") or item.get("image_id")
                    texts = item.get("text", [])
                    
                    if not image_id:
                        continue
                    
                    try:
                        # 生成标签
                        label = generate_product_label(texts, model, processor)
                        
                        # 写入结果
                        result = {
                            "image_id": image_id,
                            "label": label,
                            "original_text": texts
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        
                        chunk_processed += 1
                        total_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing item {image_id}: {e}")
                        continue
            
            logger.info(f"Chunk {chunk_count} processed: {chunk_processed} samples")
            
            # 清理内存
            import gc
            gc.collect()
            if torch.cuda.is_available() and not USE_CPU:
                torch.cuda.empty_cache()
        
        logger.info(f"Completed processing {jsonl_file}: {total_processed} samples")
    
    # 生成汇总统计
    generate_summary_statistics()


def generate_summary_statistics():
    """生成标签统计信息"""
    import collections
    
    all_labels = []
    file_stats = {}
    
    # 统计每个文件的标签
    for jsonl_file in ["IC_train.jsonl", "IC_valid.jsonl", "IC_test.jsonl"]:
        output_file = f"{os.path.splitext(jsonl_file)[0]}_labels.jsonl"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        
        if not os.path.exists(output_path):
            continue
        
        labels = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                labels.append(item["label"])
        
        # 统计标签频率
        label_counts = collections.Counter(labels)
        file_stats[jsonl_file] = {
            "total_samples": len(labels),
            "unique_labels": len(label_counts),
            "top_labels": dict(label_counts.most_common(10))
        }
        
        all_labels.extend(labels)
    
    # 整体统计
    total_label_counts = collections.Counter(all_labels)
    
    # 保存统计结果
    stats_path = os.path.join(OUTPUT_DIR, "label_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "file_stats": file_stats,
            "overall": {
                "total_samples": len(all_labels),
                "unique_labels": len(total_label_counts),
                "top_labels": dict(total_label_counts.most_common(20))
            }
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Label statistics generated and saved to {stats_path}")


if __name__ == "__main__":
    process_all_data()
