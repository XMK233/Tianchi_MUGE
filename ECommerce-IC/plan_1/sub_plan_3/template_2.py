import os
import base64
from io import BytesIO
from PIL import Image
import re
import numpy as np
from collections import defaultdict, Counter
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, TrainingArguments, Trainer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import logging
import argparse
import gc
import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("template_2")

# 脚本所在目录，用于构建正确的文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))

###### 基础工具

def decode_base64_to_image(base64_str: str) -> Image.Image:
    img_data = base64.urlsafe_b64decode(base64_str)
    return Image.open(BytesIO(img_data)).convert("RGB")

def chinese_char_tokenize(text: str) -> list:
    """
    中文按字切分，连续字母/数字视为一个 token
    示例: "V领包臀裙80后" → ["V", "领", "包", "臀", "裙", "80", "后"]
    """
    tokens = []
    i = 0
    while i < len(text):
        if re.match(r'[a-zA-Z0-9]', text[i]):
            j = i
            while j < len(text) and re.match(r'[a-zA-Z0-9]', text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            tokens.append(text[i])
            i += 1
    return tokens

##### 中文 CIDEr-D 实现（简化版）

class CiderScorer:
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []   # reference captions per image
        self.cands = []   # candidate captions

    def compute_doc_freq(self):
        """计算 IDF 所需的文档频率"""
        doc_freq = defaultdict(int)
        for refs in self.crefs:
            # 对每个图像的参考描述集，统计 unique n-grams
            words = set()
            for ref in refs:
                for ngram in self._get_ngrams(ref, self.n):
                    words.add(ngram)
            for word in words:
                doc_freq[word] += 1
        return doc_freq

    def _get_ngrams(self, sentence, n):
        tokens = chinese_char_tokenize(sentence)
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    def add_example(self, refs, cand):
        self.crefs.append(refs)
        self.cands.append(cand)

    def compute_score(self):
        doc_freq = self.compute_doc_freq()
        N = len(self.crefs)  # 图像数量
        idf = {}
        for w in doc_freq:
            idf[w] = np.log(N / doc_freq[w])

        scores = []
        for refs, cand in zip(self.crefs, self.cands):
            # 计算候选描述的 TF-IDF 向量
            cand_vec = self._tfidf_vector(cand, idf)
            ref_vecs = [self._tfidf_vector(ref, idf) for ref in refs]
            # 与每个参考描述计算余弦相似度，取平均
            sims = []
            for rvec in ref_vecs:
                sim = self._cosine(cand_vec, rvec)
                sims.append(sim)
            scores.append(np.mean(sims))
        return np.mean(scores) * 10.0  # CIDEr 通常 ×10

    def _tfidf_vector(self, sent, idf):
        ngrams = self._get_ngrams(sent, self.n)
        tf = Counter(ngrams)
        vec = defaultdict(float)
        total = sum(tf.values())
        for ng, count in tf.items():
            if ng in idf:
                vec[ng] = (count / total) * idf[ng]
        return vec

    def _cosine(self, v1, v2):
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k] * v2[k] for k in common)
        norm1 = np.sqrt(sum(v**2 for v in v1.values()))
        norm2 = np.sqrt(sum(v**2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

###### 数据集类

class ECommerceCaptionDataset(Dataset):
    def __init__(self, tsv_path, jsonl_path, processor, max_length=512, start_line=0, num_lines=None):
        self.processor = processor
        self.max_length = max_length
        
        # 加载图像 base64
        self.img_dict = {}
        with open(tsv_path, 'r') as f:
            for i, line in enumerate(f):
                if num_lines is not None and i >= start_line + num_lines:
                    break
                if i >= start_line:
                    img_id, b64 = line.strip().split('\t')
                    self.img_dict[img_id] = b64
        
        # 加载 caption
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                img_id = data['image_id']
                if img_id in self.img_dict:
                    target = np.random.choice(data['text'])  # 随机选择一条描述作为目标
                    self.samples.append((img_id, target, data['text']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, target_caption, all_captions = self.samples[idx]
        image = decode_base64_to_image(self.img_dict[img_id])
        
        # 构造对话格式（正确的多模态内容格式）
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "请为这件商品生成一段吸引人的描述。"}
            ]},
            {"role": "assistant", "content": target_caption}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # 生成 labels，将输入文本作为标签，忽略 pad_token
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # 修正：使用 self.processor
        inputs["labels"] = labels
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, all_captions  # all_captions 用于 CIDEr 计算

# 数据整理函数，确保正确合并batch - 特别处理Qwen2.5-VL需要的图像特征
def data_collator(batch):
    # 从batch中提取所有inputs
    inputs_list = [item[0] for item in batch]
    all_captions_list = [item[1] for item in batch]
    
    # 合并inputs字典
    merged_inputs = {}
    for key in inputs_list[0].keys():
        if key == "pixel_values":
            # 图像像素值需要特殊处理，使用torch.stack
            merged_inputs[key] = torch.stack([item[key] for item in inputs_list])
        elif key == "input_ids" or key == "attention_mask" or key == "labels":
            # 对于序列数据，使用pad_sequence来确保长度一致
            merged_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                [item[key] for item in inputs_list],
                batch_first=True,
                padding_value=processor.tokenizer.pad_token_id
            )
        else:
            # 其他类型的数据默认处理
            merged_inputs[key] = torch.stack([item[key] for item in inputs_list])
    
    return merged_inputs, all_captions_list

# 释放显存函数
def free_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

###### 训练、验证、推理函数

# 训练函数
def run_training_rounds(
    train_tsv, train_jsonl, rounds, per_round_lines, model, processor, 
    output_dir, lora_config, train_bs=1, gradient_accumulation_steps=8, 
    learning_rate=2e-4, num_train_epochs=3, fp16=True, start_round=0
):
    """分轮次训练模型"""
    
    for round_idx in range(start_round, rounds):
        log.info(f"开始训练第 {round_idx + 1}/{rounds} 轮")
        
        # 检查当前轮次是否已完成
        round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
        if os.path.exists(round_dir):
            log.info(f"第 {round_idx + 1} 轮已完成，跳过")
            continue
        
        # 加载当前轮次的数据
        start_line = round_idx * per_round_lines
        log.info(f"加载数据：从第 {start_line} 行开始，共 {per_round_lines} 行")
        
        train_dataset = ECommerceCaptionDataset(
            train_tsv, 
            train_jsonl, 
            processor, 
            start_line=start_line,
            num_lines=per_round_lines
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_bs,
            shuffle=True,
            collate_fn=data_collator
        )
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 训练循环
        model.train()
        total_loss = 0
        total_steps = 0
        
        for epoch in range(num_train_epochs):
            log.info(f"第 {round_idx + 1} 轮，第 {epoch + 1}/{num_train_epochs} 个epoch")
            
            for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(train_loader)):
                # 将输入移动到设备
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # 前向传播
                with torch.cuda.amp.autocast(enabled=fp16):
                    outputs = model(**inputs)
                    loss = outputs.loss
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * gradient_accumulation_steps
                total_steps += 1
                
                # 优化步骤
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 打印训练信息
                    if total_steps % 10 == 0:
                        log.info(f"Step {total_steps}, Loss: {total_loss / total_steps:.4f}")
            
        # 保存当前轮次的模型
        os.makedirs(round_dir, exist_ok=True)
        model.save_pretrained(round_dir)
        log.info(f"第 {round_idx + 1} 轮模型已保存到 {round_dir}")
        
        # 清理显存
        free_torch_memory()
    
    # 保存最终模型
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    log.info(f"最终模型已保存到 {final_dir}")

# 验证函数
def run_validation(valid_tsv, valid_jsonl, rounds, per_round_lines, model, processor, infer_bs=1):
    """分轮次验证模型"""
    
    model.eval()
    all_preds = []
    all_refs = []
    
    for round_idx in range(rounds):
        log.info(f"开始验证第 {round_idx + 1}/{rounds} 轮")
        
        # 加载当前轮次的数据
        start_line = round_idx * per_round_lines
        log.info(f"加载数据：从第 {start_line} 行开始，共 {per_round_lines} 行")
        
        valid_dataset = ECommerceCaptionDataset(
            valid_tsv, 
            valid_jsonl, 
            processor, 
            start_line=start_line,
            num_lines=per_round_lines
        )
        
        # 创建数据加载器
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=infer_bs,
            shuffle=False,
            collate_fn=data_collator
        )
        
        # 验证循环
        for batch_idx, (inputs, all_captions) in enumerate(tqdm.tqdm(valid_loader)):
            # 将输入移动到设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # 解码生成的文本
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            
            # 保存预测结果和参考结果
            all_preds.extend(generated_texts)
            all_refs.extend(all_captions)
        
        # 清理显存
        free_torch_memory()
    
    # 计算 CIDEr 分数
    scorer = CiderScorer()
    for pred, refs in zip(all_preds, all_refs):
        scorer.add_example(refs, pred)
    cider_score = scorer.compute_score()
    
    log.info(f"CIDEr 分数: {cider_score:.4f}")
    return cider_score

# 推理函数
def run_inference(test_tsv, test_jsonl, rounds, per_round_lines, model, processor, output_jsonl, infer_bs=1):
    """分轮次推理生成描述"""
    
    model.eval()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for round_idx in range(rounds):
            log.info(f"开始推理第 {round_idx + 1}/{rounds} 轮")
            
            # 加载当前轮次的数据
            start_line = round_idx * per_round_lines
            log.info(f"加载数据：从第 {start_line} 行开始，共 {per_round_lines} 行")
            
            # 加载图像 base64
            img_dict = {}
            with open(test_tsv, 'r') as f:
                for i, line in enumerate(f):
                    if i >= start_line + per_round_lines:
                        break
                    if i >= start_line:
                        img_id, b64 = line.strip().split('\t')
                        img_dict[img_id] = b64
            
            # 加载测试集数据
            test_samples = []
            with open(test_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    img_id = data['image_id']
                    if img_id in img_dict:
                        test_samples.append((img_id, img_dict[img_id]))
            
            # 批量推理
            for i in tqdm.tqdm(range(0, len(test_samples), infer_bs)):
                batch_samples = test_samples[i:i+infer_bs]
                
                # 处理批量图像和输入
                images = []
                img_ids = []
                for img_id, b64 in batch_samples:
                    images.append(decode_base64_to_image(b64))
                    img_ids.append(img_id)
                
                # 构造输入
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "请为这件商品生成一段吸引人的描述。"}
                ]}]
                prompts = [processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) for _ in range(len(batch_samples))]
                
                inputs = processor(
                    images=images,
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # 生成描述
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码生成的文本
                generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                
                # 保存结果
                for img_id, text in zip(img_ids, generated_texts):
                    result = {"image_id": img_id, "text": text}
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 清理显存
            free_torch_memory()
    
    log.info(f"推理结果已保存到 {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行电商图像描述生成的训练、验证和推理")
    parser.add_argument("--mode", choices=["train", "validate", "infer"], default="train", help="运行模式")
    parser.add_argument("--rounds", type=int, default=3, help="训练/验证/推理的轮次数")
    parser.add_argument("--per_round_lines", type=int, default=1000, help="每轮加载的样本行数")
    parser.add_argument("--train_bs", type=int, default=1, help="训练批次大小")
    parser.add_argument("--infer_bs", type=int, default=2, help="推理批次大小")
    parser.add_argument("--force_retrain", action="store_true", help="强制重新训练，不加载已有模型")
    args = parser.parse_args()
    
    # 配置参数
    model_id = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
    base_dir = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-IC/"
    output_dir = os.path.join(script_dir, "sft_qwen25vl_lora_improved")
    
    # 数据路径
    train_tsv = os.path.join(base_dir, "IC_train_rnd_3w.tsv")
    train_jsonl = os.path.join(base_dir, "IC_train.jsonl")
    valid_tsv = os.path.join(base_dir, "IC_valid_rnd_2k.tsv")
    valid_jsonl = os.path.join(base_dir, "IC_valid.jsonl")
    test_tsv = os.path.join(base_dir, "IC_test.tsv")
    test_jsonl = os.path.join(base_dir, "IC_test.jsonl")
    
    # 加载模型和处理器
    log.info("加载模型和处理器...")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id, image_size=224)  # 降低图像分辨率
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",  # 强制使用CUDA
        dtype=torch.float16,  # 使用float16减少内存使用
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2"  # 若支持
    )
    
    # 添加gradient checkpointing以减少内存使用
    model.gradient_checkpointing_enable()
    
    # 注入 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 检查是否需要加载已有模型
    start_round = 0
    if not args.force_retrain and os.path.exists(output_dir):
        # 检查是否有已完成的轮次
        round_dirs = [d for d in os.listdir(output_dir) if d.startswith("round_")]
        if round_dirs:
            # 找到最大的轮次号
            max_round = max(int(d.split("_")[1]) for d in round_dirs)
            start_round = max_round
            
            # 加载最新轮次的模型
            latest_round_dir = os.path.join(output_dir, f"round_{max_round}")
            log.info(f"加载最新轮次 {max_round} 的模型: {latest_round_dir}")
            model = get_peft_model(model, lora_config)
            model.load_adapter(latest_round_dir, adapter_name="default", is_trainable=True)
            model.to(torch.float16).to("cuda")
    else:
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # 运行不同模式
    if args.mode == "train":
        run_training_rounds(
            train_tsv, train_jsonl, args.rounds, args.per_round_lines, 
            model, processor, output_dir, lora_config,
            train_bs=args.train_bs,
            gradient_accumulation_steps=8,  # 梯度累积提升显存利用率
            learning_rate=2e-4,
            num_train_epochs=3,
            fp16=True,
            start_round=start_round
        )
    elif args.mode == "validate":
        run_validation(
            valid_tsv, valid_jsonl, args.rounds, args.per_round_lines, 
            model, processor, infer_bs=args.infer_bs
        )
    elif args.mode == "infer":
        output_jsonl = os.path.join(script_dir, "submission_improved.jsonl")
        run_inference(
            test_tsv, test_jsonl, args.rounds, args.per_round_lines, 
            model, processor, output_jsonl, infer_bs=args.infer_bs
        )
    
    log.info("任务完成！")