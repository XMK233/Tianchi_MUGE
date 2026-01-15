# 临时测试脚本，用于验证模型保存路径
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import base64
from PIL import Image
import json

# 设置基本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"脚本所在目录: {script_dir}")

# 加载模型和 processor
model_id = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
processor = Qwen2_5_VLProcessor.from_pretrained(model_id, image_size=224)

# 针对16GB GPU优化的模型加载配置
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",  # 强制使用CUDA
    dtype=torch.float16,  # 使用float16而不是bfloat16来减少内存使用
    low_cpu_mem_usage=True,
)

# 启用梯度检查点以减少内存使用
model.gradient_checkpointing_enable()

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 应用 LoRA 适配器
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 简单的测试数据
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.length = 2
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # 创建一个简单的输入
        inputs = processor(
            images=Image.new('RGB', (224, 224), color='white'),
            text="测试输入",
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs["labels"] = inputs["input_ids"].clone()
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, ["测试描述"]

# 训练参数
training_args = TrainingArguments(
    output_dir=os.path.join(script_dir, "sft_qwen25vl_lora"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_steps=1,  # 设置只训练一个step
    logging_steps=1,
    save_strategy="steps",
    save_steps=1,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=True,
)

# 自定义训练器
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# 初始化训练器
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=DummyDataset(),
    data_collator=lambda batch: batch[0][0],  # 简单的数据整理
)

# 训练一个 step 然后保存
print("开始训练...")
trainer.train()

# 保存最终模型
model.save_pretrained(os.path.join(script_dir, "sft_qwen25vl_lora/final"))
print("模型保存完成！")

# 检查文件是否存在
final_dir = os.path.join(script_dir, "sft_qwen25vl_lora/final")
if os.path.exists(final_dir):
    print(f"模型文件已保存到: {final_dir}")
    print("文件列表:", os.listdir(final_dir))
else:
    print(f"错误: 模型文件未保存到 {final_dir}")