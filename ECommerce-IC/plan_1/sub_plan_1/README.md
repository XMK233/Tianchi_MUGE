# sub_plan_1 基线：Qwen2.5-VL-3B + SFT + SCST(CIDEr-D)

本子计划提供一个可落地的电商图像中文描述基线，采用 2024–2025 年主流多模态大模型微调技术：
- 模型：`Qwen2.5-VL-3B`（Instruct 变体更利于生成）
- 训练：先监督微调（SFT）再自批评序列训练（SCST），SCST 以 `CIDEr-D` 为奖励
- 参数高效：LoRA/QLoRA（依赖 `peft`）

目录结构：
- `configs/`：SFT 与 SCST 的示例配置
- `utils/`：数据集加载与中文分词、图像解码等工具
- `train/`：SFT 与 SCST 训练脚本
- `infer/`：生成与后处理脚本
- `eval/`：CIDEr-D 实现与评测入口（只实现 CIDEr-D，BLEU/ROUGE 可后续补充）
- `scripts/`：一键运行示例脚本

依赖（建议）：
- `torch>=2.1`, `transformers>=4.43`, `accelerate`, `peft`, `datasets`, `Pillow`

数据格式（与总方案一致）：
- `captions.jsonl`：每行 `{ "image_id": str, "caption": str }`（如有多参考，可重复 image_id）
- `images.tsv`：`image_id\t<base64>`，图片为 PNG/JPG 的 Base64 编码

快速开始：
1) SFT
```
bash scripts/run_sft.sh \
  --train_jsonl /path/to/captions_train.jsonl \
  --images_tsv  /path/to/images_train.tsv \
  --output_dir  /path/to/outputs/sft
```

2) SCST（CIDEr-D 奖励）
```
bash scripts/run_scst.sh \
  --train_jsonl /path/to/captions_train.jsonl \
  --images_tsv  /path/to/images_train.tsv \
  --sft_model   /path/to/outputs/sft \
  --output_dir  /path/to/outputs/scst
```

3) 生成与后处理
```
bash scripts/run_generate.sh \
  --test_jsonl /path/to/captions_test.jsonl \
  --images_tsv /path/to/images_test.tsv \
  --model_dir  /path/to/outputs/scst \
  --output_jsonl /path/to/preds.jsonl
```

4) 评测（CIDEr-D）
```
bash scripts/run_eval.sh \
  --refs_jsonl /path/to/captions_ref.jsonl \
  --preds_jsonl /path/to/preds.jsonl
```

注意：
- 若无 GPU 或显存较小，建议先跑少量样例验证流程；LoRA/QLoRA 需 `peft` 支持。
- Qwen2.5-VL 需从 HuggingFace 拉取权重，首次运行会自动下载。

