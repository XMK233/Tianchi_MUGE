# ECommerce-IC 电商图像描述生成方案（2024–2025 多模态大模型微调）

本方案针对数据集 `ECommerce-IC`（图片Base64/TSV、caption JSONL、多参考描述），提供基于多模态大模型（MM-LLM）的端到端解决路径，覆盖数据管线、模型与训练策略、推理与后处理、评测与提交格式，并给出可落地的文件结构与执行方式。

## 目标与约束
- 目标：对每张商品图生成可信、吸引用户的中文描述；主指标 `CIDEr-D`，辅指标 `BLEU-4`、`ROUGE-L`。
- 约束：
  - 数据：`IC_train/valid/test.tsv`（Base64图片），`IC_train/valid.jsonl`（多参考描述），`IC_test.jsonl`（仅 `image_id`）。
  - 分字评测：计算 n-gram 时中文按“分字”，连续的字母或数字视作一个 token。
  - 输出格式：按官方要求输出 `jsonl`，每行 `{"image_id": ..., "text": ...}`。

## 总体方案（现代 MM-LLM）
- 模型路线优选：
  - `Qwen2.5-VL`（3B/7B）：中文能力与电商语域更强，开放权重，适合国内语料与场景；支持高分辨率与长文本。
  - 备选：`InternVL 2.5`、`LLaVA-OneVision`、`MiniCPM-V 2.6`。如资源有限，可选 3B 级模型做 LoRA/QLoRA 微调。
- 训练策略：两阶段训练提升效果与稳健性
  1) SFT（监督微调）：用多参考描述做标准交叉熵训练，对齐“图→文本”指令风格；
  2) SCST 或 CIDEr-RL（自批评序列训练）：以 `CIDEr-D` 为回报优化，进一步贴合主指标；可辅以 `DPO/KTO`（基于参考/偏好对）提升可读性与风格。
- 参数高效微调：`LoRA/QLoRA` 仅训练语言侧投影和部分 Cross-Attn/MLP 层，结合 `bf16` + `FSDP/DeepSpeed` 节省显存与加速。
- 推理策略：Beam Search（beam=5~8）+ 长度惩罚 + 重复惩罚，必要时结合 `CLIPScore` 或文本质量 rerank。
- 后处理：中文去冗余、规范数字/单位、去除特殊 token/空格/重复片段。

## 架构与数据管线
- 输入：
  - 从 `TSV` 解码 Base64 → `PIL.Image`；按模型视觉编码器规范做增强与尺寸（如 `Qwen2.5-VL` 默认 448 或 768 的短边，训练时可固定 `image_size=448`）。
  - 文本（训练参考）：从 `JSONL` 读取 `image_id → [text...]`；训练时按“多参考采样”或“拼接多参考”策略使用。
- 视觉编码：使用模型自带 ViT（如 CLIP/ViT-Large/高分辨率适配），产生视觉 token，送入语言解码。
- 指令格式：统一成对模型友好的指令样式，例如：
  - 用户：`请为这张电商商品图生成中文描述，突出品类、卖点与风格。`，系统/开发者提示中加入“安全与事实遵循”与“简洁吸引力”的偏好。
- 分词与评测：训练时使用模型自带 tokenizer（BPE/WordPiece）；评测计算时独立实现“中文分字 + 连续字母/数字归一 token”的规则。

## 训练与推理策略
- SFT（监督微调）：
  - 目标：`CrossEntropy + LabelSmoothing(0.1)`；多参考可随机采样一条或在一个 batch 内展开多条，配合 `loss` 归一；
  - 数据增强：图像 `RandAugment/ColorJitter/随机裁切` 适度；文本规范化（去品牌露出、去噪、不改事实）。
  - 优化器与调度：AdamW，`lr=1e-4 ~ 2e-5`（视模型与 LoRA 范围），`cosine/polynomial` + `warmup`（5%）；`grad_clip=1.0`；
  - 分布式：`bf16`、`FSDP`（仅 LoRA 层全量与 embedding 层梯度），或 `DeepSpeed ZeRO-2/3`；
  - 训练步数：50k 图片、最多10参考，等效样本量可到 200k～500k（展开后）；训练 `1–3` 轮，视收敛。
- SCST / CIDEr-RL：
  - 用当前模型生成（自采样/beam）与参考比较，计算 `CIDEr-D` 回报，对比基准句（greedy）做优势函数更新；可混合 `SFT`（一定比例）保持稳定性。
- 推理：
  - 解码：`beam=5~8`、`max_len=30`、`length_penalty=0.8~1.2`、`repetition_penalty=1.2`；
  - rerank（可选）：以 `CLIPScore`/`Image-Text Matching` 对 beam 结果重排序；
  - 归一化与后处理：去空格、数字单位规范、去重复短语，保持电商风格但不虚构。

## 评测与提交
- 指标实现：
  - `CIDEr-D`：使用官方实现或 `pycocoevalcap` 的 `cider` 变体，中文处理为“分字 + 连续字母/数字视作一个 token”；
  - `BLEU-4`：以中文分字 n-gram；
  - `ROUGE-L`：基于 LCS 的中文分字版本；
- 产物：`results/caption_gen.jsonl`，每行 `{"image_id": ..., "text": ...}`；在验证集上报告三项指标，主看 `CIDEr-D`。

## 目录与文件设计
建议在 `ECommerce-IC/plan_1` 下采用以下结构：

```
ECommerce-IC/plan_1/
├── README.md                 # 本方案文档（当前文件）
├── configs/
│   ├── qwen25vl_lora.yaml    # 模型、LoRA、优化与数据路径配置
│   └── infer.yaml            # 推理与后处理配置
├── data/
│   ├── prepare_data.py       # 读取 tsv/jsonl、base64 解码、生成训练/验证索引
│   ├── dataset.py            # HF datasets/自定义 PyTorch Dataset（多参考采样）
│   └── collate.py            # 图像预处理（根据模型视觉编码器）、batch 构造
├── train/
│   ├── finetune_sft.py       # SFT 训练入口（LoRA/QLoRA、FSDP/DeepSpeed）
│   ├── finetune_scst.py      # SCST/CIDEr-RL 阶段训练入口
│   └── utils.py              # 训练循环、日志与保存、分布式工具
├── infer/
│   ├── generate.py           # 推理与 beam search、rerank、后处理，输出 jsonl
│   └── postprocess.py        # 中文去冗余、数字/单位规范、特殊符号清理
├── eval/
│   ├── cider.py              # CIDEr-D（中文分字）
│   ├── bleu.py               # BLEU-4（中文分字）
│   ├── rouge_l.py            # ROUGE-L（中文分字）
│   └── evaluate.py           # 统一评测入口，读参考与预测，输出指标
└── scripts/
    ├── train_sft.sh          # 读取 configs，启动 SFT 训练
    ├── train_scst.sh         # 读取 configs，启动 RL 训练
    ├── generate.sh           # 批量推理与结果输出
    └── eval.sh               # 验证集评测与指标汇总
```

> 说明：此结构兼容 HuggingFace Transformers + PEFT + Diffusers/vision backbone，或 OpenMMLab/MMPretrain 生态。视觉编码器由所选 MM-LLM 内置，不建议手工替换以免错配。

## 关键实现要点
- 多参考使用：
  - 训练：`一图多参考`可随机采样一条提高多样性，或在同 batch 展开多条、共享图像特征（提升稳定性）。
  - 评测：按官方规则，中文分字 + 连续字母/数字为单 token，确保与评测一致。
- LoRA/QLoRA 配置建议：
  - `target_modules`: `attention.q_proj/k_proj/v_proj/o_proj`、`mlp.down_proj/up_proj`（语言侧为主，必要时跨注意力）；
  - `r=16~64`、`alpha=32~64`、`dropout=0.05`；`bnb 4bit` + `nf4` 量化。
- 资源与加速：
  - 3B/7B：A800(80G)×1 可跑 QLoRA，A100(40G)×2 可跑 LoRA；用 `FSDP`/`ZeRO` 合理切分；
  - `bf16` 优先，`fp16` 备选；开启 `FlashAttention-2`（如模型支持）。
- 安全与事实：
  - 指令模板中约束“不得YY虚构成分，不做夸张承诺”，训练时文本清洗减少夸张营销语对事实描述的干扰。

## 执行示例（伪命令）
- 数据准备：
  - `python data/prepare_data.py --data-root /mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-IC --out data_cache/`
- SFT 训练：
  - `bash scripts/train_sft.sh --config configs/qwen25vl_lora.yaml`
- RL 训练（可选）：
  - `bash scripts/train_scst.sh --config configs/qwen25vl_lora.yaml`
- 生成：
  - `bash scripts/generate.sh --config configs/infer.yaml --subset test`
- 评测（验证集）：
  - `bash scripts/eval.sh --pred results/caption_gen.jsonl --gold IC_valid.jsonl`

## 路线图
- Week 1：数据管线与 SFT 基线（LoRA/QLoRA）打通，验证集 CIDEr-D≥1.20（示例阈值，随数据而变）。
- Week 2：引入 SCST/CIDEr-RL，beam 调参与后处理优化，CIDEr-D 提升 0.05–0.15。
- Week 3：加入 rerank、风格控制与文本规范化，最终确定提交版本。

## 风险与替代
- 显存不足：降模型规模（3B），使用 QLoRA 与更小 `r`；
- 收敛慢：增广更多“中短句”参考或引入外部中文商品文案预训练（小心版权与赛规）；
- 评测不一致：严格采用“中文分字 + 连续字母/数字为单 token”的实现，避免与官方指标偏差。

---

如需，我可以按上述结构创建脚手架文件并补齐关键脚本，实现端到端训练与生成。