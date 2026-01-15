下面给出利用 coarse_label 的多种方案，分为三类：新增 embedding、改进 loss、其他策略。每一类都说明在哪个位置接入，如何落地到现有代码 [10_丰-基于9_只用5000个样本来训练验证_baseline.py](file:///mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/10_丰-基于9_只用5000个样本来训练验证_baseline.py)，并强调目标是让图更清晰、FID 更低。

**Embedding 方案**
- 标签嵌入拼接到条件通道
  - 在 `ClassConditionedUViT` 中新增 `nn.Embedding(num_classes, class_emb_size_label)`，将 coarse_label 映射为向量后与当前 `c_feat` 拼接，再展开为 2D 条件通道与 latent 拼接。
  - 位置：`ClassConditionedUViT.__init__` 与 `forward`（文件: plan_2/10_丰-基于9_只用5000个样本来训练验证_baseline.py:233 起）。
  - 效果：为“类别先验”提供独立语义通道，有助于收敛到更清晰的大轮廓。
- FiLM/AdaIN 条件调制
  - 用 coarse_label 的嵌入生成每个卷积块的 `scale`/`bias`，对特征作 Feature-wise Linear Modulation。
  - 位置：`down1/down2/down3/up1/up2/up3` 的前向里，在激活前插入 `y = y * gamma(label) + beta(label)`；参数由一个小 MLP 从标签嵌入生成。
  - 效果：让“类信息”直接控制特征分布，显著增强结构一致性（衣服、鞋等的轮廓）。
- MoE 路由由标签主导
  - 将 `gate_in` 改为 `concat(label_emb, t_emb)` 或者 `concat(label_emb, text_pool, t_emb)`，标签对专家选择起主导作用，文本作细化。
  - 位置：`ClassConditionedUViT.forward` 的门控输入构造（约 plan_2/10 文件:372）。
  - 效果：不同 coarse 类走不同专家通路，减少“模糊混类”的情况。

**Loss 改进**
- 辅助分类头（多任务）
  - 在瓶颈或 `p3_pool` 上接一个线性分类头预测 coarse_label，使用 `CrossEntropyLoss`。
  - 位置：在 `ClassConditionedUViT` 增加 `self.cls_head = nn.Linear(128, num_classes)`，训练时在 `DiffusionModelWrapper.forward` 返回 `loss = mse + λ * cls_loss`。
  - 效果：逼迫中间特征携带清晰类别语义，强化主体轮廓，让图不再发散。
- InfoNCE/对比约束（标签原型）
  - 维护每类一个“原型向量”（可学习或动量更新），将图像特征与对应类原型做正样本、与其他类做负样本，`InfoNCE` 或 `SupConLoss`。
  - 位置：在 `DiffusionModelWrapper.forward` 提取 `p3_pool`，计算对比损失并加权到总损失。
  - 效果：收紧类内表征，拉远类间表征，提升清晰度与可辨识性。
- 边缘/结构感知的正则
  - 在解码图 `img` 上加轻量的结构损失：`TVLoss`、Sobel/Laplacian 边缘的一致性（与真实图或同类统计）或 SSIM 组件。
  - 位置：训练阶段从 `vae.decode(noisy_latent_step)` 获取中间重建图，计算结构损失加入总 loss（权重要小，避免破坏扩散噪声学习）。
  - 效果：鼓励锐利边缘，降低“糊”感（对 FID 也通常有正面作用）。

**其他策略**
- 类均衡采样与 Batch 构型
  - 在 `ECommerceDataset` 层面读入 coarse_label，DataLoader 采用分层采样或每 batch 保证类分布均衡。
  - 效果：避免训练集中被某些类主导，提升各类清晰度一致性。
- Classifier-Free Guidance（文本+标签双引导）
  - 训练时随机 drop 文本或标签条件，采样时用 guidance scale 叠加：`x_cond = w_text*x(text) + w_label*x(label)` 与无条件输出作差提升条件性。
  - 位置：`DiffusionModelWrapper.forward` 支持条件丢弃；`sample` 中实现 CFG 混合。
  - 效果：既保留细节（文本），又强化主体（标签），清晰度与类一致性更好。
- 两阶段采样（先类后文）
  - 第一阶段用 coarse_label 条件采样到中间噪声；第二阶段引入文本条件微调细节。
  - 位置：`sample` 中分两段 timesteps，前半只用 label 条件，后半叠加 text 条件。
  - 效果：先打好轮廓，再加细节，显著减少模糊。
- 采样步数与噪声日程优化
  - 适当提高 `num_inference_steps`（如 50→75/100），或用更平滑的噪声 schedule（如 `cosine`），并叠加少量去噪后的锐化滤波（unsharp mask）。
  - 效果：更稳定的反演，图像更清晰。

**落地建议（接入点与数据读取）**
- 读取 coarse_label
  - 在脚本启动时加载 `/plan_2/1-利用文本进行分类/classification_results_coarse.csv`，构造 `img_id → coarse_label` 映射。
  - 在 `ECommerceDataset.__getitem__`（plan_2/10 文件:129 起）通过 `img_id` 取 `label_id`，返回字段 `coarse_label_id`。
- 模型接入
  - 在 `ClassConditionedUViT.__init__` 新增 `self.label_emb = nn.Embedding(num_classes, class_emb_size)`；在 `forward` 将 `label_emb` 融合到条件与/或门控。
  - 在 `DiffusionModelWrapper.forward` 追加辅助损失（分类或对比）；在 `sample` 支持 CFG 与两阶段采样。
- 训练
  - 调整 `train(...)` 函数（plan_2/10 文件:601 起）从 batch 取 `coarse_label_id`；优化器与保存检查点维持不变。
  - 保持现有 VAE 与数据预处理，避免引入新的颜色/尺寸不一致。

以上方案可以组合使用：首选“标签嵌入 + 辅助分类头 + 类均衡采样 + 两阶段采样”，这是工程上投入较小、收益明显的组合；若希望进一步降低 FID，再加上轻量的对比损失与 CFG 混合。在你确定方案后，我可以直接把这些改动以最小侵入的方式接入到当前 [10_丰] 脚本里。