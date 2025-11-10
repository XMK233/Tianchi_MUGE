import torch
import torch.nn as nn


# 2. 优化版InfoNCE损失函数
def optimized_info_nce_loss(text_feats, image_feats, temperature):
    # 温度裁剪
    safe_temp = torch.clamp(temperature, min=1e-4, max=1.0)
    # 计算相似度矩阵
    logits = torch.mm(text_feats, image_feats.t()) / safe_temp
    # 标签
    labels = torch.arange(logits.size(0), device=logits.device)
    # 双向交叉熵
    loss_t = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.t(), labels)
    return (loss_t + loss_i) * 0.5

# 3. 关键优化配置
# 在您的代码中替换：
# 1. 将原FeatureFusion中的投影头替换为OptimizedMLPProjector
# 2. 使用AdamW优化器，学习率降低到5e-4
# 3. 添加梯度裁剪：torch.nn.utils.clip_grad_norm_(projection_params, max_norm=1.0)
# 4. 增加训练轮数到10-20轮
# 5. 使用学习率调度器：lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# 6. 考虑降低dropout率到0.05

# 使用示例：
# text_projector = OptimizedMLPProjector(text_in_dim, 1024, 512, dropout=0.05)
# image_projector = OptimizedMLPProjector(image_in_dim, 1024, 512, dropout=0.05)