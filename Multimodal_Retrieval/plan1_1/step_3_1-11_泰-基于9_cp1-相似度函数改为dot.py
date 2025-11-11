#!/usr/bin/env python
# coding: utf-8

# * early: 早融合
# * projection：后融合

# In[1]:

import argparse
parser = argparse.ArgumentParser(description="设置一些参数")
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["test", "actual_run", ], 
    default="test",
    help="运行还是测试模式"
)
args = parser.parse_args()

import os
import sys
import math
from typing import List, Dict, Tuple
import torch
import timm
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import importlib

# 可选：Faiss 加速检索（安全导入，避免 NumPy 版本问题导致报错中断）
HAS_FAISS = False
faiss = None
try:
    spec = importlib.util.find_spec('faiss')
    if spec is not None:
        faiss = importlib.import_module('faiss')
        HAS_FAISS = True
except BaseException as e:
    print(f'Faiss unavailable: {e.__class__.__name__}')
    HAS_FAISS = False

# 设置环境变量（与基线一致，按需修改为本地镜像/缓存）
cache_dir = "/mnt/d/HuggingFaceModels/"
os.environ['TORCH_HOME'] = cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 导入数据加载器（使用 plan1_1/data_loader.py）
sys.path.append(os.path.abspath(os.path.join('.', 'Multimodal_Retrieval', 'plan1_1')))
from data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[2]:


save_dir = '/mnt/d/forCoding_data/Tianchi_MUGE/trained_models/weights'
save_path = os.path.join(save_dir, 'step_3_1_1__.pth')


# In[ ]:





# In[3]:


# import time
# while True:
#     if os.path.exists("step_3_1-7_师_cp1-基于6_cp2-fc前特征也就是gap.finishflag"):
#         break
#     time.sleep(5)


# In[ ]:





# ## 模型与特征模块
# 文本特征使用基于 `attention_mask` 的 mean-pooling（排除 padding），相比仅用 [CLS] 更稳健。训练时允许梯度。
# 

# In[4]:


import torch.nn.functional as F
from torch import nn
# 1. 优化版两层MLP投影头（核心组件）
class OptimizedMLPProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, use_bn=True):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        # Kaiming初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


# In[5]:


class TextFeatureExtractor:
    def __init__(self, model_name='bert-base-chinese', device='cpu', cache_dir=None, pooling='mean'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)#
        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).to(device)#, local_files_only=True
        self.pooling = pooling
        if self.pooling == 'attentive':
            self.attn = nn.Linear(768, 1).to(self.device)
        # 默认 eval，训练时将对子模块单独切换 train
        self.model.eval()

    def encode_with_grad(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 768), dtype=torch.float32, device=self.device)
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state  # [B, L, 768]
        attention_mask = inputs.get('attention_mask', None)
        if self.pooling == 'cls':
            return token_embeddings[:, 0, :]
        if self.pooling == 'attentive':
            # 运行时再确保注意力层与当前张量设备一致，增强健壮性
            self.attn = self.attn.to(token_embeddings.device)
            scores = self.attn(token_embeddings).squeeze(-1)  # [B, L]
            if attention_mask is not None:
                mask_pad = (attention_mask == 0)
                scores = scores.masked_fill(mask_pad, torch.finfo(scores.dtype).min)
            # 在 FP16 下用 FP32 计算 softmax 更稳健
            weights = torch.softmax(scores.to(torch.float32), dim=1).to(scores.dtype).unsqueeze(-1)  # [B, L, 1]
            return (token_embeddings * weights).sum(dim=1)
        if attention_mask is None:
            # 兜底：无 mask 时退化为 CLS
            return token_embeddings[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # [B, L, 1]
        summed = (token_embeddings * mask).sum(dim=1)  # [B, 768]
        lengths = mask.sum(dim=1).clamp(min=1)        # [B, 1]
        mean_pooled = summed / lengths
        return mean_pooled

# class ImageFeatureExtractor:
#     def __init__(self, model_name='resnet50', device='cpu', cache_dir=None):
#         self.device = device
#         self.model = timm.create_model(
#             model_name, pretrained=True, num_classes=0,
#             cache_dir=cache_dir
#         ).to(device)
#         self.model.eval()
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def encode_with_grad(self, images: List[Image.Image]) -> torch.Tensor:
#         if not images:
#             return torch.empty((0, 2048), dtype=torch.float32, device=self.device)
#         tensors = torch.stack([self.transform(img.convert('RGB')) for img in images]).to(self.device)
#         feats = self.model(tensors)
#         return feats
# image_extractor = ImageFeatureExtractor(device=device, cache_dir=cache_dir)

from safetensors.torch import load_file
class ImageFeatureExtractor:
    '''
    改进版，使得 timm 不要每次都去连 huggingface；
    并新增 feature_source 开关以对比 GAP 后（fc 前）与 fc 后特征。

    注意：本模型在创建时设置 num_classes=0（无分类头），
    timm 的 forward 将返回 GAP 后的特征；若选择 'fc' 且存在真实分类头，
    则返回 fc 后向量；否则退化为 GAP 特征（与 'gap' 等价）。
    '''
    def __init__(self, model_name='resnet101', device='cpu', weights_path=None, cache_dir=None, feature_source: str = 'gap'):
        self.device = device
        self.feature_source = feature_source  # 'gap' 为全局池化后（fc 前）；'fc' 为分类层后
        # 保持原行为：使用 num_classes=0 得到 backbone 的特征输出
        if weights_path is None:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0, cache_dir=cache_dir)
        else:
            self.model = timm.create_model(
                model_name, pretrained=False, num_classes=0, cache_dir=cache_dir,
                pretrained_cfg_overlay={'file': weights_path}
            )

        if weights_path is not None:
            if weights_path.endswith('.safetensors'):
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def output_dim(self) -> int:
        """
        返回当前 feature_source 下的输出维度。
        - gap: 使用 backbone 的 num_features。
        - fc: 若存在真实分类头，返回 num_classes；否则退化为 num_features。
        """
        in_dim = getattr(self.model, 'num_features', 2048)
        if self.feature_source == 'fc':
            num_classes = getattr(self.model, 'num_classes', None)
            if isinstance(num_classes, int) and num_classes > 0:
                in_dim = num_classes
        return in_dim

    def encode_with_grad(self, images: List[Image.Image]) -> torch.Tensor:
        '''
        根据 feature_source 返回不同阶段的图像特征：
        - 'gap'：layer4 后经全局平均池化的特征（fc 前特征，维度通常为 num_features，例如 ResNet101 的 2048）。
        - 'fc'：若存在真实分类头则返回其输出（fc 后特征）；否则回退为 GAP 特征。
        '''
        if not images:
            return torch.empty((0, self.output_dim()), dtype=torch.float32, device=self.device)

        tensors = torch.stack([self.transform(img.convert('RGB')) for img in images]).to(self.device)

        if self.feature_source == 'gap':
            # 使用 forward_features + global_pool 获取 GAP 后的特征（fc 前）
            if hasattr(self.model, 'forward_features'):
                x = self.model.forward_features(tensors)
                if hasattr(self.model, 'global_pool'):
                    x = self.model.global_pool(x)
                else:
                    # 兜底：若模型未暴露 global_pool，则用自适应池化
                    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
                return x
            # 兜底：直接使用 forward（在 num_classes=0 时也会返回 GAP 特征）
            return self.model(tensors)

        # feature_source == 'fc'：若存在真实分类头，返回其输出；否则退化为 GAP 特征
        # 由于本模型创建时 num_classes=0，forward 等同于 GAP 特征；
        # 若你希望比较真实 fc 后的向量，可改为以 num_classes=1000 创建，并相应调整下游维度。
        return self.model(tensors)

class FeatureFusion:
    # 类作用：将原始文本/图像特征投影到共同的子空间（projection_dim）
    # 参数:
    # - fusion_method: 融合方式，当前支持 'projection'
    # - projection_dim: 目标投影维度
    # - device: 设备
    # - hidden_dim: 两层 MLP 的中间隐藏层维度
    # - dropout: Dropout 概率
    # - text_in_dim/image_in_dim: 输入维度（默认文本768/图像2048）
    def __init__(self, fusion_method='projection', projection_dim=512, device=None, hidden_dim=1024, dropout=0.1, text_in_dim=768, image_in_dim=2048):
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        if fusion_method == 'projection':
            # self.text_projector = torch.nn.Sequential(
            #     torch.nn.Linear(text_in_dim, hidden_dim),
            #     torch.nn.GELU(),
            #     torch.nn.Dropout(p=dropout),
            #     torch.nn.Linear(hidden_dim, projection_dim)
            # ).to(self.device)
            # self.image_projector = torch.nn.Sequential(
            #     torch.nn.Linear(image_in_dim, hidden_dim),
            #     torch.nn.GELU(),
            #     torch.nn.Dropout(p=dropout),
            #     torch.nn.Linear(hidden_dim, projection_dim)
            # ).to(self.device)

            # self.text_projector = OptimizedMLPProjector(text_in_dim, hidden_dim, projection_dim, dropout=dropout).to(self.device)
            # self.image_projector = OptimizedMLPProjector(image_in_dim, hidden_dim, projection_dim, dropout=dropout).to(self.device)

            self.text_projector = torch.nn.Linear(text_in_dim, projection_dim).to(self.device)
            self.image_projector = torch.nn.Linear(image_in_dim, projection_dim).to(self.device)

        elif fusion_method == 'early':
            # 修改点：新增早融合统一评分网络：拼接 [text; image] 后用两层MLP输出分数
            self.pair_projector = OptimizedMLPProjector(text_in_dim + image_in_dim, hidden_dim, 1, dropout=dropout).to(self.device)
            ## torch.nn.Linear(text_in_dim + image_in_dim, 1).to(self.device)
            ## OptimizedMLPProjector(text_in_dim + image_in_dim, hidden_dim, 1, dropout=dropout).to(self.device)

    def fuse_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        # 后融合：返回投影后的文本特征；早融合：直接返回原始文本特征用于后续拼接
        return self.text_projector(text_features) if self.fusion_method == 'projection' else text_features
    def fuse_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        # 后融合：返回投影后的图像特征；早融合：直接返回原始图像特征用于后续拼接
        return self.image_projector(image_features) if self.fusion_method == 'projection' else image_features
    def pair_scores_matrix(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # 早融合：对所有 (文本, 图像) 对拼接并统一打分，返回 [B_text, B_image] 分数矩阵
        assert self.fusion_method == 'early', "pair_scores_matrix 仅在 'early' 模式下使用"
        Bt = text_features.size(0)
        Bi = image_features.size(0)
        t_exp = text_features.unsqueeze(1).expand(Bt, Bi, text_features.size(1))
        i_exp = image_features.unsqueeze(0).expand(Bt, Bi, image_features.size(1))
        cat = torch.cat([t_exp, i_exp], dim=-1)
        scores = self.pair_projector(cat.view(Bt * Bi, -1)).view(Bt, Bi)
        return scores

class SimilarityCalculator:
    def __init__(self, similarity_type='cosine'):
        self.similarity_type = similarity_type
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(features, p=2, dim=1)
    def calculate_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        if self.similarity_type == 'cosine':
            t_n = self.normalize_features(text_features)
            i_n = self.normalize_features(image_features)
            return torch.mm(t_n, i_n.t())
        return torch.mm(text_features, image_features.t())

class CrossModalRetrievalModel:
    def __init__(self, text_extractor, image_extractor, fusion_method='projection', projection_dim=512, similarity_type='cosine', normalize_features=True, device=None):
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        # 动态适配图像特征维度：依据 image_extractor.feature_source 决定输入维度
        img_in_dim = getattr(self.image_extractor.model, 'num_features', 2048)
        if getattr(self.image_extractor, 'feature_source', 'gap') == 'fc':
            num_classes = getattr(self.image_extractor.model, 'num_classes', None)
            if isinstance(num_classes, int) and num_classes > 0:
                img_in_dim = num_classes
        self.fusion = FeatureFusion(fusion_method, projection_dim, device, image_in_dim=img_in_dim)
        self.sim = SimilarityCalculator(similarity_type)
        self.normalize_features = normalize_features
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1) if self.normalize_features else x
    def extract_and_fuse_text_features(self, texts: List[str]) -> torch.Tensor:
        # 评估阶段禁用 BN/Dropout 的训练行为
        with torch.no_grad():
            t = self.text_extractor.encode_with_grad(texts)
        if self.fusion.fusion_method == 'projection':
            self.fusion.text_projector.eval()
            return self._norm(self.fusion.fuse_text_features(t))
        # 早融合：返回原始文本特征
        return t
    def extract_and_fuse_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        # 评估阶段禁用 BN/Dropout 的训练行为
        with torch.no_grad():
            i = self.image_extractor.encode_with_grad(images)
        if self.fusion.fusion_method == 'projection':
            self.fusion.image_projector.eval()
            return self._norm(self.fusion.fuse_image_features(i))
        # 早融合：返回原始图像特征
        return i
    def compute_logits(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # 修改点：新增统一的 logits 计算接口
        if self.fusion.fusion_method == 'early':
            return self.fusion.pair_scores_matrix(text_features, image_features)
        t_proj = self._norm(self.fusion.fuse_text_features(text_features))
        i_proj = self._norm(self.fusion.fuse_image_features(image_features))
        return torch.mm(t_proj, i_proj.t())
    def build_image_index(self, images_dict: Dict[str, Image.Image], batch_size: int = 32) -> Dict[str, torch.Tensor]:
        feats = {}
        keys = list(images_dict.keys())
        for s in range(0, len(keys), batch_size):
            batch_ids = keys[s:s+batch_size]
            batch_imgs = [images_dict[k] for k in batch_ids if images_dict[k] is not None]
            valid_ids = [k for k in batch_ids if images_dict[k] is not None]
            if not batch_imgs:
                continue
            bf = self.extract_and_fuse_image_features(batch_imgs)
            for j, img_id in enumerate(valid_ids):
                feats[img_id] = bf[j].detach().cpu()
        return feats

def info_nce_loss(text_feats: torch.Tensor, image_feats: torch.Tensor, temp: float) -> torch.Tensor:
    # 修改点：在 'early' 模式下，外部会传入原始特征并用统一打分接口产生 logits
    logits = torch.mm(text_feats, image_feats.t()) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t = torch.nn.functional.cross_entropy(logits, labels)
    loss_i = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_t + loss_i) * 0.5


# In[ ]:





# ## 顶层解冻与优化器分组
# 仅解冻顶层，降低学习率，控制训练稳定性。
# 

# In[6]:


def unfreeze_text_top_layers(text_extractor: TextFeatureExtractor, last_n_layers: int = 2):
    for p in text_extractor.model.parameters():
        p.requires_grad = False
    enc = text_extractor.model.encoder
    total_layers = len(enc.layer)
    for i in range(total_layers - last_n_layers, total_layers):
        for p in enc.layer[i].parameters():
            p.requires_grad = True
        enc.layer[i].train()
    if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        for p in text_extractor.model.pooler.parameters():
            p.requires_grad = True
        text_extractor.model.pooler.train()
    text_extractor.model.eval()

def unfreeze_image_top_block(image_extractor: ImageFeatureExtractor, unfreeze_layer4: bool = True):
    for p in image_extractor.model.parameters():
        p.requires_grad = False
    if unfreeze_layer4 and hasattr(image_extractor.model, 'layer4'):
        for p in image_extractor.model.layer4.parameters():
            p.requires_grad = True
        image_extractor.model.layer4.train()
    image_extractor.model.eval()

# 修改点（依据 3.1.3 的改进方案 L12）：
# 新增通用图像解冻函数，可在顶层解冻与全解冻之间切换，以便对比训练稳定性与检索效果。
def unfreeze_image(image_extractor: ImageFeatureExtractor, mode: str = 'top'):
    """
    mode='top'：仅解冻 layer4（顶层 block），保持大部分骨干冻结，训练更稳健；
    mode='full'：全解冻图像骨干（包含 stem/各层 block），适配更强的适应能力但需更稳的优化策略。
    """
    # 先全部冻结
    for p in image_extractor.model.parameters():
        p.requires_grad = False
    if mode == 'top':
        if hasattr(image_extractor.model, 'layer4'):
            for p in image_extractor.model.layer4.parameters():
                p.requires_grad = True
            image_extractor.model.layer4.train()
    else:  # mode == 'full'
        # 解冻 stem 与各层 block
        for name in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(image_extractor.model, name):
                mod = getattr(image_extractor.model, name)
                for p in mod.parameters():
                    p.requires_grad = True
                mod.train()
    image_extractor.model.eval()
def build_optimizer(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                   lr_proj: float = 1e-3, lr_text_top: float = 5e-5, lr_img_top: float = 1e-4, weight_decay: float = 1e-4):
    params = []
    # 修改点：根据融合方式设置优化器分组
    if model.fusion.fusion_method == 'early':
        params.append({
            'params': list(model.fusion.pair_projector.parameters()),
            'lr': lr_proj,
            'weight_decay': weight_decay
        })
    else:
        params.append({
            'params': list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
            'lr': lr_proj,
            'weight_decay': weight_decay
        })
    text_top_params = []
    enc = text_extractor.model.encoder
    for mod in enc.layer[-2:]:
        text_top_params += list(mod.parameters())
    if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        text_top_params += list(text_extractor.model.pooler.parameters())
    params.append({
        'params': [p for p in text_top_params if p.requires_grad],
        'lr': lr_text_top,
        'weight_decay': 0.0
    })
    img_top_params = []
    if hasattr(image_extractor.model, 'layer4'):
        img_top_params += list(image_extractor.model.layer4.parameters())
    params.append({
        'params': [p for p in img_top_params if p.requires_grad],
        'lr': lr_img_top,
        'weight_decay': 0.0
    })
    optimizer = torch.optim.Adam(params)
    return optimizer

# LLRD（Layer-wise LR Decay）优化器构建：为BERT顶层设置逐层衰减学习率
# 修改点（依据 3.1.3 的改进方案 L12）：
# 为图像编码器新增 LLRD 的全解冻分层学习率：layer4 > layer3 > layer2 > layer1 > stem。
def build_llrd_optimizer(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                         lr_proj: float = 1e-3, lr_text_max: float = 5e-5, lr_img_top: float = 1e-4, decay: float = 0.9,
                         last_n_layers: int = 2, weight_decay: float = 1e-4, img_unfreeze_mode: str = 'top'):
    params = []
    # 投影层分组：后融合为两个线性层；早融合为统一 pair_projector
    if model.fusion.fusion_method == 'early':
        params.append({
            'params': list(model.fusion.pair_projector.parameters()),
            'lr': lr_proj,
            'weight_decay': weight_decay
        })
    else:
        params.append({
            'params': list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
            'lr': lr_proj,
            'weight_decay': weight_decay
        })
    # 文本顶层：逐层衰减（最顶层lr=lr_text_max，其次乘以decay）
    enc = text_extractor.model.encoder
    total_layers = len(enc.layer)
    start_idx = max(0, total_layers - last_n_layers)
    # 从顶层到次顶层设置lr
    order = 0
    for i in range(total_layers - 1, start_idx - 1, -1):
        group_lr = lr_text_max * (decay ** order)
        params.append({
            'params': [p for p in enc.layer[i].parameters() if p.requires_grad],
            'lr': group_lr,
            'weight_decay': 0.0
        })
        order += 1
    # pooler（若存在），使用与最顶层一致的lr
    if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        params.append({
            'params': [p for p in text_extractor.model.pooler.parameters() if p.requires_grad],
            'lr': lr_text_max,
            'weight_decay': 0.0
        })
    # 将注意力池化权重也纳入文本顶层优化
    if hasattr(text_extractor, 'attn'):
        params.append({
            'params': [p for p in text_extractor.attn.parameters() if p.requires_grad],
            'lr': lr_text_max,
            'weight_decay': 0.0
        })
    # 图像侧 LLRD：
    if img_unfreeze_mode == 'top':
        # 仅顶层 block 使用较大学习率
        if hasattr(image_extractor.model, 'layer4'):
            params.append({
                'params': [p for p in image_extractor.model.layer4.parameters() if p.requires_grad],
                'lr': lr_img_top,
                'weight_decay': 0.0
            })
    else:
        # 全解冻：按照 ResNet 层次递减学习率
        # 从顶到底依次 layer4, layer3, layer2, layer1, stem(conv1+bn1)
        order = 0
        for name in ['layer4', 'layer3', 'layer2', 'layer1']:
            if hasattr(image_extractor.model, name):
                group_lr = lr_img_top * (decay ** order)
                params.append({
                    'params': [p for p in getattr(image_extractor.model, name).parameters() if p.requires_grad],
                    'lr': group_lr,
                    'weight_decay': 0.0
                })
                order += 1
        # stem（conv1+bn1），使用更小 lr
        stem_params = []
        if hasattr(image_extractor.model, 'conv1'):
            stem_params += list(image_extractor.model.conv1.parameters())
        if hasattr(image_extractor.model, 'bn1'):
            stem_params += list(image_extractor.model.bn1.parameters())
        if stem_params:
            params.append({
                'params': [p for p in stem_params if p.requires_grad],
                'lr': lr_img_top * (decay ** 4),
                'weight_decay': 0.0
            })
    optimizer = torch.optim.Adam(params)
    return optimizer

# Warmup + Cosine 学习率调度器
def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_ratio: float, min_lr_ratio: float, total_steps: int):
    warmup_steps = max(1, int(total_steps * max(0.0, min(warmup_ratio, 0.5))))
    min_ratio = max(0.0, min(min_lr_ratio, 1.0))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay from 1.0 -> min_ratio
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ## 数据加载与训练参数
# 保持与基线一致的查询数据加载；图片按批次流式以控制显存。默认使用较小批次以便顺利运行。
# 

# In[7]:


loader = DataLoader()
train_df = loader.load_queries(split='train')
valid_df = loader.load_queries(split='valid')

if args.mode == "test":
    # 训练与流式参数（默认较小，确保顺利运行；可按需增大）
    train_image_batch_size = 500
    max_train_batches = 1
    epochs_per_batch = 1
    train_step_batch_size = 32
    valid_imgs_max_samples = 100
else:
    # 训练与流式参数（按需调整）：实际用
    train_image_batch_size = 15000 ## 一个大batch有这么多图片样本。
    max_train_batches = 10 ## 总共加载多少个大batch。
    epochs_per_batch = 3 ## 每个大batch训练几个epoch。
    train_step_batch_size = 32 ## 每个大batch里面训练的时候的小batch_size是多少。
    valid_imgs_max_samples = 30000

use_amp = True
temperature = 0.07

# 微调与调度参数
last_n_layers = 8  # 顶层解冻层数
warmup_ratio = 0.1  # 预热比例
min_lr_ratio = 0.1  # 余弦最低学习率相对比例
use_grad_checkpoint = False  # 可选：启用BERT梯度检查点以降低显存


# ## 初始化模型并执行顶层解冻
# 解冻文本最后2层与池化；解冻图像 `layer4`。
# 

# In[8]:


image_extractor = ImageFeatureExtractor(
    device=device,
    feature_source='fc',  # 'gap' 使用全局池化后的 fc 前特征；改为 'fc' 可对比分类头后的向量
    model_name='resnet101', 
    weights_path='/mnt/d/HuggingFaceModels/models--timm--resnet50.a1_in1k/snapshots/767268603ca0cb0bfe326fa87277f19c419566ef/model.safetensors'

)
text_extractor = TextFeatureExtractor(
    model_name = "hfl/chinese-roberta-wwm-ext", 
    device=device, 
    cache_dir=cache_dir,
    pooling='attentive'
)


# 修改点（依据 3.1.3 的改进方案 L16）：将融合方式切换为“后融合”（projection），
# 即文本与图像分别投影并归一化后，再用相似度计算器打分；便于与“早融合”对比。
model = CrossModalRetrievalModel(
    text_extractor, image_extractor, 
    fusion_method='projection', projection_dim=512, similarity_type='dot', normalize_features=False, device=device
)

# 可选：启用BERT梯度检查点以降低显存
if use_grad_checkpoint and hasattr(text_extractor.model, 'gradient_checkpointing_enable'):
    text_extractor.model.gradient_checkpointing_enable()

unfreeze_text_top_layers(text_extractor, last_n_layers=last_n_layers)
# 修改点（依据 3.1.3 的改进方案 L12）：切换图像解冻模式以做实验对比
img_unfreeze_mode = 'full'  # 可选 'top'（仅layer4）或 'full'（全解冻）
unfreeze_image(image_extractor, mode=img_unfreeze_mode)

# 使用LLRD优化器：为文本顶层设置逐层衰减的学习率
# 修改点（依据 3.1.3 的改进方案 L12）：为全解冻提供分层学习率（LLRD）
optim = build_llrd_optimizer(model, text_extractor, image_extractor,
                             lr_proj=1e-3, lr_text_max=5e-5, lr_img_top=1e-4, decay=0.9,
                             last_n_layers=last_n_layers, weight_decay=1e-4, img_unfreeze_mode=img_unfreeze_mode)
scaler = GradScaler(enabled=(device.type == 'cuda' and use_amp))
print('Optim groups:', len(optim.param_groups))


# ## 训练循环：按批次流式构建配对并微调顶层
# 仅使用配对中的第一张可用图片；文本与图像编码器顶层参与反向传播。
# 

# In[9]:


def build_batch_pairs(train_df, img_dict: Dict[str, Image.Image]) -> List[Tuple[str, Image.Image, str]]:
    pairs = []
    if 'item_ids' in train_df.columns:
        for _, row in train_df.iterrows():
            q = row.get('query_text', None)
            ids = row.get('item_ids', [])
            if not q or not isinstance(ids, list) or not ids:
                continue
            chosen_img = None
            chosen_id = None
            for iid in ids:
                sid = str(iid)
                if sid in img_dict and img_dict[sid] is not None:
                    chosen_img = img_dict[sid]
                    chosen_id = sid
                    break
            if chosen_img is not None:
                pairs.append((q, chosen_img, chosen_id))
    return pairs

def train_one_batch(pairs: List[Tuple[str, Image.Image, str]], epochs: int, step_bs: int):
    model.fusion.text_projector.train()
    model.fusion.image_projector.train()
    text_extractor.model.encoder.layer[-1].train()
    text_extractor.model.encoder.layer[-2].train()
    if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        text_extractor.model.pooler.train()
    if hasattr(image_extractor.model, 'layer4'):
        image_extractor.model.layer4.train()

    # 为当前大batch构建 Warmup+Cosine 学习率调度器（按总steps）
    steps_per_epoch = math.ceil(len(pairs) / max(1, step_bs))
    total_steps = epochs * max(1, steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(optim, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio, total_steps=total_steps)

    for e in range(epochs):
        running_loss = 0.0
        steps = 0
        for s in range(0, len(pairs), step_bs):
            batch = pairs[s:s+step_bs]
            if not batch:
                continue
            texts = [t for (t, _, _) in batch]
            imgs = [im for (_, im, _) in batch]

            optim.zero_grad()
            if use_amp and device.type == 'cuda':
                with autocast(enabled=True):
                    # 修改点：支持早融合与后融合两种路径
                    t_feats = text_extractor.encode_with_grad(texts)
                    i_feats = image_extractor.encode_with_grad(imgs)
                    if model.fusion.fusion_method == 'early':
                        # 早融合：拼接后统一打分得到 logits，再除以温度进入 InfoNCE
                        logits = model.compute_logits(t_feats, i_feats) / temperature
                        labels = torch.arange(logits.size(0), device=logits.device)
                        loss_t = torch.nn.functional.cross_entropy(logits, labels)
                        loss_i = torch.nn.functional.cross_entropy(logits.t(), labels)
                        loss = (loss_t + loss_i) * 0.5
                    else:
                        # 后融合：各自投影并归一化后做相似度再 InfoNCE
                        t_proj = model._norm(model.fusion.fuse_text_features(t_feats))
                        i_proj = model._norm(model.fusion.fuse_image_features(i_feats))
                        loss = info_nce_loss(t_proj, i_proj, temp=temperature)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
                    max_norm=5.0
                )
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            else:
                t_feats = text_extractor.encode_with_grad(texts)
                i_feats = image_extractor.encode_with_grad(imgs)
                if model.fusion.fusion_method == 'early':
                    logits = model.compute_logits(t_feats, i_feats) / temperature
                    labels = torch.arange(logits.size(0), device=logits.device)
                    loss_t = torch.nn.functional.cross_entropy(logits, labels)
                    loss_i = torch.nn.functional.cross_entropy(logits.t(), labels)
                    loss = (loss_t + loss_i) * 0.5
                else:
                    t_proj = model._norm(model.fusion.fuse_text_features(t_feats))
                    i_proj = model._norm(model.fusion.fuse_image_features(i_feats))
                    loss = info_nce_loss(t_proj, i_proj, temp=temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
                    max_norm=5.0
                )
                optim.step()
                scheduler.step()
            running_loss += loss.item()
            steps += 1
            # if (steps % 100) == 0:
            #     print('Current LRs:', [pg['lr'] for pg in optim.param_groups])
        print(f"Epoch {e+1}/{epochs}: avg loss={running_loss/max(steps,1):.4f}")

# 流式加载图片与训练
batch_idx = 0
for image_batch in loader.load_images_batch(split='train', batch_size=train_image_batch_size, max_batches=max_train_batches):
    batch_idx += 1
    img_map = {item['img_id']: item['image'] for item in image_batch}
    pairs = build_batch_pairs(train_df, img_map)
    print(f"Batch {batch_idx}: images={len(img_map)}, usable_pairs={len(pairs)}")
    if not pairs:
        del img_map
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        continue
    train_one_batch(pairs, epochs=epochs_per_batch, step_bs=train_step_batch_size)
    del img_map
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ## 保存：投影层 + 已解冻顶层 + 优化器
# 保存 BERT 的最后2层 + pooler、ResNet50 的 layer4、投影层、优化器。
# 

# In[10]:


def save_unfreeze_checkpoint(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                             optimizer: torch.optim.Optimizer, save_path: str, last_n_layers: int):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        'projection_dim': model.fusion.projection_dim,
        'last_n_layers': last_n_layers,
        'fusion': {
            'text_projector': model.fusion.text_projector.state_dict(),
            'image_projector': model.fusion.image_projector.state_dict(),
        },
        'text_unfrozen': {},
        'image_unfrozen': {},
        'optimizer': optimizer.state_dict(),
    }
    enc = text_extractor.model.encoder
    total_layers = len(enc.layer)
    start_idx = max(0, total_layers - last_n_layers)
    for i in range(start_idx, total_layers):
        ckpt['text_unfrozen'][f'encoder_layer_{i}'] = enc.layer[i].state_dict()
    if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        ckpt['text_unfrozen']['pooler'] = text_extractor.model.pooler.state_dict()
    if hasattr(image_extractor.model, 'layer4'):
        ckpt['image_unfrozen']['layer4'] = image_extractor.model.layer4.state_dict()
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to: {save_path}")

# 保存一次
save_unfreeze_checkpoint(model, text_extractor, image_extractor, optim, save_path, last_n_layers)


# ## 加载：恢复解冻顶层与投影层权重，继续训练
# 加载后会自动再次执行顶层解冻，并恢复优化器状态（如提供）。
# 

# In[11]:


def load_unfreeze_checkpoint(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                             optimizer: torch.optim.Optimizer, load_path: str):
    ckpt = torch.load(load_path, map_location='cpu')
    model.fusion.text_projector.load_state_dict(ckpt['fusion']['text_projector'])
    model.fusion.image_projector.load_state_dict(ckpt['fusion']['image_projector'])
    ln = ckpt.get('last_n_layers', last_n_layers)
    unfreeze_text_top_layers(text_extractor, last_n_layers=ln)
    unfreeze_image_top_block(image_extractor, unfreeze_layer4=True)
    enc = text_extractor.model.encoder
    for k, v in ckpt['text_unfrozen'].items():
        if k.startswith('encoder_layer_'):
            idx = int(k.split('_')[-1])
            if 0 <= idx < len(enc.layer):
                enc.layer[idx].load_state_dict(v)
    if 'pooler' in ckpt['text_unfrozen'] and hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
        text_extractor.model.pooler.load_state_dict(ckpt['text_unfrozen']['pooler'])
    if 'layer4' in ckpt['image_unfrozen'] and hasattr(image_extractor.model, 'layer4'):
        image_extractor.model.layer4.load_state_dict(ckpt['image_unfrozen']['layer4'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"Checkpoint loaded from: {load_path}")

# 测试加载
# load_unfreeze_checkpoint(model, text_extractor, image_extractor, optim, save_path)


# ## 验证评估：Recall@1/5/10 与 MeanRecall
# - 基于验证集构建图像索引
# - 对每条查询计算相似度并统计召回（优先使用 FAISS；不可用则 Torch 回退）
# 

# In[12]:


valid_imgs = loader.create_img_id_to_image_dict(
    split='valid', 
    max_samples=valid_imgs_max_samples,
    # batch_size=3000,
    # max_batches=10
)

valid_queries = []
if 'item_ids' in valid_df.columns:
    for _, row in valid_df.iterrows():
        q = row.get('query_text', None)
        ids = [str(i) for i in row.get('item_ids', [])] if isinstance(row.get('item_ids', []), list) else []
        if q and ids:
            valid_queries.append((q, ids))
print(f'Usable valid queries: {len(valid_queries)}')

image_index = model.build_image_index(valid_imgs, batch_size=32)
all_image_ids = list(image_index.keys())
# 修改点：早融合下不做图像侧投影与归一化索引，直接缓存原始图像特征；后融合保持为 512 维投影
if model.fusion.fusion_method == 'early':
    all_image_feats = torch.stack([image_index[i] for i in all_image_ids]) if all_image_ids else torch.empty((0, getattr(model.image_extractor, 'output_dim', lambda:2048)()) )
else:
    all_image_feats = torch.stack([image_index[i] for i in all_image_ids]) if all_image_ids else torch.empty((0, 512))
faiss_index = None
if HAS_FAISS and all_image_feats.size(0) > 0:
    d = all_image_feats.size(1)
    faiss_index = faiss.IndexFlatIP(d)
    feats_np = all_image_feats.detach().cpu().numpy().astype('float32')
    faiss_index.add(feats_np)

all_image_feats = all_image_feats.to(device)

def compute_recall_at_k(k_values, queries):
    recalls = {k: 0 for k in k_values}
    total = 0
    for q_text, gt_ids in tqdm(queries, desc='Evaluate'):
        if all_image_feats.size(0) == 0:
            continue
        if model.fusion.fusion_method == 'early':
            # 早融合：逐块计算文本与所有候选图像的 pairwise 分数
            q_feat = model.extract_and_fuse_text_features([q_text])
            scores = model.compute_logits(q_feat, all_image_feats)  # [1, N]
            _, top_idx = torch.topk(scores[0], k=max(k_values))
            top_ids = [all_image_ids[i] for i in top_idx.tolist()]
        else:
            # 后融合：使用相似度计算器（cosine/dot）
            q_feat = model.extract_and_fuse_text_features([q_text])
            if faiss_index is not None:
                q_np = q_feat.detach().cpu().numpy().astype('float32')
                _, I = faiss_index.search(q_np, max(k_values))
                top_idx = I[0].tolist()
                top_ids = [all_image_ids[i] for i in top_idx]
            else:
                sims = model.sim.calculate_similarity(q_feat, all_image_feats)
                _, top_idx = torch.topk(sims[0], k=max(k_values))
                top_ids = [all_image_ids[i] for i in top_idx.tolist()]
        total += 1
        for k in k_values:
            if any(g in set(top_ids[:k]) for g in gt_ids):
                recalls[k] += 1
    return {k: (recalls[k] / total if total > 0 else 0.0) for k in k_values}, total

rec, total_q = compute_recall_at_k([1,5,10], valid_queries)
mean_recall = (rec.get(1,0)+rec.get(5,0)+rec.get(10,0))/3 if total_q>0 else 0.0
print(f'Recall@1={rec.get(1,0):.4f}, Recall@5={rec.get(5,0):.4f}, Recall@10={rec.get(10,0):.4f}, MeanRecall={mean_recall:.4f} (N={total_q})')


# In[ ]:





# In[13]:


with open(f"{os.path.basename(__file__).split('.')[0]}.finishflag", "w") as f:
    f.write("finish")

# import IPython
# def kill_current_kernel():
#     '''杀死当前的kernel释放内存空间。'''
#     IPython.Application.instance().kernel.do_shutdown(True)
# kill_current_kernel()


# In[ ]:





# In[ ]:




