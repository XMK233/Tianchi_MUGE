#!/usr/bin/env python
# coding: utf-8

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
parser.add_argument(
    "--image_aug",
    type=str,
    choices=["none", "jitter_flip", "grayscale_blur"],
    default="none",
    help="图像增强方式选择"
)
parser.add_argument(
    "--text_aug",
    type=str,
    choices=["none", "word_dropout", "random_swap"],
    default="none",
    help="文本增强方式选择"
)
parser.add_argument(
    "--text_aug_prob",
    type=float,
    default=0.3,
    help="文本增强应用概率（训练时）"
)
parser.add_argument(
    "--image_aug_prob",
    type=float,
    default=0.5,
    help="图像增强强度或概率（训练时，部分增强内部固定p）"
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
#     if os.path.exists("step_3_1-6_讼_cp2-基于5_cp2-换图像模型2.finishflag"):
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
    def __init__(self, model_name='bert-base-chinese', device='cpu', cache_dir=None, pooling='mean',
                 text_aug_mode: str = 'none', text_aug_prob: float = 0.0):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)#
        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).to(device)#, local_files_only=True
        self.pooling = pooling
        self.text_aug_mode = text_aug_mode
        self.text_aug_prob = max(0.0, min(1.0, float(text_aug_prob)))
        if self.pooling == 'attentive':
            self.attn = nn.Linear(768, 1).to(self.device)
        # 默认 eval，训练时将对子模块单独切换 train
        self.model.eval()

    def encode_with_grad(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 768), dtype=torch.float32, device=self.device)
        if isinstance(texts, str):
            texts = [texts]
        # 训练阶段才进行文本增强（no_grad 下不增强）
        if torch.is_grad_enabled() and self.text_aug_mode != 'none':
            texts = self._augment_texts(texts)
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

    def _augment_texts(self, texts: List[str]) -> List[str]:
        auged = []
        for t in texts:
            if self.text_aug_mode == 'word_dropout':
                if torch.rand(1).item() <= self.text_aug_prob:
                    tokens = t.split()
                    if len(tokens) >= 2:
                        drop_p = 0.15
                        kept = [w for w in tokens if torch.rand(1).item() > drop_p]
                        if len(kept) == 0:
                            kept = tokens
                        t = ' '.join(kept)
            elif self.text_aug_mode == 'random_swap':
                if torch.rand(1).item() <= self.text_aug_prob:
                    tokens = t.split()
                    if len(tokens) >= 2:
                        i = torch.randint(0, len(tokens), (1,)).item()
                        j = torch.randint(0, len(tokens), (1,)).item()
                        if i != j:
                            tokens[i], tokens[j] = tokens[j], tokens[i]
                            t = ' '.join(tokens)
            auged.append(t)
        return auged

from safetensors.torch import load_file
class ImageFeatureExtractor:
    '''
    改进版，使得timm不要每次都去连huggingface。
    '''
    def __init__(self, model_name='resnet50', device='cpu', weights_path=None, cache_dir=None,
                 aug_mode: str = 'none', aug_prob: float = 0.0):
        self.device = device

        if weights_path is None:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0, cache_dir=cache_dir)
        else:
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes=0, cache_dir=cache_dir,
                pretrained_cfg_overlay={'file': weights_path}
            )
            ## 或者如下亦可：
            # self.model = timm.create_model(
            #     model_name, pretrained=False, num_classes=0, cache_dir=cache_dir,
            # )
            # if weights_path.endswith('.safetensors'):
            #     state_dict = load_file(weights_path)
            # else:
            #     state_dict = torch.load(weights_path, map_location='cpu')
            # self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(device)
        self.model.eval()
        # 增强配置
        self.aug_mode = aug_mode
        self.aug_prob = max(0.0, min(1.0, float(aug_prob)))
        # 评估变换（与原始一致）
        self.transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # 训练增强变换：根据模式选择
        if self.aug_mode == 'jitter_flip':
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self.aug_mode == 'grayscale_blur':
            self.transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_train = self.transform_eval

    def encode_with_grad(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            in_dim = getattr(self.model, 'num_features', 2048)
            return torch.empty((0, in_dim), dtype=torch.float32, device=self.device)
        # 训练阶段使用训练增强，评估阶段使用评估变换
        transform = self.transform_train if torch.is_grad_enabled() else self.transform_eval
        tensors = torch.stack([transform(img.convert('RGB')) for img in images]).to(self.device)
        feats = self.model(tensors)
        return feats

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

    def fuse_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.text_projector(text_features) if self.fusion_method == 'projection' else text_features
    def fuse_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.image_projector(image_features) if self.fusion_method == 'projection' else image_features

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
        # 动态适配图像特征维度：ConvNeXt-Tiny 为 768，ResNet50 为 2048
        img_in_dim = getattr(self.image_extractor.model, 'num_features', 2048)
        self.fusion = FeatureFusion(fusion_method, projection_dim, device, image_in_dim=img_in_dim)
        self.sim = SimilarityCalculator(similarity_type)
        self.normalize_features = normalize_features
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1) if self.normalize_features else x
    def extract_and_fuse_text_features(self, texts: List[str]) -> torch.Tensor:
        # 评估阶段禁用 BN/Dropout 的训练行为
        self.fusion.text_projector.eval()
        with torch.no_grad():
            t = self.text_extractor.encode_with_grad(texts)
        return self._norm(self.fusion.fuse_text_features(t))
    def extract_and_fuse_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        # 评估阶段禁用 BN/Dropout 的训练行为
        self.fusion.image_projector.eval()
        with torch.no_grad():
            i = self.image_extractor.encode_with_grad(images)
        return self._norm(self.fusion.fuse_image_features(i))
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
    if hasattr(image_extractor.model, 'stages'):
        for p in image_extractor.model.stages[-1].parameters():
            p.requires_grad = True
        image_extractor.model.stages[-1].train()
    elif unfreeze_layer4 and hasattr(image_extractor.model, 'layer4'):
        for p in image_extractor.model.layer4.parameters():
            p.requires_grad = True
        image_extractor.model.layer4.train()
    image_extractor.model.eval()

def build_optimizer(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                   lr_proj: float = 1e-3, lr_text_top: float = 5e-5, lr_img_top: float = 1e-4, weight_decay: float = 1e-4):
    params = []
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
    if hasattr(image_extractor.model, 'stages'):
        img_top_params += list(image_extractor.model.stages[-1].parameters())
    elif hasattr(image_extractor.model, 'layer4'):
        img_top_params += list(image_extractor.model.layer4.parameters())
    params.append({
        'params': [p for p in img_top_params if p.requires_grad],
        'lr': lr_img_top,
        'weight_decay': 0.0
    })
    optimizer = torch.optim.Adam(params)
    return optimizer

# LLRD（Layer-wise LR Decay）优化器构建：为BERT顶层设置逐层衰减学习率
def build_llrd_optimizer(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                         lr_proj: float = 1e-3, lr_text_max: float = 5e-5, lr_img_top: float = 1e-4, decay: float = 0.9,
                         last_n_layers: int = 2, weight_decay: float = 1e-4):
    params = []
    # 投影层（文本/图像）
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
    # 图像顶层（ConvNeXt stages[-1] 或 ResNet layer4）
    if hasattr(image_extractor.model, 'stages'):
        params.append({
            'params': [p for p in image_extractor.model.stages[-1].parameters() if p.requires_grad],
            'lr': lr_img_top,
            'weight_decay': 0.0
        })
    elif hasattr(image_extractor.model, 'layer4'):
        params.append({
            'params': [p for p in image_extractor.model.layer4.parameters() if p.requires_grad],
            'lr': lr_img_top,
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
    epochs_per_batch = 6 ## 每个大batch训练几个epoch。
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
    model_name='convnext_tiny', 
    weights_path='/mnt/d/HuggingFaceModels/models--timm--convnext_tiny.in12k_ft_in1k/snapshots/aa096f03029c7f0ec052013f64c819b34f8ad790/model.safetensors',
    aug_mode=args.image_aug,
    aug_prob=args.image_aug_prob
)


# In[9]:


text_extractor = TextFeatureExtractor(
    model_name = "hfl/chinese-roberta-wwm-ext", 
    device=device, 
    cache_dir=cache_dir,
    pooling='attentive',
    text_aug_mode=args.text_aug,
    text_aug_prob=args.text_aug_prob
)

model = CrossModalRetrievalModel(
    text_extractor, image_extractor, 
    fusion_method='projection', projection_dim=512, similarity_type='cosine', normalize_features=True, device=device
)

# 可选：启用BERT梯度检查点以降低显存
if use_grad_checkpoint and hasattr(text_extractor.model, 'gradient_checkpointing_enable'):
    text_extractor.model.gradient_checkpointing_enable()

unfreeze_text_top_layers(text_extractor, last_n_layers=last_n_layers)
unfreeze_image_top_block(image_extractor, unfreeze_layer4=True)

# 使用LLRD优化器：为文本顶层设置逐层衰减的学习率
optim = build_llrd_optimizer(model, text_extractor, image_extractor,
                             lr_proj=1e-3, lr_text_max=5e-5, lr_img_top=1e-4, decay=0.9,
                             last_n_layers=last_n_layers, weight_decay=1e-4)
scaler = GradScaler(enabled=(device.type == 'cuda' and use_amp))
print('Optim groups:', len(optim.param_groups))


# ## 训练循环：按批次流式构建配对并微调顶层
# 仅使用配对中的第一张可用图片；文本与图像编码器顶层参与反向传播。
# 

# In[10]:


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
    if hasattr(image_extractor.model, 'stages'):
        image_extractor.model.stages[-1].train()
    elif hasattr(image_extractor.model, 'layer4'):
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
                    t_feats = text_extractor.encode_with_grad(texts)
                    i_feats = image_extractor.encode_with_grad(imgs)
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

# In[11]:


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

# In[12]:


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

# In[13]:


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





# In[14]:


with open(f"{os.path.basename(__file__).split('.')[0]}.finishflag", "w") as f:
    f.write("finish")

# import IPython
# def kill_current_kernel():
#     '''杀死当前的kernel释放内存空间。'''
#     IPython.Application.instance().kernel.do_shutdown(True)
# kill_current_kernel()


# In[ ]:





# In[ ]:




