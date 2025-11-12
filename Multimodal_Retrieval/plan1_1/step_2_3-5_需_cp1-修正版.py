#!/usr/bin/env python
# coding: utf-8


# In[2]:
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
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
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

clip_model_name = 'openai/clip-vit-base-patch32'


# In[ ]:





# ## 模型与特征模块
# 保持原有类名与结构，内部实现改为使用 CLIP 的 `get_text_features` 与 `get_image_features`。
# 

# In[3]:


class TextFeatureExtractor:
    def __init__(self, device='cpu', cache_dir=None, clip_model_name='openai/clip-vit-base-patch32', clip_model=None, clip_processor=None):
        self.device = device
        self.processor = clip_processor or CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True)
        self.model = clip_model or CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).to(device)
        self.model.eval()
        # 输出维度（CLIP 投影维度）
        self.out_dim = getattr(self.model.config, 'projection_dim', 512)

    def encode_with_grad(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.out_dim), dtype=torch.float32, device=self.device)
        inputs = self.processor(text=texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)  # [B, D]
        return feats

class ImageFeatureExtractor:
    def __init__(self, device='cpu', cache_dir=None, clip_model_name='openai/clip-vit-base-patch32', clip_model=None, clip_processor=None):
        self.device = device
        self.processor = clip_processor or CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True)
        self.model = clip_model or CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).to(device)
        self.model.eval()
        self.out_dim = getattr(self.model.config, 'projection_dim', 512)

    def encode_with_grad(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.out_dim), dtype=torch.float32, device=self.device)
        inputs = self.processor(images=images, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)  # [B, D]
        return feats

class FeatureFusion:
    def __init__(self, fusion_method='projection', projection_dim=512, device=None, hidden_dim=1024, dropout=0.1, text_in_dim=512, image_in_dim=512):
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        if fusion_method == 'projection':
            self.text_projector = torch.nn.Sequential(
                torch.nn.Linear(text_in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(hidden_dim, projection_dim)
            ).to(self.device)
            self.image_projector = torch.nn.Sequential(
                torch.nn.Linear(image_in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(hidden_dim, projection_dim)
            ).to(self.device)
            # self.text_projector = torch.nn.Linear(text_in_dim, projection_dim).to(self.device)
            # self.image_projector = torch.nn.Linear(image_in_dim, projection_dim).to(self.device)

    def fuse_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.text_projector(text_features) if self.fusion_method == 'projection' else text_features

    def fuse_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.image_projector(image_features) if self.fusion_method == 'projection' else image_features

class SimilarityCalculator:
    def __init__(self, similarity_type='cosine'):
        self.similarity_type = similarity_type
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        f = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nn.functional.normalize(f, p=2, dim=1, eps=1e-6)
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
        text_in_dim = getattr(text_extractor, 'out_dim', 512)
        image_in_dim = getattr(image_extractor, 'out_dim', 512)
        self.fusion = FeatureFusion(fusion_method, projection_dim, device, text_in_dim=text_in_dim, image_in_dim=image_in_dim)
        self.sim = SimilarityCalculator(similarity_type)
        self.normalize_features = normalize_features
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-6)
    def extract_and_fuse_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            t = self.text_extractor.encode_with_grad(texts)
        return self._norm(self.fusion.fuse_text_features(t))
    def extract_and_fuse_image_features(self, images: List[Image.Image]) -> torch.Tensor:
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
    logits = torch.mm(text_feats, image_feats.t()).float() / float(temp)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = torch.clamp(logits, -100.0, 100.0)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t = torch.nn.functional.cross_entropy(logits, labels)
    loss_i = torch.nn.functional.cross_entropy(logits.t(), labels)
    return (loss_t + loss_i) * 0.5


# ## 优化器（最小改动）
# 仅优化两层 MLP 投影头参数，保持与原 notebook 相同的接口与结构。
# 

# In[4]:


def build_optimizer(model: CrossModalRetrievalModel, lr_proj: float = 1e-3, weight_decay: float = 1e-4):
    params = []
    params.append({
        'params': list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
        'lr': lr_proj,
        'weight_decay': weight_decay
    })
    optimizer = torch.optim.Adam(params)
    return optimizer


# ## 数据加载与训练参数
# 保持与原 notebook 一致的流式图片加载与训练参数接口。
# 

# In[5]:


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
    epochs_per_batch = 1 ## 每个大batch训练几个epoch。
    train_step_batch_size = 32 ## 每个大batch里面训练的时候的小batch_size是多少。
    valid_imgs_max_samples = 30000

use_amp = True
temperature = 0.07


# ## 初始化模型
# 

# In[6]:


# 共享一个 CLIP 模型与处理器，减少显存与加载开销
shared_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True)
shared_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir, local_files_only=True).to(device)
text_extractor = TextFeatureExtractor(device=device, cache_dir=cache_dir, clip_model_name=clip_model_name, clip_model=shared_model, clip_processor=shared_processor)
image_extractor = ImageFeatureExtractor(device=device, cache_dir=cache_dir, clip_model_name=clip_model_name, clip_model=shared_model, clip_processor=shared_processor)
model = CrossModalRetrievalModel(
    text_extractor, image_extractor, 
    fusion_method='projection', projection_dim=512, similarity_type='cosine', normalize_features=True, device=device
)

optim = build_optimizer(model, lr_proj=1e-3, weight_decay=1e-4)
scaler = GradScaler(enabled=(device.type == 'cuda' and use_amp))
print('Optim groups:', len(optim.param_groups))


# ## 训练循环（流式）
# 结构与原 notebook 保持一致：构建 (query, image) 配对，AMP 与裁剪保持。
# 

# In[7]:


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
            temp = temperature
            if use_amp and device.type == 'cuda':
                with autocast(enabled=True):
                    t_feats = text_extractor.encode_with_grad(texts)
                    i_feats = image_extractor.encode_with_grad(imgs)
                    t_proj = model._norm(model.fusion.fuse_text_features(t_feats))
                    i_proj = model._norm(model.fusion.fuse_image_features(i_feats))
                    loss = info_nce_loss(t_proj, i_proj, temp)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
                    max_norm=5.0
                )
                scaler.step(optim)
                scaler.update()
            else:
                t_feats = text_extractor.encode_with_grad(texts)
                i_feats = image_extractor.encode_with_grad(imgs)
                t_proj = model._norm(model.fusion.fuse_text_features(t_feats))
                i_proj = model._norm(model.fusion.fuse_image_features(i_feats))
                loss = info_nce_loss(t_proj, i_proj, temp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.text_projector.parameters()) + list(model.fusion.image_projector.parameters()),
                    max_norm=5.0
                )
                optim.step()
            running_loss += loss.item()
            steps += 1
        print(f"Epoch {e+1}/{epochs}: avg loss={running_loss/max(steps,1):.4f}")

# 流式加载与训练
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


# In[ ]:





# ## 保存：投影层 + 已解冻顶层 + 优化器
# 保存 BERT 的最后2层 + pooler、ResNet50 的 layer4、投影层、优化器。
# 

# In[8]:


save_dir = '/mnt/d/forCoding_data/Tianchi_MUGE/trained_models/weights'
save_path = os.path.join(save_dir, 'step_2_3_5.pth')

def save_unfreeze_checkpoint(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                             optimizer: torch.optim.Optimizer, save_path: str, last_n_layers: int):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        'projection_dim': model.fusion.projection_dim,
        # 'last_n_layers': last_n_layers,
        'fusion': {
            'text_projector': model.fusion.text_projector.state_dict(),
            'image_projector': model.fusion.image_projector.state_dict(),
        },
        # 'text_unfrozen': {},
        # 'image_unfrozen': {},
        # 'optimizer': optimizer.state_dict(),
    }
    # enc = text_extractor.model.encoder
    # total_layers = len(enc.layer)
    # start_idx = max(0, total_layers - last_n_layers)
    # for i in range(start_idx, total_layers):
    #     ckpt['text_unfrozen'][f'encoder_layer_{i}'] = enc.layer[i].state_dict()
    # if hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
    #     ckpt['text_unfrozen']['pooler'] = text_extractor.model.pooler.state_dict()
    # if hasattr(image_extractor.model, 'layer4'):
    #     ckpt['image_unfrozen']['layer4'] = image_extractor.model.layer4.state_dict()
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to: {save_path}")

# 保存一次
save_unfreeze_checkpoint(model, text_extractor, image_extractor, optim, save_path, 2)


# ## 加载：恢复解冻顶层与投影层权重，继续训练
# 加载后会自动再次执行顶层解冻，并恢复优化器状态（如提供）。
# 

# In[9]:


def load_unfreeze_checkpoint(model: CrossModalRetrievalModel, text_extractor: TextFeatureExtractor, image_extractor: ImageFeatureExtractor,
                             optimizer: torch.optim.Optimizer, load_path: str):
    ckpt = torch.load(load_path, map_location='cpu')
    model.fusion.text_projector.load_state_dict(ckpt['fusion']['text_projector'])
    model.fusion.image_projector.load_state_dict(ckpt['fusion']['image_projector'])
    # ln = ckpt.get('last_n_layers', 2)
    # unfreeze_text_top_layers(text_extractor, last_n_layers=ln)
    # unfreeze_image_top_block(image_extractor, unfreeze_layer4=True)
    # enc = text_extractor.model.encoder
    # for k, v in ckpt['text_unfrozen'].items():
    #     if k.startswith('encoder_layer_'):
    #         idx = int(k.split('_')[-1])
    #         if 0 <= idx < len(enc.layer):
    #             enc.layer[idx].load_state_dict(v)
    # if 'pooler' in ckpt['text_unfrozen'] and hasattr(text_extractor.model, 'pooler') and text_extractor.model.pooler is not None:
    #     text_extractor.model.pooler.load_state_dict(ckpt['text_unfrozen']['pooler'])
    # if 'layer4' in ckpt['image_unfrozen'] and hasattr(image_extractor.model, 'layer4'):
    #     image_extractor.model.layer4.load_state_dict(ckpt['image_unfrozen']['layer4'])
    # if optimizer is not None and 'optimizer' in ckpt:
    #     optimizer.load_state_dict(ckpt['optimizer'])
    # print(f"Checkpoint loaded from: {load_path}")

# 测试加载
load_unfreeze_checkpoint(model, text_extractor, image_extractor, optim, save_path)

# 重要：验证前切换到 eval，关闭投影头中的 Dropout，确保索引与查询一致
model.fusion.text_projector.eval()
model.fusion.image_projector.eval()


# ## 验证评估：Recall@1/5/10 与 MeanRecall
# - 基于验证集构建图像索引
# - 对每条查询计算相似度并统计召回（优先使用 FAISS；不可用则 Torch 回退）
# 

# In[11]:


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

# 关键修正：确保索引覆盖所有查询的GT图片，避免采样导致GT缺失
required_gt_ids = set()
for _, ids in valid_queries:
    for i in ids:
        required_gt_ids.add(i)
missing_ids = required_gt_ids - set(valid_imgs.keys())
if missing_ids:
    print(f"Augmenting index to include missing GT images: {len(missing_ids)}")
    # 流式遍历验证集图片，补齐缺失的GT图片（最多扫描一定批次数避免过慢）
    scan_batches = 50
    scanned = 0
    for image_batch in loader.load_images_batch(split='valid', batch_size=3000, max_batches=scan_batches):
        scanned += 1
        for item in image_batch:
            img_id = item['img_id']
            if img_id in missing_ids and item['image'] is not None:
                valid_imgs[img_id] = item['image']
        missing_ids = missing_ids - set(valid_imgs.keys())
        if not missing_ids:
            break
    print(f"After augmentation, remaining missing GT: {len(missing_ids)}")

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

# 仅评估那些在索引中至少存在一个GT图片的查询（通常此时应覆盖全部或绝大多数）
id_set = set(all_image_ids)
filtered_queries = []
for q, ids in valid_queries:
    ids_in_index = [gid for gid in ids if gid in id_set]
    if ids_in_index:
        filtered_queries.append((q, ids_in_index))
print(f"Filtered valid queries: {len(filtered_queries)} (with GT present in index)")
valid_queries = filtered_queries

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





# In[ ]:


with open(f"{os.path.basename(__file__).split('.')[0]}.finishflag", "w") as f:
    f.write("finish")

# import IPython
# def kill_current_kernel():
#     '''杀死当前的kernel释放内存空间。'''
#     IPython.Application.instance().kernel.do_shutdown(True)
# kill_current_kernel()


# In[ ]:





# In[ ]:




