import argparse
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--n_epochs', type=int, default=100)
_args, _unknown = _arg_parser.parse_known_args()

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import os
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError
import pandas as pd
import base64
import io
from transformers import BertTokenizer, BertModel

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# class SafeImageFolder(datasets.ImageFolder):
#     def __init__(self, root, transform=None, include_prefixes=None, only_jpg_with_underscore=False):
#         self.include_prefixes = include_prefixes
#         self.only_jpg_with_underscore = only_jpg_with_underscore
#         super().__init__(root=root, transform=transform)
#         kept, skipped = [], 0
#         for path, target in self.samples:
#             fname = os.path.basename(path)
#             dname = os.path.basename(os.path.dirname(path))
#             prefix = dname.split()[0] if dname else ""
#             if self.include_prefixes is not None and prefix not in self.include_prefixes:
#                 continue
#             if self.only_jpg_with_underscore:
#                 if not fname.lower().endswith(".jpg") or "_" not in fname or fname.startswith("._"):
#                     continue
#             try:
#                 with Image.open(path) as im:
#                     im.verify()
#             except (UnidentifiedImageError, OSError):
#                 skipped += 1
#                 continue
#             kept.append((path, target))
#         present_classes = sorted({os.path.basename(os.path.dirname(p)) for p, _ in kept})
#         class_to_idx = {cls: i for i, cls in enumerate(present_classes)}
#         remapped = [(p, class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in kept]
#         self.classes = present_classes
#         self.class_to_idx = class_to_idx
#         self.samples = remapped
#         self.imgs = remapped
#         self.targets = [t for _, t in remapped]
#     def find_classes(self, directory):
#         classes = []
#         for d in sorted(os.listdir(directory)):
#             p = os.path.join(directory, d)
#             if not os.path.isdir(p):
#                 continue
#             prefix = d.split()[0] if d else ""
#             if self.include_prefixes is None or prefix in self.include_prefixes:
#                 classes.append(d)
#         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#         return classes, class_to_idx

# root = "/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志"


_base = os.path.splitext(os.path.basename(__file__))[0]
_save_path = os.path.join(
    # os.path.dirname(__file__), 
    "/mnt/d/forCoding_data/Tianchi_MUGE/plan_3/trained_models",
    f"{_base}_model.pth"
)
n_epochs = _args.n_epochs
start_epoch = 0

# 与 MNIST 类似的变换（返回 C×H×W 的 float 张量）
crop_x, crop_y, crop_w, crop_h = 50, 100, 400, 550 
## 如果不想裁切，后面两个设置很大的数就行了。
# 0, 0, 256, 256
def crop_im(im):
    W, H = im.size
    x0 = max(0, min(W - 1, crop_x))
    y0 = max(0, min(H - 1, crop_y))
    x1 = max(x0 + 1, min(W, x0 + crop_w))
    y1 = max(y0 + 1, min(H, y0 + crop_h))
    return im.crop((x0, y0, x1, y1))

img_sz = (256, 256)
tf = transforms.Compose([
    # transforms.Lambda(crop_im),
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
include = [str(i) for i in range(2, 23)]
# 适配 TSV+CSV 的数据加载
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.cls1.tsv"
CSV_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/1-利用文本进行分类/classification_results_coarse.csv"

class TSVCSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path, csv_path, transform=None):
        self.transform = transform
        # 加载 CSV（coarse_label）
        try:
            meta = pd.read_csv(csv_path, sep="\t")
            if "coarse_label" not in meta.columns:
                meta = pd.read_csv(csv_path, sep=",")
        except Exception as e:
            raise RuntimeError(f"读取CSV失败: {e}")
        # 类别映射
        classes = sorted(meta["coarse_label"].astype(str).unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        # 加载 TSV（img_id -> base64）
        img_map = {}
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_map[parts[0]] = parts[1]
        # 过滤存在图片的样本
        meta = meta[meta["img_id"].astype(str).isin(img_map.keys())]
        self.samples = []
        for _, row in meta.iterrows():
            img_id = str(row["img_id"])
            label_str = str(row["coarse_label"])
            label_id = self.class_to_idx[label_str]
            text = str(row["text"])
            self.samples.append((img_id, img_map[img_id], label_id, text))
        self.targets = [t for _, _, t, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, b64, label, text = self.samples[idx]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, text

dataset = TSVCSVImageDataset(TSV_PATH, CSV_PATH, transform=tf)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        m = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, m), nn.GELU(), nn.Linear(m, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

def build_2d_sincos(h, w, dim):
    y = torch.arange(h).float()
    x = torch.arange(w).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(dim // 4).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    pos_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(pos_x), torch.cos(pos_x), torch.sin(pos_y), torch.cos(pos_y)], dim=1)
    if emb.shape[1] < dim:
        pad = torch.zeros(emb.shape[0], dim - emb.shape[1])
        emb = torch.cat([emb, pad], dim=1)
    return emb

class ClassConditionedUViT(nn.Module):
    """
    MoE 版本的 ClassConditionedUViT：
    - 下采样编码器共享。
    - 在瓶颈的 ViT 序列处使用多个专家，每个专家处理不同的内容风格。
    - 门控网络使用类别嵌入、时间步嵌入和全局特征决定专家权重。
    其他接口保持不变。
    """
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4, vit_dim=256, vit_heads=4, vit_layers=6, num_experts=8, time_emb_size=32, text_emb_size=768):
        super().__init__()
        # 条件嵌入
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.t_emb = nn.Embedding(1000, time_emb_size)
        
        # 编码器（共享）
        # 改为拼接 text_emb
        in_ch = 1 + text_emb_size
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU()
        )
        self.pool1 = nn.Conv2d(64, 64, 3, 2, 1)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.pool2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.pool3 = nn.Conv2d(128, 128, 3, 2, 1)

        # ViT 输入投影（共享）
        self.vproj_in = nn.Conv2d(128, vit_dim, 1)

        # MoE 专家集合：每个专家一套 ViTBlock + LayerNorm + 输出投影
        self.num_experts = num_experts
        experts = []
        for _ in range(num_experts):
            expert = nn.ModuleDict({
                'blocks': nn.ModuleList([ViTBlock(vit_dim, vit_heads) for _ in range(vit_layers)]),
                'norm': nn.LayerNorm(vit_dim),
                'proj_out': nn.Conv2d(vit_dim, 128, 1)
            })
            experts.append(expert)
        self.experts = nn.ModuleList(experts)

        # 门控网络：基于类别嵌入 + 时间步嵌入 + 全局特征
        gate_in_dim = 128 + class_emb_size + time_emb_size
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 128), nn.GELU(),
            nn.Linear(128, num_experts)
        )

        # 解码器（共享）
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU()
        )
        self.out = nn.Conv2d(64, 1, 1)

        # 预计算位置编码（与原实现保持一致）
        gh = img_sz[0] // 8
        gw = img_sz[1] // 8
        pe = build_2d_sincos(gh, gw, vit_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0), persistent=False)

    def forward(self, x, t, class_labels, text_emb):
        bs, _, h, w = x.shape
        # 类别条件不再拼接到输入，而是使用 text_emb
        # text_emb: [bs, 768]
        t2d = text_emb.view(bs, -1, 1, 1).expand(bs, -1, h, w)
        z = torch.cat([x, t2d], 1)

        # 编码器
        d1 = self.down1(z)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # ViT tokens
        vb = self.vproj_in(p3)
        gh = vb.shape[2]
        gw = vb.shape[3]
        s = vb.flatten(2).transpose(1, 2)
        s = s + self.pos_embed[:, : gh * gw, :].to(s.device)

        # 门控权重（per-sample）
        if t.dim() == 0:
            t = t.view(1).repeat(bs)
        t = t.long().clamp(0, 999)
        t_feat = self.t_emb(t)  # [bs, time_emb_size]
        c_feat = self.class_emb(class_labels)  # [bs, class_emb_size]
        p3_pool = F.adaptive_avg_pool2d(p3, 1).view(bs, 128)  # [bs, 128]
        gate_in = torch.cat([p3_pool, c_feat, t_feat], dim=1)
        gate_logits = self.gate(gate_in)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [bs, num_experts]

        # 专家前向并加权融合
        yb_sum = None
        for e, expert in enumerate(self.experts):
            se = s
            for blk in expert['blocks']:
                se = blk(se)
            se = expert['norm'](se)
            se = se.transpose(1, 2).reshape(bs, -1, gh, gw)
            yb_e = expert['proj_out'](se)  # [bs, 128, gh, gw]
            w_e = gate_w[:, e].view(bs, 1, 1, 1)
            yb_e = yb_e * w_e
            yb_sum = yb_e if yb_sum is None else (yb_sum + yb_e)

        # 解码器
        u1 = F.interpolate(yb_sum, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d3], 1)
        u1 = self.up1(u1)
        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], 1)
        u2 = self.up2(u2)
        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d1], 1)
        u3 = self.up3(u3)
        return self.out(u3)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')

# 加载BERT
bert_path = "/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(bert_path)
text_encoder = BertModel.from_pretrained(bert_path).to(device)
text_encoder.eval()

net = ClassConditionedUViT().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
net_ema = ClassConditionedUViT().to(device)
# 若存在已保存的模型文件，则加载并继续训练
if os.path.exists(_save_path):
    ckpt = torch.load(_save_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            try:
                res = net.load_state_dict(ckpt['model_state_dict'], strict=False)
                if 'ema_state_dict' in ckpt:
                    net_ema.load_state_dict(ckpt['ema_state_dict'], strict=False)
                else:
                    net_ema.load_state_dict(net.state_dict())
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch']) + 1
                    print(f"已加载模型，历史已训练 epoch 数: {int(ckpt['epoch']) + 1}")
                    if 'loss' in ckpt:
                        print(f"上一轮 ({int(ckpt['epoch'])}) Loss: {ckpt['loss']:.6f}")
                else:
                    print("已加载模型，但未记录历史 epoch 数，视为 0")
                print(f"已检测到现有模型，加载并续训: {_save_path}")
            except RuntimeError as e:
                print(f"模型结构不匹配（可能是参数修改导致），放弃加载旧模型，从头训练: {e}")
                start_epoch = 0
                # Re-init weights if needed, but they are already random
    else:
        # 兼容直接保存整个 state_dict 的情况
        try:
            res = net.load_state_dict(ckpt, strict=False)
            net_ema.load_state_dict(ckpt, strict=False)
            print(f"已从原始 state_dict 加载并续训: {_save_path}")
        except Exception as e:
            print(f"加载模型失败，改为从头开始训练: {e}")
            net_ema.load_state_dict(net.state_dict())
else:
    net_ema.load_state_dict(net.state_dict())
ema_decay = 0.999

losses = []
for epoch in range(start_epoch, n_epochs):
    for x, y, text in tqdm(train_dataloader):
        x = x.to(device) * 2 - 1
        y = y.to(device)
        
        # 获取 text embedding
        with torch.no_grad():
            inputs = tokenizer(list(text), return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = text_encoder(**inputs)
            text_emb = outputs.pooler_output # [bs, 768]
            
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        pred = net(noisy_x, timesteps, y, text_emb)
        velocity = noise_scheduler.get_velocity(x, noise, timesteps)
        alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
        snr = alphas / (1 - alphas)
        weight = torch.sqrt(torch.clamp(snr, max=5.0))
        loss = (weight * (pred - velocity) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        losses.append(loss.item())
    avg_loss = sum(losses[-100:]) / 100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
    # 每训练完一个epoch就保存一次模型，覆盖原文件
    torch.save({
        "model_state_dict": net.state_dict(),
        "ema_state_dict": net_ema.state_dict(),
        "epoch": epoch,
        "loss": avg_loss
    }, _save_path)

    if (epoch + 1) % 5 == 0:

        xg = torch.randn(3 * 2, 1, img_sz[0], img_sz[1]).to(device)
        yg = torch.tensor([[i] * 3 for i in range(2)]).flatten().to(device)
        
        # 准备采样文本
        sample_texts = [dataset.classes[i] for i in range(2) for _ in range(3)]
        with torch.no_grad():
            inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = text_encoder(**inputs)
            text_emb_sample = outputs.pooler_output

        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(60)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                net_ema.eval()
                residual_cond = net_ema(xg, t.to(xg.device), yg, text_emb_sample)
                drop_mask = torch.rand_like(yg.float()) < 0.1
                yg_uncond = yg.clone()
                yg_uncond[drop_mask] = 0
                residual_uncond = net_ema(xg, t.to(xg.device), yg_uncond, text_emb_sample)
                guidance_scale = 2.0
                residual = residual_uncond + guidance_scale * (residual_cond - residual_uncond)
            xg = ddim.step(residual, t, xg).prev_sample
        out_name = os.path.join(os.path.dirname(__file__), f'samples_uvit_bw_epoch{epoch+1}.png')
        torchvision.utils.save_image(
            xg.detach().cpu().clip(-1, 1),
            out_name,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
plt.plot(losses)



VAL_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.cls1.tsv"
VAL_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
text_map = {}
with open(VAL_TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            text_map[parts[0]] = parts[1]
meta = pd.read_csv(CSV_PATH, sep="\t")
if "coarse_label" not in meta.columns:
    meta = pd.read_csv(CSV_PATH, sep=",")
coarse_map = {}
if "img_id" in meta.columns and "coarse_label" in meta.columns:
    for _, row in meta.iterrows():
        coarse_map[str(row["img_id"])] = str(row["coarse_label"])
out_dir = os.path.join(os.path.dirname(__file__), _base)
os.makedirs(out_dir, exist_ok=True)
max_count = 50
page_size = 10
page_index = 1
items = []
count = 0
with open(VAL_IMG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if count >= max_count:
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img_id = parts[0]
        img_b64 = parts[1]
        coarse_label = coarse_map.get(str(img_id), None)
        label_id = dataset.class_to_idx.get(coarse_label, 0)
        
        # 获取文本并编码
        text_str = text_map.get(str(img_id), "")
        with torch.no_grad():
            inputs = tokenizer([text_str], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = text_encoder(**inputs)
            text_emb_val = outputs.pooler_output

        val_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        proc_t = tf(val_img)
        x = torch.randn(1, 1, img_sz[0], img_sz[1]).to(device)
        y = torch.tensor([label_id]).to(device)
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(30)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                residual = net(x, t.to(x.device), y, text_emb_val)
            x = ddim.step(residual, t, x).prev_sample
        gen = x.detach().cpu().clip(-1, 1)[0, 0]
        gen = (gen + 1.0) / 2.0
        to_pil = transforms.ToPILImage()
        gen_pil = to_pil(gen.unsqueeze(0))
        buf1 = io.BytesIO()
        gen_pil.save(buf1, format="PNG")
        gen_b64 = base64.b64encode(buf1.getvalue()).decode("utf-8")
        proc_pil = to_pil(proc_t)
        buf2 = io.BytesIO()
        proc_pil.save(buf2, format="PNG")
        val_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")
        item_html = f"""
<div style="display:flex;gap:24px;margin-bottom:16px;">
  <div>
    <div>coarse_label: {coarse_label}</div>
    <div>text: {text_map.get(str(img_id), "")}</div>
  </div>
  <div>
    <div>生成图片</div>
    <img src="data:image/png;base64,{gen_b64}" />
  </div>
  <div>
    <div>验证图片（预处理后）</div>
    <img src="data:image/png;base64,{val_b64}" />
  </div>
</div>
"""
        items.append(item_html)
        count += 1
        if len(items) == page_size:
            page_html = "<html><body>" + "".join(items) + "</body></html>"
            out_html = os.path.join(out_dir, f"page_{page_index}.html")
            with open(out_html, "w", encoding="utf-8") as fw:
                fw.write(page_html)
            items = []
            page_index += 1
if len(items) > 0:
    page_html = "<html><body>" + "".join(items) + "</body></html>"
    out_html = os.path.join(out_dir, f"page_{page_index}.html")
    with open(out_html, "w", encoding="utf-8") as fw:
        fw.write(page_html)
