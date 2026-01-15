import argparse
import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import base64
import io
import csv
from transformers import BertTokenizer, BertModel

# 解析参数
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--n_epochs', type=int, default=100)
_args, _unknown = _arg_parser.parse_known_args()

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集配置
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.cls1.tsv"
CSV_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/1-利用文本进行分类/classification_results_coarse.csv"
BERT_PATH = "/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/"

# 自定义数据集
class MUGEDataset(Dataset):
    def __init__(self, tsv_path, csv_path, tokenizer, text_encoder, transform=None):
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        # 加载CSV标签映射
        print(f"Loading labels from {csv_path}...")
        try:
            self.meta = pd.read_csv(csv_path, sep='\t')
            if 'coarse_label' not in self.meta.columns: # Fallback if separator is comma
                self.meta = pd.read_csv(csv_path, sep=',')
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise e

        # 构建标签映射
        self.unique_labels = sorted(self.meta['coarse_label'].unique().tolist())
        self.label_map = {l: i for i, l in enumerate(self.unique_labels)}
        print(f"Found {len(self.unique_labels)} classes: {self.unique_labels}")
        
        # 加载TSV图片
        print(f"Loading images from {tsv_path}...")
        self.images = {}
        # TSV可能很大，但为了简单起见，这里一次性加载进内存
        # 如果内存不足，需要改为基于索引的读取方式
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading TSV"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.images[parts[0]] = parts[1] # img_id -> base64
        
        # 过滤掉没有图片的元数据
        initial_len = len(self.meta)
        self.meta = self.meta[self.meta['img_id'].astype(str).isin(self.images.keys())]
        print(f"Kept {len(self.meta)}/{initial_len} samples with images.")
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_id = str(row['img_id'])
        text = row['text']
        label_str = row['coarse_label']
        label_id = self.label_map[label_str]
        
        # 处理图片
        b64_str = self.images[img_id]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
            img = img.convert('RGB') # 确保RGB
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            # 返回全黑图作为fallback
            img = Image.new('RGB', (256, 256))
            
        if self.transform:
            img = self.transform(img)
            
        # 处理文本嵌入
        # 注意：在DataLoader worker中运行模型可能会有问题，这里假设num_workers=0
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=32
        )
        
        input_ids = inputs.input_ids.to(self.text_encoder.device)
        attention_mask = inputs.attention_mask.to(self.text_encoder.device)
        
        with torch.no_grad():
            out = self.text_encoder(input_ids, attention_mask=attention_mask)
            # 使用pooler_output或last_hidden_state的均值
            # 20_乾使用 last_hidden_state (seq_len, 768)
            # 为了方便拼接，我们需要一个全局向量，或者在模型里处理序列
            # 20_乾在模型forward里做了 mean(dim=1)
            # 这里我们返回 (SeqLen, 768)，在模型里处理
            text_emb = out.last_hidden_state.squeeze(0).cpu() 
            
        return img, label_id, text_emb

# 图像预处理
img_sz = (256, 256)
tf = transforms.Compose([
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1), # 保持原代码的灰度设置
    transforms.ToTensor(),
])

# ViT Block (保持不变)
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

# Positional Encoding (保持不变)
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

# 改进后的 ClassConditionedUViT
class ClassConditionedUViT(nn.Module):
    """
    MoE 版本的 ClassConditionedUViT：
    - 支持文本嵌入 + 类别嵌入
    - MoE 路由依赖类别嵌入
    """
    def __init__(self, num_classes, class_emb_size=128, text_emb_size=768, text_proj_dim=128, 
                 vit_dim=256, vit_heads=4, vit_layers=6, num_experts=4, time_emb_size=32):
        super().__init__()
        # 条件嵌入
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.t_emb = nn.Embedding(1000, time_emb_size)
        
        # 文本投影层：将BERT的768维投影到较小维度，以便拼接
        self.text_proj = nn.Linear(text_emb_size, text_proj_dim)
        
        # 编码器（共享）
        # 输入通道: 1 (灰度图) + class_emb_size + text_proj_dim
        in_ch = 1 + class_emb_size + text_proj_dim
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

        # MoE 专家集合
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
        # 需求：利用coarse_label的embedding信息来选择
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

        # 预计算位置编码
        gh = img_sz[0] // 8
        gw = img_sz[1] // 8
        pe = build_2d_sincos(gh, gw, vit_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0), persistent=False)

    def forward(self, x, t, class_labels, text_emb):
        bs, _, h, w = x.shape
        
        # 1. 类别嵌入
        c_feat = self.class_emb(class_labels) # [B, class_emb_size]
        c2d = c_feat.view(bs, -1, 1, 1).expand(bs, -1, h, w)
        
        # 2. 文本嵌入处理 (参考20_乾: mean pooling)
        text_mean = text_emb.mean(dim=1) # [B, 768]
        t_feat = self.text_proj(text_mean) # [B, text_proj_dim]
        t2d = t_feat.view(bs, -1, 1, 1).expand(bs, -1, h, w)
        
        # 3. 拼接所有条件到输入
        # x: [B, 1, H, W]
        # c2d: [B, class_emb_size, H, W]
        # t2d: [B, text_proj_dim, H, W]
        z = torch.cat([x, c2d, t2d], 1)

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
        time_feat = self.t_emb(t)  # [bs, time_emb_size]
        
        # Gate input: Global feature + Class embedding + Time embedding
        # 满足需求：利用coarse_label的embedding信息(c_feat)来选择
        p3_pool = F.adaptive_avg_pool2d(p3, 1).view(bs, 128)  # [bs, 128]
        gate_in = torch.cat([p3_pool, c_feat, time_feat], dim=1)
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

def main():
    print("Initializing BERT...")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text_encoder = BertModel.from_pretrained(BERT_PATH).to(device)
    text_encoder.eval()

    print("Initializing Dataset...")
    dataset = MUGEDataset(TSV_PATH, CSV_PATH, tokenizer, text_encoder, transform=tf)
    # num_workers=0 因为text_encoder在Dataset里使用了CUDA，多进程会导致错误
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    num_classes = len(dataset.unique_labels)
    print(f"Num classes: {num_classes}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
    
    # 增加class_emb_size到128，以便承载更多信息，同时与text_proj_dim匹配
    net = ClassConditionedUViT(num_classes=num_classes, class_emb_size=128, text_proj_dim=128).to(device)
    
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    net_ema = ClassConditionedUViT(num_classes=num_classes, class_emb_size=128, text_proj_dim=128).to(device)

    _base = os.path.splitext(os.path.basename(__file__))[0]
    _save_path = os.path.join(os.path.dirname(__file__), f"{_base}_model.pth")
    
    start_epoch = 0
    n_epochs = _args.n_epochs

    # 加载模型
    if os.path.exists(_save_path):
        print(f"Loading checkpoint from {_save_path}")
        ckpt = torch.load(_save_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                net.load_state_dict(ckpt['model_state_dict'])
            if 'ema_state_dict' in ckpt:
                net_ema.load_state_dict(ckpt['ema_state_dict'])
            else:
                net_ema.load_state_dict(net.state_dict())
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1 # 从下一轮开始
                print(f"Resuming from epoch {start_epoch}")
        else:
            # 兼容旧格式
            try:
                net.load_state_dict(ckpt, strict=False)
                net_ema.load_state_dict(ckpt, strict=False)
            except:
                print("Failed to load legacy checkpoint, starting fresh.")
                net_ema.load_state_dict(net.state_dict())
    else:
        net_ema.load_state_dict(net.state_dict())

    ema_decay = 0.999
    losses = []

    print(f"Starting training from epoch {start_epoch} to {n_epochs}...")
    
    for epoch in range(start_epoch, n_epochs):
        net.train()
        epoch_losses = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for x, label_id, text_emb in pbar:
            x = x.to(device) * 2 - 1
            label_id = label_id.to(device)
            text_emb = text_emb.to(device) # [B, SeqLen, 768]
            
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            
            # Forward pass with text embedding
            pred = net(noisy_x, timesteps, label_id, text_emb)
            
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
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f'Finished epoch {epoch}. Average loss: {avg_loss:05f}')

        # 保存模型 (包含epoch信息)
        torch.save({
            "model_state_dict": net.state_dict(),
            "ema_state_dict": net_ema.state_dict(),
            "epoch": epoch
        }, _save_path)
        print(f"Saved model to {_save_path}")

        # 采样可视化 (仅在每5轮或最后一轮)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            # 简单采样测试 (使用第一个batch的label和text作为条件)
            xg = torch.randn(4, 1, img_sz[0], img_sz[1]).to(device)
            # 构造测试条件
            sample_labels = torch.tensor([0, 1, 2, 3] if num_classes >=4 else [0]*4).to(device)
            # 需要text_emb。这里简单起见，从当前batch里取或者随机
            # 为演示，我们使用零向量作为text_emb的占位符（实际上应该用真实文本）
            # 或者复用最后一个batch的text_emb
            sample_text_emb = text_emb[:4] if text_emb.shape[0] >= 4 else text_emb.repeat(4,1,1)[:4]
            
            ddim = DDIMScheduler.from_config(noise_scheduler.config)
            ddim.set_timesteps(60)
            
            for i, t in tqdm(enumerate(ddim.timesteps), desc="Sampling"):
                with torch.no_grad():
                    net_ema.eval()
                    residual = net_ema(xg, t.to(xg.device), sample_labels, sample_text_emb)
                xg = ddim.step(residual, t, xg).prev_sample
            
            out_name = os.path.join(os.path.dirname(__file__), f'samples_uvit_bw_epoch{epoch+1}.png')
            torchvision.utils.save_image(
                xg.detach().cpu().clip(-1, 1),
                out_name,
                nrow=2,
                normalize=True,
                value_range=(-1, 1)
            )
            
    plt.plot(losses)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'loss_curve.png'))

if __name__ == "__main__":
    main()
