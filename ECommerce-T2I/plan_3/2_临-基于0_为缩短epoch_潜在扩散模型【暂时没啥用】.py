import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm.auto import tqdm
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import base64
import io
import math

# 参数设置
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--n_epochs', type=int, default=50)
_arg_parser.add_argument('--batch_size', type=int, default=32)
_arg_parser.add_argument('--learning_rate', type=float, default=2e-4)
_args, _unknown = _arg_parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 基础配置
_base = os.path.splitext(os.path.basename(__file__))[0]
_save_path = os.path.join(
    "/mnt/d/forCoding_data/Tianchi_MUGE/plan_3/trained_models",
    f"{_base}_model.pth"
)
n_epochs = _args.n_epochs
batch_size = _args.batch_size
learning_rate = _args.learning_rate

# 数据预处理
img_sz = (64, 64)  # 降低分辨率加速训练
tf = transforms.Compose([
    transforms.Resize(img_sz),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 数据路径
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.cls1.tsv"
CSV_PATH = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_2/1-利用文本进行分类/classification_results_coarse.csv"

# VAE 编码器/解码器
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, 1, 1)  # 8x8
        )
    
    def forward(self, x):
        return self.encoder(x)

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),     # 64x64
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.decoder(z)

# 增强的条件嵌入
class EnhancedConditionEmbedding(nn.Module):
    def __init__(self, num_classes, emb_size=64):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
    
    def forward(self, labels):
        emb = self.class_emb(labels)
        return self.mlp(emb)

# 时间步嵌入
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # 确保在正确的设备上创建张量
        device = t.device if hasattr(t, 'device') else 'cpu'
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 确保t有正确的维度
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

# 高效的UNet架构
class EfficientUNet(nn.Module):
    def __init__(self, cond_dim=256, latent_dim=4):
        super().__init__()
        
        # 时间步嵌入
        self.time_emb = TimeEmbedding(256)
        self.time_proj = nn.Linear(256, 256)
        
        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, 256)
        
        # 下采样
        self.down1 = self._block(latent_dim + 256, 128)
        self.down2 = self._block(128, 256)
        self.down3 = self._block(256, 512)
        
        # 中间层
        self.mid = self._block(512, 512)
        
        # 上采样
        self.up3 = self._block(512 + 512, 256)
        self.up2 = self._block(256 + 256, 128)
        self.up1 = self._block(128 + 128, latent_dim)
        
        # 最终输出
        self.out = nn.Conv2d(latent_dim, latent_dim, 1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU()
        )
    
    def forward(self, z, t, cond):
        # 时间步和条件嵌入
        t_emb = self.time_proj(self.time_emb(t))
        cond_emb = self.cond_proj(cond)
        
        # 合并条件信息
        cond_expanded = (t_emb + cond_emb).unsqueeze(-1).unsqueeze(-1)
        cond_expanded = cond_expanded.expand(-1, -1, z.shape[2], z.shape[3])
        x = torch.cat([z, cond_expanded], dim=1)
        
        # 下采样路径
        d1 = self.down1(x)
        d2 = self.down2(F.avg_pool2d(d1, 2))
        d3 = self.down3(F.avg_pool2d(d2, 2))
        
        # 中间层
        m = self.mid(F.avg_pool2d(d3, 2))
        
        # 上采样路径
        u3 = self.up3(torch.cat([F.interpolate(m, scale_factor=2), d3], dim=1))
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2), d2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), d1], dim=1))
        
        return self.out(u1)

# 数据集类
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
            self.samples.append((img_id, img_map[img_id], label_id))
        self.targets = [t for _, _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, b64, label = self.samples[idx]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            img = Image.new("RGB", img_sz)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# 主训练函数
def main():
    # 数据集
    dataset = TSVCSVImageDataset(TSV_PATH, CSV_PATH, transform=tf)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    vae_encoder = VAEEncoder().to(device)
    vae_decoder = VAEDecoder().to(device)
    cond_embed = EnhancedConditionEmbedding(len(dataset.classes)).to(device)
    unet = EfficientUNet().to(device)
    
    # 检查是否存在模型文件，如果存在则加载
    start_epoch = 0
    if os.path.exists(_save_path):
        print(f"加载已有模型: {_save_path}")
        checkpoint = torch.load(_save_path, map_location=device)
        vae_encoder.load_state_dict(checkpoint['vae_encoder'])
        vae_decoder.load_state_dict(checkpoint['vae_decoder'])
        cond_embed.load_state_dict(checkpoint['cond_embed'])
        unet.load_state_dict(checkpoint['unet'])
        start_epoch = checkpoint['epoch']
        print(f"从第 {start_epoch} 轮开始继续训练")
    
    # 优化器
    params = list(vae_encoder.parameters()) + list(vae_decoder.parameters()) + \
             list(cond_embed.parameters()) + list(unet.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    
    # 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='scaled_linear',
        prediction_type='v_prediction'
    )
    
    # 训练循环
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0
        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x = x.to(device)
            y = y.to(device)
            
            # VAE编码
            with torch.no_grad():
                z = vae_encoder(x)
            
            # 条件嵌入
            cond = cond_embed(y)
            
            # 扩散过程
            noise = torch.randn_like(z)
            timesteps = torch.randint(0, 999, (z.shape[0],)).long().to(device)
            noisy_z = noise_scheduler.add_noise(z, noise, timesteps)
            
            # 预测
            pred = unet(noisy_z, timesteps, cond)
            
            # 计算损失
            velocity = noise_scheduler.get_velocity(z, noise, timesteps)
            loss = F.mse_loss(pred, velocity)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
        # 保存模型（每个epoch都保存）
        torch.save({
            "vae_encoder": vae_encoder.state_dict(),
            "vae_decoder": vae_decoder.state_dict(),
            "cond_embed": cond_embed.state_dict(),
            "unet": unet.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }, _save_path)
        
        # 生成样本（每2个epoch生成一次用于测试）
        if (epoch + 1) % 2 == 0:
            generate_samples(vae_decoder, unet, cond_embed, dataset, epoch+1, noise_scheduler)

# 生成样本函数
def generate_samples(vae_decoder, unet, cond_embed, dataset, epoch, noise_scheduler):
    unet.eval()
    cond_embed.eval()
    
    with torch.no_grad():
        # 为每个类别生成样本
        for class_id in range(min(5, len(dataset.classes))):
            z = torch.randn(1, 4, 8, 8).to(device)
            y = torch.tensor([class_id]).to(device)
            cond = cond_embed(y)
            
            ddim = DDIMScheduler.from_config(noise_scheduler.config)
            ddim.set_timesteps(50)
            
            for t in ddim.timesteps:
                t = t.to(device)  # 确保时间步长在正确的设备上
                pred = unet(z, t, cond)
                z = ddim.step(pred, t, z).prev_sample
            
            # VAE解码
            gen_img = vae_decoder(z)
            gen_img = (gen_img + 1) / 2  # 转换到[0,1]范围
            
            # 保存生成的图片
            out_path = os.path.join(os.path.dirname(__file__), 
                                  f"sample_class{class_id}_epoch{epoch}.png")
            torchvision.utils.save_image(gen_img, out_path)
    
    unet.train()
    cond_embed.train()

if __name__ == "__main__":
    main()