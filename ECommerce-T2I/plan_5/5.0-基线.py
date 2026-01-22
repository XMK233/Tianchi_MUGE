import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--n_epochs', type=int, default=100)
_arg_parser.add_argument('--train_mode', type=str, default='resume', choices=['fresh', 'resume'])
_args, _unknown = _arg_parser.parse_known_args()

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import os
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError
import pandas as pd
import base64
import io
from transformers import AutoTokenizer, AutoModel

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("未安装 peft 库，请先执行: pip install peft") from e

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

_base = os.path.splitext(os.path.basename(__file__))[0]
_save_dir = "/mnt/d/forCoding_data/Tianchi_MUGE/plan_5/trained_models"
# 如果保存目录不存在，则创建
os.makedirs(_save_dir, exist_ok=True)
_save_path = os.path.join(
    _save_dir,
    f"{_base}_model.pth"
)
n_epochs = _args.n_epochs
start_epoch = 0
start_step = 0

img_sz = (256, 256)
tf = transforms.Compose([
    # transforms.Lambda(crop_im),
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
include = [str(i) for i in range(2, 23)]
# 适配 TSV + 文本 TSV 的数据加载（不再依赖 coarse_label）
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.cls1.tsv"
TEXT_TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"

class TSVCSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path, text_path, transform=None):
        self.transform = transform
        # 加载文本 TSV（img_id -> text）
        text_map = {}
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text_map[str(parts[0])] = parts[1]
        # 加载 TSV（img_id -> base64）
        img_map = {}
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_map[parts[0]] = parts[1]
        # 仅保留既有图片又有文本的样本
        self.samples = []
        for img_id, b64 in img_map.items():
            text = text_map.get(str(img_id), "")
            self.samples.append((img_id, b64, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, b64, text = self.samples[idx]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        if self.transform is not None:
            img = self.transform(img)
        return img, text

dataset = TSVCSVImageDataset(TSV_PATH, TEXT_TSV_PATH, transform=tf)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

## 定义模型
class ImageAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        # 编码器：将图片压缩到潜空间
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # 256->128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128->64
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1), # 64->32
            nn.ReLU(),
        )
        # 解码器：将潜空间还原为图片
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1), # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 128->256
            nn.Sigmoid(), # 输出0-1之间的像素值
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# -----------------------------
# 2. 简化的文本编码器
# 实际项目中建议用 CLIP 或 BERT，这里用一个简单的嵌入层模拟
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256, max_len=77):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def forward(self, text_ids):
        # text_ids: [batch_size, seq_len]
        embeddings = self.embedding(text_ids) 
        # 简单平均池化得到句子向量
        text_features = embeddings.mean(dim=1) 
        return text_features # [batch_size, embed_dim]

class DiffusionUNet(nn.Module):
    def __init__(self, image_channels=3, text_embed_dim=256):
        super().__init__()
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(text_embed_dim, text_embed_dim*2),
            nn.SiLU(),
            nn.Linear(text_embed_dim*2, text_embed_dim)
        )
        
        # 简单的卷积层来模拟UNet
        self.conv1 = nn.Conv2d(image_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64 + text_embed_dim, 128, 3, padding=1) # 图像特征 + 文本特征拼接
        self.conv3 = nn.Conv2d(128, image_channels, 3, padding=1)
        
        # 文本投影层，用来融合文本信息
        self.text_proj = nn.Linear(text_embed_dim, 64) 

    def forward(self, x, t, text_emb):
        # x: 图像 [B, C, H, W]
        # t: 时间步
        # text_emb: 文本特征 [B, D]
        
        # 处理时间步
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # [B, D, 1, 1]
        
        # 处理文本
        text_emb = self.text_proj(text_emb)
        text_emb = text_emb.unsqueeze(-1).unsqueeze(-1) # [B, 64, 1, 1]
        
        # 简单的前向传播
        h = F.relu(self.conv1(x))
        
        # 融合文本信息 (这里简化了，实际是交叉注意力)
        # 我们把文本特征广播到和图像一样的大小然后拼接
        h = torch.cat([h, text_emb.expand(-1, -1, h.shape[-2], h.shape[-1])], dim=1)
        
        h = F.relu(self.conv2(h))
        return self.conv3(h)

class TextToImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 256
        self.latent_dim = 256
        
        # 1. 图像压缩器 (模拟 VQ-GAN 或 AutoEncoder)
        self.image_ae = ImageAutoEncoder(latent_dim=self.latent_dim)
        
        # 2. 文本理解器
        self.text_enc = TextEncoder()
        
        # 3. 核心生成器
        self.diffusion_net = DiffusionUNet(image_channels=3, text_embed_dim=256)

    def forward(self, image, text_ids):
        # 1. (可选) 压缩图像到潜空间，为了简单这里直接用原图
        # latent = self.image_ae.encode(image)
        
        # 2. 编码文本
        text_features = self.text_enc(text_ids) # [B, 256]
        
        # 3. 模拟扩散过程：给图片加噪
        noise = torch.randn_like(image)
        t = torch.rand(image.size(0)).to(image.device) # 随机时间步
        
        # 4. 去噪网络预测噪声
        # 这里需要把时间步 t 处理一下，简单线性变换
        t_mlp = torch.sin(t * 10).unsqueeze(1).expand(-1, 256)
        predicted_noise = self.diffusion_net(image + noise * 0.1, t_mlp, text_features)
        
        return predicted_noise, noise # 我们希望 predicted_noise 尽可能接近 noise

## 训练模型
def train_loop(dataloader, model, optimizer, loss_fn):
    model.train()
    for batch in dataloader:
        images = batch['image'] # 形状: [B, 3, 256, 256]
        texts = batch['text_ids'] # 形状: [B, 77]
        
        # 前向传播
        pred_noise, true_noise = model(images, texts)
        
        # 计算损失：让模型预测的噪声和真实的噪声一致
        loss = loss_fn(pred_noise, true_noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}")

## 验证模型，观察效果
class FIDCalculator:
    def __init__(self):
        self.inception = None
        self.load_inception()
    
    def load_inception(self):
        try:
            import timm
            cache_dir = "/mnt/d/HuggingFaceModels/"
            self.inception = timm.create_model(
                "inception_v3",
                pretrained=False,
                num_classes=0,
                cache_dir=cache_dir,
                pretrained_cfg_overlay={
                    "file": "/mnt/d/ModelScopeModels/timm/inception_v3.tv_in1k/model.safetensors"
                },
            )
            self.inception.eval().to(device)
            print("Inception v3模型加载成功")
        except Exception as e:
            print(f"加载Inception v3模型时出错: {e}")
    
    def get_features(self, images):
        if self.inception is None:
            return None
        with torch.no_grad():
            images = torch.clamp(images, 0, 1)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            resize = transforms.Resize((299, 299))
            resized_images = resize(images)
            features = self.inception(resized_images)
        return features
    
    def calculate_fid(self, real_features, generated_features):
        if real_features is None or generated_features is None:
            return float("inf")
        
        real_features = real_features - torch.mean(real_features)
        real_features = real_features / (torch.std(real_features) + 1e-6)
        generated_features = generated_features - torch.mean(generated_features)
        generated_features = generated_features / (torch.std(generated_features) + 1e-6)
        
        mu1 = torch.mean(real_features, dim=0)
        mu2 = torch.mean(generated_features, dim=0)
        mu_diff = torch.sum((mu1 - mu2) ** 2)
        
        var1 = torch.var(real_features, dim=0, unbiased=False) + 1e-4
        var2 = torch.var(generated_features, dim=0, unbiased=False) + 1e-4
        
        trace_term = torch.sum(var1 + var2)
        
        log_var1 = torch.log(var1)
        log_var2 = torch.log(var2)
        log_sqrt_term = 0.5 * (log_var1 + log_var2)
        exp_log_sqrt = torch.exp(log_sqrt_term)
        cross_term = 2 * torch.sum(exp_log_sqrt)
        
        fid = mu_diff + trace_term - cross_term
        return max(0.0, fid.item())


def calculate_and_print_fid(real_images, generated_images, fid_calculator):
    print("\n=== 计算FID分数 ===")
    
    if len(real_images) == 0 or len(generated_images) == 0:
        print("没有足够的图像用于计算FID")
        return
    
    try:
        real_images_tensor = torch.stack(real_images).to(device)
        real_features = fid_calculator.get_features(real_images_tensor)
        
        generated_images_tensor = torch.stack(generated_images).to(device)
        generated_features = fid_calculator.get_features(generated_images_tensor)
        
        fid_score = fid_calculator.calculate_fid(real_features, generated_features)
        
        print(f"FID分数: {fid_score:.4f}")
    except Exception as e:
        print(f"计算FID时出错: {e}")


VAL_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.cls1.tsv"
VAL_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
text_map = {}
with open(VAL_TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            text_map[parts[0]] = parts[1]
out_dir = os.path.join(os.path.dirname(__file__), _base)
os.makedirs(out_dir, exist_ok=True)
max_count = 200
page_size = 10
page_index = 1
items = []
count = 0
real_images_for_fid = []
gen_images_for_fid = []
with open(VAL_IMG_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, "生成图片中(总共{}张)".format(max_count)):
        if count >= max_count:
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img_id = parts[0]
        img_b64 = parts[1]
        
        text_str = text_map.get(str(img_id), "")
        text_emb_val = encode_text_batch([text_str], tokenizer, text_encoder, device, grad=False)

        val_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        proc_t = tf(val_img)
        x = torch.randn(1, 1, img_sz[0], img_sz[1]).to(device)
        y = torch.zeros(1, dtype=torch.long).to(device)
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(30)
        for i, t in enumerate(ddim.timesteps):
            with torch.no_grad():
                residual = net(x, t.to(x.device), y, text_emb_val)
            x = ddim.step(residual, t, x).prev_sample
        gen = x.detach().cpu().clip(-1, 1)[0, 0]
        gen = (gen + 1.0) / 2.0
        real_images_for_fid.append(proc_t)
        to_pil = transforms.ToPILImage()
        gen_for_fid = gen.unsqueeze(0)
        gen_images_for_fid.append(gen_for_fid)
        gen_pil = to_pil(gen_for_fid)
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
    <div>img_id: {img_id}</div>
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

fid_calculator = FIDCalculator()
calculate_and_print_fid(real_images_for_fid, gen_images_for_fid, fid_calculator)
