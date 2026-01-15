import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import BertTokenizer, BertModel
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from tqdm import tqdm
import subprocess

# 添加当前目录到路径，以便导入chunk_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunk_loader import ChunkLoader, decode_base64_image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 全局参数
IMAGE_SIZE = 256
MAX_TEXT_LENGTH = 64
VOCAB_SIZE = 1024  # VQGAN codebook size
HIDDEN_DIM = 512
NUM_LAYERS = 4
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
CHUNK_SIZE = 10  # 测试用，加载10张图片

# 数据文件路径
TRAIN_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
TRAIN_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
VAL_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.tsv"
VAL_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.text.tsv"

# 2. ECommerceDataset类
class ECommerceDataset(Dataset):
    def __init__(self, img_path, text_path, tokenizer, text_encoder, chunk_size=10, start_line=0):
        self.loader = ChunkLoader(img_path, text_path, chunk_size=chunk_size)
        self.data = self.loader.get_chunk(start_line=start_line, chunk_size=chunk_size)
        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 使用外部传入的文本编码器
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_id, image_base64, description = self.data[idx]
        
        # 处理图像
        try:
            img = decode_base64_image(image_base64)
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"处理图像时出错: {e}")
            # 返回零张量作为替代
            img_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        
        # 处理文本
        text_inputs = self.tokenizer(
            description,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 获取文本嵌入
        with torch.no_grad():
            text_emb = self.text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device)
            ).last_hidden_state
        
        return {
            "img_id": img_id,
            "image": img_tensor,
            "text_emb": text_emb.squeeze(0),  # [MAX_TEXT_LENGTH, 768]
            "description": description
        }

# 2. SD VAE模型包装类
class SDVAEWrapper:
    def __init__(self, model_path="/mnt/d/ModelScopeModels/stabilityai/sd-vae-ft-mse/"):
        self.model_path = model_path
        self.vae = None
        self.load_model()
    
    def load_model(self):
        """加载SD VAE模型"""
        try:
            # 从diffusers导入AutoencoderKL
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            ).to(device)
            self.vae.eval()
            print("SD VAE模型加载成功")
        except ImportError:
            print("未安装diffusers，无法加载SD VAE模型")
        except Exception as e:
            print(f"加载SD VAE模型时出错: {e}")
    
    def encode(self, img_tensor):
        """将图像编码为latent表示"""
        if self.vae is None:
            raise RuntimeError("SD VAE模型未加载")
        
        with torch.no_grad():
            # 确保输入形状正确 [B, C, H, W]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # SD VAE输入需要在[-1, 1]范围
            img_tensor = img_tensor * 2 - 1
            
            # 编码为latent
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            # 缩放latent以匹配训练分布
            latent = latent * self.vae.config.scaling_factor
            
            return latent.squeeze(0)  # [4, 32, 32]
    
    def decode(self, latent):
        """将latent表示解码为图像"""
        if self.vae is None:
            raise RuntimeError("SD VAE模型未加载")
        
        with torch.no_grad():
            # 确保输入形状正确 [4, 32, 32]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            
            # 反向缩放latent
            latent = latent / self.vae.config.scaling_factor
            
            # 解码为图像
            img = self.vae.decode(latent).sample
            # 将图像从[-1, 1]转换到[0, 1]
            img = (img + 1) / 2
            
            return img[0]  # [3, 256, 256]

# 3. 文本到图像Transformer模型 (适配SD VAE连续latent)
class TextToImageTransformer(nn.Module):
    def __init__(self, text_dim=768, hidden=512, n_layers=4, latent_channels=4, latent_size=32):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)  # 将文本嵌入投影到hidden维度
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.flattened_latent_dim = latent_channels * latent_size * latent_size
        
        # 位置嵌入 (for flattened latent)
        self.pos_emb = nn.Embedding(self.flattened_latent_dim, hidden)  # 位置嵌入
        
        # Transformer Decoder
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=hidden, nhead=8),
            num_layers=n_layers,
        )
        
        self.head = nn.Linear(hidden, 1)  # 输出层 - 预测连续值
    
    def forward(self, text_emb, latents):
        """
        前向传播
        text_emb: [B, T, D] - 文本嵌入
        latents: [B, C, H, W] - 图像latent表示
        """
        # 投影文本嵌入
        text_emb = self.text_proj(text_emb)  # [B, T, hidden]
        
        # 展平latent
        batch_size, channels, height, width = latents.shape
        flattened_latents = latents.view(batch_size, -1, 1)  # [B, L, 1], L = C*H*W
        seq_len = flattened_latents.shape[1]
        
        # 生成位置嵌入
        pos = torch.arange(seq_len, device=latents.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.pos_emb(pos)  # [B, L, hidden]
        
        # 投影输入latent并添加位置嵌入
        hidden = text_emb.shape[-1]  # 获取hidden维度
        input_emb = torch.zeros(batch_size, seq_len, hidden, device=latents.device)
        for i in range(seq_len):
            input_emb[:, i, :] = pos_emb[:, i, :] * flattened_latents[:, i, 0].unsqueeze(1)
        
        # Transformer解码
        output = self.transformer(
            tgt=input_emb.permute(1, 0, 2),  # [L, B, hidden]
            memory=text_emb.permute(1, 0, 2),  # [T, B, hidden]
        )
        
        # 输出层 - 预测连续值
        predictions = self.head(output.permute(1, 0, 2)).squeeze(-1)  # [B, L]
        predictions = predictions.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        return predictions
    
    def generate_causal_mask(self, size):
        """生成因果掩码"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 4. 自回归生成函数 (适配SD VAE连续latent)
def autoregressive_generate(model, text_emb, latent_channels=4, latent_size=32):
    """
    生成图像latent表示
    """
    model.eval()
    
    # 初始latent - 从正态分布采样
    batch_size = text_emb.shape[0]
    generated_latents = torch.randn(
        batch_size, latent_channels, latent_size, latent_size
    ).to(device)
    
    # 使用模型生成优化的latent
    with torch.no_grad():
        predicted_latents = model(text_emb, generated_latents)
    
    return predicted_latents

# 5. FID计算器
class FIDCalculator:
    def __init__(self):
        self.inception = None
        self.load_inception()
    
    def load_inception(self):
        """加载Inception v3模型"""
        try:
            import torchvision.models as models
            self.inception = models.inception_v3(pretrained=True, transform_input=False)
            # 替换分类层为特征提取层
            self.inception.fc = nn.Identity()
            self.inception.eval().to(device)
            print("Inception v3模型加载成功")
        except Exception as e:
            print(f"加载Inception v3模型时出错: {e}")
    
    def get_features(self, images):
        """提取图像特征"""
        if self.inception is None:
            raise RuntimeError("Inception v3模型未加载")
        
        with torch.no_grad():
            # 确保输入图像在[0, 1]范围内
            images = torch.clamp(images, 0, 1)
            # Inception v3需要输入大小为299x299
            resize = T.Resize((299, 299))
            resized_images = resize(images)
            # 提取特征
            features = self.inception(resized_images)
        
        return features
    
    def calculate_fid(self, real_features, generated_features):
        """计算FID分数"""
        # 计算均值和协方差
        mu1 = torch.mean(real_features, dim=0)
        mu2 = torch.mean(generated_features, dim=0)
        
        sigma1 = torch.cov(real_features.T)
        sigma2 = torch.cov(generated_features.T)
        
        # 计算FID
        fid = torch.norm(mu1 - mu2)**2 + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(torch.matmul(sigma1, sigma2)))
        
        return fid.item()

# 6. 训练函数
def train(tokenizer, text_encoder):
    print("\n=== 开始训练 ===")
    
    # 1. 加载数据
    print("加载训练数据...")
    train_dataset = ECommerceDataset(TRAIN_IMG_PATH, TRAIN_TEXT_PATH, tokenizer, text_encoder, chunk_size=CHUNK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("加载验证数据...")
    val_dataset = ECommerceDataset(VAL_IMG_PATH, VAL_TEXT_PATH, tokenizer, text_encoder, chunk_size=CHUNK_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 加载SD VAE模型
    vqgan = SDVAEWrapper()
    
    # 3. 初始化模型
    model = TextToImageTransformer(
        text_dim=768,  # BERT隐藏层维度
        hidden=HIDDEN_DIM,
        n_layers=NUM_LAYERS,
        latent_channels=4,  # SD VAE latent channels
        latent_size=32  # SD VAE latent size for 256x256 images
    ).to(device)
    
    # 4. 初始化优化器和梯度缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"训练 epoch {epoch+1}/{NUM_EPOCHS}"):
            images = batch["image"].to(device)
            text_emb = batch["text_emb"].to(device)
            
            # 使用SD VAE编码图像为latent
            try:
                latents = vqgan.encode(images)  # [B, 4, 32, 32]
                
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    predicted_latents = model(text_emb, latents)
                    loss = F.mse_loss(predicted_latents, latents)
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, 平均损失: {avg_loss:.4f}")
    
    print("\n=== 训练完成 ===")
    return model, vqgan

# 7. 推理函数
def generate_images(model, vqgan, text_encoder, tokenizer, descriptions, num_samples=1):
    """
    根据文本描述生成图像
    """
    print("\n=== 开始生成图像 ===")
    
    model.eval()
    generated_images = []
    
    for desc in tqdm(descriptions, desc="生成图像"):
        # 编码文本
        text_inputs = tokenizer(
            desc,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_emb = text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device)
            ).last_hidden_state
        
        # 生成图像latent
        generated_latents = autoregressive_generate(model, text_emb)
        
        # 解码为图像
        try:
            generated_image = vqgan.decode(generated_latents[0])
            # 转换为[0, 1]范围
            generated_image = torch.clamp(generated_image, -1, 1) * 0.5 + 0.5
            generated_images.append(generated_image.cpu())
        except Exception as e:
            print(f"生成图像时出错: {e}")
            # 添加空白图像
            generated_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
    
    return generated_images

# 8. 计算FID函数
def calculate_and_print_fid(real_images, generated_images, fid_calculator):
    """
    计算并打印FID分数
    """
    print("\n=== 计算FID分数 ===")
    
    if len(real_images) == 0 or len(generated_images) == 0:
        print("没有足够的图像用于计算FID")
        return
    
    try:
        # 提取真实图像特征
        real_images_tensor = torch.stack(real_images).to(device)
        real_features = fid_calculator.get_features(real_images_tensor)
        
        # 提取生成图像特征
        generated_images_tensor = torch.stack(generated_images).to(device)
        generated_features = fid_calculator.get_features(generated_images_tensor)
        
        # 计算FID
        fid_score = fid_calculator.calculate_fid(real_features, generated_features)
        print(f"FID分数: {fid_score:.4f}")
        
    except Exception as e:
        print(f"计算FID时出错: {e}")

# 主函数
if __name__ == "__main__":
    print("电商文本到图像生成 - 初试")
    
    # 加载语言模型（移到外部）
    print("加载语言模型...")
    tokenizer = BertTokenizer.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/")
    text_encoder = BertModel.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/").to(device)
    text_encoder.eval()
    print("语言模型加载完成")
    
    # 1. 训练模型
    model, vqgan = train(tokenizer, text_encoder)
    
    # 2. 准备验证数据用于生成
    print("\n准备验证数据...")
    val_dataset = ECommerceDataset(VAL_IMG_PATH, VAL_TEXT_PATH, tokenizer, text_encoder, chunk_size=CHUNK_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 获取验证集的文本描述和真实图像
    descriptions = []
    real_images = []
    for batch in val_loader:
        descriptions.extend(batch["description"])
        real_images.extend([img.cpu() for img in batch["image"]])
    
    # 限制数量用于测试
    descriptions = descriptions[:CHUNK_SIZE]
    real_images = real_images[:CHUNK_SIZE]
    
    print(f"准备生成 {len(descriptions)} 张图像")
    
    # 3. 生成图像
    generated_images = generate_images(model, vqgan, text_encoder, tokenizer, descriptions)
    
    # # 4. 计算FID分数
    # fid_calculator = FIDCalculator()
    # calculate_and_print_fid(real_images, generated_images, fid_calculator)
    
    # 5. 保存一些生成的图像用于查看
    print("\n保存生成的图像...")
    os.makedirs("generated_images", exist_ok=True)
    
    for i, (img_tensor, desc) in enumerate(zip(generated_images, descriptions)):
        # 转换为PIL图像
        img_pil = T.ToPILImage()(img_tensor)
        
        # 保存图像
        img_path = f"generated_images/img_{i+1}.png"
        img_pil.save(img_path)
        
        # 保存描述
        desc_path = f"generated_images/img_{i+1}_desc.txt"
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(desc)
    
    print(f"生成的图像已保存到 generated_images 目录")
    print("\n所有测试完成!")
