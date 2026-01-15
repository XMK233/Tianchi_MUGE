import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

# 导入 diffusers 相关库
try:
    from diffusers import UNet2DConditionModel, DDPMScheduler
    print("成功导入 diffusers 库")
except ImportError:
    print("未安装 diffusers 库，请先安装: pip install diffusers")
    sys.exit(1)

# 添加当前目录到路径，以便导入chunk_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunk_loader import ChunkLoader, decode_base64_image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 全局参数
IMAGE_SIZE = 256
MAX_TEXT_LENGTH = 64
HIDDEN_DIM = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5  # 增加训练轮数
TRAIN_CHUNK_SIZE = 1000 ## 训练集加载多少
TOTAL_IMAGES = 1000 ## 训练样本总共加载多少
CHUNK_SIZE = 200  ## 验证集总共加载多少

# 数据文件路径
TRAIN_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
TRAIN_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
VAL_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.tsv"
VAL_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.text.tsv"

schm_type = "v8"
FORCE_RETRAIN = True
MODEL_SAVE_DIR = f"/mnt/d/forCoding_data/Tianchi_MUGE/plan_1/models/{schm_type}"

# 如果FORCE_RETRAIN为True，则删除模型目录强行重新训练
if FORCE_RETRAIN:
    if os.path.exists(MODEL_SAVE_DIR):
        print(f"强制重新训练，删除模型目录: {MODEL_SAVE_DIR}")
        shutil.rmtree(MODEL_SAVE_DIR)
    else:
        print(f"模型目录不存在，直接开始训练: {MODEL_SAVE_DIR}")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 1. ECommerceDataset类
class ECommerceDataset(Dataset):
    def __init__(self, img_path, text_path, tokenizer, text_encoder, chunk_size=10, start_line=0):
        self.loader = ChunkLoader(img_path, text_path, chunk_size=chunk_size)
        self.data = self.loader.get_chunk(start_line=start_line, chunk_size=chunk_size)
        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Grayscale(num_output_channels=3), # 确保是3通道
            T.ToTensor(), # [0, 1]
            T.Normalize([0.5], [0.5]) # [0, 1] -> [-1, 1]
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
            "image": img_tensor, # [-1, 1]
            "text_emb": text_emb.squeeze(0),  # [MAX_TEXT_LENGTH, 768]
            "description": description
        }

# 2. Diffusion Model Wrapper (UNet + Scheduler) - 直接处理图像
class DiffusionModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化 UNet
        # 直接处理 256x256 的图像，需要更多的下采样层
        self.unet = UNet2DConditionModel(
            sample_size=IMAGE_SIZE, # 256
            in_channels=3, # RGB
            out_channels=3, # RGB
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",              # 256 -> 128
                "CrossAttnDownBlock2D",     # 128 -> 64
                "CrossAttnDownBlock2D",     # 64 -> 32
                "CrossAttnDownBlock2D",     # 32 -> 16
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",       # 16 -> 32
                "CrossAttnUpBlock2D",       # 32 -> 64
                "CrossAttnUpBlock2D",       # 64 -> 128
                "UpBlock2D"                 # 128 -> 256
            ),
            cross_attention_dim=768, # BERT embedding dimension
        )
        
        # 初始化 Scheduler (DDPM)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon" # 预测噪声
        )
        
    def forward(self, images, text_emb):
        """
        训练时的前向传播：计算噪声预测损失
        images: [B, 3, 256, 256]
        text_emb: [B, SeqLen, 768]
        """
        # 1. 采样时间步 t
        bsz = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=images.device
        ).long()
        
        # 2. 采样噪声 noise
        noise = torch.randn_like(images)
        
        # 3. 加噪 (Forward Diffusion Process)
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        
        # 4. 预测噪声 (UNet)
        # encoder_hidden_states 接收 text_emb
        noise_pred = self.unet(noisy_images, timesteps, encoder_hidden_states=text_emb).sample
        
        # 5. 计算损失 (MSE Loss between actual noise and predicted noise)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

    @torch.no_grad()
    def sample(self, text_emb, num_inference_steps=50):
        """
        生成时的采样过程
        """
        bsz = text_emb.shape[0]
        
        # 1. 初始噪声
        images = torch.randn(
            bsz, self.unet.config.in_channels, IMAGE_SIZE, IMAGE_SIZE, device=text_emb.device
        )
        
        # 2. 设置推理步数
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 3. 逐步去噪
        for t in self.scheduler.timesteps: # tqdm(self.scheduler.timesteps, desc="Sampling"):
            # 预测噪声
            noise_pred = self.unet(images, t, encoder_hidden_states=text_emb).sample
            
            # 计算上一步的 image (x_{t-1})
            images = self.scheduler.step(noise_pred, t, images).prev_sample
            
        return images

# 3. FID计算器 (保持不变，用于评估)
class FIDCalculator:
    def __init__(self):
        self.inception = None
        self.load_inception()
    
    def load_inception(self):
        """加载Inception v3模型"""
        try:
            import timm
            cache_dir = "/mnt/d/HuggingFaceModels/"
            self.inception = timm.create_model(
                'inception_v3', pretrained=False, num_classes=0, cache_dir=cache_dir,
                pretrained_cfg_overlay={'file': '/mnt/d/ModelScopeModels/timm/inception_v3.tv_in1k/model.safetensors'}
            )
            self.inception.eval().to(device)
            print("Inception v3模型加载成功")
        except Exception as e:
            print(f"加载Inception v3模型时出错: {e}")
    
    def get_features(self, images):
        if self.inception is None: return None
        with torch.no_grad():
            images = torch.clamp(images, 0, 1)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            resize = T.Resize((299, 299))
            resized_images = resize(images)
            features = self.inception(resized_images)
        return features
    
    def calculate_fid(self, real_features, generated_features):
        if real_features is None or generated_features is None: return float('inf')
        
        # 简单归一化和计算
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

# 4. 训练函数
def train(tokenizer, text_encoder):
    print("\n=== 开始训练 (Image Diffusion Mode) ===")
    
    # 初始化 Diffusion Model Wrapper
    model = DiffusionModelWrapper().to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # 2. 解析已有检查点以确定最新进度
    latest_chunk = 0
    latest_epoch = -1
    latest_ckpt_path = None
    
    print(f"检查模型保存目录: {MODEL_SAVE_DIR}")
    if os.path.exists(MODEL_SAVE_DIR):
        ckpt_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.ckpt')]
        if ckpt_files:
            print(f"发现 {len(ckpt_files)} 个检查点文件")
            for ckpt_file in ckpt_files:
                try:
                    # 格式：diffusion_no_vae_chunk_{chunk_idx}_epoch_{epoch}.ckpt
                    parts = ckpt_file.split('_')
                    # diffusion, no, vae, chunk, {idx}, epoch, {epoch}.ckpt -> length 7
                    if len(parts) >= 7 and parts[3] == 'chunk' and parts[5] == 'epoch':
                        chunk_idx = int(parts[4])
                        epoch = int(parts[6].split('.')[0])
                        
                        if (chunk_idx > latest_chunk) or (chunk_idx == latest_chunk and epoch > latest_epoch):
                            latest_chunk = chunk_idx
                            latest_epoch = epoch
                            latest_ckpt_path = os.path.join(MODEL_SAVE_DIR, ckpt_file)
                except Exception as e:
                    print(f"解析检查点文件名 {ckpt_file} 时出错: {e}")

    # 3. 加载最新检查点（如果存在）
    if latest_ckpt_path:
        print(f"\n=== 加载最新检查点: {latest_ckpt_path} ===")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'chunk_idx', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                print(f"检查点缺少必要的键: {missing_keys}")
                raise KeyError(f"Missing keys: {missing_keys}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            latest_chunk = checkpoint['chunk_idx']
            latest_epoch = checkpoint['epoch']
            
            print(f"成功加载检查点，继续从分片 {latest_chunk}, epoch {latest_epoch+1} 开始训练")
            print(f"检查点保存时的损失: {checkpoint.get('loss', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            print("将从头开始训练")
            latest_chunk = 0
            latest_epoch = -1
    else:
        print("没有发现检查点，将从头开始训练")
    
    # 训练参数
    total_images = TOTAL_IMAGES
    chunk_size = TRAIN_CHUNK_SIZE
    total_chunks = (total_images + chunk_size - 1) // chunk_size
    
    print(f"总分片数: {total_chunks}, 批次大小: {BATCH_SIZE}")
    
    for chunk_idx in range(latest_chunk, total_chunks):
        print(f"\n=== 加载第 {chunk_idx+1}/{total_chunks} 个分片 ===")
        start_line = chunk_idx * chunk_size
        
        try:
            train_dataset = ECommerceDataset(
                TRAIN_IMG_PATH, 
                TRAIN_TEXT_PATH, 
                tokenizer, 
                text_encoder, 
                chunk_size=chunk_size,
                start_line=start_line
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        except Exception as e:
            print(f"加载分片出错: {e}")
            continue
            
        # 计算起始 epoch
        start_epoch = latest_epoch + 1 if chunk_idx == latest_chunk else 0
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"Chunk {chunk_idx+1} - Epoch {epoch+1}")
            for batch in pbar:
                images = batch["image"].to(device) # [-1, 1]
                text_emb = batch["text_emb"].to(device) # [B, SeqLen, 768]
                
                try:
                    # 混合精度训练 Diffusion
                    with torch.cuda.amp.autocast():
                        loss = model(images, text_emb)
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({"loss": loss.item()})
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # 保存检查点
            ckpt_path = os.path.join(MODEL_SAVE_DIR, f"diffusion_no_vae_chunk_{chunk_idx}_epoch_{epoch}.ckpt")
            print(f"保存检查点到 {ckpt_path}")
            torch.save({
                'chunk_idx': chunk_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'learning_rate': LEARNING_RATE,
                'total_images': total_images,
                'chunk_size': chunk_size
            }, ckpt_path)
            
        del train_dataset, train_loader
        torch.cuda.empty_cache()
    
    print("\n=== 所有分片训练完成 ===")
    return model

# 5. 生成函数
def generate_images(model, text_encoder, tokenizer, descriptions):
    print("\n=== 开始生成图像 ===")
    model.eval()
    generated_images = []
    
    for desc in tqdm(descriptions, desc="生成中"):
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
        
        # 采样 (Image Diffusion)
        try:
            images = model.sample(text_emb, num_inference_steps=50)
            
            # 后处理: [-1, 1] -> [0, 1]
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
            generated_images.append(images[0].cpu())
        except Exception as e:
            print(f"生成失败: {e}")
            generated_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
            
    return generated_images

# 主函数
if __name__ == "__main__":
    print("电商文本到图像生成 - 无VAE直接Diffusion版")
    
    # 加载语言模型
    tokenizer = BertTokenizer.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/")
    text_encoder = BertModel.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/").to(device)
    text_encoder.eval()
    
    # 训练
    model = train(tokenizer, text_encoder)
    
    # 验证生成
    print("\n准备验证数据...")
    val_dataset = ECommerceDataset(VAL_IMG_PATH, VAL_TEXT_PATH, tokenizer, text_encoder, chunk_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    descriptions = []
    real_images = []
    for batch in val_loader:
        descriptions.extend(batch["description"])
        real_images.extend([img.cpu() for img in batch["image"]])
        if len(descriptions) >= 4: break # 仅测试少量
    
    generated_images = generate_images(model, text_encoder, tokenizer, descriptions)
    
    # 保存结果
    os.makedirs(f"generated_{schm_type}", exist_ok=True)
    for i, (img_tensor, real_img_tensor, desc) in enumerate(zip(generated_images, real_images, descriptions)):
        # 保存生成的图像
        img_pil = T.ToPILImage()(img_tensor)
        img_pil.save(f"generated_{schm_type}/img_{i+1}.png")
        
        # 保存调整大小后的原始图像
        real_img_pil = T.ToPILImage()(T.Resize((IMAGE_SIZE, IMAGE_SIZE))(real_img_tensor))
        real_img_pil.save(f"generated_{schm_type}/img_{i+1}_real.png")
        
        # 保存描述
        with open(f"generated_{schm_type}/img_{i+1}_desc.txt", "w", encoding="utf-8") as f:
            f.write(desc)
            
    print(f"生成完成，结果保存在 generated_{schm_type} 目录")
