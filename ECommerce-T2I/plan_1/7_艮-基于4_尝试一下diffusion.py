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
    from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
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
LEARNING_RATE = 1e-4  # Diffusion通常需要稍大的LR或更精细的调整
NUM_EPOCHS = 20  # 增加训练轮数
TRAIN_CHUNK_SIZE = 1000 ## 训练集加载多少
TOTAL_IMAGES = 7000 ## 训练样本总共加载多少
CHUNK_SIZE = 200  ## 验证集总共加载多少

# 数据文件路径
TRAIN_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
TRAIN_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
VAL_IMG_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.tsv"
VAL_TEXT_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.text.tsv"

schm_type = "v7"
FORCE_RETRAIN = False
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
            T.Grayscale(num_output_channels=3), # 转换为3通道以适配VAE
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

# 2. SD VAE模型包装类
class SDVAEWrapper:
    def __init__(self, model_path="/mnt/d/ModelScopeModels/stabilityai/sd-vae-ft-mse/"):
        self.model_path = model_path
        self.vae = None
        self.load_model()
    
    def load_model(self):
        """加载SD VAE模型"""
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            ).to(device)
            self.vae.eval()
            print("SD VAE模型加载成功")
        except Exception as e:
            print(f"加载SD VAE模型时出错: {e}")
            # 如果加载失败，可能会导致后续错误
    
    def encode(self, img_tensor):
        """
        将图像编码为latent表示
        img_tensor: [B, 3, H, W] 范围 [-1, 1]
        """
        if self.vae is None:
            raise RuntimeError("SD VAE模型未加载")
        
        with torch.no_grad():
            # 确保输入形状正确 [B, C, H, W]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 编码为latent
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            # 缩放latent以匹配训练分布
            latent = latent * self.vae.config.scaling_factor
            
            return latent # [B, 4, 32, 32]
    
    def decode(self, latent):
        """
        将latent表示解码为图像
        latent: [B, 4, 32, 32]
        """
        if self.vae is None:
            raise RuntimeError("SD VAE模型未加载")
        
        with torch.no_grad():
            # 确保输入形状正确
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            
            # 反向缩放latent
            latent = latent / self.vae.config.scaling_factor
            
            # 解码为图像
            img = self.vae.decode(latent).sample
            
            # 将图像从[-1, 1]转换到[0, 1] (仅用于显示)
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # 转灰度用于兼容后续处理 (如果需要)
            # img_gray = torch.sum(img * torch.tensor([0.299, 0.587, 0.114]).to(device).view(1, 3, 1, 1), dim=1, keepdim=True)
            
            return img  # [B, 3, 256, 256]

# 3. Diffusion Model Wrapper (UNet + Scheduler)
class DiffusionModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化 UNet
        # 这是一个简化的UNet配置，适配32x32 latent
        self.unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D"
            ),
            cross_attention_dim=768, # BERT embedding dimension
        )
        
        # 初始化 Scheduler (DDPM)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        
    def forward(self, latents, text_emb):
        """
        训练时的前向传播：计算噪声预测损失
        latents: [B, 4, 32, 32]
        text_emb: [B, SeqLen, 768]
        """
        # 1. 采样时间步 t
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        
        # 2. 采样噪声 noise
        noise = torch.randn_like(latents)
        
        # 3. 加噪 (Forward Diffusion Process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. 预测噪声 (UNet)
        # encoder_hidden_states 接收 text_emb
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
        
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
        latents = torch.randn(
            bsz, self.unet.config.in_channels, 32, 32, device=text_emb.device
        )
        
        # 2. 设置推理步数
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 3. 逐步去噪
        for t in self.scheduler.timesteps: #tqdm(self.scheduler.timesteps, desc="Sampling"):
            # 预测噪声
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_emb).sample
            
            # 计算上一步的 latent (x_{t-1})
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents

# 4. FID计算器 (保持不变，用于评估)
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
        
        # 简单归一化和计算 (省略详细注释以节省篇幅，逻辑保持一致)
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

# 5. 计算FID函数
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

# 6. 训练函数
def train(tokenizer, text_encoder):
    print("\n=== 开始训练 (Diffusion Mode) ===")
    
    # 1. 检查模型保存目录
    print(f"检查模型保存目录: {MODEL_SAVE_DIR}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 2. 解析已有检查点以确定最新进度
    latest_chunk = 0
    latest_epoch = -1
    latest_ckpt_path = None
    
    if os.path.exists(MODEL_SAVE_DIR):
        ckpt_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.ckpt')]
        if ckpt_files:
            print(f"发现 {len(ckpt_files)} 个检查点文件")
            
            # 解析每个检查点文件名
            for ckpt_file in ckpt_files:
                try:
                    # 格式：model_chunk_{chunk_idx}_epoch_{epoch}.ckpt
                    parts = ckpt_file.split('_')
                    if len(parts) >= 5 and parts[0] == 'model' and parts[1] == 'chunk' and parts[3] == 'epoch':
                        chunk_idx = int(parts[2])
                        epoch = int(parts[4].split('.')[0])
                        
                        # 更新最新进度
                        if (chunk_idx > latest_chunk) or (chunk_idx == latest_chunk and epoch > latest_epoch):
                            latest_chunk = chunk_idx
                            latest_epoch = epoch
                            latest_ckpt_path = os.path.join(MODEL_SAVE_DIR, ckpt_file)
                except Exception as e:
                    print(f"解析检查点文件名 {ckpt_file} 时出错: {e}")
    
    # 3. 初始化 VAE
    vae = SDVAEWrapper()
    
    # 4. 初始化 Diffusion Model Wrapper
    model = DiffusionModelWrapper().to(device)
    
    # 5. 初始化优化器和梯度缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. 加载最新检查点（如果存在）
    if latest_ckpt_path:
        print(f"\n=== 加载最新检查点: {latest_ckpt_path} ===")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            # 验证检查点包含必要的键
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'chunk_idx', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                print(f"检查点缺少必要的键: {missing_keys}")
                raise KeyError(f"Missing keys: {missing_keys}")
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型状态加载成功")
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态加载成功")
            
            # 加载梯度缩放器状态
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("梯度缩放器状态加载成功")
            
            # 更新最新进度
            latest_chunk = checkpoint['chunk_idx']
            latest_epoch = checkpoint['epoch']
            
            print(f"成功加载检查点，继续从分片 {latest_chunk}, epoch {latest_epoch+1} 开始训练")
            print(f"检查点保存时的损失: {checkpoint.get('loss', 'N/A'):.4f}")
            
        except KeyError as e:
            print(f"检查点格式错误: {e}")
            print("将从头开始训练")
            latest_chunk = 0
            latest_epoch = -1
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            print("将从头开始训练")
            latest_chunk = 0
            latest_epoch = -1
    else:
        print("没有发现检查点，将从头开始训练")
    
    # 7. 训练参数设置
    total_images = TOTAL_IMAGES
    chunk_size = TRAIN_CHUNK_SIZE
    total_chunks = (total_images + chunk_size - 1) // chunk_size
    
    print(f"\n=== 训练参数设置 ===")
    print(f"总训练图片数: {total_images}")
    print(f"每次加载图片数: {chunk_size}")
    print(f"总分片数: {total_chunks}")
    print(f"每个分片训练轮数: {NUM_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    
    # 8. 循环加载每个分片进行训练，从最新分片开始
    for chunk_idx in range(latest_chunk, total_chunks):
        print(f"\n=== 加载第 {chunk_idx+1}/{total_chunks} 个分片 ({chunk_size} 张图片) ===")
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
            
            # 初始化学习率调度器（每个分片重新初始化）
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))
            
            if len(train_dataset) == 0:
                print(f"第 {chunk_idx+1} 个分片没有数据，跳过")
                continue
                
        except Exception as e:
            print(f"加载分片出错: {e}")
            continue
            
        # 在当前分片上进行训练，跳过已经完成的epoch
        start_epoch = latest_epoch + 1 if chunk_idx == latest_chunk else 0
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"分片 {chunk_idx+1}/{total_chunks} - epoch {epoch+1}/{NUM_EPOCHS}")
            for batch in pbar:
                images = batch["image"].to(device) # [-1, 1]
                text_emb = batch["text_emb"].to(device) # [B, SeqLen, 768]
                
                try:
                    # 1. 编码图像为 Latent
                    latents = vae.encode(images) # [B, 4, 32, 32]
                    
                    # 2. 混合精度训练 Diffusion
                    with torch.cuda.amp.autocast():
                        loss = model(latents, text_emb)
                    
                    # 3. 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # 更新学习率
                    scheduler.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({"loss": loss.item()})
                    
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"分片 {chunk_idx+1}/{total_chunks} - Epoch {epoch+1}/{NUM_EPOCHS}, 平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            ckpt_path = os.path.join(MODEL_SAVE_DIR, f"model_chunk_{chunk_idx}_epoch_{epoch}.ckpt")
            print(f"\n=== 保存检查点到 {ckpt_path} ===")
            
            checkpoint = {
                'chunk_idx': chunk_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'learning_rate': LEARNING_RATE,
                'total_images': total_images,
                'chunk_size': chunk_size
            }
            
            try:
                torch.save(checkpoint, ckpt_path)
                print("检查点保存成功")
            except Exception as e:
                print(f"保存检查点时出错: {e}")
            
        del train_dataset, train_loader
        torch.cuda.empty_cache()  # 清理GPU缓存
        print("清理缓存完成")
    
    print("\n=== 所有分片训练完成 ===")
    return model, vae

# 6. 生成函数
def generate_images(model, vae, text_encoder, tokenizer, descriptions):
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
        
        # 采样 (Latent Diffusion)
        try:
            latents = model.sample(text_emb, num_inference_steps=30) # 减少步数以加快演示
            
            # 解码
            img = vae.decode(latents)
            generated_images.append(img.cpu().squeeze(0))
        except Exception as e:
            print(f"生成失败: {e}")
            generated_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
            
    return generated_images

# 主函数
if __name__ == "__main__":
    print("电商文本到图像生成 - Diffusion 版")
    
    # 加载语言模型
    tokenizer = BertTokenizer.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/")
    text_encoder = BertModel.from_pretrained("/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/").to(device)
    text_encoder.eval()
    
    # 训练
    model, vae = train(tokenizer, text_encoder)
    
    # 验证生成
    print("\n准备验证数据...")
    val_dataset = ECommerceDataset(VAL_IMG_PATH, VAL_TEXT_PATH, tokenizer, text_encoder, chunk_size=CHUNK_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # val_dataset = ECommerceDataset(TRAIN_IMG_PATH, TRAIN_TEXT_PATH, tokenizer, text_encoder, chunk_size=CHUNK_SIZE)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    descriptions = []
    real_images = []
    for batch in val_loader:
        descriptions.extend(batch["description"])
        real_images.extend([img.cpu() for img in batch["image"]])
        # if len(descriptions) >= 4: break # 仅测试少量
    
    generated_images = generate_images(model, vae, text_encoder, tokenizer, descriptions)
    
    # 计算FID分数
    fid_calculator = FIDCalculator()
    calculate_and_print_fid(real_images, generated_images, fid_calculator)
    
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
