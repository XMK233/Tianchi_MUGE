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
_save_dir = "/mnt/d/forCoding_data/Tianchi_MUGE/plan_4_1/trained_models"
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
TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/hanfu_train.img.tsv"
TEXT_TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/hanfu_train.text.tsv"
VAL_TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/hanfu_val.img.tsv"
VAL_TEXT_TSV_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/hanfu_val.text.tsv"

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

class TransfusionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, cond):
        # Self Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Cross Attention
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(x_norm, cond, cond)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x

class TransfusionBackbone(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=1, dim=512, depth=8, heads=8, 
                 num_classes=1, text_emb_size=768, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        
        self.class_emb = nn.Embedding(num_classes, dim)
        self.time_emb = nn.Embedding(1000, dim)
        self.text_proj = nn.Linear(text_emb_size, dim)
        
        self.blocks = nn.ModuleList([
            TransfusionBlock(dim, heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * in_ch)
        
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, class_labels, text_emb):
        B, C, H, W = x.shape
        
        # Patchify
        # x: [B, 1, 256, 256] -> [B, dim, 16, 16] -> [B, dim, 256] -> [B, 256, dim]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Conditions
        if t.dim() == 0:
            t = t.view(1).expand(B)
        t = t.long().clamp(0, 999)
        t_emb = self.time_emb(t).unsqueeze(1) # [B, 1, dim]
        
        c_emb = self.class_emb(class_labels).unsqueeze(1) # [B, 1, dim]
        
        txt_emb = self.text_proj(text_emb).unsqueeze(1) # [B, 1, dim]
        
        # Concatenate conditions: [B, 3, dim]
        cond = torch.cat([t_emb, c_emb, txt_emb], dim=1)
        
        for blk in self.blocks:
            x = blk(x, cond)
            
        x = self.norm(x)
        x = self.head(x)
        
        # Unpatchify
        # x: [B, N, patch_dim] -> [B, C, H, W]
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        x = x.view(B, h_p, w_p, self.patch_size, self.patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x


def load_vlm(path, device):
    base_path = path
    config_path = os.path.join(base_path, "config.json")
    if not os.path.exists(config_path):
        snapshots_dir = os.path.join(base_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for name in os.listdir(snapshots_dir):
                cand = os.path.join(snapshots_dir, name)
                if os.path.exists(os.path.join(cand, "config.json")):
                    base_path = cand
                    break
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(base_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return tokenizer, model


def encode_text_batch(text_list, tokenizer, model, device, grad=False):
    inputs = tokenizer(list(text_list), return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if grad:
        outputs = model(**inputs)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states"):
        hidden = outputs.hidden_states[-1]
    else:
        raise RuntimeError("Qwen 模型输出中找不到 last_hidden_state/hidden_states")
    text_emb = hidden[:, -1, :].to(torch.float32)
    norm = text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    text_emb = text_emb / norm
    return text_emb


local_model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')

tokenizer, base_text_model = load_vlm(local_model_path, device)
# 开启梯度检查点，显著降低显存占用（以计算换显存）
if hasattr(base_text_model, "gradient_checkpointing_enable"):
    base_text_model.gradient_checkpointing_enable()
    print("已开启 text_encoder 梯度检查点 (Gradient Checkpointing)")

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", 
        # "v_proj", "o_proj"
        ],
)
text_encoder = get_peft_model(base_text_model, lora_config)
text_encoder.to(device)
text_encoder.train()

hidden_size = getattr(getattr(text_encoder, "config", None), "hidden_size", None)
if hidden_size is None or hidden_size <= 0:
    dummy_emb = encode_text_batch(["占位符"], tokenizer, text_encoder, device, grad=False)
    hidden_size = dummy_emb.shape[-1]

print(f"使用 Qwen 文本编码器并应用 LoRA，隐藏维度为 {hidden_size}")

# ddpm_model_path = "/mnt/d/HuggingFaceModels/models--google--ddpm-cifar10-32"

net = TransfusionBackbone(text_emb_size=hidden_size).to(device)
loss_fn = nn.MSELoss()

# 优化器包含 net 和 text_encoder 的可训练参数
params_to_optimize = list(net.parameters()) + [p for p in text_encoder.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params_to_optimize, lr=1e-4, weight_decay=1e-4)

if device == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

net_ema = TransfusionBackbone(text_emb_size=hidden_size).to(device)
# 根据 train_mode 决定是否加载已存在的模型
if _args.train_mode == 'resume' and os.path.exists(_save_path):
    try:
        ckpt = torch.load(_save_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                try:
                    res = net.load_state_dict(ckpt['model_state_dict'], strict=False)
                    if 'ema_state_dict' in ckpt:
                        net_ema.load_state_dict(ckpt['ema_state_dict'], strict=False)
                    else:
                        net_ema.load_state_dict(net.state_dict())
                    
                    # 加载 text_encoder
                    if 'text_encoder_state_dict' in ckpt:
                        text_encoder.load_state_dict(ckpt['text_encoder_state_dict'], strict=False)
                        print("已加载 text_encoder 参数")
                    else:
                        print("未找到 text_encoder 参数，使用预训练默认参数")
                    
                    if scaler is not None and 'scaler_state_dict' in ckpt:
                        try:
                            scaler.load_state_dict(ckpt['scaler_state_dict'])
                            print("已加载 scaler 状态")
                        except Exception as e:
                            print(f"加载 scaler 状态失败，将使用新的 scaler: {e}")

                    # 检查加载的参数是否包含 NaN/Inf
                    for name, param in net.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            raise RuntimeError(f"加载的参数 {name} 包含 NaN 或 Inf，模型已损坏")
                    
                    if 'epoch' in ckpt:
                        start_epoch = int(ckpt['epoch'])
                        start_step = int(ckpt.get('step', 0))
                        if 'loss' in ckpt:
                            print(f"断点位置 epoch={start_epoch}, step={start_step}, loss={ckpt['loss']:.6f}")
                        else:
                            print(f"断点位置 epoch={start_epoch}, step={start_step}")
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
    except (RuntimeError, EOFError, OSError) as e:
        print(f"模型文件可能已损坏，无法加载: {e}")
        print("将从头开始训练，并会覆盖损坏的文件。")
        start_epoch = 0
        net_ema.load_state_dict(net.state_dict())
else:
    net_ema.load_state_dict(net.state_dict())
    if _args.train_mode == 'fresh':
        print("检测到 train_mode=fresh，忽略已有模型并从头开始训练。")
        start_epoch = 0
        start_step = 0
ema_decay = 0.999

losses = []
avg_loss = None
torch.cuda.empty_cache() # 训练开始前清理显存
for epoch in range(start_epoch, n_epochs):
    if epoch == start_epoch:
        step_offset = start_step
    else:
        step_offset = 0
        
    for step, (x, text) in enumerate(tqdm(train_dataloader)):
        if step < step_offset:
            continue
        x = x.to(device) * 2 - 1
        
        if device == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                text_emb = encode_text_batch(text, tokenizer, text_encoder, device, grad=True)
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                dummy_y = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                pred = net(noisy_x, timesteps, dummy_y, text_emb)
                velocity = noise_scheduler.get_velocity(x, noise, timesteps)
                alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
                snr = alphas / (1 - alphas)
                weight = torch.sqrt(torch.clamp(snr, max=5.0))
                loss = (weight * (pred - velocity) ** 2).mean()
        else:
            text_emb = encode_text_batch(text, tokenizer, text_encoder, device, grad=True)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            dummy_y = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            pred = net(noisy_x, timesteps, dummy_y, text_emb)
            velocity = noise_scheduler.get_velocity(x, noise, timesteps)
            alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
            snr = alphas / (1 - alphas)
            weight = torch.sqrt(torch.clamp(snr, max=5.0))
            loss = (weight * (pred - velocity) ** 2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at step {step}, skipping update.")
            opt.zero_grad()
            continue

        opt.zero_grad()
        if device == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        
        losses.append(loss.item())
    if len(losses) >= 100:
        avg_loss = sum(losses[-100:]) / 100
        print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
    
    # 每20个epoch保存一次模型，覆盖原文件
    # _tmp_save_path = _save_path + ".tmp"
    if (epoch + 1) % 20 == 0:
        try:
            if avg_loss is not None:
                save_loss = avg_loss
            elif losses:
                save_loss = sum(losses) / len(losses)
            else:
                save_loss = 0.0
            save_dict = {
                "model_state_dict": net.state_dict(),
                "ema_state_dict": net_ema.state_dict(),
                "text_encoder_state_dict": text_encoder.state_dict(),
                "epoch": epoch,
                "step": step, # 记录 epoch 结束时的 step
                "loss": save_loss
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, _save_path)
            # os.replace(_tmp_save_path, _save_path)
        except Exception as e:
            print(f"保存模型失败: {e}")

    # if (epoch + 1) % 5 == 0:
    if False:

        xg = torch.randn(3 * 2, 1, img_sz[0], img_sz[1]).to(device)
        yg = torch.zeros(xg.shape[0], dtype=torch.long).to(device)
        
        # 准备采样文本
        sample_texts = []
        for i in range(xg.shape[0]):
            sample_texts.append(dataset.samples[i % len(dataset)][2])
        text_encoder.eval()
        with torch.no_grad():
            text_emb_sample = encode_text_batch(sample_texts, tokenizer, text_encoder, device, grad=False)
        text_encoder.train()

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
        out_name = os.path.join(os.path.dirname(__file__), f'samples_bw_epoch{epoch+1}.png')
        torchvision.utils.save_image(
            xg.detach().cpu().clip(-1, 1),
            out_name,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
plt.plot(losses)



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


VAL_IMG_PATH = VAL_TSV_PATH # "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_val.img.cls1.tsv"
VAL_TEXT_PATH = VAL_TEXT_TSV_PATH # "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
text_map = {}
with open(VAL_TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            text_map[parts[0]] = parts[1]
out_dir = os.path.join(os.path.dirname(__file__), _base)
os.makedirs(out_dir, exist_ok=True)
max_count = 400
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
