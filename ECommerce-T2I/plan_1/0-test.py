import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

# ----------------------------
# 1. 加载预训练 VAE（自动从 Hugging Face 下载）
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained(
    "/mnt/d/ModelScopeModels/stabilityai/sd-vae-ft-mse/",
    torch_dtype=torch.float32  # 也可用 float16 节省显存
).to(device)

# ----------------------------
# 2. 生成随机 latent（符合 VAE 的 latent 分布）
# ----------------------------
# SD-VAE 的 latent 尺寸：channels=4, height=32, width=32 → 对应 256x256 图像
# （因为压缩率是 8: 256 / 8 = 32）
batch_size = 1
latent_channels = 4
latent_size = 32  # for 256x256 output

# 从标准正态分布采样（VAE 的 latent 空间近似高斯）
rand_latent = torch.randn(
    batch_size, latent_channels, latent_size, latent_size
).to(device)

# ----------------------------
# 3. 解码 latent → 图像
# ----------------------------
with torch.no_grad():
    decoded = vae.decode(rand_latent).sample  # [1, 3, 256, 256]

# 反归一化：[-1, 1] → [0, 1]
decoded = (decoded / 2 + 0.5).clamp(0, 1)
image_tensor = decoded[0]  # [3, 256, 256]

# 转为 PIL 图像
to_pil = T.ToPILImage()
pil_image = to_pil(image_tensor.cpu())

# ----------------------------
# 4. 保存或显示
# ----------------------------
pil_image.save("random_vae_image.png")
print("✅ 随机图像已保存为 'random_vae_image.png'")

# 可选：显示图像（在支持 GUI 的环境中）
# pil_image.show()