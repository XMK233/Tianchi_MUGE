local_model_path = "/mnt/d/HuggingFaceModels/models--Qwen--Qwen2.5-VL-3B-Instruct"
text = "华之屋恒香芝士味椰子味酥皮注心蛋卷办公室休闲零食小吃罐装年货"

import torch
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os


class SimpleTextToImageGenerator(nn.Module):
    def __init__(self, emb_dim, base_channels=64):
        super().__init__()
        self.fc = nn.Linear(emb_dim, base_channels * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(base_channels // 8, 1, 4, 2, 1)
        self.act = nn.GELU()

    def forward(self, emb):
        x = self.fc(emb)
        x = x.view(-1, 64, 16, 16)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
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
    model.eval()
    return tokenizer, model


def encode_text(text_str, tokenizer, model, device):
    inputs = tokenizer(text_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states"):
        hidden = outputs.hidden_states[-1]
    else:
        raise RuntimeError("Qwen 模型输出中找不到 last_hidden_state/hidden_states")
    text_emb = hidden[:, -1, :].to(torch.float32)
    return text_emb


def generate_image_from_text(text_str, save_path="vlm_t2i_out.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_vlm(local_model_path, device)
    text_emb = encode_text(text_str, tokenizer, model, device)
    emb_dim = text_emb.shape[-1]
    generator = SimpleTextToImageGenerator(emb_dim=emb_dim).to(device)
    with torch.no_grad():
        img_tensor = generator(text_emb.to(device))[0, 0]
    img_tensor = img_tensor.clamp(-1, 1)
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = img_tensor.unsqueeze(0)
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor)
    img = img.resize((256, 256), Image.BILINEAR)
    img = img.convert("L")
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    img.save(save_path)
    return save_path


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "vlm_t2i_256x256_gray.png")
    generate_image_from_text(text, save_path=output_path)
