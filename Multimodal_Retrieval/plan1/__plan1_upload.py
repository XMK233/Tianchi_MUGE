import os
import sys
import torch
from transformers import BertTokenizer, BertModel
import timm
from PIL import Image
import numpy as np
from torchvision import transforms

# 设置 Hugging Face 和 timm 的缓存目录
cache_dir = "/mnt/d/HuggingFaceModels/"
os.environ['TORCH_HOME'] = cache_dir


# 禁用 transformers 冗余日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"


# 将 data_loader.py 的路径添加到系统路径中
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader import DataLoader

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化 DataLoader
loader = DataLoader()

# 加载少量验证集数据作为示例
valid_queries_df = loader.load_queries(split='valid')
sample_queries = valid_queries_df.head(3)  # 取前3个查询作为示例

# 加载对应的图片
all_item_ids = []
for ids in sample_queries['item_ids']:
    all_item_ids.extend(ids)

# 使用 create_img_id_to_image_dict 加载特定ID的图片
# 注意：这里为了演示，我们加载所有验证集图片，然后筛选。在实际应用中，应优化为按需加载。
valid_images_dict = loader.create_img_id_to_image_dict(split='valid') # 限制样本量以节省时间

sample_images = []
sample_image_ids = []
for img_id in all_item_ids:
    if str(img_id) in valid_images_dict:
        sample_images.append(valid_images_dict[str(img_id)])
        sample_image_ids.append(str(img_id))

print(f"加载了 {len(sample_queries)} 条查询和 {len(sample_images)} 张图片作为示例。")

from transformers import CLIPProcessor, CLIPModel
class CLIPFeatureExtractor:
    def __init__(self, model_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese", device=None, cache_dir=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Model {model_name} loaded on {self.device}")

    def extract_text_features(self, text_list):
        # 确保文本列表不为空且所有元素都是字符串
        if not text_list or not all(isinstance(text, str) for text in text_list):
            raise ValueError("文本列表必须包含非空字符串")
        
        # 处理文本
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 使用模型的标准方法提取文本特征
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def extract_image_features(self, image_list):
        # 确保图像列表不为空且所有元素都有效
        if not image_list or not all(img is not None for img in image_list):
            raise ValueError("图像列表必须包含有效图像")
        
        # 处理图像
        inputs = self.processor(images=image_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 使用模型的标准方法提取图像特征
            image_features = self.model.get_image_features(**inputs)
        return image_features

# 初始化特征提取器
feature_extractor = CLIPFeatureExtractor(cache_dir=cache_dir)

# 提取文本特征
if not sample_queries.empty:
    sample_texts = sample_queries['query_text'].tolist()
    text_features = feature_extractor.extract_text_features(sample_texts)
    print(f"提取的文本特征维度: {text_features.shape}")

# 提取图像特征
if sample_images:
    # 确保所有图像都已正确加载
    valid_sample_images = [img for img in sample_images if img is not None]
    if valid_sample_images:
        image_features = feature_extractor.extract_image_features(valid_sample_images)
        print(f"提取的图像特征维度: {image_features.shape}")
    else:
        print("没有有效的图像可供处理。")

