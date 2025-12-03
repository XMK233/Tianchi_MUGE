import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

# 设置环境变量
cache_dir = "/mnt/d/HuggingFaceModels/"
os.environ['TORCH_HOME'] = cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# 特征提取器适配器，将特征提取器适配为检索模型使用的格式
class FeatureExtractorAdapter:
    """
    特征提取器适配器，使特征提取器兼容检索模型接口
    """
    def __init__(self, extractor, feature_type='text'):
        """
        初始化特征提取器适配器
        
        Args:
            extractor: 特征提取器实例
            feature_type: 特征类型，'text'或'image'
        """
        self.extractor = extractor
        self.feature_type = feature_type
    
    def extract_features(self, inputs):
        """
        提取特征的统一接口
        
        Args:
            inputs: 文本列表或图像列表
            
        Returns:
            特征张量
        """
        if self.feature_type == 'text':
            return self.extractor.extract_text_features(inputs)
        elif self.feature_type == 'image':
            return self.extractor.extract_image_features(inputs)
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")

# 从step_2_1中导入特征提取器（如果需要单独运行，我们也会定义默认的提取器）
try:
    # 假设在同一目录下有特征提取器的实现
    from step_2_1_特征提取模块设计_普通模型 import TextFeatureExtractor, ImageFeatureExtractor
except ImportError:
    # 如果导入失败，我们在这里定义完整的特征提取器
    from transformers import BertTokenizer, BertModel
    import timm
    from torchvision import transforms
    
    class TextFeatureExtractor:
        """
        文本特征提取器，使用BERT模型提取文本特征
        """
        def __init__(self, model_name='bert-base-chinese', device='cpu', cache_dir=None):
            """
            初始化文本特征提取器
            
            Args:
                model_name: 预训练模型名称
                device: 运行设备
                cache_dir: 模型缓存目录
            """
            self.device = device
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir,
                local_files_only=False
            )
            self.model = BertModel.from_pretrained(
                model_name, cache_dir=cache_dir,
                local_files_only=False
            ).to(device)
            self.model.eval()
            print(f"文本模型 {model_name} 加载完成。")
        
        def extract_text_features(self, texts):
            """
            提取文本特征
            
            Args:
                texts: 文本列表
                
            Returns:
                文本特征张量
            """
            # 输入验证
            if not texts:
                return torch.tensor([], dtype=torch.float32, device=self.device)
            
            # 处理单个文本的情况
            if isinstance(texts, str):
                texts = [texts]
            
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]
        
        # 兼容接口
        def extract_features(self, texts):
            return self.extract_text_features(texts)
    
    class ImageFeatureExtractor:
        """
        图像特征提取器，使用ResNet模型提取图像特征
        """
        def __init__(self, model_name='resnet50', device='cpu', cache_dir=None):
            """
            初始化图像特征提取器
            
            Args:
                model_name: 预训练模型名称
                device: 运行设备
                cache_dir: 模型缓存目录
            """
            self.device = device
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes=0,
                cache_dir=cache_dir
            ).to(device)
            self.model.eval()
            print(f"图像模型 {model_name} 加载完成。")

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        def extract_image_features(self, images):
            """
            提取图像特征
            
            Args:
                images: PIL图像列表
                
            Returns:
                图像特征张量
            """
            # 输入验证
            if not images:
                return torch.tensor([], dtype=torch.float32, device=self.device)
            
            # 处理单个图像的情况
            if not isinstance(images, list):
                images = [images]
            
            # 确保输入是PIL图像或可以转换为PIL图像的对象
            image_tensors = []
            for img in images:
                # 确保是PIL图像
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img) if isinstance(img, np.ndarray) else Image.open(img)
                img = img.convert('RGB')  # 确保是RGB格式
                image_tensors.append(self.transform(img))
            
            image_tensors = torch.stack(image_tensors).to(self.device)
            with torch.no_grad():
                features = self.model(image_tensors)
            return features
        
        # 兼容接口
        def extract_features(self, images):
            return self.extract_image_features(images)

# 1. 特征融合策略
class FeatureFusion:
    """
    实现多种跨模态特征融合策略
    """
    def __init__(self, fusion_method='projection', projection_dim=512, device=None):
        """
        初始化特征融合器
        
        Args:
            fusion_method: 融合方法，支持 'projection'(投影到共享空间)、'concatenation'(拼接)
            projection_dim: 共享投影空间的维度
            device: 运行设备
        """
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化投影层（如果需要）
        if fusion_method == 'projection':
            # 假设文本特征维度为768（BERT输出），图像特征维度为2048（ResNet50输出）
            self.text_projector = torch.nn.Linear(768, projection_dim).to(self.device)
            self.image_projector = torch.nn.Linear(2048, projection_dim).to(self.device)
    
    def fuse_text_features(self, text_features):
        """
        处理文本特征
        
        Args:
            text_features: 文本特征张量 [batch_size, text_feature_dim]
            
        Returns:
            融合后的文本特征
        """
        if self.fusion_method == 'projection':
            # 投影到共享空间
            return self.text_projector(text_features)
        else:
            # 直接返回原始特征
            return text_features
    
    def fuse_image_features(self, image_features):
        """
        处理图像特征
        
        Args:
            image_features: 图像特征张量 [batch_size, image_feature_dim]
            
        Returns:
            融合后的图像特征
        """
        if self.fusion_method == 'projection':
            # 投影到共享空间
            return self.image_projector(image_features)
        else:
            # 直接返回原始特征
            return image_features

# 2. 相似度计算
class SimilarityCalculator:
    """
    计算文本特征和图像特征之间的相似度
    """
    def __init__(self, similarity_type='cosine'):
        """
        初始化相似度计算器
        
        Args:
            similarity_type: 相似度类型，支持 'cosine'(余弦相似度)、'dot'(点积)、'euclidean'(欧几里得距离)、'manhattan'(曼哈顿距离)
        """
        self.similarity_type = similarity_type
    
    def normalize_features(self, features):
        """
        对特征进行L2归一化
        
        Args:
            features: 特征张量
            
        Returns:
            归一化后的特征
        """
        return torch.nn.functional.normalize(features, p=2, dim=1)
    
    def calculate_similarity(self, text_features, image_features):
        """
        计算文本特征和图像特征之间的相似度矩阵
        
        Args:
            text_features: 文本特征张量 [text_batch_size, feature_dim]
            image_features: 图像特征张量 [image_batch_size, feature_dim]
            
        Returns:
            相似度矩阵 [text_batch_size, image_batch_size]
            注意：对于距离度量（如欧几里得、曼哈顿），返回的是负距离，以便与相似度排序逻辑保持一致
        """
        if self.similarity_type == 'cosine':
            # 余弦相似度 = 归一化后的点积
            text_features_normalized = self.normalize_features(text_features)
            image_features_normalized = self.normalize_features(image_features)
            return torch.mm(text_features_normalized, image_features_normalized.t())
        elif self.similarity_type == 'dot':
            # 直接计算点积
            return torch.mm(text_features, image_features.t())
        elif self.similarity_type == 'euclidean':
            # 计算欧几里得距离（返回负距离，以便与相似度排序逻辑保持一致）
            # 使用广播计算每个文本特征与所有图像特征的距离
            text_batch_size = text_features.size(0)
            image_batch_size = image_features.size(0)
            
            # 扩展维度以便广播
            text_features_expanded = text_features.unsqueeze(1).expand(text_batch_size, image_batch_size, -1)
            image_features_expanded = image_features.unsqueeze(0).expand(text_batch_size, image_batch_size, -1)
            
            # 计算欧几里得距离
            distances = torch.sqrt(torch.sum((text_features_expanded - image_features_expanded) ** 2, dim=2))
            
            # 返回负距离，因为我们希望相似度高的排在前面
            return -distances
        elif self.similarity_type == 'manhattan':
            # 计算曼哈顿距离（返回负距离，以便与相似度排序逻辑保持一致）
            text_batch_size = text_features.size(0)
            image_batch_size = image_features.size(0)
            
            # 扩展维度以便广播
            text_features_expanded = text_features.unsqueeze(1).expand(text_batch_size, image_batch_size, -1)
            image_features_expanded = image_features.unsqueeze(0).expand(text_batch_size, image_batch_size, -1)
            
            # 计算曼哈顿距离
            distances = torch.sum(torch.abs(text_features_expanded - image_features_expanded), dim=2)
            
            # 返回负距离，因为我们希望相似度高的排在前面
            return -distances
        else:
            raise ValueError(f"不支持的相似度类型: {self.similarity_type}")

# 3. 检索模型
class CrossModalRetrievalModel:
    """
    跨模态检索模型，整合特征提取、融合和相似度计算，将文本和图像映射到同一嵌入空间
    """
    def __init__(self, text_extractor, image_extractor, fusion_method='projection', 
                 projection_dim=512, similarity_type='cosine', normalize_features=True, device=None):
        """
        初始化检索模型
        
        Args:
            text_extractor: 文本特征提取器
            image_extractor: 图像特征提取器
            fusion_method: 特征融合方法
            projection_dim: 投影空间维度
            similarity_type: 相似度计算类型
            normalize_features: 是否在融合后对特征进行标准化
            device: 运行设备
        """
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        self.fusion = FeatureFusion(fusion_method, projection_dim, device)
        self.similarity_calculator = SimilarityCalculator(similarity_type)
        self.normalize_features = normalize_features
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 记录特征维度信息
        self.text_feature_dim = None
        self.image_feature_dim = None
        self.common_space_dim = projection_dim if fusion_method == 'projection' else None
    
    def _normalize(self, features):
        """
        对特征进行标准化
        
        Args:
            features: 特征张量
            
        Returns:
            标准化后的特征
        """
        if self.normalize_features:
            return torch.nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def extract_and_fuse_text_features(self, texts, record_dim=True):
        """
        提取并融合文本特征
        
        Args:
            texts: 文本列表
            record_dim: 是否记录特征维度
            
        Returns:
            融合后的文本特征
        """
        # 提取文本特征
        text_features = self.text_extractor.extract_features(texts)
        
        # 记录特征维度（首次调用时）
        if record_dim and self.text_feature_dim is None:
            self.text_feature_dim = text_features.shape[1]
        
        # 融合特征
        fused_features = self.fusion.fuse_text_features(text_features)
        
        # 标准化（如果需要）
        return self._normalize(fused_features)
    
    def extract_and_fuse_image_features(self, images, record_dim=True):
        """
        提取并融合图像特征
        
        Args:
            images: 图像列表
            record_dim: 是否记录特征维度
            
        Returns:
            融合后的图像特征
        """
        # 提取图像特征
        image_features = self.image_extractor.extract_features(images)
        
        # 记录特征维度（首次调用时）
        if record_dim and self.image_feature_dim is None:
            self.image_feature_dim = image_features.shape[1]
        
        # 融合特征
        fused_features = self.fusion.fuse_image_features(image_features)
        
        # 标准化（如果需要）
        return self._normalize(fused_features)
    
    def retrieve_images(self, query_texts, image_features_dict, top_k=10, return_scores=False):
        """
        根据查询文本检索相关图像
        
        Args:
            query_texts: 查询文本列表
            image_features_dict: 图像ID到特征的映射字典
            top_k: 返回前k个结果
            return_scores: 是否返回相似度分数
            
        Returns:
            如果return_scores=True: 每个查询对应的检索结果列表 [(query_idx, [(image_id, similarity), ...]), ...]
            否则: 每个查询对应的图像ID列表 [[image_id1, image_id2, ...], ...]
        """
        # 获取所有图像ID和特征
        image_ids = list(image_features_dict.keys())
        all_image_features = torch.stack([image_features_dict[img_id] for img_id in image_ids])
        
        # 提取并融合查询文本特征
        text_features = self.extract_and_fuse_text_features(query_texts)
        
        # 计算相似度
        similarities = self.similarity_calculator.calculate_similarity(text_features, all_image_features)
        
        # 获取Top-k结果
        results = []
        for i, query_sims in enumerate(similarities):
            # 按相似度降序排序
            top_indices = torch.topk(query_sims, min(top_k, len(image_ids))).indices
            
            if return_scores:
                query_results = [(image_ids[idx], query_sims[idx].item()) for idx in top_indices]
                results.append((i, query_results))
            else:
                query_results = [image_ids[idx] for idx in top_indices]
                results.append(query_results)
        
        return results
    
    def batch_retrieve(self, query_texts, image_features_dict, top_k=10, batch_size=32):
        """
        批量检索图像，处理大量查询时更高效
        
        Args:
            query_texts: 查询文本列表
            image_features_dict: 图像ID到特征的映射字典
            top_k: 返回前k个结果
            batch_size: 查询批处理大小
            
        Returns:
            每个查询对应的图像ID列表 [[image_id1, image_id2, ...], ...]
        """
        # 获取所有图像ID和特征
        image_ids = list(image_features_dict.keys())
        all_image_features = torch.stack([image_features_dict[img_id] for img_id in image_ids])
        
        results = []
        # 批量处理查询
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i+batch_size]
            
            # 提取并融合查询文本特征
            text_features = self.extract_and_fuse_text_features(batch_texts)
            
            # 计算相似度
            similarities = self.similarity_calculator.calculate_similarity(text_features, all_image_features)
            
            # 获取Top-k结果
            for query_sims in similarities:
                # 按相似度降序排序
                top_indices = torch.topk(query_sims, min(top_k, len(image_ids))).indices
                query_results = [image_ids[idx] for idx in top_indices]
                results.append(query_results)
        
        return results
    
    def build_image_index(self, images_dict, batch_size=32, show_progress=False):
        """
        为图像集合构建特征索引
        
        Args:
            images_dict: 图像ID到PIL图像的映射字典
            batch_size: 批处理大小
            show_progress: 是否显示处理进度
            
        Returns:
            图像ID到特征的映射字典
        """
        image_features_dict = {}
        image_ids = list(images_dict.keys())
        total_batches = (len(image_ids) + batch_size - 1) // batch_size
        
        # 批量处理图像以节省内存
        for i in range(0, len(image_ids), batch_size):
            if show_progress:
                print(f"处理批次 {i//batch_size + 1}/{total_batches}")
            
            batch_ids = image_ids[i:i+batch_size]
            batch_images = [images_dict[img_id] for img_id in batch_ids]
            
            # 提取并融合图像特征
            batch_features = self.extract_and_fuse_image_features(batch_images)
            
            # 存储特征
            for j, img_id in enumerate(batch_ids):
                image_features_dict[img_id] = batch_features[j]
        
        return image_features_dict
    
    def get_feature_spaces_info(self):
        """
        获取特征空间信息
        
        Returns:
            包含特征维度信息的字典
        """
        return {
            "text_feature_dim": self.text_feature_dim,
            "image_feature_dim": self.image_feature_dim,
            "common_space_dim": self.common_space_dim,
            "fusion_method": self.fusion.fusion_method,
            "normalize_features": self.normalize_features
        }

# 4. 评估函数
def evaluate_retrieval(model, queries_df, images_dict, top_k_list=[1, 5, 10]):
    """
    评估检索模型性能
    
    Args:
        model: 检索模型
        queries_df: 包含查询文本和对应商品ID的DataFrame
        images_dict: 图像ID到PIL图像的映射字典
        top_k_list: 评估的K值列表
        
    Returns:
        各K值对应的Recall指标字典
    """
    # 构建图像特征索引
    print("正在构建图像特征索引...")
    image_features_dict = model.build_image_index(images_dict)
    
    # 准备查询文本
    query_texts = queries_df['query_text'].tolist()
    
    # 获取最大的top_k
    max_top_k = max(top_k_list)
    
    # 执行检索
    print(f"正在执行检索，获取Top-{max_top_k}结果...")
    retrieval_results = model.retrieve_images(query_texts, image_features_dict, top_k=max_top_k)
    
    # 计算Recall指标
    recalls = {f'Recall@{k}': 0.0 for k in top_k_list}
    total_queries = len(queries_df)
    
    for i, (query_idx, top_results) in enumerate(retrieval_results):
        # 获取当前查询的真实商品ID
        ground_truth_ids = [str(id) for id in queries_df.iloc[query_idx]['item_ids']]
        # 获取检索结果中的图像ID
        retrieved_ids = [str(result[0]) for result in top_results]
        
        # 对每个top_k计算Recall
        for k in top_k_list:
            top_retrieved = retrieved_ids[:k]
            # 检查前k个结果中是否包含至少一个真实ID
            if any(gt_id in top_retrieved for gt_id in ground_truth_ids):
                recalls[f'Recall@{k}'] += 1.0
    
    # 计算最终Recall值
    for k in top_k_list:
        recalls[f'Recall@{k}'] /= total_queries
    
    # 计算MeanRecall
    recalls['MeanRecall'] = sum(recalls.values()) / (len(recalls) - 1)  # 减去MeanRecall本身
    
    return recalls

def evaluate_retrieval_results(retrieval_results, ground_truth, k_values=[1, 5, 10]):
    """
    评估检索结果
    
    Args:
        retrieval_results: 检索结果列表，每个元素是一个检索到的ID列表
        ground_truth: 真实匹配关系，字典格式 {查询ID: [相关图像ID列表]}
        k_values: 评估不同的top-k值
        
    Returns:
        包含不同评估指标的字典
    """
    metrics = {}
    
    for k in k_values:
        precision_scores = []
        recall_scores = []
        map_scores = []  # 平均准确率
        
        for query_idx, retrieved_ids in enumerate(retrieval_results):
            retrieved_ids_k = retrieved_ids[:k]
            relevant_ids = ground_truth.get(query_idx, [])
            
            if not relevant_ids:
                continue
            
            # 计算精确率
            relevant_retrieved = len(set(retrieved_ids_k) & set(relevant_ids))
            precision = relevant_retrieved / k if k > 0 else 0
            precision_scores.append(precision)
            
            # 计算召回率
            recall = relevant_retrieved / len(relevant_ids) if len(relevant_ids) > 0 else 0
            recall_scores.append(recall)
            
            # 计算平均准确率（MAP@k）
            ap = 0
            hits = 0
            for i, retrieved_id in enumerate(retrieved_ids_k):
                if retrieved_id in relevant_ids:
                    hits += 1
                    ap += hits / (i + 1)
            ap = ap / min(len(relevant_ids), k) if relevant_ids else 0
            map_scores.append(ap)
        
        # 计算平均值
        metrics[f'precision@{k}'] = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        metrics[f'recall@{k}'] = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        metrics[f'map@{k}'] = sum(map_scores) / len(map_scores) if map_scores else 0
    
    return metrics

def create_test_data(num_texts=10, num_images=50, num_relevant=3):
    """
    创建测试数据
    
    Args:
        num_texts: 测试文本数量
        num_images: 测试图像数量
        num_relevant: 每个文本相关的图像数量
        
    Returns:
        测试文本列表、图像字典和真实匹配关系
    """
    # 创建测试文本
    test_texts = [f"测试文本 {i}" for i in range(num_texts)]
    
    # 创建模拟图像（使用PIL创建空白图像）
    test_images = {}
    for i in range(num_images):
        # 创建一个简单的PIL图像对象
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        test_images[f"image_{i}"] = img
    
    # 创建真实匹配关系
    ground_truth = {}
    for i in range(num_texts):
        # 为每个文本随机选择相关图像
        relevant_indices = np.random.choice(range(num_images), size=num_relevant, replace=False)
        ground_truth[i] = [f"image_{idx}" for idx in relevant_indices]
    
    return test_texts, test_images, ground_truth

def create_retrieval_model(fusion_method='projection', projection_dim=512, 
                          similarity_type='cosine', normalize_features=True, device=None):
    """
    创建检索模型的工厂函数
    
    Args:
        fusion_method: 特征融合方法
        projection_dim: 投影空间维度
        similarity_type: 相似度计算类型
        normalize_features: 是否标准化特征
        device: 运行设备
        
    Returns:
        初始化的检索模型
    """
    # 使用自定义的特征提取器
    text_extractor = TextFeatureExtractor(device=device, cache_dir=cache_dir)
    image_extractor = ImageFeatureExtractor(device=device, cache_dir=cache_dir)
    
    # 创建检索模型
    model = CrossModalRetrievalModel(
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        fusion_method=fusion_method,
        projection_dim=projection_dim,
        similarity_type=similarity_type,
        normalize_features=normalize_features,
        device=device
    )
    
    return model

def test_feature_extraction(model, test_texts, test_images):
    """
    测试特征提取功能
    
    Args:
        model: 检索模型
        test_texts: 测试文本列表
        test_images: 测试图像字典
    """
    print("\n=== 测试特征提取 ===")
    
    # 提取文本特征
    text_features = model.extract_and_fuse_text_features(test_texts[:3])
    print(f"文本特征形状: {text_features.shape}")
    
    # 提取图像特征
    sample_images = list(test_images.values())[:3]
    image_features = model.extract_and_fuse_image_features(sample_images)
    print(f"图像特征形状: {image_features.shape}")
    
    # 检查特征空间信息
    space_info = model.get_feature_spaces_info()
    print(f"特征空间信息: {space_info}")

def test_similarity_calculation(model, test_texts, test_images):
    """
    测试相似度计算功能
    
    Args:
        model: 检索模型
        test_texts: 测试文本列表
        test_images: 测试图像字典
    """
    print("\n=== 测试相似度计算 ===")
    
    # 提取特征
    text_features = model.extract_and_fuse_text_features(test_texts[:5])
    sample_images = list(test_images.values())[:5]
    image_features = model.extract_and_fuse_image_features(sample_images)
    
    # 计算相似度
    similarities = model.similarity_calculator.calculate_similarity(text_features, image_features)
    print(f"相似度矩阵形状: {similarities.shape}")
    print(f"相似度矩阵示例:\n{similarities[:3, :3]}")

def test_retrieval_functionality(model, test_texts, test_images, ground_truth):
    """
    测试检索功能
    
    Args:
        model: 检索模型
        test_texts: 测试文本列表
        test_images: 测试图像字典
        ground_truth: 真实匹配关系
    """
    print("\n=== 测试检索功能 ===")
    
    # 构建图像索引
    print("构建图像索引...")
    image_index = model.build_image_index(test_images, batch_size=16, show_progress=True)
    print(f"图像索引包含 {len(image_index)} 个图像特征")
    
    # 进行检索
    print("执行检索...")
    retrieval_results = model.retrieve_images(test_texts[:5], image_index, top_k=10)
    print(f"检索结果数量: {len(retrieval_results)}")
    
    # 显示一些检索结果
    for i, results in enumerate(retrieval_results[:3]):
        print(f"查询 {i} 的前5个结果: {results[:5]}")
    
    # 评估检索性能
    print("\n=== 评估检索性能 ===")
    metrics = evaluate_retrieval_results(retrieval_results, ground_truth)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

def compare_different_methods(test_texts, test_images, ground_truth, device=None):
    """
    比较不同的融合方法和相似度计算方法
    
    Args:
        test_texts: 测试文本列表
        test_images: 测试图像字典
        ground_truth: 真实匹配关系
        device: 运行设备
    """
    print("\n=== 比较不同方法 ===")
    
    # 测试不同的融合方法
    fusion_methods = ['projection', 'concatenation']
    similarity_types = ['cosine', 'dot']
    
    comparison_results = []
    
    for fusion_method in fusion_methods:
        for similarity_type in similarity_types:
            print(f"\n测试融合方法: {fusion_method}, 相似度类型: {similarity_type}")
            
            # 创建模型
            model = create_retrieval_model(
                fusion_method=fusion_method,
                similarity_type=similarity_type,
                device=device
            )
            
            # 构建图像索引
            image_index = model.build_image_index(test_images, batch_size=16)
            
            # 进行检索
            retrieval_results = model.retrieve_images(test_texts[:5], image_index, top_k=10)
            
            # 评估
            metrics = evaluate_retrieval_results(retrieval_results, ground_truth)
            metrics['fusion_method'] = fusion_method
            metrics['similarity_type'] = similarity_type
            
            comparison_results.append(metrics)
            
            print(f"map@10: {metrics.get('map@10', 0):.4f}")
    
    return comparison_results

# 5. 主函数示例
def main():
    """
    主函数，运行所有测试
    """
    try:
        print("==== 跨模态检索模型测试 ====")
        
        # 检查是否有可用的GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建测试数据
        print("创建测试数据...")
        test_texts, test_images, ground_truth = create_test_data(num_texts=20, num_images=100)
        print(f"创建了 {len(test_texts)} 条测试文本和 {len(test_images)} 张测试图像")
        
        # 创建检索模型
        print("创建检索模型...")
        model = create_retrieval_model(device=device)
        
        # 运行各项测试
        test_feature_extraction(model, test_texts, test_images)
        test_similarity_calculation(model, test_texts, test_images)
        test_retrieval_functionality(model, test_texts, test_images, ground_truth)
        
        # 比较不同方法（可选，耗时较长）
        # comparison_results = compare_different_methods(test_texts, test_images, ground_truth, device=device)
        
        print("\n==== 测试完成 ====")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()