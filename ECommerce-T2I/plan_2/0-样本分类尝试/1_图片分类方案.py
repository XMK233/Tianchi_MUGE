#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图像的电商商品分类系统
使用预训练模型提取图像特征，结合聚类算法进行分类
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from chunk_loader import ChunkLoader


class ImageFeatureExtractor:
    """图像特征提取器"""
    
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化特征提取器
        Args:
            model_name: 预训练模型名称
            device: 设备类型
        """
        self.device = device
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # 移除最后的分类层
            self.feature_dim = 2048
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier = nn.Identity()
            self.feature_dim = 1280
        elif model_name == 'vit':
            self.model = models.vit_b_16(pretrained=True)
            self.model.heads.head = nn.Identity()
            self.feature_dim = 768
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """
        提取单张图像的特征
        Args:
            image: PIL图像
        Returns:
            特征向量 (numpy array)
        """
        try:
            # 预处理
            if isinstance(image, str):  # 图像路径
                image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):  # base64编码
                from base64 import b64decode
                image = Image.open(BytesIO(b64decode(image))).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 特征提取
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
        except Exception as e:
            print(f"图像特征提取失败: {e}")
            return np.zeros(self.feature_dim)
    
    def batch_extract_features(self, images):
        """
        批量提取图像特征
        Args:
            images: 图像列表
        Returns:
            特征矩阵 (numpy array)
        """
        features = []
        for i, image in enumerate(images):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(images)} 张图像...")
            
            feature = self.extract_features(image)
            features.append(feature)
        
        return np.array(features)


class VisualClassifier:
    """基于视觉特征的分类器"""
    
    def __init__(self, feature_extractor, n_clusters=10):
        """
        初始化分类器
        Args:
            feature_extractor: 图像特征提取器
            n_clusters: 聚类数量
        """
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = None
        self.fitted = False
        
        # 商品类型映射（基于视觉特征）
        self.category_mapping = {
            0: "服装-上衣",
            1: "服装-下装", 
            2: "服装-连衣裙",
            3: "鞋类-运动鞋",
            4: "鞋类-皮鞋",
            5: "箱包-手提包",
            6: "箱包-双肩包",
            7: "配饰-首饰",
            8: "配饰-帽子",
            9: "其他商品"
        }
    
    def preprocess_features(self, features):
        """
        特征预处理
        Args:
            features: 原始特征矩阵
        Returns:
            预处理后的特征
        """
        # 特征标准化
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # PCA降维（如果特征维度太高且样本数足够）
        if features.shape[1] > 512 and features.shape[0] > features.shape[1]:
            self.pca = PCA(n_components=min(512, features.shape[0]-1))
            features = self.pca.fit_transform(features)
        elif features.shape[1] > 512:
            # 如果样本数不足，则只保留部分特征
            self.pca = PCA(n_components=min(features.shape[0]-1, features.shape[1]))
            features = self.pca.fit_transform(features)
        
        return features
    
    def find_optimal_clusters(self, features, max_clusters=20):
        """
        寻找最优聚类数量
        Args:
            features: 特征矩阵
            max_clusters: 最大聚类数量
        Returns:
            最优聚类数量
        """
        print("正在寻找最优聚类数量...")
        
        # 特征预处理
        features = self.preprocess_features(features)
        
        # 计算不同聚类数的轮廓系数
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(features) // 10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            score = silhouette_score(features, cluster_labels)
            silhouette_scores.append(score)
            print(f"K={k}: 轮廓系数 = {score:.3f}")
        
        # 选择最优K值
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"最优聚类数量: {optimal_k}")
        
        return optimal_k
    
    def fit(self, features):
        """
        训练聚类模型
        Args:
            features: 图像特征矩阵
        """
        print("开始训练聚类模型...")
        
        # 特征预处理
        features = self.preprocess_features(features)
        
        # 寻找最优聚类数量
        optimal_k = self.find_optimal_clusters(features, self.n_clusters)
        self.n_clusters = optimal_k
        
        # 训练KMeans模型
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features)
        
        self.fitted = True
        print(f"聚类模型训练完成，共 {self.n_clusters} 个类别")
        
        return cluster_labels
    
    def predict(self, features):
        """
        预测图像类别
        Args:
            features: 图像特征
        Returns:
            预测类别
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 特征预处理
        features = self.preprocess_features(features)
        
        # 预测
        cluster_labels = self.kmeans.predict(features)
        
        # 映射到商品类别
        categories = [self.category_mapping.get(label, f"类别{label}") for label in cluster_labels]
        
        return categories, cluster_labels
    
    def get_cluster_info(self, features, cluster_labels):
        """
        获取聚类信息
        Args:
            features: 特征矩阵
            cluster_labels: 聚类标签
        Returns:
            聚类信息字典
        """
        cluster_info = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            
            # 计算聚类统计信息
            cluster_size = np.sum(cluster_mask)
            cluster_center = np.mean(cluster_features, axis=0)
            
            # 计算类内方差
            intra_variance = np.mean(np.linalg.norm(cluster_features - cluster_center, axis=1) ** 2)
            
            cluster_info[f"聚类_{cluster_id}"] = {
                "样本数量": int(cluster_size),
                "比例": f"{cluster_size / len(features) * 100:.2f}%",
                "类内方差": f"{intra_variance:.4f}",
                "映射类别": self.category_mapping.get(cluster_id, f"类别{cluster_id}")
            }
        
        return cluster_info


class ColorHistogramAnalyzer:
    """颜色直方图分析器"""
    
    def __init__(self, n_bins=8):
        """
        初始化颜色分析器
        Args:
            n_bins: 颜色直方图的bin数量
        """
        self.n_bins = n_bins
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    
    def extract_color_histogram(self, image):
        """
        提取颜色直方图
        Args:
            image: PIL图像
        Returns:
            颜色直方图 (numpy array)
        """
        try:
            image_tensor = self.transform(image)  # [3, 64, 64]
            
            # 分别处理RGB三个通道
            histograms = []
            for channel in range(3):
                channel_data = image_tensor[channel].numpy().flatten()
                hist, _ = np.histogram(channel_data, bins=self.n_bins, range=(0, 1))
                histograms.append(hist)
            
            # 拼接并归一化
            color_hist = np.concatenate(histograms)
            color_hist = color_hist / (np.sum(color_hist) + 1e-8)
            
            return color_hist
        except Exception as e:
            print(f"颜色直方图提取失败: {e}")
            return np.zeros(self.n_bins * 3)
    
    def classify_by_color(self, color_hist):
        """
        基于颜色特征进行简单分类
        Args:
            color_hist: 颜色直方图
        Returns:
            颜色分类
        """
        # 分析RGB通道分布
        r_hist = color_hist[:self.n_bins]
        g_hist = color_hist[self.n_bins:2*self.n_bins]
        b_hist = color_hist[2*self.n_bins:]
        
        # 计算主导颜色
        dominant_r = np.argmax(r_hist)
        dominant_g = np.argmax(g_hist)
        dominant_b = np.argmax(b_hist)
        
        # 简单分类规则
        if dominant_r > dominant_g and dominant_r > dominant_b:
            if np.sum(r_hist[5:]) > 0.3:  # 红色系商品
                return "红色商品"
            else:
                return "暖色调商品"
        elif dominant_g > dominant_r and dominant_g > dominant_b:
            return "绿色/自然色商品"
        elif dominant_b > dominant_r and dominant_b > dominant_g:
            return "蓝色/冷色调商品"
        else:
            return "多色调商品"


def analyze_images_batch(data_list, sample_size=200):
    """
    批量分析图像
    Args:
        data_list: 数据列表 [(img_id, image_base64, description), ...]
        sample_size: 样本大小
    Returns:
        分析结果
    """
    print(f"开始分析 {min(len(data_list), sample_size)} 个样本...")
    
    # 随机采样
    if len(data_list) > sample_size:
        import random
        data_list = random.sample(data_list, sample_size)
    
    # 解码图像
    from base64 import b64decode
    from io import BytesIO
    
    images = []
    valid_indices = []
    
    for i, (img_id, image_base64, description) in enumerate(data_list):
        try:
            image = Image.open(BytesIO(b64decode(image_base64))).convert('RGB')
            images.append(image)
            valid_indices.append(i)
        except Exception as e:
            print(f"图像解码失败 (索引{i}): {e}")
            continue
    
    print(f"成功解码 {len(images)} 张图像")
    
    # 初始化分析器
    feature_extractor = ImageFeatureExtractor('resnet50')
    visual_classifier = VisualClassifier(feature_extractor, n_clusters=10)
    color_analyzer = ColorHistogramAnalyzer()
    
    # 提取特征
    print("提取图像特征...")
    features = feature_extractor.batch_extract_features(images)
    
    # 训练分类器
    cluster_labels = visual_classifier.fit(features)
    
    # 颜色分析
    print("分析颜色特征...")
    color_categories = []
    for image in images:
        color_hist = color_analyzer.extract_color_histogram(image)
        color_cat = color_analyzer.classify_by_color(color_hist)
        color_categories.append(color_cat)
    
    # 组合分析结果
    results = []
    for i, (image, cluster_label, color_cat) in enumerate(zip(images, cluster_labels, color_categories)):
        results.append({
            "index": valid_indices[i],
            "img_id": data_list[valid_indices[i]][0],
            "description": data_list[valid_indices[i]][2],
            "visual_category": visual_classifier.category_mapping.get(cluster_label, f"类别{cluster_label}"),
            "color_category": color_cat,
            "cluster_id": int(cluster_label)
        })
    
    # 生成统计报告
    visual_dist = Counter([r["visual_category"] for r in results])
    color_dist = Counter([r["color_category"] for r in results])
    
    cluster_info = visual_classifier.get_cluster_info(features, cluster_labels)
    
    analysis_report = {
        "视觉分类分布": dict(visual_dist),
        "颜色分类分布": dict(color_dist),
        "聚类详细信息": cluster_info,
        "总样本数": len(results),
        "特征维度": features.shape[1]
    }
    
    return results, analysis_report


def main():
    """主函数 - 演示图片分类流程"""
    print("=== 基于图像的电商商品分类系统 ===\n")
    
    # 数据文件路径
    image_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
    text_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
    
    # 初始化数据加载器
    loader = ChunkLoader(image_file_path, text_file_path, chunk_size=50)
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_data = loader.get_chunk(start_line=0, chunk_size=50)
    
    if not test_data:
        print("未找到数据，请检查文件路径")
        return
    
    print(f"成功加载 {len(test_data)} 条测试数据\n")
    
    # 执行图像分类分析
    results, analysis_report = analyze_images_batch(test_data, sample_size=30)
    
    # 显示结果
    print("\n=== 图像分类结果示例 ===")
    for i, result in enumerate(results[:10]):
        print(f"\n{i+1}. 描述: {result['description'][:50]}...")
        print(f"   视觉分类: {result['visual_category']}")
        print(f"   颜色分类: {result['color_category']}")
        print(f"   聚类ID: {result['cluster_id']}")
    
    # 显示统计报告
    print("\n=== 视觉分类分布 ===")
    for category, count in analysis_report["视觉分类分布"].items():
        percentage = count / len(results) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n=== 颜色分类分布 ===")
    for category, count in analysis_report["颜色分类分布"].items():
        percentage = count / len(results) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n=== 聚类详细信息 ===")
    for cluster_id, info in analysis_report["聚类详细信息"].items():
        print(f"{cluster_id}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # 保存结果
    output_file = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/图像分类结果.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_report": analysis_report,
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n图像分类结果已保存到: {output_file}")
    print("\n=== 图像分类完成 ===")


if __name__ == "__main__":
    main()