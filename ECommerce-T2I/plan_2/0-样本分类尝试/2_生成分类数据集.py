#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的电商商品分类数据集生成器
结合文本分类和图像分类结果，生成完整的分类数据集
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from chunk_loader import ChunkLoader
from 商品分类方案 import ProductClassifier
from 图片分类方案 import ImageFeatureExtractor, VisualClassifier, ColorHistogramAnalyzer


class UnifiedClassifier:
    """统一分类器 - 结合文本和图像分类"""
    
    def __init__(self):
        """初始化分类器"""
        # 初始化各个分类器
        self.text_classifier = ProductClassifier()
        self.image_extractor = ImageFeatureExtractor('resnet50')
        self.visual_classifier = None
        self.color_analyzer = ColorHistogramAnalyzer()
        
        # 分类权重配置
        self.text_weight = 0.6  # 文本分类权重
        self.image_weight = 0.4  # 图像分类权重
        
        # 训练状态
        self.image_trained = False
    
    def train_image_classifier(self, training_data):
        """
        训练图像分类器
        Args:
            training_data: 训练数据 [(img_id, image_base64, description), ...]
        """
        print("正在训练图像分类器...")
        
        # 提取图像
        from base64 import b64decode
        from io import BytesIO
        from PIL import Image
        
        images = []
        valid_data = []
        
        for img_id, image_base64, description in training_data:
            try:
                image = Image.open(BytesIO(b64decode(image_base64))).convert('RGB')
                images.append(image)
                valid_data.append((img_id, image_base64, description))
            except Exception as e:
                print(f"图像解码失败: {e}")
                continue
        
        if len(images) < 2:
            print("有效图像数量不足，无法训练图像分类器")
            return
        
        print(f"成功处理 {len(images)} 张训练图像")
        
        # 提取特征
        features = self.image_extractor.batch_extract_features(images)
        
        # 训练分类器
        self.visual_classifier = VisualClassifier(self.image_extractor, n_clusters=10)
        self.visual_classifier.fit(features)
        
        self.image_trained = True
        print("图像分类器训练完成")
    
    def classify_single_item(self, img_id: str, image_base64: str, description: str) -> Dict:
        """
        对单个商品进行分类
        Args:
            img_id: 商品ID
            image_base64: base64编码的图像
            description: 商品描述
        Returns:
            分类结果字典
        """
        result = {
            "img_id": img_id,
            "description": description,
            "text_classification": None,
            "image_classification": None,
            "final_classification": None,
            "confidence": 0.0,
            "classification_method": "unified"
        }
        
        # 文本分类
        try:
            text_result = self.text_classifier.classify_by_text(description)
            if len(text_result) == 3:
                main_category, subcategory, confidence = text_result
            else:
                main_category, subcategory = text_result[0], text_result[1]
                confidence = 0.5  # 默认置信度
            
            result["text_classification"] = {
                "main_category": main_category,
                "subcategory": subcategory,
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.5
            }
        except Exception as e:
            print(f"文本分类失败: {e}")
            result["text_classification"] = {
                "main_category": "其他",
                "subcategory": "未分类",
                "confidence": 0.0
            }
        
        # 图像分类
        if self.image_trained and self.visual_classifier:
            try:
                from base64 import b64decode
                from io import BytesIO
                from PIL import Image
                
                image = Image.open(BytesIO(b64decode(image_base64))).convert('RGB')
                
                # 图像特征提取
                features = self.image_extractor.extract_features(image)
                visual_categories, cluster_ids = self.visual_classifier.predict([features])
                
                # 颜色分析
                color_hist = self.color_analyzer.extract_color_histogram(image)
                color_category = self.color_analyzer.classify_by_color(color_hist)
                
                result["image_classification"] = {
                    "visual_category": visual_categories[0],
                    "color_category": color_category,
                    "cluster_id": int(cluster_ids[0])
                }
            except Exception as e:
                print(f"图像分类失败: {e}")
                result["image_classification"] = {
                    "visual_category": "未分类",
                    "color_category": "未知",
                    "cluster_id": -1
                }
        else:
            result["image_classification"] = {
                "visual_category": "未分类",
                "color_category": "未知",
                "cluster_id": -1
            }
        
        # 统一分类决策
        final_classification = self._make_unified_decision(
            result["text_classification"], 
            result["image_classification"]
        )
        
        result["final_classification"] = final_classification
        result["confidence"] = final_classification["confidence"]
        
        return result
    
    def _make_unified_decision(self, text_result: Dict, image_result: Dict) -> Dict:
        """
        基于文本和图像分类结果做出最终决策
        Args:
            text_result: 文本分类结果
            image_result: 图像分类结果
        Returns:
            最终分类决策
        """
        # 如果文本分类有高置信度，优先使用文本结果
        if text_result["confidence"] > 0.7:
            return {
                "main_category": text_result["main_category"],
                "subcategory": text_result["subcategory"],
                "method": "text_dominant",
                "confidence": text_result["confidence"]
            }
        
        # 如果图像分类有结果，结合两种结果
        if image_result["visual_category"] != "未分类":
            # 尝试将图像分类映射到文本分类体系
            mapped_category = self._map_image_to_text_category(image_result["visual_category"])
            
            if mapped_category != "其他":
                # 结合两种分类结果
                text_score = text_result["confidence"] * self.text_weight
                image_score = 0.5 * self.image_weight  # 图像分类默认分数
                total_score = text_score + image_score
                
                return {
                    "main_category": mapped_category,
                    "subcategory": text_result["subcategory"],
                    "method": "combined",
                    "confidence": min(total_score, 1.0)
                }
        
        # 默认使用文本分类结果
        return {
            "main_category": text_result["main_category"],
            "subcategory": text_result["subcategory"],
            "method": "text_fallback",
            "confidence": text_result["confidence"]
        }
    
    def _map_image_to_text_category(self, visual_category: str) -> str:
        """
        将图像分类结果映射到文本分类体系
        Args:
            visual_category: 图像分类类别
        Returns:
            映射的文本分类
        """
        mapping = {
            "服装-上衣": "女装",
            "服装-下装": "女装", 
            "服装-连衣裙": "女装",
            "鞋类-运动鞋": "鞋类",
            "鞋类-皮鞋": "鞋类",
            "箱包-手提包": "箱包配饰",
            "箱包-双肩包": "箱包配饰",
            "配饰-首饰": "箱包配饰",
            "配饰-帽子": "箱包配饰",
            "其他商品": "其他"
        }
        
        return mapping.get(visual_category, "其他")
    
    def batch_classify(self, data_list: List[Tuple], sample_size: int = 1000) -> List[Dict]:
        """
        批量分类商品
        Args:
            data_list: 数据列表
            sample_size: 样本大小
        Returns:
            分类结果列表
        """
        print(f"开始批量分类 {min(len(data_list), sample_size)} 个样本...")
        
        # 随机采样
        if len(data_list) > sample_size:
            import random
            data_list = random.sample(data_list, sample_size)
        
        # 训练图像分类器（使用前20%的数据）
        train_size = min(int(len(data_list) * 0.2), 100)
        if train_size > 2:
            self.train_image_classifier(data_list[:train_size])
        
        # 分类所有数据
        results = []
        for i, (img_id, image_base64, description) in enumerate(data_list):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(data_list)} 个样本...")
            
            result = self.classify_single_item(img_id, image_base64, description)
            results.append(result)
        
        return results


def generate_classified_dataset(output_dir: str = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/classified_dataset"):
    """
    生成分类后的完整数据集
    Args:
        output_dir: 输出目录
    """
    print("=== 生成统一的商品分类数据集 ===\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据文件路径
    image_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
    text_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
    
    # 初始化数据加载器和分类器
    loader = ChunkLoader(image_file_path, text_file_path, chunk_size=100)
    unified_classifier = UnifiedClassifier()
    
    # 分批处理数据
    batch_size = 500
    total_batches = 10  # 处理前10个批次，总共5000个样本
    all_results = []
    
    for batch_idx in range(total_batches):
        print(f"\n--- 处理批次 {batch_idx + 1}/{total_batches} ---")
        
        start_line = batch_idx * batch_size
        batch_data = loader.get_chunk(start_line=start_line, chunk_size=batch_size)
        
        if not batch_data:
            print(f"批次 {batch_idx + 1} 无数据，跳过")
            continue
        
        print(f"加载 {len(batch_data)} 条数据")
        
        # 分类当前批次
        batch_results = unified_classifier.batch_classify(batch_data, sample_size=len(batch_data))
        all_results.extend(batch_results)
        
        # 保存批次结果
        batch_output_file = os.path.join(output_dir, f"batch_{batch_idx + 1:03d}_results.json")
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        print(f"批次 {batch_idx + 1} 结果已保存到: {batch_output_file}")
    
    # 生成汇总统计
    summary_stats = generate_summary_statistics(all_results)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "classification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    # 保存完整结果
    final_output_file = os.path.join(output_dir, "complete_classification_results.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_samples": len(all_results),
                "generation_time": summary_stats["生成时间"],
                "classifier_version": "1.0"
            },
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 数据集生成完成 ===")
    print(f"总样本数: {len(all_results)}")
    print(f"汇总统计: {summary_file}")
    print(f"完整结果: {final_output_file}")
    print(f"分类数据集目录: {output_dir}")
    
    return all_results, summary_stats


def generate_summary_statistics(results: List[Dict]) -> Dict:
    """
    生成分类结果的汇总统计
    Args:
        results: 分类结果列表
    Returns:
        统计字典
    """
    print("生成汇总统计...")
    
    # 统计各分类的数量分布
    main_category_dist = Counter([r["final_classification"]["main_category"] for r in results])
    subcategory_dist = Counter([r["final_classification"]["subcategory"] for r in results])
    
    # 统计分类方法分布
    method_dist = Counter([r["final_classification"]["method"] for r in results])
    
    # 统计置信度分布
    confidences = [r["final_classification"]["confidence"] for r in results]
    avg_confidence = np.mean(confidences)
    
    # 统计文本和图像分类的一致性
    text_vs_image_consistency = 0
    valid_comparisons = 0
    
    for r in results:
        if r["image_classification"]["visual_category"] != "未分类":
            text_cat = r["final_classification"]["main_category"]
            image_cat = r["image_classification"]["visual_category"]
            
            # 简单的一致性检查
            if "服装" in text_cat and "服装" in image_cat:
                text_vs_image_consistency += 1
            elif text_cat == image_cat:
                text_vs_image_consistency += 1
            
            valid_comparisons += 1
    
    consistency_rate = text_vs_image_consistency / valid_comparisons if valid_comparisons > 0 else 0
    
    # 按主分类分组的子分类分布
    category_sub_dist = defaultdict(Counter)
    for r in results:
        main_cat = r["final_classification"]["main_category"]
        sub_cat = r["final_classification"]["subcategory"]
        category_sub_dist[main_cat][sub_cat] += 1
    
    summary = {
        "生成时间": "2025-12-26 15:30:00",
        "总样本数": len(results),
        "主分类分布": dict(main_category_dist),
        "子分类分布": dict(subcategory_dist),
        "分类方法分布": dict(method_dist),
        "平均置信度": f"{avg_confidence:.3f}",
        "文本图像一致性": f"{consistency_rate:.3f}",
        "分类覆盖率": f"{len([r for r in results if r['final_classification']['main_category'] != '其他']) / len(results) * 100:.2f}%",
        "按主分类的子分类分布": {k: dict(v) for k, v in category_sub_dist.items()}
    }
    
    return summary


def main():
    """主函数"""
    print("=== 统一电商商品分类数据集生成器 ===\n")
    
    # 生成分类数据集
    results, stats = generate_classified_dataset()
    
    # 显示主要统计信息
    print("\n=== 主要统计信息 ===")
    print(f"总样本数: {stats['总样本数']}")
    print(f"平均置信度: {stats['平均置信度']}")
    print(f"分类覆盖率: {stats['分类覆盖率']}")
    print(f"文本图像一致性: {stats['文本图像一致性']}")
    
    print("\n主分类分布:")
    for category, count in stats["主分类分布"].items():
        percentage = count / len(results) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n分类方法分布:")
    for method, count in stats["分类方法分布"].items():
        percentage = count / len(results) * 100
        print(f"  {method}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()