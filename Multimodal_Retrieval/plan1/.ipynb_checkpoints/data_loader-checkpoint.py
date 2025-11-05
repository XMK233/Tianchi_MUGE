'''
实现的功能：

1. 数据加载函数 ：
   
   - 处理 tsv 格式的图片数据
   - 处理 jsonl 格式的 query 数据
   - 批量加载机制，避免内存溢出
2. Base64 图片解析 ：
   
   - 将 base64 编码转换为 PIL 图像
   - 统一图像格式为 RGB
   - 调整图像尺寸到指定大小（默认 224x224）
3. 数据统计功能 ：
   
   - 统计查询数量、平均查询长度
   - 统计图片数量
   - 验证数据一致性
4. 内存优化设计 （考虑到您32G内存和16G显存的限制）：
   
   - 批处理加载图片数据
   - 垃圾回收机制定期清理内存
   - 可选的最大样本数限制
   - 按需加载图片，避免一次性加载全部数据
5. 示例数据生成 ：
   
   - 当实际数据文件不存在时，自动创建示例数据用于测试
您可以通过运行这个脚本来测试数据加载功能，或者将其作为模块导入到您的其他代码中使用。脚本中已经包含了使用示例，展示了如何统计数据集信息、验证数据一致性以及加载示例图片。
'''

import os
import json
import base64
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import logging
from tqdm import tqdm
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    电商图文检索数据加载器
    用于处理tsv格式的图片数据和jsonl格式的query数据
    """
    
    def __init__(self, data_dir=None, batch_size=1000, image_size=(224, 224)):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            batch_size: 批处理大小，控制内存使用
            image_size: 图像尺寸
        """
        self.data_dir = data_dir or '/mnt/d/forCoding_data/Tianchi_MUGE/originalData/Multimodal_Retrieval'
        self.batch_size = batch_size
        self.image_size = image_size
        
        # 数据文件路径
        self.file_paths = {
            'train_imgs': os.path.join(self.data_dir, 'MR_train_imgs.tsv'),
            'train_queries': os.path.join(self.data_dir, 'MR_train_queries.jsonl'),
            'valid_imgs': os.path.join(self.data_dir, 'MR_valid_imgs.tsv'),
            'valid_queries': os.path.join(self.data_dir, 'MR_valid_queries.jsonl'),
            'test_imgs': os.path.join(self.data_dir, 'MR_test_imgs.tsv'),
            'test_queries': os.path.join(self.data_dir, 'MR_test_queries.jsonl')
        }
        
        logger.info(f"初始化数据加载器，数据目录: {self.data_dir}")
    
    def decode_image(self, base64_str):
        """
        解码base64编码的图片
        
        Args:
            base64_str: base64编码的字符串
            
        Returns:
            PIL.Image: 解码后的图像
        """
        try:
            # 解码base64字符串
            img_data = base64.urlsafe_b64decode(base64_str)
            # 转换为PIL图像
            img = Image.open(BytesIO(img_data))
            # 转换为RGB格式（处理灰度图）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # 调整图像尺寸
            img = img.resize(self.image_size)
            return img
        except Exception as e:
            logger.error(f"图像解码失败: {e}")
            return None
    
    def load_queries(self, split='train'):
        """
        加载query数据
        
        Args:
            split: 数据集类型，可选'train', 'valid', 'test'
            
        Returns:
            dict: 包含query_id、query_text和item_ids的字典
        """
        file_path = self.file_paths[f'{split}_queries']
        
        if not os.path.exists(file_path):
            logger.warning(f"查询文件不存在: {file_path}")
            # 创建示例数据用于测试
            logger.info("创建示例查询数据用于测试...")
            sample_queries = []
            if split == 'train':
                sample_queries = [
                    {"query_id": 1, "query_text": "胖妹妹松紧腰长裤", "item_ids": [1001]},
                    {"query_id": 2, "query_text": "大码长款棉麻女衬衫", "item_ids": [1002]},
                    {"query_id": 3, "query_text": "高级感托特包斜挎", "item_ids": [1003, 1004]}
                ]
            else:
                sample_queries = [
                    {"query_id": 1001, "query_text": "厚底系带帆布鞋女"},
                    {"query_id": 1002, "query_text": "日式粗陶咖啡杯"},
                    {"query_id": 1003, "query_text": "豹纹雪纺半身"}
                ]
            
            # 保存示例数据
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                for query in sample_queries:
                    f.write(json.dumps(query, ensure_ascii=False) + '\n')
            logger.info(f"已创建示例查询数据: {file_path}")
        
        logger.info(f"加载{split}查询数据: {file_path}")
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"加载{split}查询数据"):
                    try:
                        query = json.loads(line.strip())
                        queries.append(query)
                    except json.JSONDecodeError:
                        logger.warning(f"无效的JSON行: {line.strip()}")
            
            # 转换为DataFrame便于处理
            queries_df = pd.DataFrame(queries)
            logger.info(f"成功加载{split}查询数据，共{len(queries_df)}条")
            return queries_df
            
        except Exception as e:
            logger.error(f"加载查询数据失败: {e}")
            return pd.DataFrame()
    
    def load_images_batch(self, split='train', max_samples=None):
        """
        批量加载图片数据，控制内存使用
        
        Args:
            split: 数据集类型，可选'train', 'valid', 'test'
            max_samples: 最大样本数，用于测试
            
        Yields:
            dict: 包含图片id和图像数据的字典批次
        """
        file_path = self.file_paths[f'{split}_imgs']
        
        if not os.path.exists(file_path):
            logger.warning(f"图片文件不存在: {file_path}")
            # 创建示例数据用于测试
            logger.info("创建示例图片数据用于测试...")
            # 这里我们只创建图片ID，不创建实际的base64编码
            sample_img_ids = [f"{split}_img_{i}" for i in range(10)]
            
            # 保存示例数据
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                for img_id in sample_img_ids:
                    # 使用占位符作为base64编码
                    f.write(f"{img_id}\tplaceholder_base64_data\n")
            logger.info(f"已创建示例图片数据: {file_path}")
        
        logger.info(f"批量加载{split}图片数据: {file_path}")
        
        try:
            batch = []
            count = 0
            
            # 首先计算总行数
            total_lines = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            
            if max_samples:
                total_lines = min(total_lines, max_samples)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"加载{split}图片数据", total=total_lines):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_id, base64_data = parts[0], parts[1]
                            
                            # 对于测试数据，我们可以跳过实际的图像解码以节省内存
                            if base64_data != 'placeholder_base64_data':
                                img = self.decode_image(base64_data)
                                if img:
                                    batch.append({'img_id': img_id, 'image': img})
                            else:
                                # 对于占位符数据，我们创建一个空图像
                                batch.append({'img_id': img_id, 'image': None})
                            
                            count += 1
                            
                            # 当达到批处理大小时，返回批次并重置
                            if len(batch) >= self.batch_size:
                                yield batch
                                batch = []
                                # 清理内存
                                gc.collect()
                            
                            # 如果达到最大样本数，停止处理
                            if max_samples and count >= max_samples:
                                break
                    except Exception as e:
                        logger.error(f"处理图片行失败: {e}")
            
            # 处理最后一个批次
            if batch:
                yield batch
            
        except Exception as e:
            logger.error(f"批量加载图片数据失败: {e}")
    
    def create_img_id_to_image_dict(self, split='train', max_samples=None):
        """
        创建图片ID到图像的映射字典
        
        Args:
            split: 数据集类型，可选'train', 'valid', 'test'
            max_samples: 最大样本数，用于测试
            
        Returns:
            dict: 图片ID到图像的映射
        """
        img_dict = {}
        
        # 由于内存限制，对于训练集，我们可以只加载一部分数据
        if split == 'train' and not max_samples:
            logger.info("训练集可能很大，建议指定max_samples参数限制加载数量")
            max_samples = 10000  # 默认限制为10000张图片
        
        for batch in self.load_images_batch(split=split, max_samples=max_samples):
            for item in batch:
                img_dict[item['img_id']] = item['image']
            
            # 打印内存使用情况
            # logger.info(f"已加载{len(img_dict)}张图片到内存")
        
        logger.info(f"成功创建{split}图片映射字典，共{len(img_dict)}张图片")
        return img_dict
    
    def load_dataset_statistics(self, split='train'):
        """
        加载数据集统计信息，而不加载实际图像数据
        
        Args:
            split: 数据集类型，可选'train', 'valid', 'test'
            
        Returns:
            dict: 数据集统计信息
        """
        logger.info(f"统计{split}数据集信息")
        
        # 统计query信息
        queries_df = self.load_queries(split=split)
        query_stats = {
            'query_count': len(queries_df),
            'avg_query_length': queries_df['query_text'].str.len().mean() if 'query_text' in queries_df.columns else 0
        }
        
        # 如果有item_ids列，统计相关信息
        if 'item_ids' in queries_df.columns:
            query_stats['has_item_ids'] = True
            query_stats['avg_items_per_query'] = queries_df['item_ids'].apply(len).mean()
        else:
            query_stats['has_item_ids'] = False
        
        # 统计图片数量（不加载实际图片）
        img_file_path = self.file_paths[f'{split}_imgs']
        img_count = 0
        
        if os.path.exists(img_file_path):
            with open(img_file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    img_count += 1
        
        stats = {
            'split': split,
            'query_statistics': query_stats,
            'image_count': img_count
        }
        
        logger.info(f"{split}数据集统计: {stats}")
        return stats
    
    def validate_data_consistency(self, split='train'):
        """
        验证数据一致性，检查query引用的图片ID是否存在
        
        Args:
            split: 数据集类型，可选'train', 'valid'
            
        Returns:
            dict: 一致性检查结果
        """
        if split == 'test':
            logger.info("测试集不包含item_ids，跳过一致性检查")
            return {}
        
        logger.info(f"验证{split}数据集一致性")
        
        # 获取所有图片ID
        img_file_path = self.file_paths[f'{split}_imgs']
        img_ids = set()
        
        if os.path.exists(img_file_path):
            with open(img_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="收集图片ID"):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 1:
                            img_ids.add(parts[0])
                    except:
                        pass
        
        # 验证query引用的图片ID
        queries_df = self.load_queries(split=split)
        if 'item_ids' not in queries_df.columns:
            logger.warning("查询数据中没有item_ids列")
            return {}
        
        missing_ids = set()
        total_references = 0
        
        for _, row in tqdm(queries_df.iterrows(), desc="验证查询引用", total=len(queries_df)):
            if isinstance(row['item_ids'], list):
                for item_id in row['item_ids']:
                    item_id_str = str(item_id)
                    total_references += 1
                    if item_id_str not in img_ids:
                        missing_ids.add(item_id_str)
        
        consistency_result = {
            'total_image_ids': len(img_ids),
            'total_item_references': total_references,
            'missing_image_ids_count': len(missing_ids),
            'missing_image_ids_sample': list(missing_ids)[:10] if len(missing_ids) > 10 else list(missing_ids),
            'consistency_rate': 1 - (len(missing_ids) / total_references) if total_references > 0 else 1.0
        }
        
        logger.info(f"{split}数据集一致性检查结果: {consistency_result}")
        return consistency_result

# 使用示例
if __name__ == "__main__":
    # 初始化数据加载器
    loader = DataLoader(batch_size=500, image_size=(224, 224))
    
    # 1. 统计数据集信息
    logger.info("========== 统计数据集信息 ==========")
    for split in ['train', 'valid', 'test']:
        try:
            stats = loader.load_dataset_statistics(split=split)
            logger.info(f"{split}集统计结果: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"统计{split}集信息失败: {e}")
    
    # 2. 验证数据一致性
    logger.info("\n========== 验证数据一致性 ==========")
    for split in ['train', 'valid']:
        try:
            consistency = loader.validate_data_consistency(split=split)
            logger.info(f"{split}集一致性结果: {json.dumps(consistency, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"验证{split}集一致性失败: {e}")
    
    # 3. 示例：加载一小部分验证集图片用于测试
    logger.info("\n========== 加载示例验证集图片 ==========")
    try:
        # 只加载10张图片作为示例
        img_dict = loader.create_img_id_to_image_dict(split='valid', max_samples=10)
        logger.info(f"成功加载{len(img_dict)}张验证集图片")
        
        # 显示图片信息
        for img_id, img in list(img_dict.items())[:3]:  # 只显示前3张
            if img:
                logger.info(f"图片 {img_id}: 尺寸={img.size}, 模式={img.mode}")
            else:
                logger.info(f"图片 {img_id}: 未加载图像数据")
    
    except Exception as e:
        logger.error(f"加载示例图片失败: {e}")
    
    logger.info("\n数据加载完成！")