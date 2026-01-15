#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商商品自动分类系统
基于文本描述和商品关键词进行商品分类
"""

import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import jieba
import jieba.analyse
from chunk_loader import ChunkLoader


class ProductClassifier:
    """商品分类器"""
    
    def __init__(self):
        """初始化分类器，定义分类体系和关键词"""
        # 一级分类定义
        self.categories = {
            "女装": {
                "keywords": ["女", "女士", "女装", "姑娘", "女生", "妹", "姐"],
                "subcategories": {
                    "连衣裙": ["连衣裙", "裙子", "长裙", "短裙", "半身裙", "A字裙", "百褶裙", "纱裙"],
                    "上衣": ["T恤", "t恤", "衬衫", "卫衣", "毛衣", "针织衫", "外套", "夹克", "马甲"],
                    "裤子": ["裤子", "长裤", "短裤", "阔腿裤", "牛仔裤", "打底裤", "哈伦裤", "运动裤"],
                    "套装": ["套装", "两件套", "三件套", "套头", "连体"],
                    "内衣": ["内衣", "文胸", "内裤", "睡衣", "家居服", "打底衫"]
                }
            },
            "男装": {
                "keywords": ["男", "男士", "男装", "男人", "男款"],
                "subcategories": {
                    "上衣": ["T恤", "t恤", "衬衫", "卫衣", "毛衣", "外套", "夹克", "风衣"],
                    "裤子": ["裤子", "长裤", "短裤", "牛仔裤", "休闲裤", "运动裤"],
                    "套装": ["套装", "两件套", "西服", "正装"]
                }
            },
            "童装": {
                "keywords": ["童", "儿童", "宝宝", "婴儿", "小孩", "少儿", "男童", "女童"],
                "subcategories": {
                    "上衣": ["T恤", "t恤", "衬衫", "卫衣", "毛衣"],
                    "裤子": ["裤子", "短裤", "长裤"],
                    "裙子": ["连衣裙", "裙子", "公主裙"],
                    "套装": ["套装", "连体衣", "爬服"]
                }
            },
            "鞋类": {
                "keywords": ["鞋", "靴", "高跟鞋", "单鞋", "凉鞋", "运动鞋", "皮鞋", "帆布鞋"],
                "subcategories": {
                    "女鞋": ["高跟鞋", "单鞋", "凉鞋", "皮鞋", "帆布鞋", "小白鞋", "豆豆鞋"],
                    "男鞋": ["皮鞋", "运动鞋", "休闲鞋", "帆布鞋", "板鞋"],
                    "童鞋": ["童鞋", "宝宝鞋", "学步鞋"],
                    "靴子": ["靴子", "短靴", "长靴", "马丁靴", "雪地靴"]
                }
            },
            "箱包配饰": {
                "keywords": ["包", "包包", "背包", "钱包", "配饰", "首饰", "戒指", "耳环", "项链", "帽子", "围巾"],
                "subcategories": {
                    "女包": ["手提包", "斜挎包", "双肩包", "单肩包", "水桶包", "小包", "迷你包"],
                    "男包": ["公文包", "双肩包", "钱包", "手拿包"],
                    "首饰": ["戒指", "耳环", "项链", "手镯", "脚链"],
                    "帽子": ["帽子", "棒球帽", "贝雷帽", "毛线帽"],
                    "其他配饰": ["围巾", "丝巾", "手表", "眼镜"]
                }
            },
            "家居用品": {
                "keywords": ["家居", "家具", "装饰", "摆件", "灯", "沙发", "床", "窗帘", "餐具", "花瓶"],
                "subcategories": {
                    "灯具": ["吊灯", "台灯", "吸顶灯", "壁灯", "落地灯"],
                    "家具": ["沙发", "床", "椅子", "桌子", "柜子"],
                    "装饰品": ["摆件", "花瓶", "装饰画", "挂件"],
                    "厨房用品": ["餐具", "碗", "盘子", "杯子", "锅具"],
                    "布艺": ["窗帘", "地毯", "抱枕", "床品"]
                }
            },
            "数码配件": {
                "keywords": ["手机", "数码", "电子", "配件", "壳", "充电器", "数据线"],
                "subcategories": {
                    "手机配件": ["手机壳", "充电器", "数据线", "耳机"],
                    "数码产品": ["相机", "音响", "键盘", "鼠标"]
                }
            },
            "美妆护肤": {
                "keywords": ["护肤", "美妆", "化妆品", "面膜", "口红", "洗面奶", "精华", "面霜"],
                "subcategories": {
                    "面部护理": ["洗面奶", "爽肤水", "精华", "面霜", "乳液"],
                    "彩妆": ["口红", "粉底", "眼影", "睫毛膏", "腮红"],
                    "面膜": ["面膜", "眼膜", "颈膜"]
                }
            },
            "宠物用品": {
                "keywords": ["宠物", "狗", "猫", "鸟", "鱼", "狗粮", "猫粮", "玩具"],
                "subcategories": {
                    "宠物食品": ["狗粮", "猫粮", "零食", "罐头"],
                    "宠物用品": ["狗碗", "猫砂", "玩具", "牵引绳", "衣服"],
                    "宠物护理": ["沐浴露", "指甲剪", "梳子"]
                }
            },
            "其他": {
                "keywords": [],
                "subcategories": {}
            }
        }
        
        # 构建关键词映射
        self.keyword_to_category = {}
        self.keyword_to_subcategory = {}
        
        for category, info in self.categories.items():
            for keyword in info["keywords"]:
                self.keyword_to_category[keyword] = category
            
            for subcategory in info["subcategories"]:
                for keyword in info["subcategories"][subcategory]:
                    self.keyword_to_subcategory[keyword] = subcategory
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 使用jieba分词
        words = jieba.cut(text)
        # 过滤停用词和单字符
        filtered_words = [word for word in words if len(word) > 1 and not self._is_stop_word(word)]
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(" ".join(filtered_words), topK=top_k, withWeight=False)
        return keywords
    
    def _is_stop_word(self, word: str) -> bool:
        """判断是否为停用词"""
        stop_words = {"新款", "正品", "时尚", "潮流", "韩版", "欧美", "日韩", "爆款", "热销", 
                     "百搭", "显瘦", "显高", "宽松", "修身", "洋气", "气质", "优雅", "甜美",
                     "个性", "简约", "休闲", "运动", "商务", "正装", "复古", "清新", "可爱",
                     "舒适", "透气", "保暖", "防滑", "耐磨", "轻便", "厚实", "柔软", "光滑",
                     "精致", "高档", "优质", "精选", "推荐", "必备", "超值", "特价", "促销",
                     "包邮", "限时", "抢购", "秒杀", "新品", "上市", "到货", "现货", "现货",
                     "2021", "2020", "夏季", "春季", "秋季", "冬季", "春夏", "秋冬"}
        return word in stop_words
    
    def classify_by_text(self, text: str) -> Tuple[str, str, str]:
        """基于文本进行分类"""
        if not text or not text.strip():
            return "其他", "其他", ""
        
        text_lower = text.lower()
        text_keywords = self.extract_keywords(text)
        
        # 评分字典
        category_scores = defaultdict(int)
        subcategory_scores = defaultdict(int)
        
        # 遍历分类关键词
        for keyword in text_keywords:
            # 一级分类匹配
            for cat_keyword in self.keyword_to_category:
                if cat_keyword in text or cat_keyword in text_keywords:
                    category = self.keyword_to_category[cat_keyword]
                    category_scores[category] += 2
            
            # 二级分类匹配
            for sub_keyword in self.keyword_to_subcategory:
                if sub_keyword in text or sub_keyword in text_keywords:
                    subcategory = self.keyword_to_subcategory[sub_keyword]
                    subcategory_scores[subcategory] += 3
        
        # 特殊规则匹配
        special_rules = {
            "女装": ["女", "女士", "女装", "姑娘", "妹", "姐"],
            "男装": ["男", "男士", "男装", "男人", "男款"],
            "童装": ["童", "儿童", "宝宝", "婴儿", "小孩", "少儿", "男童", "女童"],
            "鞋类": ["鞋", "靴", "高跟鞋", "单鞋", "凉鞋"],
            "箱包配饰": ["包", "戒指", "耳环", "项链", "帽子", "围巾"],
            "家居用品": ["家居", "家具", "装饰", "灯", "沙发", "床", "窗帘", "餐具"],
            "数码配件": ["手机", "数码", "壳", "充电器"],
            "美妆护肤": ["护肤", "美妆", "面膜", "口红", "洗面奶"],
            "宠物用品": ["宠物", "狗粮", "猫粮"]
        }
        
        for category, keywords in special_rules.items():
            for keyword in keywords:
                if keyword in text:
                    category_scores[category] += 5
        
        # 确定一级分类
        if not category_scores:
            main_category = "其他"
        else:
            main_category = max(category_scores, key=category_scores.get)
        
        # 确定二级分类
        subcategory_map = {
            "女装": ["连衣裙", "上衣", "裤子", "套装", "内衣"],
            "男装": ["上衣", "裤子", "套装"],
            "童装": ["上衣", "裤子", "裙子", "套装"],
            "鞋类": ["女鞋", "男鞋", "童鞋", "靴子"],
            "箱包配饰": ["女包", "男包", "首饰", "帽子", "其他配饰"],
            "家居用品": ["灯具", "家具", "装饰品", "厨房用品", "布艺"],
            "数码配件": ["手机配件", "数码产品"],
            "美妆护肤": ["面部护理", "彩妆", "面膜"],
            "宠物用品": ["宠物食品", "宠物用品", "宠物护理"],
            "其他": []
        }
        
        valid_subcategories = subcategory_map.get(main_category, [])
        valid_subcategory_scores = {k: v for k, v in subcategory_scores.items() if k in valid_subcategories}
        
        if not valid_subcategory_scores:
            subcategory = "其他" if main_category != "其他" else "其他"
        else:
            subcategory = max(valid_subcategory_scores, key=valid_subcategory_scores.get)
        
        # 生成置信度
        confidence = max(category_scores.values()) if category_scores else 0
        
        return main_category, subcategory, f"置信度: {confidence}"
    
    def batch_classify(self, data_list: List[Tuple]) -> List[Dict]:
        """批量分类"""
        results = []
        for i, (img_id, image_base64, description) in enumerate(data_list):
            main_category, subcategory, confidence = self.classify_by_text(description)
            results.append({
                "index": i,
                "img_id": img_id,
                "description": description,
                "main_category": main_category,
                "subcategory": subcategory,
                "confidence": confidence
            })
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 条数据...")
        
        return results
    
    def analyze_distribution(self, results: List[Dict]) -> Dict:
        """分析分类结果分布"""
        main_dist = Counter([r["main_category"] for r in results])
        sub_dist = Counter([r["subcategory"] for r in results])
        
        # 按主分类统计子分类分布
        category_sub_dist = defaultdict(lambda: defaultdict(int))
        for r in results:
            category_sub_dist[r["main_category"]][r["subcategory"]] += 1
        
        analysis = {
            "主分类分布": dict(main_dist),
            "子分类分布": dict(sub_dist),
            "主分类下子分类分布": {k: dict(v) for k, v in category_sub_dist.items()},
            "总样本数": len(results),
            "分类覆盖率": f"{len([r for r in results if r['main_category'] != '其他']) / len(results) * 100:.2f}%"
        }
        
        return analysis


def main():
    """主函数 - 演示分类流程"""
    print("=== 电商商品自动分类系统 ===\n")
    
    # 初始化分类器
    classifier = ProductClassifier()
    
    # 数据文件路径
    image_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.img.tsv"
    text_file_path = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
    
    # 初始化数据加载器
    loader = ChunkLoader(image_file_path, text_file_path, chunk_size=20)
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_data = loader.get_chunk(start_line=0, chunk_size=20)
    
    if not test_data:
        print("未找到数据，请检查文件路径")
        return
    
    print(f"成功加载 {len(test_data)} 条测试数据\n")
    
    # 执行分类
    print("开始分类...")
    results = classifier.batch_classify(test_data)
    
    # 显示结果
    print("\n=== 分类结果示例 ===")
    for i, result in enumerate(results[:10]):
        print(f"\n{i+1}. 描述: {result['description'][:50]}...")
        print(f"   主分类: {result['main_category']}")
        print(f"   子分类: {result['subcategory']}")
        print(f"   置信度: {result['confidence']}")
    
    # 分析分布
    print("\n=== 分类分布分析 ===")
    analysis = classifier.analyze_distribution(results)
    
    print(f"总样本数: {analysis['总样本数']}")
    print(f"分类覆盖率: {analysis['分类覆盖率']}")
    
    print("\n主分类分布:")
    for category, count in analysis['主分类分布'].items():
        percentage = count / len(results) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n子分类分布 (前10个):")
    for subcategory, count in sorted(analysis['子分类分布'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / len(results) * 100
        print(f"  {subcategory}: {count} ({percentage:.1f}%)")
    
    # 保存详细结果
    output_file = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/分类结果.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis": analysis,
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 生成分类后的数据集
    generate_classified_dataset(results)
    
    print("\n=== 分类完成 ===")


def generate_classified_dataset(results: List[Dict]):
    """生成按分类整理的数据集"""
    print("\n正在生成分类后的数据集...")
    
    # 按主分类和子分类组织数据
    classified_data = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        main_cat = result["main_category"]
        sub_cat = result["subcategory"]
        classified_data[main_cat][sub_cat].append(result)
    
    # 保存分类后的数据
    output_dir = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/分类数据集"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成分类统计报告
    stats_report = []
    stats_report.append("# 电商商品分类统计报告\n")
    stats_report.append(f"分类时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    stats_report.append(f"总样本数: {len(results)}\n")
    stats_report.append(f"分类覆盖率: {len([r for r in results if r['main_category'] != '其他']) / len(results) * 100:.2f}%\n\n")
    
    # 为每个分类创建文件
    for main_category, subcategories in classified_data.items():
        main_cat_dir = os.path.join(output_dir, main_category)
        os.makedirs(main_cat_dir, exist_ok=True)
        
        stats_report.append(f"## {main_category}\n")
        
        for subcategory, items in subcategories.items():
            if not items:
                continue
                
            sub_cat_dir = os.path.join(main_cat_dir, subcategory)
            os.makedirs(sub_cat_dir, exist_ok=True)
            
            # 保存该子分类的数据列表
            data_file = os.path.join(sub_cat_dir, f"{subcategory}_数据.txt")
            with open(data_file, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write(f"{item['img_id']}\t{item['description']}\n")
            
            stats_report.append(f"- {subcategory}: {len(items)} 个商品\n")
            
            print(f"已保存 {main_category}/{subcategory}: {len(items)} 个商品")
        
        stats_report.append("\n")
    
    # 保存统计报告
    stats_file = os.path.join(output_dir, "分类统计报告.md")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.writelines(stats_report)
    
    print(f"分类数据集已保存到: {output_dir}")
    print(f"统计报告已保存到: {stats_file}")


if __name__ == "__main__":
    main()