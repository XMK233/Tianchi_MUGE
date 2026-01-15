import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import jieba
import jieba.posseg as pseg
import re
from collections import Counter, defaultdict
import json
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 配置路径
DATA_PATH = "/mnt/d/forCoding_data/Tianchi_MUGE/originalData/ECommerce-T2I/T2I_train.text.tsv"
MODEL_PATH = "/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/"
OUTPUT_DIR = "/mnt/d/forCoding_code/Tianchi_MUGE/ECommerce-T2I/plan_1/1-利用文本进行分类"

# 采样配置
NUM_BATCHES = 10
BATCH_SIZE_LOAD = 500
TOTAL_SAMPLES = NUM_BATCHES * BATCH_SIZE_LOAD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text_for_bert(text):
    """
    预处理文本给BERT：虽然BERT可以处理所有字符，但用户希望关注中文。
    这里主要用于后续的标签提取，BERT输入保持原样或轻微清洗。
    """
    return text.strip()

def extract_chinese_keywords(texts):
    """
    从一组文本中提取核心中文关键词（名词）
    """
    all_words = []
    for text in texts:
        # 仅保留中文
        chinese_text = "".join(re.findall(r'[\u4e00-\u9fa5]+', text))
        if not chinese_text:
            continue
            
        # 分词并标注词性
        words = pseg.cut(chinese_text)
        for w, flag in words:
            # 保留名词(n)，过滤掉单字（除非是非常核心的词如“鞋”）
            if flag.startswith('n') and len(w) > 0:
                all_words.append(w)
    return all_words

def analyze_cluster_label(texts):
    """
    分析聚类标签，尝试归纳出通用类别（如“鞋”、“衣服”）
    """
    words = extract_chinese_keywords(texts)
    
    # 过滤掉营销词汇
    MARKETING_WORDS = {
        '新款', '韩版', '时尚', '气质', '品牌', '现货', '官方', '旗舰店', '专柜', '正品', 
        '男女', '儿童', '宝宝', '学生', '2021', '夏季', '春季', '秋冬', 'ins', '网红',
        '设计', '宽松', '显瘦', '大码', '加绒', '加厚', '修身', '百搭', '复古', '洋气',
        '风格', '系列', '定制', '同款', '少女', '可爱', '性感', '日系', '港味', '欧美',
        '新品', '特价', '包邮', '组合', '套装', '礼盒', '家用', '专用', '通用', '智能',
        '客厅', '卧室', '厨房', '卫生间', '阳台', '宿舍', '办公室', '户外', '居家',
        '颜色', '尺码', '详情', '链接', '拍下', '备注', '收藏', '关注', '店铺', '优惠'
    }

    # 过滤词汇
    filtered_words = [w for w in words if w not in MARKETING_WORDS and len(w) >= 2]
    
    word_counts = Counter(filtered_words)
    
    if not word_counts:
        return "未知类别"
    
    # 获取最高频的3个词作为标签组合
    top_words_tuples = word_counts.most_common(3)
    top_words = [w for w, c in top_words_tuples]
    
    return "_".join(top_words)

def main():
    print(f"使用的设备: {DEVICE}")
    print(f"加载模型: {MODEL_PATH}")
    
    # 1. 加载模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    
    # 2. 加载数据
    print(f"正在加载数据 (前 {TOTAL_SAMPLES} 条)...")
    data = []
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= TOTAL_SAMPLES:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id, desc = parts[0], parts[1]
                    data.append({'img_id': img_id, 'text': desc})
    except FileNotFoundError:
        print(f"错误：找不到文件 {DATA_PATH}")
        return

    df = pd.DataFrame(data)
    print(f"成功加载 {len(df)} 条数据")
    
    # 3. 提取特征
    print("正在提取BERT特征...")
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['text'][i:i+batch_size].tolist()
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 Mean Pooling 代替 [CLS]
            # attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            embeddings.append(mean_embeddings)
            
    features = np.vstack(embeddings)
    
    # 降维（可选，但推荐用于加速聚类和去除噪声）
    print("正在进行PCA降维...")
    pca = PCA(n_components=50) # 保留50维
    features_reduced = pca.fit_transform(features)
    
    # 4. 寻找最优聚类数 (K)
    print("正在寻找最优聚类数...")
    best_score = -1
    best_k = 20 # 默认值稍微调大
    # 尝试 15 到 60 个类别，强制细分
    search_range = range(15, 60) 
    
    scores = []
    for k in search_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_reduced)
        score = silhouette_score(features_reduced, labels)
        scores.append(score)
        print(f"K={k}, 轮廓系数={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"最优聚类数: {best_k} (轮廓系数: {best_score:.4f})")
    
    # 5. 执行最终聚类
    print(f"使用 K={best_k} 进行最终聚类...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_reduced)
    
    df['cluster'] = cluster_labels
    
    # 6. 生成类别标签
    print("正在生成类别标签...")
    cluster_names = {}
    
    # 统计每个簇的信息
    cluster_info = []
    
    for c_id in range(best_k):
        cluster_texts = df[df['cluster'] == c_id]['text'].tolist()
        label_name = analyze_cluster_label(cluster_texts)
        cluster_names[c_id] = label_name
        
        print(f"簇 {c_id}: 自动命名 -> {label_name} (样本数: {len(cluster_texts)})")
        print(f"  示例文本: {cluster_texts[:2]}")
        
        cluster_info.append({
            "cluster_id": c_id,
            "label_name": label_name,
            "count": len(cluster_texts),
            "example_texts": cluster_texts[:5]
        })
        
    df['predicted_label'] = df['cluster'].map(cluster_names)
    
    # 7. 保存结果
    output_file_csv = os.path.join(OUTPUT_DIR, "classification_results.csv")
    output_file_json = os.path.join(OUTPUT_DIR, "classification_summary.json")
    
    # 保存详细映射列表
    df[['img_id', 'text', 'cluster', 'predicted_label']].to_csv(output_file_csv, index=False, sep='\t')
    
    # 保存汇总信息
    with open(output_file_json, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
        
    print(f"\n任务完成！")
    print(f"结果已保存至:\n1. {output_file_csv}\n2. {output_file_json}")

if __name__ == "__main__":
    main()
