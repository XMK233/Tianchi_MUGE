"""
评测指标实现（简化版、符合项目 Readme 的中文分词规则）：

- CIDEr（近似 CIDEr-D）：
  - 中文按“分字”处理；连续的字母或数字视为一个 token；去除空白与常见标点。
  - 使用 1-4 gram 的 TF-IDF 向量并计算余弦相似度；对多参考文本取平均；对 n-gram 等权平均。
  - IDF 在整个参考语料（所有样本的所有参考句子）上统计文档频率。

注意：这是占位与联调用的近似实现，非官方 pycocoevalcap CIDEr-D。
"""

import math
import re


# 标点与空白集合：中文分字、连续字母数字作为单 token，其余视为分隔
# 这里将常见中英文标点与空白一次性放入集合，避免语法歧义
_PUNCTS = set("，。！？；：、‘’“”（）《》【】—…,.!?;:()[]{}<>\"'` \t\n\r")


def _tokenize_zh(text):
    """按 Readme 要求进行中文分词：
    - 连续字母或数字作为一个 token
    - 中文及其它字符按“分字”处理
    - 去除空白与常见标点
    """
    if not isinstance(text, str):
        return []
    s = text.strip()
    if not s:
        return []
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        # 跳过空白与标点
        if ch in _PUNCTS:
            i += 1
            continue
        # 连续字母或数字聚合
        if re.match(r"[A-Za-z0-9]", ch):
            j = i + 1
            while j < len(s) and re.match(r"[A-Za-z0-9]", s[j]):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # 其它字符按“分字”
        tokens.append(ch)
        i += 1
    return tokens


def _ngrams(tokens, n):
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _build_idf(all_refs_list, n_values=(1, 2, 3, 4)):
    """根据所有参考文本构建 n-gram 的 IDF 统计。
    all_refs_list: List[List[str]]，每个样本的参考文本列表
    返回：{n: {ngram_tuple: idf_value}}
    """
    df_maps = {n: {} for n in n_values}
    # 文档频率：以每条参考文本为一个文档进行计数
    doc_count = {n: 0 for n in n_values}
    for refs in all_refs_list:
        if not isinstance(refs, list):
            refs = [] if refs is None else [refs]
        for r in refs:
            tokens = _tokenize_zh(r)
            for n in n_values:
                doc_count[n] += 1
                seen = set(_ngrams(tokens, n))
                for g in seen:
                    df_maps[n][g] = df_maps[n].get(g, 0) + 1
    # 平滑 IDF
    idf_maps = {}
    for n in n_values:
        N = max(1, doc_count[n])
        idf_maps[n] = {g: math.log(N / (c + 1)) for g, c in df_maps[n].items()}
    return idf_maps


def _tf_vec(text, n, idf_map):
    """构建单句的 TF-IDF 向量（字典）。"""
    vec = {}
    tokens = _tokenize_zh(text)
    for g in _ngrams(tokens, n):
        vec[g] = vec.get(g, 0.0) + 1.0
    for k in list(vec.keys()):
        vec[k] *= idf_map.get(k, 0.0)
    return vec


def _avg_ref_vec(refs, n, idf_map):
    """对多参考文本的 TF-IDF 向量求平均。"""
    if not isinstance(refs, list):
        refs = [] if refs is None else [refs]
    if not refs:
        return {}
    acc = {}
    for r in refs:
        v = _tf_vec(r or "", n, idf_map)
        for k, val in v.items():
            acc[k] = acc.get(k, 0.0) + val
    # 平均
    for k in list(acc.keys()):
        acc[k] /= max(1, len(refs))
    return acc


def _cosine(a, b):
    if not a or not b:
        return 0.0
    dot = 0.0
    if len(a) <= len(b):
        for k, va in a.items():
            vb = b.get(k)
            if vb:
                dot += va * vb
    else:
        for k, vb in b.items():
            va = a.get(k)
            if va:
                dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def compute_cider(pred_texts, ref_texts_list, n_values=(1, 2, 3, 4)):
    """
    计算近似 CIDEr 分数：
    - pred_texts: List[str]
    - ref_texts_list: List[List[str] | str | None]，每个样本的参考文本（列表或单字符串或 None）
    - 返回：float，所有样本的平均分
    """
    if not pred_texts or not ref_texts_list:
        return 0.0
    # 构建 IDF（基于所有参考语料）
    idf_maps = _build_idf(ref_texts_list, n_values=n_values)

    scores = []
    for p, refs in zip(pred_texts, ref_texts_list):
        # 对每个 n 计算 TF-IDF 余弦相似度
        sims = []
        for n in n_values:
            idf_map = idf_maps[n]
            vp = _tf_vec(p or "", n, idf_map)
            vr = _avg_ref_vec(refs, n, idf_map)
            sims.append(_cosine(vp, vr))
        # 等权平均 1-4 gram
        scores.append(sum(sims) / len(sims))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
