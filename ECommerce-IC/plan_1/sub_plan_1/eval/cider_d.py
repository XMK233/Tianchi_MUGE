import math
from collections import Counter, defaultdict
from typing import List, Dict

from ..utils.zh_tokenizer import zh_basic_tokenize, make_ngrams


class CiderD:
    def __init__(self, refs: Dict[str, List[str]], n_max: int = 4, sigma: float = 6.0):
        self.n_max = n_max
        self.sigma = sigma
        # 构建参考的 DF 与 IDF
        self.idf = [{} for _ in range(n_max)]
        self._build_idf(refs)

    def _build_idf(self, refs: Dict[str, List[str]]):
        # 文档数：参考条目总数
        D = sum(len(v) for v in refs.values())
        df_list = [defaultdict(int) for _ in range(self.n_max)]
        for _, ref_list in refs.items():
            for ref in ref_list:
                tokens = zh_basic_tokenize(ref)
                for n in range(1, self.n_max+1):
                    ngrams = set(make_ngrams(tokens, n))
                    for g in ngrams:
                        df_list[n-1][g] += 1
        for n in range(1, self.n_max+1):
            idf_n = {}
            for g, df in df_list[n-1].items():
                idf_n[g] = math.log(max(1, D) / max(1, df))
            self.idf[n-1] = idf_n

    def _tf_vec(self, text: str, n: int):
        tokens = zh_basic_tokenize(text)
        grams = make_ngrams(tokens, n)
        cnt = Counter(grams)
        total = float(sum(cnt.values())) or 1.0
        tf = {g: c/total for g, c in cnt.items()}
        return tf

    def _tfidf_vec(self, text: str, n: int):
        tf = self._tf_vec(text, n)
        idf = self.idf[n-1]
        return {g: tf[g] * idf.get(g, 0.0) for g in tf}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]):
        inter = set(a.keys()) & set(b.keys())
        num = sum(a[k]*b[k] for k in inter)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)

    def _gaussian_len_pen(self, cand: str, ref: str):
        lc = len(zh_basic_tokenize(cand))
        lr = len(zh_basic_tokenize(ref))
        diff = lc - lr
        return math.exp(-(diff*diff)/(2*self.sigma*self.sigma))

    def score_one(self, cand: str, refs: List[str]) -> float:
        sims = []
        for ref in refs:
            sim_n = []
            for n in range(1, self.n_max+1):
                vc = self._tfidf_vec(cand, n)
                vr = self._tfidf_vec(ref, n)
                sim_n.append(self._cosine(vc, vr))
            mean_sim = sum(sim_n) / len(sim_n)
            mean_sim *= self._gaussian_len_pen(cand, ref)
            sims.append(mean_sim)
        return sum(sims) / max(1, len(sims))

