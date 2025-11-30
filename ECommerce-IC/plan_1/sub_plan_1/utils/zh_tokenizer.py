import re


def zh_basic_tokenize(text: str):
    if not text:
        return []
    # 中文逐字，连续字母/数字合并为一个 token
    tokens = []
    buf = []
    def flush_buf():
        if buf:
            tokens.append("".join(buf))
            buf.clear()
    for ch in text:
        if re.match(r"[A-Za-z0-9]", ch):
            buf.append(ch)
        else:
            flush_buf()
            if not ch.isspace():
                tokens.append(ch)
    flush_buf()
    return tokens


def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

