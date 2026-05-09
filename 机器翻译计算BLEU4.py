

import math
from collections import Counter

def get_ngrams(text, n):
    """生成文本的n-gram列表"""
    return [tuple(text[i:i+n]) for i in range(len(text)-n+1)]

def calculate_bleu_4(candidate, reference):
    """
    计算候选翻译相对于参考翻译的BLEU-4分数
    :param candidate: 候选翻译（字符串或字符列表）
    :param reference: 参考翻译（字符串或字符列表）
    :return: BLEU-4分数（浮点数）
    """
    # 将输入转换为字符列表（适用于中文字符）
    cand_chars = list(candidate)
    ref_chars = list(reference)
    
    # 计算长度惩罚
    cand_len = len(cand_chars)
    ref_len = len(ref_chars)
    if cand_len == 0:
        return 0.0
    brevity_penalty = math.exp(1 - ref_len / cand_len) if cand_len < ref_len else 1.0

    # 计算1-gram到4-gram的精度
    p_n = []  # 存储各n-gram的精度
    for n in range(1, 5):
        cand_ngrams = get_ngrams(cand_chars, n)
        ref_ngrams = get_ngrams(ref_chars, n)
        
        if not cand_ngrams:
            # 避免除零错误
            p_n.append(0)
            continue
            
        # 统计候选n-gram在参考中的出现次数（使用计数截断）
        count = Counter(cand_ngrams)
        max_ref_count = Counter(ref_ngrams)
        clipped_count = {}
        for gram in cand_ngrams:
            clipped_count[gram] = min(count[gram], max_ref_count.get(gram, 0))
        
        # 计算精度
        total_clipped = sum(clipped_count.values())
        p_n.append(total_clipped / len(cand_ngrams))

    # 计算几何平均精度
    if min(p_n) == 0:
        return 0.0
    geometric_mean = math.exp(sum(math.log(p) for p in p_n) / 4)

    # 计算BLEU-4分数
    bleu4 = brevity_penalty * geometric_mean
    return bleu4

# 示例用法

# 示例用法
file = r"一共三列，第一列是粤方言原始文本，第二列是标准普通话翻译，第三列是模型预测的普通话翻译.txt"
file1 = open(file,"r",encoding="utf8")

all_bleu = 0
num = 0
for line in file1.readlines():
    num = num + 1
    line_list = line.strip().split("\t")
    if len(line_list) < 3:
        continue
    truth = line_list[1]
    pred = line_list[2]
    #bleu_score = calculate_bleu_4(candidate_translation, reference_translation)
    bleu_score = calculate_bleu_4(pred, truth)
    #print(truth,"  ",pred,"  ",bleu_score)
    all_bleu = all_bleu + bleu_score

all_bleu = all_bleu / num
print(all_bleu)
