# -*- coding: utf-8 -*-
"""
粤语 -> 普通话 机器翻译 BERTScore 评测脚本
支持批量评测、自动计算指标、结果保存到文件
"""

# ===================== 1. 安装并导入依赖 =====================
import subprocess
import sys

# 自动安装依赖（首次运行执行）
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# 核心依赖：bert-score + torch
#try:
#    from bert_score import score
#except ImportError:
#    print("正在安装 bert-score 和 torch...")
#    install_package("bert-score")
#    install_package("torch")


from bert_score import score

import pandas as pd

# ===================== 2. 配置评测数据 =====================
# 【修改这里】你的评测数据
# refs: 标准参考译文（人工正确翻译的普通话）
# cands: 机器翻译生成的普通话译文
# cantonese: 原始粤语句子（仅用于展示，不参与计算）

file = r"C:\Users\one\Desktop\实验结果\6号模型qwen3-8b-词典lora加上simi.txt"
file1 = open(file,"r",encoding="utf8")

reference_translations = []
machine_translations = []

for line in file1.readlines():
    line_list = line.strip().split("\t")
    if len(line_list) != 4:
        continue
    true = line_list[1]
    pred = line_list[2]

    if len(true) > 500:
        true = true[:500]
    if len(pred) > 500:
        pred = pred[:500]

    reference_translations.append(true)
    machine_translations.append(pred)

'''
reference_translations = [
    "今天天气很好，我想去逛街。",
    "他做的饭很好吃。",
    "麻烦借过一下，我要下车。",
    "这本书很有趣，你看过吗？"
]
'''

'''
machine_translations = [
    "今天天气不错，我想出去逛街。",       # 机器翻译结果1
    "他煮的饭非常美味。",                 # 机器翻译结果2
    "不好意思让一下，我要下车。",         # 机器翻译结果3
    "这本书特别有意思，你有没有看过？"    # 机器翻译结果4
]
'''

# ===================== 3. BERTScore 核心评测 =====================
print("="*60)
print("正在使用 BERTScore 评测 粤语→普通话 翻译效果...")
print(f"评测模型：bert-base-chinese (官方中文预训练模型)")
print(f"评测句子数量：{len(machine_translations)}")
print("="*60)

# 调用 BERTScore 计算指标
# lang='zh' 自动加载中文bert模型，适合粤语+普通话
P, R, F1 = score(
    cands=machine_translations,  # 机器翻译结果
    refs=reference_translations, # 人工参考译文
    lang="zh",                   # 中文/粤语适配
    verbose=True,                # 显示加载进度
    rescale_with_baseline=True,   # 标准化到 0~1 区间（更易读）
    model_type = r"bert-base-chinese",
    #rescale_baseline=False, #关键：关闭长度缩放，避免超长报错
    #max_length=512,
    device="cuda"
    
)

# ===================== 4. 输出详细结果 =====================
#print("\n" + "="*30 + " 单句评测结果 " + "="*30)
#for i in range(len(cantonese_sentences)):
#    print(f"\n【句子 {i+1}】")
#    print(f"粤语原文：{cantonese_sentences[i]}")
#    print(f"参考普通话：{reference_translations[i]}")
#    print(f"机器翻译：{machine_translations[i]}")
#    print(f"BERTScore | P: {P[i].item():.4f} | R: {R[i].item():.4f} | F1: {F1[i].item():.4f}")

# ===================== 5. 输出整体平均指标 =====================
avg_P = P.mean().item()
avg_R = R.mean().item()
avg_F1 = F1.mean().item()

print("\n" + "="*60)
print("【整体评测汇总】")
print(f"平均精确率 (Precision)：{avg_P:.4f}")
print(f"平均召回率 (Recall)：{avg_R:.4f}")
print(f"平均F1得分 (F1)：{avg_F1:.4f}")
print("="*60)

