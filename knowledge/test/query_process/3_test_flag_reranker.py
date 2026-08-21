from FlagEmbedding import FlagReranker

# 使用交叉编码器进行相关性得分。
# reranker = FlagReranker(
#     #model_name_or_path="BAAI/bge-reranker-large",
#     model_name_or_path="C:\\Users\\slkh\\.cache\\modelscope\\models\\BAAI\\bge-reranker-large",
#     device="cuda:0",      # GPU 加速
#     use_fp16=True       # 半精度推理
# )


reranker = FlagReranker(
    model_name_or_path="C:\\Users\\slkh\\.cache\\modelscope\\models\\BAAI\\bge-reranker-large",
    device="cpu",
    use_fp16=False
)

# 计算相关性得分
#  给Reranker模型计算相关性得分 ：[CLS]什么是万用表？[SEP]万用表是一种测量电压、电流、电阻的仪器[SEP]
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"]
]
# scores = reranker.compute_score(pairs)
scores = reranker.compute_score(pairs, normalize=True)
print(f'scores: {scores}')

"""
    scores: [7.8984375, -9.484375]    gpu 半精度
    scores: [7.895776271820068, -9.480398178100586]   cpu 全精度
    归一化后:
    scores: [0.9996278257931412, 7.632771039613732e-05]
"""

