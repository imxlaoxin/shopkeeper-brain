import time

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 加载模型
bge_m3 = BGEM3EmbeddingFunction(
    # model_name="BAAI/bge-m3",
    model_name=r"C:\Users\slkh\.cache\modelscope\models\BAAI\bge-m3",
    device="cuda:0",
    use_fp16=True,
)

# 2. 生成嵌入
start_time = time.time()
embeddings = bge_m3.encode_documents(["RS-12 数字万用表"])

# 3. 提取向量
dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵

# 4. 从 CSR 矩阵提取稀疏向量
start_idx = sparse_matrix.indptr[0]
end_idx = sparse_matrix.indptr[1]
token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
weights = sparse_matrix.data[start_idx:end_idx].tolist()
sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]

print(start_idx)
print(end_idx)
print(token_ids)
print(weights)
print(sparse_vector)
end_time = time.time()
print(f"耗时: {end_time - start_time} 秒")

"""
0
7
[6, 1173, 3895, 5873, 9955, 18912, 28406]
[0.00847625732421875, 0.1339111328125, 0.1796875, 0.2437744140625, 0.26416015625, 0.2181396484375, 0.1632080078125]
{6: 0.00847625732421875, 1173: 0.1339111328125, 3895: 0.1796875, 5873: 0.2437744140625, 9955: 0.26416015625, 18912: 0.2181396484375, 28406: 0.1632080078125}
耗时: 2.985412359237671 秒
"""