import json
import os
from pathlib import Path
from typing import List

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BGEEmbedding(BaseNode):
    name = 'bge_chunks_embedding_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        chunks, embedding_batch_size = self._validate_params(state['chunks'], self.config.embedding_batch_size)
        # 2. 分批循环处理切片
        self.log_step('step-2', '分批循环处理切片')
        total_length = len(chunks)
        updated_chunks: List[dict] = []  # 收集批次嵌入后的切片
        for cur_start_idx in range(0, total_length, embedding_batch_size):
            cur_last_idx = cur_start_idx + embedding_batch_size
            if cur_last_idx > total_length:
                cur_last_idx = total_length
            self.logger.info(f'处理切片范围: {cur_start_idx} 到 {cur_last_idx}')
            batch_chunks = chunks[cur_start_idx:cur_last_idx]
            updated_batch_chunks = self._bge_embedding(batch_chunks)
            updated_chunks.extend(updated_batch_chunks)

        # 4. 备份切片(看效果)
        self._backup_chunks(state, updated_chunks)

        # 5. 合并所有批次结果
        state['chunks'] = updated_chunks

        return state

    def _validate_params(self, chunks: List[dict], embedding_batch_size: int):
        self.log_step('step-1', '参数校验')
        if not chunks or not isinstance(chunks, list):
            self.logger.error('切片集合为空或不为列表类型')
            raise ValueError('切片集合为空或不为列表类型')
        if embedding_batch_size < 1:
            self.logger.error('embedding_batch_size 必须大于等于 1')
            raise ValueError('embedding_batch_size 必须大于等于 1')
        self.logger.info('step-1 结束执行')
        return chunks, embedding_batch_size

    def _bge_embedding(self, batch_chunks: List[dict]) -> List[dict]:
        self.log_step('step-3', '批量生成切片嵌入')
        batch_chunks_embedding_contents = [f'{chunk['item_name']}\n{chunk['content']}' for chunk in batch_chunks]

        try:
            bge_m3_model = AIClients.get_bge_m3_client()
        except Exception as e:
            raise EmbeddingError(f'初始化BGE M3模型异常，具体异常信息如下：{e}', self.name)

        try:
            embeddings = bge_m3_model.encode_documents(batch_chunks_embedding_contents)
            if not embeddings:
                self.logger.warning("嵌入后的结果不存在，返回原始切片...")
                return batch_chunks
        except Exception as e:
            self.logger.warning(f'生成切片嵌入异常，返回原始切片...具体异常信息如下：{e}', self.name)
            return batch_chunks
        for vec_idx in range(len(batch_chunks)):
            cur_dense_vector = embeddings["dense"][vec_idx].tolist()  # List[float], 长度 1024
            cur_sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵

            start_idx = cur_sparse_matrix.indptr[vec_idx]
            end_idx = cur_sparse_matrix.indptr[vec_idx + 1]
            token_ids = cur_sparse_matrix.indices[start_idx:end_idx].tolist()
            weights = cur_sparse_matrix.data[start_idx:end_idx].tolist()
            cur_sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]
            batch_chunks[vec_idx]['dense_vector'] = cur_dense_vector
            batch_chunks[vec_idx]['sparse_vector'] = cur_sparse_vector
        self.logger.info('step-3 结束执行')
        return batch_chunks

    def _backup_chunks(self, state: ImportGraphState, chunks: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step-4", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.info("未设置 file_dir，跳过备份")
            return

        try:
            # local_dir = Path(local_dir, state['file_title'], 'hybrid_auto')
            local_dir = Path(local_dir, state['file_title'], 'auto')
            local_dir.mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(str(local_dir), "chunks_with_vector.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用\hybrid_auto\chunks_with_item_name.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用",
        "chunks": chunk_content,
        "file_dir": r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir"
    }

    # 3. 实例化节点
    bge_node = BGEEmbedding()

    # 4. 调用process
    processed_state = bge_node.process(state)

    # bge_node.logger.info(json.dumps(processed_state, ensure_ascii=False, indent=4))