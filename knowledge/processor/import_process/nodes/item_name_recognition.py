import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from mcp.server.fastmcp.prompts.base import UserMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognition(BaseNode):
    name = 'item_name_recognitino_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_params(state['file_title'], state['chunks'], self.config.item_name_chunk_k, self.config.item_name_chunk_size)
        # 2. 构建识别上下文
        chunks_context = self._prepare_context(chunks, item_name_chunk_k, item_name_chunk_size)
        # 3. LLM识别商品名
        item_name = self._recognize_item_name(file_title, chunks_context)
        # 4. BGE-M3向量化商品名
        # dense_vector: List[float]
        # sparse_vector: Dict[int, float]
        dense_vector, sparse_vector = self.__bge_m3_embedding(item_name)
        # 5. 存储商品名向量到milvus中
        self._insert_item_name_to_milvus(file_title, item_name, dense_vector, sparse_vector)
        # 6. 回填商品名信息
        self._fill_item_name(item_name, state['chunks'], state)
        # 7. 备份带商品名的chunks.json(看效果)
        self._backup_chunks(state, state['chunks'])
        return state

    def _validate_params(self, file_title, chunks: List[dict], item_name_chunk_k: int, item_name_chunk_size: int) -> Tuple[str, List[dict], int, int]:
        self.log_step('step-1', '参数校验')
        if not file_title.strip():
            raise StateFieldError(self.name, 'file_title', str)
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(self.name, 'chunks', list)
        if not isinstance(item_name_chunk_k, int) or item_name_chunk_k <= 0:
            raise ValidationError('item_name_chunk_k参数值不合理', self.name)
        if not isinstance(item_name_chunk_size, int) or item_name_chunk_size <= 0:
            raise StateFieldError('item_name_chunk_size参数值不合理', self.name)
        self.logger.info('step-2 参数校验通过')
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks: List[dict], item_name_chunk_k: int, item_name_chunk_size: int):
        self.log_step('step-2', '构建识别上下文')
        chunks_context = ''
        contents_chunk = chunks[:item_name_chunk_k]
        for idx, content_chunk in enumerate(contents_chunk):
            candidate_context = f'\n切片-{idx}:\n\n{content_chunk['content']}'
            if len(chunks_context + candidate_context) < item_name_chunk_size:
                chunks_context += candidate_context
        self.logger.info(f'step-2 构建识别上下文完成 返回值 --> chunks_context: {chunks_context[:100]}')
        return chunks_context

    def _recognize_item_name(self, file_title: str, chunks_context: str):
        self.log_step('step-3', 'LLM识别商品名')
        user_prompt: str = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=chunks_context)
        try:
            llm_model = AIClients.get_llm_openai()
            ret = llm_model.invoke([SystemMessage(ITEM_NAME_SYSTEM_PROMPT), HumanMessage(user_prompt)])
        except Exception as e:
            self.logger.error(f'step-3 LLM构建或调用异常，将会降级处理，具体异常信息为: {e}')
            return file_title
        item_name_json_str = ret.content.strip()
        if not item_name_json_str or item_name_json_str == 'UNKNOWN':
            self.logger.warning(f'step-3 LLM识别商品名结果不合理，将会降级处理，具体结果为: {item_name_json_str}')
            return file_title
        self.logger.info(f'step-3 LLM识别商品名完成 返回值 --> {item_name_json_str}')
        item_name_json = json.loads(item_name_json_str)
        return item_name_json['item_name']

    def __bge_m3_embedding(self, item_name: str) -> Tuple[List[float] | None, Dict[int, float] | None]:
        self.log_step('step-4', 'BGE-M3向量化')
        try:
            bge_m3 = AIClients.get_bge_m3_client()

            embeddings = bge_m3.encode_documents([item_name])

            dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
            sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵

            start_idx = sparse_matrix.indptr[0]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
            weights = sparse_matrix.data[start_idx:end_idx].tolist()
            sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]
        except Exception as e:
            self.logger.error(f'step-4 BGE-M3向量化异常，具体异常信息为: {e}')
            return None, None
        self.logger.info(f'step-4 BGE-M3向量化完成 返回值 --> dense_vector: {dense_vector[:10]} sparse_vector: {sparse_vector}')
        return dense_vector, sparse_vector

    def _insert_item_name_to_milvus(self, file_title: str, item_name: str, dense_vector: List[float] | None, sparse_vector: Dict[int, float] | None):
        self.log_step('step-5', '存储商品名向量到Milvus中')
        # 1. 向量有效性检查（为空则跳过，不去创建Milvus连接）
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            raise ValidationError(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整", self.name)
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            raise Exception(f"Milvus 客户端创建失败: {e}")
        try:
            item_name_collection = self.config.item_name_collection
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)
            item_name_data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[item_name_data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f'step-5 存储商品名向量到Milvus中异常，具体异常信息为: {e}')
            raise Exception(f'step-5 存储商品名向量到Milvus中异常，具体异常信息为: {e}')
        self.logger.info(f'step-5 存储商品名向量到Milvus中完成')


    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector",
                              index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector",
                              index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_param
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name: str, chunks: List[dict], state: ImportGraphState):
        self.log_step('step-6', '回填商品名信息')
        for chunk in chunks:
            chunk['item_name'] = item_name  # chunk 级别的回填是给下游模型用的
        state['item_name'] = item_name  # state 级别的回填是给下游程序逻辑用的
        self.logger.info(f'step-6 回填商品名信息完成')

    def _backup_chunks(self, state: ImportGraphState, chunks: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step-7", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.info("未设置 file_dir，跳过备份")
            return

        try:
            local_dir = Path(local_dir, state['file_title'], 'hybrid_auto')
            local_dir.mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(str(local_dir), "chunks_with_item_name.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用\hybrid_auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用",
        "chunks": chunk_content,
        "file_dir": r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir"
    }

    # 3. 实例化节点
    node = ItemNameRecognition()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")


