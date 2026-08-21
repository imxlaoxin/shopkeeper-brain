import json
from pprint import pprint
from typing import List, Dict, Any, Tuple

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = 'vector_search_node'

    def process(self, state: QueryGraphState) -> dict:
        # 1. 参数校验
        rewritten_query, item_names = self._validate_params(state['rewritten_query'], state['item_names'])
        # 2. 获取模型/客户端
        embedding_model, milvus_client = self.get_model_and_client()
        if embedding_model is None or milvus_client is None:
            self.logger.error('获取模型/客户端为空，当前节点将直接返回state')
            return {}
        # 3. 查询向量化
        query_embedding_ret: Dict[str, Any] = self._query_embedding(rewritten_query, embedding_model)
        # 4. 构建过滤条件
        filter_expr, filter_params = self._build_filter_expr(item_names)
        try:
            # 5. 创建混合搜索请求
            self.logger.info('step-5 创建混合搜索请求')
            hybrid_requests = create_hybrid_search_requests(
                dense_vector=query_embedding_ret['dense_vector'],
                sparse_vector=query_embedding_ret['sparse_vector'],
                expr=filter_expr,
                expr_params=filter_params,
                limit=5
            )
            # 6. 执行混合检索
            self.logger.info('step-6 执行混合搜索')
            retrieved_rets = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_requests,
                norm_score=True,
                output_fields=["chunk_id", "content", "item_name", "title"]
            )
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return {}

        if not retrieved_rets or not retrieved_rets[0]:
            self.logger.error('混合搜索结果为空')
            return {}

        # self.logger.info(f'混合搜索结果: {retrieved_rets}')
        self.logger.info(f'混合搜索，长度: {len(retrieved_rets[0])}')

        return {'embedding_chunks': retrieved_rets[0]}

    def _validate_params(self, rewritten_query: str, item_names: List[str]):
        self.logger.info('step-1 参数校验')
        if not rewritten_query.strip():
            raise StateFieldError(self.name, 'rewritten_query', str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(self.name, 'item_names', list)
        return rewritten_query, item_names

    def get_model_and_client(self):
        self.logger.info('step-2 获取模型/客户端')
        try:
            embedding_model = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f'获取模型/客户端失败: {e}')
            return None, None
        return embedding_model, milvus_client

    def _query_embedding(self, rewritten_query: str, embedding_model: BGEM3EmbeddingFunction):
        self.logger.info('step-3 查询向量化')
        hybrid_vector_ret = generate_bge_m3_hybrid_vectors(embedding_model, [rewritten_query], is_query=True)
        return {
            'dense_vector': hybrid_vector_ret.dense_vectors[0],
            'sparse_vector': hybrid_vector_ret.sparse_vectors[0]
        }

    def _build_filter_expr(self, validate_item_names: List[str]) -> Tuple[str, dict]:
        # 使用过滤表达式模板，将动态值从表达式中分离，优化中日韩字符的查询性能
        self.logger.info('step-4 构建过滤条件')
        expr = "item_name in {item_names}"
        filter_params = {"item_names": validate_item_names}
        return expr, filter_params


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))

    """
    [
        {       
            "chunk_id": 467934745184964390,
            "distance": 0.7380340099334717,
            "entity": {
                "chunk_id": 467934745184964390,
                "content": "万用表RS-12的使用\n\n## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![万用表RS-12进行电阻测量时的表笔连接与读数示例](http://192.168.200.3:9000/knowledge-base-files/万用表RS-12的使用/3dce15efe5689c2d8c904dfbbb653eef71ce00b270bd08d47e3b474af2fb68a8.jpg)",
                "item_name": "RS-12 数字万用表",
                "title": "## 电阻测量"
            }
        }
    ]
    """
