"""HyDE 检索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""

import json
import logging
from typing import List, Tuple, Union, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点

    流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行 HyDE 检索

        Args:
            state: 需包含 rewritten_query 和 item_names

        Returns:
            {"hyde_embedding_chunks": [...]} 搜索结果列表
        """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus 客户端
        self.log_step("step-3", "获取嵌入模型 & milvus 客户端")
        embedding_model = AIClients.get_bge_m3_client()
        milvus_client = StorageClients.get_milvus_client()
        if not embedding_model or not milvus_client:
            self.logger.warning("获取嵌入模型或 milvus 客户端失败")
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        self.log_step("step-4", "假设性文档嵌入")
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_bge_m3_hybrid_vectors(
            embedding_model,
            embedding_documents=[embedding_document]
        )

        if not embedding_result:
            self.logger.warning("向量化结果为空，返回原state")
            return state

        # 5. 获取 item_name 的过滤表达式
        self.log_step("step-5", "获取 item_name 的过滤表达式")
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        self.log_step("step-6", "创建混合搜索请求")
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result.dense_vectors[0],
            sparse_vector=embedding_result.sparse_vectors[0],
            expr=item_name_filtered_expr
        )

        # 7. 执行混合搜索请求
        self.log_step("step-7", "执行混合搜索请求")
        reps = execute_hybrid_search_query(
            milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id", "content", "item_name", 'title']
        )

        if not reps or not reps[0]:
            self.logger.warning("混合搜索结果为空，返回原state")
            return state
        # self.logger.info(f'混合搜索结果: {reps}')
        self.logger.info(f'HyDE混合搜索，长度: {len(reps[0])}')

        # 8. 只更新 hyde_embedding_chunks
        return {"hyde_embedding_chunks": reps[0]}

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        """使用 LLM 生成假设性文档"""
        self.log_step("step-2", "LLM生成假设性文档")
        # 1. 获取 LLM 客户端
        llm_client = AIClients.get_llm_openai(False)

        # 2. 判断
        if llm_client is None:
            self.logger.warning("获取 LLM 模型失败，将返回空字符串")
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=validate_item_names,
            rewritten_query=validated_query
        )
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            # 4. 获取 AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                self.logger.warning("LLM返回空字符串")
                return ""

            self.logger.info(f"LLM返回内容: {llm_response_content}")
            return llm_response_content

        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}，返回空字符串")
            return ""

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        self.log_step("step-1", "参数校验")
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        return rewritten_query, item_names

    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        """构建商品名过滤表达式"""
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    for r in result.get('hyde_embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))

    """
    [
        {
            "chunk_id": 467934745184964388,
            "distance": 0.8129572868347168,
            "entity": {
                "item_name": "RS-12 数字万用表",
                "title": "## 直流电压测量",
                "chunk_id": 467934745184964388,
                "content": "万用表RS-12的使用\n\n## 直流电压测量\n\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n## 交流电压测量\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量时表笔正确连接被测电路的示意图](http://192.168.200.3:9000/knowledge-base-files/万用表RS-12的使用/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。"
            }
        }
    ]
    """