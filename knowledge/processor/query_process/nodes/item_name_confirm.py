import json
import re
from dataclasses import dataclass, field
from json import JSONDecodeError
from logging import Logger
from typing import List, Any, Dict, Tuple

from langchain_core.messages import HumanMessage
from pymilvus import AnnSearchRequest
from pymilvus.client.search_result import HybridHits, Hit

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE, ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors, HybridVectorsRet
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


@dataclass
class ItemNameExtractRet:
    item_names: List[str] = field(default_factory=list)
    rewritten_query: str = ''


@dataclass
class MatchItem:
    score: float
    retrieved_item_name: str


@dataclass
class MatchVectorItem:
    llm_respond_item_name: str
    matches: List[MatchItem]


type MatchVectorFuncRet = List[MatchVectorItem]


class ItemNameExtractor:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def extract_item_names(self, user_query: str, history_query: str) -> ItemNameExtractRet:
        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=user_query,
                                                        history_text=history_query if history_query else '暂无上下文')
        system_prompt = ITEM_NAME_EXTRACT_SYSTEM_PROMPT

        extracted_ret = ItemNameExtractRet()  # 默认值，rewritten_query是为了消除指代问题.

        try:
            llm_model = AIClients.get_llm_openai()
            if llm_model is None:
                self.logger.warning('LLM客户端获取为空，将返回extract_item_names默认值')
                return extracted_ret
        except Exception as e:
            self.logger.warning(f'获取LLM客户端失败，将返回extract_item_names默认值，具体异常信息如下: \n{e}\n')
            return extracted_ret
        try:
            response = llm_model.invoke([system_prompt, user_prompt])
        except Exception as e:
            self.logger.warning(f'LLM调用失败，将返回extract_item_names默认值，具体异常信息如下: \n{e}\n')
            return extracted_ret
        llm_content = response.content.strip()

        # 4. 判断 LLM 的输出
        if not llm_content.strip():
            self.logger.warning('LLM的输出为空，将返回extract_item_names默认值')
            return extracted_ret

        try:
            parsed_result = self._clean_parse(llm_content)
        except Exception as e:
            self.logger.warning(f'LLM的输出解析失败，将返回extract_item_names默认值，具体异常信息如下: \n{e}\n')
            return extracted_ret
        extracted_ret.item_names = parsed_result.get('item_names', [])
        extracted_ret.rewritten_query = parsed_result.get('rewritten_query', '')
        self.logger.info(f'LLM的输出解析结果为：{parsed_result}')
        return extracted_ret

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def match_align_filter(self, item_names: List[str], collection_name: str, dense_weight: float, sparse_weight: float,
                           max_option: int, high_confidence: float, mid_confidence: float) -> Tuple[
        List[str], List[str]]:
        # 向量匹配
        match_vector_ret: MatchVectorFuncRet = self._match_vector(item_names, collection_name, dense_weight,
                                                                  sparse_weight)
        # 评分对齐
        confirmed, options = self._item_name_score_align(match_vector_ret, high_confidence, mid_confidence, max_option)
        # 分数差过滤
        confirmed = self._item_name_score_filter(confirmed, match_vector_ret)

        return confirmed, options

    def _match_vector(self, item_names: List[str], collection_name: str, dense_weight: float, sparse_weight: float) -> \
            MatchVectorFuncRet:
        self.logger.info('[step-3] 向量匹配')
        match_result = []
        if not item_names:
            return match_result
        try:
            embedding_model = AIClients.get_bge_m3_client()
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.warning(f"Milvus或BGE-M3客户端创建失败，将返回空列表: {e}")
            return match_result
        vectors_ret: HybridVectorsRet = generate_bge_m3_hybrid_vectors(embedding_model, item_names, True)
        for dense_vector, sparse_vector, item_name in zip(vectors_ret.dense_vectors, vectors_ret.sparse_vectors,
                                                          item_names):
            hybrid_reqs: List[AnnSearchRequest] = create_hybrid_search_requests(dense_vector=dense_vector,
                                                                                sparse_vector=sparse_vector,
                                                                                limit=5)
            hybrid_search_ret = execute_hybrid_search_query(milvus_client, collection_name=collection_name,
                                                            search_requests=hybrid_reqs,
                                                            ranker_weights=(dense_weight, sparse_weight),
                                                            norm_score=True, limit=5,
                                                            output_fields=['item_name'], search_params=None)
            hybrid_hits: HybridHits = hybrid_search_ret[0]
            match_vector_item = MatchVectorItem(llm_respond_item_name=item_name, matches=[])
            for hybrid_hit in hybrid_hits:
                match_vector_item.matches.append(MatchItem(
                    score=hybrid_hit.distance,
                    retrieved_item_name=hybrid_hit.entity.item_name
                ))
            match_result.append(match_vector_item)
        self.logger.info(f'向量匹配结果为：{match_result}')
        return match_result

    def _item_name_score_align(self, match_vector_ret: MatchVectorFuncRet, high_confidence, mid_confidence, max_option):
        self.logger.info('[step-4] 评分对齐')
        # 1. 定义两个容器
        confirmed = []
        options = []
        # 存储 tuple: (score, item_name) 以便在返回时按分数全局排序
        options_with_score = []

        # 2. 遍历向量数据库查询到的所有结果
        for item_name_search_result in match_vector_ret:
            # 2.0 获取 LLM 提取的商品名
            llm_respond_item_name = item_name_search_result.llm_respond_item_name

            # 2.1 对某一商品名下找到相似的 item_name 按分数值降序
            matches: List[MatchItem] = sorted(
                item_name_search_result.matches,
                key=lambda item: item.score,
                reverse=True
            )

            # 2.2 获取 matches 中分数值能进入 confirmed 容器阈值的对象
            high_items: List[MatchItem] = [m for m in matches if m.score >= high_confidence]

            # 3. 询问是否能进入 confirmed 中
            # 对于“同一个提取出的商品名（即外层循环的单次迭代）”来说，只要存在 high_items，就不会走 mid_items 对应的逻辑
            if high_items:
                # 3.1 准备找最精准的那一个
                accurate_match_item: MatchItem = next(
                    (high_item for high_item in high_items if
                     str(high_item.retrieved_item_name) == llm_respond_item_name),
                    None
                )

                # 场景 A: 找到了精确匹配
                if accurate_match_item:
                    picked = accurate_match_item.retrieved_item_name
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 B: 只有一条高置信结果
                elif len(high_items) == 1:
                    picked = high_items[0].retrieved_item_name
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 C: 多条相似，加入 options 让用户选择
                else:
                    for h in high_items[:3]:
                        picked = h.retrieved_item_name
                        # if picked not in options and picked not in confirmed:
                        #     options.append(picked)
                        # 避免与 confirmed 冲突以及 options 内部去重
                        if picked not in confirmed and not any(opt[1] == picked for opt in options_with_score):
                            options_with_score.append((h.score, picked))

            # 4. 询问是否能进入 options 中
            else:
                mid_items = [
                    m for m in matches
                    if m.score >= mid_confidence
                       and m.retrieved_item_name not in confirmed
                       and not any(opt[1] == m.retrieved_item_name for opt in options_with_score)
                ]

                if mid_items:
                    for m in mid_items:
                        options_with_score.append((m.score, m.retrieved_item_name))

        # 全局按 score 降序排序并截取
        options_with_score.sort(key=lambda x: x[0], reverse=True)

        # 提取排名前 max_option 的名字
        options = [item[1] for item in options_with_score[:max_option]]

        self.logger.info(f'[step-4] 评分对齐结果为 --> \nconfirmed: {confirmed} \noptions: {options}')
        return confirmed, options

    def _item_name_score_filter(self, confirmed: List[str], match_vector_ret: MatchVectorFuncRet) -> List[str]:
        """
           按单个提炼实体分组过滤:
              每个 LLM 提取出的商品词各自计算各自的最大分，各自过滤低于阈值的 Candidate。

           Args:
               confirmed: 初步确认的商品名列表
               search_results: 向量检索结果

           Returns:
               过滤后的 confirmed 列表
           """
        self.logger.info('[step-5] 评分过滤')
        if len(confirmed) == 1:
            self.logger.info(f'[step-5] confirmed长度为1，直接返回结果为 --> {confirmed}')
            return confirmed
        filter_confirmed = set()

        # 按每个 LLM 提取项分别做过滤，避免跨商品比较
        for search_result in match_vector_ret:
            # 获取当前提取词对应的匹配项，且只保留在 confirm 里的
            valid_matches = [m for m in search_result.matches if m.retrieved_item_name in confirmed]
            if not valid_matches:
                continue

            # 找出当前提取项匹配到的最高分
            current_max_score = max(m.score for m in valid_matches)

            # 只有相对当前提取项最高分差值在 0.15 以内的才保留
            for m in valid_matches:
                if current_max_score - m.score <= 0.15:
                    filter_confirmed.add(m.retrieved_item_name)

        # 保持原来在 confirm 中的顺序返回
        final_confirmed = [item for item in confirmed if item in filter_confirmed]
        self.logger.info(f'[step-5] 评分过滤结果为 --> {final_confirmed}')
        return final_confirmed


class ItemNameConfirmNode(BaseNode):
    name = 'item_name_confirm_node'

    def __init__(self):
        super().__init__()
        self.item_name_extractor = ItemNameExtractor(self.logger, self.name)
        self.item_name_aligner = ItemNameAligner(self.logger, self.name)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取历史对话
        self.log_step('step-1', '获取历史对话')
        session_id = state['session_id']
        user_query = state['original_query']
        history_messages: List = self._get_recent_messages(session_id)
        history_query = "".join(
            [f'{history_message["role"]}: {history_message["text"]}\n' for history_message in history_messages])
        # 2. LLM提取商品名
        self.log_step('step-2', 'LLM提取商品名')
        extracted_ret: ItemNameExtractRet = self.item_name_extractor.extract_item_names(user_query, history_query)
        # 3. 向量匹配
        # 4. 评分对齐
        # 5. 分数差异过滤
        confirmed, options = self.item_name_aligner.match_align_filter(extracted_ret.item_names,
                                                                       self.config.item_name_collection,
                                                                       self.config.item_name_dense_weight,
                                                                       self.config.item_name_sparse_weight,
                                                                       self.config.item_name_max_options,
                                                                       self.config.item_name_high_confidence,
                                                                       self.config.item_name_mid_confidence)
        # 6. 决策更新状态
        self._decide_policy(state, confirmed, options, extracted_ret.rewritten_query)
        # 7. 历史回填
        for history_msg in history_messages:
            history_msg['item_names'] = confirmed
        state['history'] = history_messages
        return state

    def _get_recent_messages(self, session_id, limit=10) -> List:
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id, limit)
        if not history_messages:
            self.logger.info(f'当前会话: {session_id}，没有历史对话')
            return []
        self.logger.info(f'当前会话: {session_id}，历史对话：\n' + str(history_messages))
        return history_messages

    def _decide_policy(self, state: QueryGraphState, confirmed: List[str], options: List[str], rewritten_query: str):
        """根据对齐结果更新 state"""
        # TODO 待优化
        self.logger.info(f'[step-6] 决策更新状态')
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options:
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
            state['options'] = options
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    setup_logging()

    test_state: QueryGraphState = {
        "session_id": "test_session_id",
        # "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
        "original_query": "苹果手机怎么样？"
        # "original_query": "RS-12 数字万用表怎么测试电阻？以及MateStation S 12代 用户手册 中包含操作环境嘛？"
    }

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)

    node_item_name_confirm.logger.info(json.dumps(result, ensure_ascii=False, indent=4))
