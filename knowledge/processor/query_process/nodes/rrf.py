from typing import List, Dict, Any, Tuple
from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = 'rrf_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        vec_retrieved_chunks, hyde_retrieved_chunks, rrk_k, rrf_max_results = self._validate_params(
            state['embedding_chunks'], state['hyde_embedding_chunks'], self.config.rrf_k,
            self.config.rrf_max_results)
        # 2. chunks规整化
        normalized_vec_retrieved_chunks = self.normalize_chunks(vec_retrieved_chunks)
        normalized_hyde_retrieved_chunks = self.normalize_chunks(hyde_retrieved_chunks)
        # 3. rrf倒数排名融合
        chunks_with_weight = [(normalized_vec_retrieved_chunks, 1.0), (normalized_hyde_retrieved_chunks, 1.0)]
        rrf_chunks_with_score: List[Tuple[Dict[str, Any], float]] = self._rrf(chunks_with_weight, rrk_k, rrf_max_results)
        rrf_chunk = [chunk for chunk, score in rrf_chunks_with_score]
        state['rrf_chunks'] = rrf_chunk
        return state

    def _validate_params(self, vec_retrieved_chunks: List[Dict[str, Any]], hyde_retrieved_chunks: List[Dict[str, Any]],
                         rrk_k: float, rrf_max_results: int):
        self.log_step('step-1', '参数校验')
        if not vec_retrieved_chunks or not hyde_retrieved_chunks:
            raise ValueError('vec_retrieved_chunks or hyde_retrieved_chunks is empty')
        if rrk_k <= 0 or rrf_max_results <= 0:
            raise ValueError('rrk_k or rrf_max_results is invalid')
        return vec_retrieved_chunks, hyde_retrieved_chunks, rrk_k, rrf_max_results

    def normalize_chunks(self, retrieved_chunks: List[Dict[str, Any]]):
        self.log_step('step-2', 'chunks规整化')
        normalized_retrieved_chunks = [retrieved_chunk['entity'] for retrieved_chunk in retrieved_chunks if
                                       retrieved_chunk['entity']]
        return normalized_retrieved_chunks

    def _rrf(self, chunks_with_weight: List[Tuple[List[Dict[str, Any]], float]], rrk_k: float, rrf_max_results: int) -> List[Tuple[Dict[str, Any], float]]:
        self.log_step('step-3', 'rrf倒数融合排序')
        chunks_chunk_id_score_map = {}  # {'chunk_id': 'score'}
        chunks_chunk_id_chunk_map = {}  # {'chunk_id': 'chunk'}
        for chunks, weight in chunks_with_weight:
            for rank, chunk in enumerate(chunks, 1):
                cur_chunk_id = chunk['chunk_id']
                cur_chunk_score = weight / (rrk_k + rank)
                chunks_chunk_id_score_map[cur_chunk_id] = chunks_chunk_id_score_map.get(cur_chunk_id,
                                                                                        0.0) + cur_chunk_score
                chunks_chunk_id_chunk_map.setdefault(cur_chunk_id, chunk)  # setdefault: 如果 key 不存在，则添加 key，不会进行更新覆盖
        rrf_chunks_with_score = [(chunks_chunk_id_chunk_map[chunk_id], chunk_score) for chunk_id, chunk_score  in chunks_chunk_id_score_map.items()]
        rrf_chunks_with_score = sorted(rrf_chunks_with_score, key=lambda x: x[1], reverse=True)[:rrf_max_results]
        # self.logger.info(f"RRF 融合结果: {len(rrf_chunks_with_score)} 条，展示前 5 条: {rrf_chunks_with_score[:5]}")
        self.logger.info(f"RRF 融合结果: {len(rrf_chunks_with_score)} 条")
        return rrf_chunks_with_score


if __name__ == "__main__":
    setup_logging()
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
