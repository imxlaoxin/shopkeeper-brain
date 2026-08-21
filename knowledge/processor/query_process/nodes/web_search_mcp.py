import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    """
    MCP 网络搜索节点
       负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
       通过 MCP 协议调用灵积平台的通用搜索工具（bailian_web_search）。
   """
    name = 'web_search_mcp_node'

    def process(self, state: QueryGraphState) -> dict:
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)
        # 2. 执行 web_mcp_search（asyncio.run 桥接同步→异步）
        mcp_result = asyncio.run(self._execute_web_mcp_search(validated_query))
        if not mcp_result:
            return {}
        # 3. 更新 state 中的web_search_docs
        return {'web_search_docs': mcp_result}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数。

        Args:
            state: 查询图状态。

        Returns:
            校验后的 rewritten_query 和 item_names。

        Raises:
            StateFieldError: 参数校验失败时抛出。
        """
        # 1. 获取参数
        self.log_step('step-1', '参数校验')
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        # 3. 返回
        return rewritten_query, item_names

    async def _execute_web_mcp_search(self, validated_query: str):
        """异步执行 MCP 网络搜索。

        使用 async with 上下文管理器自动管理连接生命周期：
        - 进入 with 块时自动建立连接（connect）
        - 退出 with 块时自动关闭连接（cleanup），即使抛异常也会执行

        Args:
            validated_query: 校验后的查询文本。

        Returns:
            搜索结果列表，每项包含 title、url、snippet。
        """
        # 1. 定义 MCP 客户端（Streamable HTTP 模式，自动管理连接）
        self.log_step('step-2', 'MCP 网络搜索')
        async with MCPServerStreamableHttp(
                name="search_mcp",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                    "timeout": 300,
                    "terminate_on_close": True,
                },
                max_retry_attempts=2,
                cache_tools_list=True,
        ) as client:
            # 2. 调用工具
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_query, "count": 3}
            )

            # 3. 解析工具执行完的结果
            # 3.1 获取最外层的对象
            if not execute_tool_result:
                return []
            # 3.2 获取对象的 content 属性
            if not execute_tool_result.content[0]:
                return []
            # 3.3 获取 TextContent 对象的 text
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                return []
            # 3.4 反序列化
            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                # a) 获取 pages
                pages = text_content_text.get('pages', "")
                if not pages:
                    return []
                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})
                # c) 最终返回
                return search_result
            except JSONDecodeError as e:
                self.logger.error(
                    f"反序列MCP结果失败信息 {str(e.msg)} 原文{e.doc} 位置{e.pos}"
                )
                return []


if __name__ == '__main__':
    setup_logging()
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search.process(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))

    """
    [
        {
            "snippet": "2026-07-27 16:08:10 已收盘 更新实时行情 进入我的 level2 >> 查看碎股行情 沪港通资金流向 沪股通净流入520亿 港股通(沪)净流入--亿 深港通资金流向 深股通净流入520亿 港股通(深)净流入420亿 昨收盘 26.720 最高价 29.080 振幅 8.383% 价格档 0.020 今开盘 26.900 最低价 26.840 市盈率",
            "title": "小米集团-W",
            "url": "https://stock.sina.com.cn/hkstock/quotes/01810.html"
        }
    ]
    """
