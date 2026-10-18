import asyncio
import json
import logging
import os

# 排除代理，避免代理导致 500
os.environ['NO_PROXY'] = 'dashscope.aliyuncs.com,aliyuncs.com'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    name = "mcp_search_node"
    """
    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
    mcp形式调用第三方的各种通用的搜索工具。
    灵积服务平台:的通用搜索工具【bailian_web_search】
    """

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建 mcp client 并执行工具 bailian_web_search
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 只更新 state 的 web_search_docs
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        url = self.config.mcp_dashscope_base_url
        token = self.config.openai_api_key

        if not url:
            raise ValueError("config.mcp_dashscope_base_url 未配置")
        if not token:
            raise ValueError("config.openai_api_key 未配置")

        headers = {"Authorization": f"Bearer {token}"}
        timeout = httpx2.Timeout(30.0, read=300.0)

        try:
            logger.info(f"正在连接 MCP 服务：{url}")

            async with httpx2.AsyncClient(headers=headers, timeout=timeout) as http_client:
                transport = streamable_http_client(url, http_client=http_client)

                # mcp 2.x 写法：Client(transport)
                async with Client(transport) as client:
                    tools = await client.list_tools()
                    tool_names = [t.name for t in tools.tools]
                    logger.info(f"可用工具: {tool_names}")

                    if "bailian_web_search" not in tool_names:
                        logger.error("MCP 服务未提供 bailian_web_search 工具")
                        return []

                    execute_tool_result = await client.call_tool(
                        "bailian_web_search",
                        arguments={"query": validated_rewritten_query},
                    )

                    if not execute_tool_result or not execute_tool_result.content:
                        return []

                    text_content_text: str = ""
                    for item in execute_tool_result.content:
                        text = getattr(item, "text", None)
                        if text:
                            text_content_text = text
                            break

                    if not text_content_text:
                        return []

                    try:
                        data: Dict[str, Any] = json.loads(text_content_text)
                        pages = data.get("pages", [])
                        if not pages:
                            return []

                        search_result = []
                        # 原来你传了 count=3，这里等价取前 3 条
                        for page in pages[:3]:
                            snippet = page.get("snippet", "").strip()
                            title = page.get("title", "").strip()
                            url = page.get("url", "").strip()
                            search_result.append({
                                "snippet": snippet,
                                "title": title,
                                "url": url,
                            })
                        return search_result

                    except Exception as e:
                        logger.error(f"反序列化 MCP 结果失败: {e}", exc_info=True)
                        return []

        except Exception as e:
            logger.error(f"MCP 连接或调用失败: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    mcp_search = McpSearchNode()
    result = mcp_search.process(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))