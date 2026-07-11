import os
import time
import sys
import json
import asyncio
import token

from dotenv import load_dotenv

from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger



load_dotenv()


# 定义mcp的服务配置
DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

#创建McpServerStreamableHttp
async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 10
        }
    )

    #连接 - 调用 - 关闭
    try:
        await search_mcp.connect()

        #获取工具

        tools = await search_mcp.list_tools()

        for tool in tools:
            print("可用工具：", tool.name)

        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 10
            }
        )
        return result
    finally:
        await search_mcp.cleanup()



def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")


    #获取重写的问题
    query = state["rewritten_query"]

    #调用streamable 网络搜索方法
    result = asyncio.run(mcp_call_streamable(query))




    text = result.content[0].text
    print(f"MCP原始返回：{text}")

    data = json.loads(text)
    web_documents = data.get("pages",[])

    logger.info(f"mcp搜索的结果位: {web_documents}")


    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    time.sleep(1)
    # 调用mcp外部引擎
    print(f"调用外部mcp引擎")

    print("---node-web-search-mcp处理结束---")

    return {"web_search_docs":web_documents}


if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")