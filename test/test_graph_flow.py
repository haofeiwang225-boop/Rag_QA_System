import json

from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.core.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样？"
    )
final_state = None

# 若 query_app 存在，则执行流式遍历
# for event in query_app.stream(initial_state):
#     for key, value in event.items():
#         logger.info(f"节点：{key}")
#         final_state = value

# 演示直接输出初始状态（因为没有实际图）
final_state = initial_state
logger.info(f"最终状态：{json.dumps(final_state, indent=4, ensure_ascii=True)}")

# logger.info("图结构：")
query_app.get_graph().print_ascii()  # 需安装 grandalf

logger.info("===== 测试结束 =====")