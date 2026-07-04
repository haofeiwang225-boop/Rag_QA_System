import sys

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_answer_output(state: QueryGraphState):
    """
    输出最终答案。
    """
    node_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], node_name, state.get("is_stream"))

    answer = state.get("answer") or "已完成检索，等待接入答案生成逻辑。"

    add_done_task(state["session_id"], node_name, state.get("is_stream"))
    return {"answer": answer}
