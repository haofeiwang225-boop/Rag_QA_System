from app.utils.task_utils import add_done_task, add_running_task
import sys
import time

def node_search_embedding(state):
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("量内容检查答案！！")
    time.sleep(1)

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"embedding_chunks":[]}
