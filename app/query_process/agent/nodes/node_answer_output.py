import time
import sys
from app.core.logger import logger
from app.utils.sse_utils import push_to_session, SSEEvent

from app.utils.task_utils import set_task_result, add_running_task, add_done_task


# 若项目中不存在，请根据实际情况调整

def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，![baidulogo](https://example.com/demo-2.png)"
    final_text = ""

    if is_stream:
        # 流式输出：逐字推送 delta 事件
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, data={"delta": ch})
            time.sleep(0.03)
        logger.info(f"流式输出完成，总长度: {len(final_text)}")
    else:
        # 非流式输出：直接使用完整答案
        final_text = base_answer
        set_task_result(session_id, "answer", final_text)

    # 无论是否流式，最后都推送最终结果（包含图片列表）
    image_urls = [
        "http://www.baidu.com/img/bd_logo.png",
        "https://example.com/demo-2.png"
    ]
    set_task_result(session_id, "image_urls", image_urls)
    if is_stream:
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            data={
                "answer": final_text,
                "status": "completed",
                "image_urls": image_urls
            }
        )

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output 节点处理结束---")
    return {"answer": final_text, "image_urls": image_urls}
