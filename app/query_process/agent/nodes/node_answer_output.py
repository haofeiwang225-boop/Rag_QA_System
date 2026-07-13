import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000  #限制prompt的长度


# 若项目中不存在，请根据实际情况调整

def step_1_check_answer(state):

    answer = state.get("answer","")
    stream = state.get("stream","")

    if answer:
        if stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, answer)

        else:
            set_task_result(state["session_id"], "answer", answer)
        return True

    else:
        return False


def step_2_load_prompt(state):

    rewritten_query = state.get("rewritten_query",state.get("original_query"))
    reranked_docs = state.get("reranked_docs",[])
    item_names = state.get("item_names",[])
    history = state.get("history",[])

    docs = []
    used_length = 0
    #先处理chunk块的内容
    for i,doc in enumerate(reranked_docs):
        text = doc.get("text","")
        chunk_id = doc.get("chunk_id","")
        title = doc.get("title","")
        source = doc.get("source","")
        score =  doc.get("score",0)
        content = f"{{i}}[text={text}][source={source}][title={title}][score={score}]"
        used_length += len(content)

        if used_length < MAX_CONTEXT_CHARS:
            docs.append(content)

        else:
            break
        final_context = "\n\n".join(docs)

    #处理history
    history_str = ""
    if history and len(history) > 0:
        for i,message in enumerate(history):
            role = message.get("role","")
            text = message.get("text","")
            current_history = ""
            if role == "user" and text:
                current_history += f"用户:{text}\n"
            else:
                current_history += f"助手:{text}\n"

            if used_length + len(current_history) < MAX_CONTEXT_CHARS:
                history_str += current_history
            else:
                break

    else:
        history_str = "没有历史对话"

    #处理item
    item_names = ",".join(item_names)
    #question问题

    prompt = load_prompt("answer_out",
                         context=final_context,
                         history=history_str,
                         item_names=item_names,
                         question=rewritten_query
                         )





def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #看看之前state中有没有answer (item_name没有or不确定,就直接返回了)
    answer_exists = step_1_check_answer(state)

    if not answer_exists:
        #在没有的场景下
        # 生成对应的润色的prompt
        prompt = step_2_load_prompt(state)
        #使用模型润色答案 结果 文本

        #提取原来的topklist中的图片的地址 单独返回see
        #对话的聊天记录 存在mongodb
        #see- final 返回图片


    return state
