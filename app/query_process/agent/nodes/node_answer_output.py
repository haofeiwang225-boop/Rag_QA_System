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
    stream = state.get("is_stream", False)

    if answer:
        if stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})

        else:
            set_task_result(state["session_id"], "answer", answer)
        return True

    else:
        return False


def step_2_load_prompt(state):

    rewritten_query = state.get("rewritten_query",state.get("original_query"))
    reranked_docs = state.get("reranked_docs") or state.get("rrf_chunks", [])
    item_names = state.get("item_names",[])
    history = state.get("history",[])

    docs = []
    used_length = 0
    #先处理chunk块的内容
    for i,doc in enumerate(reranked_docs):
        text = doc.get("content") or doc.get("text", "")
        chunk_id = doc.get("chunk_id","")
        title = doc.get("title","")
        source = doc.get("source","")
        score =  doc.get("score",0)
        content = f"{i+1}[source={source}][title={title}][score={score}]\n\n[text={text}]"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            break

        docs.append(content)
        used_length += len(content)

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
    state["prompt"] = prompt
    return prompt

def step_3_create_answer(state, prompt):
    #1 获取模型
    model = get_llm_client()
    # 是否流式sse or set_result
    stream = state.get("is_stream", False)
    answer = ""
    #调用模型流式 or set_result
    if stream:
        for chunk in model.stream(prompt):#chunk是什么
            delta = chunk.content
            answer += delta
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": delta})
    else:
        response = model.invoke(prompt)
        answer = response.content
        set_task_result(state["session_id"], "answer", answer)

    #得到答案
    state["answer"] = answer
    #返回
    return answer


def step_4_extract_images_url(state, answer):
    images = [] #顺序保存图片地址
    set_images = set() #去重

    #定义正则 匹配到md格式的url
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")

    reranked_docs = state.get("reranked_docs") or state.get("rrf_chunks", [])
    for i,doc in enumerate(reranked_docs):
        url = doc.get("url","")
        if url:
            if url.endswith ((".png",".jpg",".jpeg",".gif",".webp")):
                if url not in set_images:
                    images.append(url)
                    set_images.add(url)

        #text 正则提取图片
        text = doc.get("content") or doc.get("text", "")
        urls = image_reg.findall(text)

        for url in urls:
            if url not in set_images:
                images.append(url)
                set_images.add(url)

    logger.info(f"已经完成图片的提取 数量{len(images)},提取内容:{images}")
    state["image_urls"] = images
    return images


def step_5_write_history(state):

    #获取要存的数据
    session_id = state["session_id"]
    answer = state.get("answer","")
    rewritten_query = state.get("rewritten_query",state.get("original_query"))
    item_names = state.get("item_names",[])




    if answer:
        save_chat_message(session_id=session_id,
                          role="assistant",
                          text=answer,
                          item_names=item_names
                          )

    logger.info(f"完成本次对话")




def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #看看之前state中有没有answer (item_name没有or不确定,就直接返回了)
    answer_exists = step_1_check_answer(state)
    answer = state.get("answer", "")
    images_url = state.get("image_urls", [])
    prompt = state.get("prompt", "")

    if not answer_exists:
        #在没有的场景下
        # 生成对应的润色的prompt
        prompt = step_2_load_prompt(state)
        #使用模型润色答案 结果 文本
        answer = step_3_create_answer(state, prompt)
        #提取原来的topklist中的图片的地址 单独返回see
        images_url = step_4_extract_images_url(state, answer)


        #see- final 返回图片
    if state.get("is_stream", False):
        push_to_session(
            state["session_id"],
            SSEEvent.FINAL,
            {
                "answer": answer,
                "status": "completed",
                "image_urls": images_url,
            },
        )

    # 对话的聊天记录 存在mongodb
    step_5_write_history(state)



    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"answer": answer, "prompt": prompt, "image_urls": images_url}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://www.baidu/img/bd_logo.png)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
