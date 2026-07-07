import json
import sys

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


load_dotenv(find_dotenv())


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _parse_llm_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()
    return json.loads(content)


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):
    history_text = ""
    for chat in history_chats:
        item_names = chat.get("item_names") or chat.get("item_name") or []
        history_text += (
            f"聊天角色:{chat.get('role', '')},"
            f"回答内容:{chat.get('text', '')},"
            f"重写问题:{chat.get('rewritten_query', '')},"
            f"关联主体:{','.join(_as_list(item_names))}\n"
        )

    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text=history_text,
        query=original_query,
    )

    llm_client = get_llm_client(json_mode=True)
    response = llm_client.invoke([HumanMessage(content=prompt)])
    dict_content = _parse_llm_json(response.content)

    item_names = dict_content.get("item_names", dict_content.get("item_name", []))
    dict_content["item_name"] = _as_list(item_names)
    dict_content["rewritten_query"] = dict_content.get("rewritten_query") or original_query

    logger.info(f"完成重写和item_name提取，结果为:{dict_content}")
    return dict_content


def step_4_query_milvus_item_names(item_names):
    item_names = _as_list(item_names)
    if not item_names:
        logger.warning("LLM未提取到item_name，跳过商品名Milvus确认")
        return []

    collection_name = milvus_config.item_name_collection
    if not collection_name:
        logger.warning("缺少ITEM_NAME_COLLECTION配置，跳过商品名Milvus确认")
        return []

    milvus_client = get_milvus_client()
    if not milvus_client.has_collection(collection_name=collection_name):
        logger.warning(f"Milvus集合[{collection_name}]不存在，跳过商品名确认")
        return []

    milvus_client.load_collection(collection_name=collection_name)
    embeddings = generate_embeddings(item_names)
    final_result = []

    for index, item_name in enumerate(item_names):
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        reqs = create_hybrid_search_requests(dense_vector, sparse_vector)

        response = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=(0.6, 0.4),
            norm_score=True,
        )

        matches = []
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
                hit_name = entity.get("item_name", "")
                score = hit.get("distance", hit.get("score", 0)) if isinstance(hit, dict) else 0
                if hit_name:
                    matches.append({"item_name": hit_name, "score": score})

        final_result.append({
            "extracted": item_name,
            "matches": matches,
        })

    return final_result


def step_5_confirmed_and_optional_item_name(query_milvus_results, threshold=0.75, option_limit=2):
    confirmed_item_names = []
    options_item_names = []
    confirmed_set = set()
    option_set = set()

    for item_meta in query_milvus_results or []:
        extracted_name = item_meta.get("extracted", "")
        matches = item_meta.get("matches", [])
        matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
        if not matches:
            continue

        top_match = matches[0]
        top_name = top_match.get("item_name")
        top_score = top_match.get("score", 0)

        if top_name and (top_score >= threshold or top_name == extracted_name):
            if top_name not in confirmed_set:
                confirmed_item_names.append(top_name)
                confirmed_set.add(top_name)
            continue

        for match in matches[:option_limit]:
            option_name = match.get("item_name")
            if option_name and option_name not in confirmed_set and option_name not in option_set:
                options_item_names.append(option_name)
                option_set.add(option_name)

    return {
        "confirmed_item_names": confirmed_item_names,
        "options_item_names": options_item_names,
    }


def step_6_deal_list(state, item_results, history_chats, rewritten_query):
    confirmed_item_names = item_results.get("confirmed_item_names", [])
    options_item_names = item_results.get("options_item_names", [])

    if confirmed_item_names:
        state["item_names"] = confirmed_item_names
        state["rewritten_query"] = rewritten_query
        state["history"] = history_chats
        state["answer"] = ""
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state

    if options_item_names:
        option_names = "、".join(options_item_names)
        state["answer"] = f"您是想咨询以下哪个商品：{option_names}？请下次提问明确商品名称！"
        logger.info(f"有可选的item_name:{options_item_names}")
        return state

    state["answer"] = "没有匹配的商品名，请重新提问！！"
    logger.info("没有匹配的item_name")
    return state


def node_item_name_confirm(state: QueryGraphState):
    print("---node_item_name_confirm---开始处理")
    add_running_task(
        state["session_id"],
        sys._getframe().f_code.co_name,
        state.get("is_stream"),
    )

    original_query = state.get("original_query", "")
    history_chats = get_recent_messages(state["session_id"], limit=10)

    try:
        item_names_and_rewritten_query = step_3_llm_item_name_and_rewrite_query(
            original_query,
            history_chats,
        )
    except Exception as e:
        logger.warning(f"商品名提取或问题重写失败，使用原始问题继续: {e}")
        item_names_and_rewritten_query = {
            "item_name": [],
            "rewritten_query": original_query,
        }

    extracted_item_names = item_names_and_rewritten_query["item_name"]
    rewritten_query = item_names_and_rewritten_query["rewritten_query"]
    logger.info(f"LLM提取item_names={extracted_item_names}, rewritten_query={rewritten_query}")
    search_item_names = extracted_item_names or [rewritten_query or original_query]
    if not extracted_item_names:
        logger.warning(f"LLM未提取到item_name，使用问题文本兜底检索Milvus：{search_item_names}")

    try:
        query_milvus_results = step_4_query_milvus_item_names(search_item_names)
        logger.info(f"Milvus商品名确认原始结果={query_milvus_results}")
        item_results = step_5_confirmed_and_optional_item_name(query_milvus_results)
        logger.info(f"Milvus商品名确认分类结果={item_results}")
    except Exception as e:
        logger.warning(f"商品名Milvus确认失败，使用LLM提取结果继续: {e}")
        item_results = {
            "confirmed_item_names": extracted_item_names,
            "options_item_names": [],
        }

    state = step_6_deal_list(state, item_results, history_chats, rewritten_query)
    final_item_names = state.get("item_names") or extracted_item_names

    try:
        save_chat_message(
            session_id=state["session_id"],
            role="user",
            text=original_query,
            rewritten_query=rewritten_query,
            item_names=final_item_names,
            image_urls=[],
        )
        if state.get("answer"):
            save_chat_message(
                session_id=state["session_id"],
                role="assistant",
                text=state["answer"],
                rewritten_query=rewritten_query,
                item_names=final_item_names,
                image_urls=[],
            )
    except Exception as e:
        logger.warning(f"保存当前查询历史失败，不影响查询流程: {e}")

    add_done_task(
        state["session_id"],
        sys._getframe().f_code.co_name,
        state.get("is_stream"),
    )
    print("---node_item_name_confirm---处理结束")

    return {
        "item_names": state.get("item_names", []),
        "rewritten_query": state.get("rewritten_query", rewritten_query),
        "history": history_chats,
        "answer": state.get("answer", ""),
    }

if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_003",
        "original_query": "H3C LA2608 室内无线网关",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
