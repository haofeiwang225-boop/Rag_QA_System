import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):

    #准备提升词
    histort_text = ""
    for chat in history_chats:
        histort_text += f"聊天角色:{chat['role']},回答内容:{chat['text']}.重写问题:{chat['rewritten_query']},关联主体.{','.join(chat['item_name'])}"
    prompt = load_prompt("rewritten_query_and_itemnames",histort_text=histort_text,query=original_query)

    #模型调用
    lm_client = get_llm_client(json_mode=True)
    messages = {
        HumanMessage(content=prompt)
    }

    response = lm_client.index(messages)

    content = response['content']

    if content.startswith("```json"):
        content = content.replace("```json","").replace("```","")

    dict_content = json.loads(content)

    if not dict_content.get("item_name"):
        dict_content["item_name"] = []

    if not dict_content.get("rewritten_query"):
        dict_content["rewritten_query"] = original_query

    logger.info(f"完成重写和item_name的提取,结果为{dict_content}")
    return dict_content

def step_4_query_milvus_item_names(item_names):


    final_result=[]

    #1获取miluvs 客户端
    milvus_client = get_milvus_client()

    #2将item_names转为稀疏稠密向量
    embeddings = generate_embeddings(item_names)


    #3混合查询 创建稀疏与稠密的annsearchrequest 设置权重重排 进行混合查询
    for index, item_name in enumerate(embeddings):
        dense_vector = embeddings['dense_vector'][index]
        sparse_embedding = embeddings['sparse_embedding'][index]

        #创建拼接annsearchrequest
        req3 = create_hybrid_search_requests(dense_vector, sparse_embedding)


        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,#他怎么知道是哪个表名呢?
            reqs=req3,
            ranker_weights=(0.5,0.5),
            norm_score=True
        )

        matchs = []
        #解析结果
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get('entity',"")
                hit_name = entity.get('item_name',"")
                score = hit.get('distance',0)
                if hit_name:
                    matchs.append({
                        "item_name": hit_name,
                        "score": score

                    }
                    )

        #封装结果 模型和查询的
        final_result.append({
            "extracted": item_name,
            "matchs": matchs

        })









    #


def process_milvus_results(query_milvus_results):
    """
    处理 Milvus 查询结果，提取确定项和可选项。

    逻辑说明：
        1. 对每个 item，将其 matches 列表按分数降序排序。
        2. 取分数最高的 1 个作为“确定项”（confirmed）。
        3. 从剩余的匹配中，取前 2 个作为“可选项”（options），供用户参考选择。
        4. 所有结果汇总到两个列表中返回。

    Args:
        query_milvus_results (list[dict]): 每个字典包含：
            - "extracted": 提取出的名称（本实现未直接使用，仅保留兼容）
            - "matches": 列表，元素为 {"score": float, "item_name": str}

    Returns:
        tuple: (confirmed_item_names, options_item_names)
            - confirmed_item_names (list[str]): 每个 item 确定的唯一名称
            - options_item_names (list[str]): 所有可选项名称（每个 item 最多 2 个）
    """
    # 1. 准备两个列表
    confirmed_item_names = []   # 确定项
    options_item_names = []     # 可选项

    # 2. 循环处理每个元数据
    for item_meta in query_milvus_results:
        extracted_name = item_meta.get("extracted")   # 本实现未强制使用，但保留
        matches = item_meta.get("matches", [])

        # 3. 按分数降序排序（高分在前）
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 如果无匹配项，跳过
        if not matches:
            continue

        # 4. 处理高分：只取分数最高的 1 个作为确定项
        for match in matches:
            if match.get("score") >= 0.85:
                confirmed_item_names.append(match.get("item_name"))
                break
            #不是大于的 有多个 放到options_item_names
            else:
                options_item_names.append(match.get("item_name"))
                continue
    return {
        "confirmed_item_names":confirmed_item_names,
        "options_item_names":options_item_names[:2]}










def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    #获取历史记录
    history_chats = get_recent_messages(state["session_id"], limit=10)
    #保存当前的聊天记录
    save_chat_message(
        session_id=state["session_id"],
        role=state["role"], 
        text=state["text"],
        rewritten_query=state["original_query", ""],
        item_names = state["item_names",[]],
        image_urls= state["image_urls",[]],
    )

    #利用llm 1 提取item_names 2 重写提问内容
    item_names_and_rewritten_query = step_3_llm_item_name_and_rewrite_query(state["original_query"],history_chats)

    item_names = item_names_and_rewritten_query["item_name"]
    #milvus向量查询 item_names
    query_milvus_results = step_4_query_milvus_item_names(item_names)

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}