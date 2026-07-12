import time
import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_3_reciprocal_rank_fusion(source_with_weight, top: int=5):
    #结果是(chunk,score)
    #返回chunks列表
    score_dict = {} # id: socre
    chunk_dict = {} #id : chunk

    for source, weight in source_with_weight:
        #source = [{id:实体主键,distance:分数 ...}]
        for rank, chunk in enumerate(source,start=1):
            #计算当前chunk的得分
            #获取chunk_id
            entity = chunk.get("entity") or {}
            chunk_id = chunk.get("id") or entity.get("chunk_id")
            if chunk_id is None:
                logger.warning("跳过缺少 id/chunk_id 的 RRF 检索结果: %s", chunk)
                continue

            # 首次出现的 chunk_id 还没有对应键，累计时以 0 分作为初始值。
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (60 + rank) * weight)

            #获取chunk块
            chunk_dict[chunk_id] = chunk


    # score和chunk 合并
    mearge = []
    for id, score in score_dict.items():
        chunk = chunk_dict[id]
        mearge.append((chunk, score))

    mearge.sort(key=lambda x: x[1], reverse=True)

    mearge = mearge[:top]

    rank_mearge = []
    for chunk, score in mearge:
        rank_mearge.append(chunk)
    return rank_mearge

def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #取同源的数据
    embedding_chunks = state["embedding_chunks"] #原问题的混合向量检索结果
    hyde_chunks = state["hyde_embedding_chunks"] #根据llm给出的答案和问题 取向量数据库中搜素

    #数据合并
    source_with_weight = [
        (embedding_chunks,0.8),
        (hyde_chunks,0.2),
    ]

    #rrf
    rrf_response = step_3_reciprocal_rank_fusion(source_with_weight)

    #将排序后的加入到rrf_chunks
    state["rrf_chunks"] = rrf_response

    time.sleep(1)
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"rrf_chunks": rrf_response}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果
    mock_embedding_chunks = [
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.9
        },
        {
            "id": "doc_2",
            "pk": "pk_2",
            "file_title": "维修指南.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容2：遇到故障请联系...",
            "score": 0.8
        },
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.7
        }
    ]

    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.85
        },
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.82
        },
        {
            "id": "doc_4",
            "pk": "pk_4",
            "file_title": "安全须知.docx",
            "item_name": "HAK 180 烫金机",
            "content": "内容4：操作时请佩戴手套...",
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)

        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]

        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
