from app.core.logger import logger
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.utils.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task

load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25 # 第一个与第二个相减 再除以第一个值
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5 #最大间断分值 相间的数 相差超过这个分数 就截取前n个


def step_1_merge_rrf_mcp(state):
    rrf_chunks = state.get("rrf_chunks", [])
    mcp_web_documents = state.get("mcp_web_documents",[])

    chunk_list = []

    #处理rrf
    for result in rrf_chunks:  # result 的内容是 id, distance entity
        if not isinstance(result, dict):
            continue

        chunk = result.get("entity")
        if not isinstance(chunk, dict):
            chunk = result

        text = chunk.get('content')
        title = chunk.get('title')
        chunk_id = chunk.get('chunk_id')
        chunk_list.append(
            {
                'chunk_id': chunk_id,
                'title': title,
                'content': text,
                "source": "local"
            }
        )
    #处理 mcp
    for entity in mcp_web_documents:
        text = entity['snippet']
        title = entity['title']
        url = entity['url']
        chunk_list.append({
            "content": text,
            "title": title,
            "url": url,
            "source": "web"
        })

    logger.info(f"多路融合,最终数据为{chunk_list}")
    return chunk_list
def step_2_rerank_doc_list(doc_list, state):
    #获取原来的问题
    rewritten_query = state.get("rewritten_query", state.get("original_query",""))

    #获取文件对应的答案
    text_list = [text["content"] for text in doc_list]

    #加载ranker模型
    reranker_model = get_reranker_model()

    #处理数据, 设置为问题加答案 装到列表 调用打分
    ranker_data = []
    for text in text_list:
        ranker_data.append((rewritten_query,text))

    scores = reranker_model.compute_score(ranker_data, normalize=True)
    # 将原来的数据添加对应的分
    doc_list_with_socre = []
    for score, item in zip(scores, doc_list):
        item["score"] = score
        doc_list_with_socre.append(item)

    doc_list_with_socre.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"已经完成排序和打分,最终结果为{doc_list_with_socre}")
    return doc_list_with_socre


def step_3_topk_and_gap(ranker_score_list):

    '''[
            {
                text: 内容 snippet content,
                chunk_id: chunk_id rrf有 mcp None,
                title: title ,
                url : rrfNone mcp url ,
                source: web -> mcp || local -> rrf ,
                score: rerank打的分
            }
        ]
    '''

    max_topk = RERANK_MAX_TOPK
    min_topk = RERANK_MIN_TOPK
    gap_ratio = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS

    #先截取最大的maxtopk数据
    topk = min(max_topk, len(ranker_score_list))

    #对截取的数据进行断崖处理
    for i in range(min_topk-1, max_topk-1): #应该从最小的min_tok-1开始
        score_1 = ranker_score_list[i].get("score", 0.0)
        score_2 = ranker_score_list[i+1].get("score", 0.0)

        gap = score_1 - score_2 #算放断崖
        gap_ratio = gap / abs(gap_ratio + 1e-6) #除0 与 断崖

        if gap >= gap_abs or gap_ratio >= gap_ratio:
            topk = i + 1
            logger.info(f"数据集合{i}和{i + 1}")
            break

    #取topk个数据
    return ranker_score_list[:topk]








    #截取确定的topk 返回


def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    将rrf融合排序的结果再与mcp  借用llm的能力一起打分
    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    #1 获取数据 和并到一个 列表中 rrf + mcp
    '''
    rrf={                                                                                   
      "id": 123,                                                                      
      "distance": 0.8,                                                                
      "entity": {                                                                     
          "chunk_id": "abc",                                                          
          "content": "文档内容"                                                       
      }                                                                               
  }   
    mcp = {snippet: 内容,title:标题,url:关联的文章或者图片的地址}
    {
        text:内容 snippet content,
        chunk_id: chunk_id rrf有 mcp None,
        title: title ,
        url: rrfNone mcp url ,
        source: web -> mcp || local -> rrf
    }
'''
    doc_list = step_1_merge_rrf_mcp(state)

    '''
    [
        {
            text: 内容 snippet content,
            chunk_id: chunk_id rrf有 mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp || local -> rrf ,
            score: rerank打的分
        }
    ]    
    '''
    #2 使用ranker 对结果排序
    ranker_score_list = step_2_rerank_doc_list(doc_list, state)

    #3 使用算法 进行放断崖及top k 处理
    final_list = step_3_topk_and_gap(ranker_score_list)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 结果装到state中
    return {"reranked_docs": final_list}
    #




if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('content')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")