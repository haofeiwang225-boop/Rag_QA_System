import copy
from typing import List

from typing_extensions import TypedDict


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义查询流程中流转的数据结构。
    """
    session_id: str
    original_query: str

    # 检索过程中的中间数据
    embedding_chunks: list
    hyde_embedding_chunks: list
    kg_chunks: list
    web_search_docs: list

    # 排序过程中的数据
    rrf_chunks: list
    reranked_docs: list

    # 生成过程中的数据
    prompt: str
    answer: str

    # 辅助信息
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "kg_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程默认状态，支持覆盖字段。
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """
    获取干净的查询流程默认状态。
    """
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: dict, **overrides) -> QueryGraphState:
    """
    复制现有状态并覆盖指定字段，避免污染原状态。
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样？",
        is_stream=False,
    )
    print("初始化状态: ", state)

    new_state = copy_query_state(
        state,
        original_query="修改后的问题",
    )
    print("复制后的状态: ", new_state)
