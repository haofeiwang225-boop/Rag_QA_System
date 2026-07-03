import os
import sys
from pathlib import Path
from typing import List, Dict, Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[4]))

# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_2_prepare_collections(state):
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus客户端初始化失败")

    # 不存在表的话创建
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 添加自增主键字段：INT64类型，唯一标识每条数据
        schema.add_field(
            field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
        )
        # 添加商品名字段：VARCHAR类型，最大长度65535，去重依据
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="part", datatype=DataType.INT8
        )
        # 添加稠密向量字段：FLOAT_VECTOR，1024维（BGE-M3固定维度）
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
        )
        # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，变长
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        index_params = milvus_client.prepare_index_params()

        # 添加索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"m": 32, "efConstruction": 300},
        )

        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+IP，关闭量化保证精度
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
            index_type="SPARSE_INVERTED_INDEX",
            # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
            metric_type="IP",
            # DAAT_MAXSCORE：稀疏向量检索时，只计算可能得高分的维度，跳过大量0值，速度更快。
            # quantization="none"：稀疏向量里的权重是小数，不做压缩，保证精度不丢。
            params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"},
        )

        # 创建集合：Schema + 索引参数
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            f"Milvus集合[{milvus_config.chunks_collection}]创建成功，包含Schema和向量索引"
        )

    return milvus_client


def step_3_del_old_data(milvus_client, item_name):

    safe_item_name = str(item_name).replace("\\", "\\\\").replace('"', '\\"')
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)
    milvus_client.delete(collection_name=milvus_config.chunks_collection,
                         filter=f'item_name == "{safe_item_name}"')


def step_4_insert_collection(milvus_client, chunks):
    insert_result= milvus_client.insert(collection_name=milvus_config.chunks_collection, data=chunks)
    milvus_client.flush(collection_name=milvus_config.chunks_collection)

    #成功插入了几条
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"成功插入{insert_count}")


    ids = insert_result.get("ids", [])
    #id回显
    if len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]
    return chunks


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)

    try:
        chunks = state.get("chunks", [])
        if not chunks:
            logger.error(f">>>{node_name}没有chunks数据")
            raise ValueError("chunks无数据")
        #没有集合,创建集合
        milvus_client = step_2_prepare_collections(state)

        #删除旧数据
        step_3_del_old_data(milvus_client, chunks[0]['item_name'])

        #插入chunks数据
        with_id_chunks = step_4_insert_collection(milvus_client, chunks)

        state["chunks"] = with_id_chunks

    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【导入milvus节点】{node_name}，错误信息：{str(e)}", exc_info=True)

    return state
if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"[OK] Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("[FAIL] 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
