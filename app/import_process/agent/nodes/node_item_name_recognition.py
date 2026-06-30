# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import sys
from pathlib import Path
from typing import List, Dict

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[4]))

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import DataType

# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

from app.conf.milvus_config import milvus_config

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState

# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client

# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client

# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import generate_embeddings

# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task

# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger

# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_inputs(state):
    """
    步骤 1: 接收并校验流程输入（商品名称识别的前置数据处理）
    核心作用：
        1. 从流程状态中提取文件标题、文本切片核心数据
        2. 做多层空值兜底，避免后续流程因空值报错
        3. 基础数据类型校验，保证下游流程输入有效性
    依赖的状态数据（上游节点产出）：
        - state["file_title"]: 上游提取的文件标题（优先使用）
        - state["file_name"]: 原始文件名（file_title为空时兜底）
        - state["chunks"]: 文本切片列表（每个切片为字典，含title/content等字段）
    返回值：
        Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
    """
    file_title = state.get("file_title", "") or state.get("file_name", "")
    chunks = state.get("chunks") or []
    # 二次兜底：file_title仍为空时，尝试从第一个有效切片中提取
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("state中无有效file_title，已从第一个切片中提取兜底标题")

        # 空值日志提示：文件标题为空时不中断流程，仅记录警告
    if not file_title:
        logger.warning("state中缺少file_title和file_name，后续大模型识别可能精度下降")

        # 数据类型校验：确保chunks为有效非空列表，否则返回空列表
    if not isinstance(chunks, list) or not chunks:
        logger.warning("state中chunks为空或非列表类型，无法进行商品名称识别")
        return file_title, []

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


def step_2_build_context(
    chunks: List[Dict],
    k: int = DEFAULT_ITEM_NAME_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS,
) -> str:
    parts = []
    total_chars = 0
    for index, chunk in enumerate(chunks[:k]):
        chunk_title = chunk.get("title", "")
        chunk_content = (chunk.get("content", "") or "")[:SINGLE_CHUNK_CONTENT_MAX_LEN]
        data = f"切片{index},标题:{chunk_title},内容{chunk_content}"
        remaining_chars = max_chars - total_chars
        if remaining_chars <= 0:
            break
        parts.append(data[:remaining_chars])
        total_chars += min(len(data), remaining_chars)

    context = "\n\n".join(parts)  # 使用\n\n对列表进行拼接
    final_context = context[:max_chars]
    return final_context


def step_3_call_llm(file_title, context):
    human_prompt = load_prompt(
        "item_name_recognition", file_title=file_title, context=context
    )
    system_prompt = load_prompt("product_recognition_system")

    llm = get_llm_client(json_mode=False)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    res = llm.invoke(messages)
    item_name = (res.content or "").strip()
    if not item_name:
        item_name = file_title

    return item_name


def step_4_update_chunks(state, chunks, item_name):
    """
    步骤 4: 回填商品名称到流程状态和所有文本切片
    核心作用：
        1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
        2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
        3. 状态同步：更新state中的chunks，确保切片修改全局生效
    设计思路：
        所有切片关联同一商品名称，保证后续向量入库、检索时的维度一致性
    参数：
        state: 流程状态对象（ImportGraphState），全局数据载体
        chunks: 校验后的文本切片列表（步骤1输出）
        item_name: 步骤3识别并清洗后的商品名称
    """
    # 将商品名称存入全局状态，供下游节点调用
    state["item_name"] = item_name
    # 遍历所有切片，为每个切片添加商品名字段，保证数据全链路一致
    for chunk in chunks:
        chunk["item_name"] = item_name
    # 同步更新state中的切片列表，确保修改全局生效
    state["chunks"] = chunks
    logger.info(
        f"步骤4：商品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}"
    )


def step_5_generate_vectors(item_name):
    item_name = (item_name or "").strip()
    if not item_name:
        raise ValueError("商品名称为空，无法生成向量")

    vectors = generate_embeddings([item_name])

    dense_vectors, sparse_vector = vectors["dense"][0], vectors["sparse"][0]
    return dense_vectors, sparse_vector


def step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector):
    collection_name = milvus_config.item_name_collection
    if not collection_name:
        raise ValueError("缺少ITEM_NAME_COLLECTION环境变量配置")

    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus客户端初始化失败")

    # 不存在表的话创建
    if not milvus_client.has_collection(collection_name=collection_name):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 添加自增主键字段：INT64类型，唯一标识每条数据
        schema.add_field(
            field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
        )
        # 添加商品名字段：VARCHAR类型，最大长度65535，去重依据
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
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
            params={"m": 16, "efConstruction": 200},
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
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引"
        )

    milvus_client.load_collection(collection_name=collection_name)
    safe_item_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=collection_name,
        filter=f'item_name == "{safe_item_name}"',
    )

    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }

    insert_result = milvus_client.insert(collection_name=collection_name, data=[item])
    milvus_client.flush(collection_name=collection_name)
    logger.info(f"商品名称已写入Milvus集合[{collection_name}]：{insert_result}")


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【商品名称识别】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state.get("task_id", ""), node_name)

    try:
        file_title, chunks = step_1_get_inputs(state)

        # 构建llm上下文
        # 作用：截取前N个切片的内容，拼接成大模型可阅读的上下文，用于辅助识别
        # 输出：拼接后的上下文字符串
        context = step_2_build_context(chunks)
        # llm识别主体
        # 作用：构造Prompt，调用LLM从上下文和标题中提取最核心的商品名称
        # 输出：识别出的商品名称字符串（如 "iPhone 15 Pro"）
        item_name = step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片
        # 作用：将识别结果写入状态字典，并同步更新到每一个Chunk对象的元数据中
        # 输出：状态字典新增item_name，chunks列表被就地修改
        step_4_update_chunks(state, chunks, item_name)

        # 生成向量 稀疏和稠密
        # 作用：调用BGE-M3模型，为商品名称生成稠密语义向量和稀疏关键词向量
        # 输出：dense_vector（List[float]）、sparse_vector（Dict[int, float]）
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

        # 存入向量数据库
        # 作用：将商品名称及其双路向量存入Milvus的 item_names 集合，用于后续检索
        # 输出：无返回值，数据已持久化
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 节点执行完成日志
        logger.info(
            f">>> 核心节点执行完成：【商品名称识别】{node_name}，识别结果：{item_name}，已存入Milvus"
        )

    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(
            f">>> 核心节点执行失败：【商品名称识别】{node_name}，错误信息：{str(e)}",
            exc_info=True,
        )
        # 仅在识别阶段没有产出商品名时设置兜底值，避免覆盖已回填到chunks中的有效识别结果
        state.setdefault("item_name", "未知商品")

    # 返回更新后的状态（供下游节点使用）
    return state


def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState(
            {
                "task_id": "test_task_123456",  # 测试任务ID
                "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
                "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
                # 模拟文本切片列表（上游切片节点产出，含title/content字段）
                "chunks": [
                    {
                        "title": "产品简介",
                        "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。",
                    },
                    {
                        "title": "拍照功能",
                        "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。",
                    },
                    {
                        "title": "电池参数",
                        "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。",
                    },
                ],
            }
        )

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(
            f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}"
        )

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            if not milvus_client.has_collection(collection_name=collection_name):
                logger.warning(f"Milvus集合[{collection_name}]不存在，跳过本地测试检索验证")
                return

            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get("item_name")
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"],
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
