import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500



def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    【步骤1】获取并预处理输入数据
    功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
    :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
    :return: 标准化后的MD内容/文件标题/单个Chunk最大长度（无内容则返回None,None,None）
    """
    # 从状态中提取MD原始内容
    content = state.get("md_content")
    # 空内容兜底：无MD内容则直接返回，终止后续处理
    if not content:
        logger.warning("状态字典中无有效MD内容，终止文档切分")
        return None, None, None

    # 基础标准化：统一换行符，避免Windows/Linux换行符差异导致的后续处理异常
    # 原始混合换行："# HL3070说明书\r\n## 产品概述\nHL3070是扫描枪\r\n\r\n### 操作步骤"
    # 统一后："# HL3070说明书\n## 产品概述\nHL3070是扫描枪\n\n### 操作步骤"
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # 提取文件标题：有则用，无则默认"Unknown File"
    file_title = state.get("file_title", "Unknown File")
    # 提取最大Chunk长度：有则用状态中的配置，无则用全局默认值
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")
    return content, file_title, max_len


def step_2_split_by_titles(content, file_title):
    """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """
    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # #{1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）
    title_pattern = r'^\s*#{1,6}\s+.+'

    # 将MD内容按换行符拆分为行列表，逐行处理
    lines = content.split("\n")
    sections = []  # 最终切分的章节列表
    current_title = ""  # 当前章节标题
    current_lines = []  # 当前章节的行缓存
    title_count = 0  # 有效标题数量（非代码块内）
    in_code_block = False  # 代码块标记：避免误判代码块内的#为标题

    for line in lines:
        strip_line = line.strip()
        #判断是代码块
        if strip_line.startswith('```') or line.startswith('~~~'):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue
        #不是代码块
        is_title = (not in_code_block) and re.match(title_pattern, strip_line)
        if is_title: # 第二次不为空,但不是标题了

            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title,
                })


            title_count += 1
            current_title = strip_line #标题名称
            current_lines = [current_title]


        else:
            current_lines.append(line)

    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })
    return sections, title_count, len(lines)


def split_long_section(section, DEFAULT_MAX_CONTENT_LENGTH):

    sub_sections = []
    content = section.get("content") or ""
    title = section.get("title")
    file_title = section.get("file_title")

    if len(content) > DEFAULT_MAX_CONTENT_LENGTH:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_MAX_CONTENT_LENGTH,
            chunk_overlap=100,
            separators=['\n\n', '\n','。','!',' ','']
        )
        for index, chunk in enumerate(splitter.split_text(content), start=1):
            text = chunk.strip()
            sub_sections.append({
                "title": f"{title}_{index}",
                "content": text,
                "file_title": file_title,
                "parent_title": title,
                "part": index
            })
    else:
        sub_sections.append({
            "title": title,
            "content": content.strip(),
            "file_title": file_title,
            "parent_title": section.get("parent_title") or title,
            "part": section.get("part") or 1
        })
    return sub_sections


def merge_shout_sections(final_sections, min_length):
    """
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
        :param final_sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
        :param min_length: 最小长度阈值，低于此值的Chunk会被合并
        :return: 合并后的Chunk列表，长度适中，保留元信息
        """
    merged_sections = []
    pre_section = None

    for section in final_sections:
        if pre_section is  None:
            pre_section = section
            continue

        is_current_short = len(pre_section.get("content")) < min_length
        is_same_parent_title = section.get("parent_title") == pre_section.get("parent_title") and section.get("parent_title")

        if is_current_short and is_same_parent_title:
            pre_section["content"] += "\n\n" + section.get("content", "")

        else:
            merged_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections


def step_3_handle_no_title(sections, DEFAULT_MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH):
    """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param DEFAULT_MAX_CONTENT_LENGTH: 超过了要切
        :param sections: 步骤2切分后的章节列表
        :param MIN_CONTENT_LENGTH: 小于要合并
        :param file_title: 所属文件标题
        :return: sections
        """
    final_sections = []#存储后的快
    #超过的切
    for section in sections:
        #
        sub_section = split_long_section(section, DEFAULT_MAX_CONTENT_LENGTH)
        #不够切的
        final_sections.extend(sub_section)



    #小于的合并
    final_sections = merge_shout_sections(final_sections, min_length=MIN_CONTENT_LENGTH)
    #补属性和参数
    for final_section in final_sections:
        final_section["part"] = final_section.get("part") or 1
        final_section["parent_title"] = final_section.get("parent_title") or final_section.get("title")

    return final_sections


def step_4_backup_chunks(state, sections):
    local_dir = state.get("local_dir") or os.getcwd()
    os.makedirs(local_dir, exist_ok=True)
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)





def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】文档切分主节点（node_document_split）
    整体流程：加载输入→按MD标题初切→无标题兜底→长切短合→统计输出→结果备份
    核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
    后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等
    :param state: 项目状态字典（ImportGraphState），必须包含md_content/task_id；可选local_dir/max_content_length/file_title
    :return: 更新后的状态字典，新增chunks键（存储最终处理后的Chunk列表，每个Chunk为含title/content/parent_title的字典）
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)

    try:
        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题/Chunk最大长度，统一换行符消除系统差异，做空值兜底
        # 输出：标准化后的md_content、文件标题、单个Chunk最大长度；无有效MD内容则直接终止节点执行
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> 节点执行终止：{node_name}（无有效MD内容）")
            return state

        # ===================================== 步骤2：按MD标题进行初次切分 =====================================
        # 作用：基于Markdown标题（#/##/###）切分文档为独立章节，自动跳过代码块内的伪标题，保证章节语义完整
        # 输出：初切后的章节列表、识别到的有效标题数量、MD原始文本总行数（为后续统计/日志使用）
        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        if title_count == 0:
            sections = [{"title": "没有标题", "content": content, "file_title": file_title}]


        #切割md文件
        sections = step_3_handle_no_title(sections, DEFAULT_MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH)



        state["chunks"] = sections
        step_4_backup_chunks(state,sections)


    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【文档切分】{node_name}，错误信息：{str(e)}", exc_info=True)



    return state



if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")


