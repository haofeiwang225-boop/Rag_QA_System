import os
import shutil
import sys
import zipfile
import time
from pathlib import Path

import requests

from app.core.logger import logger
from app.import_process.agent.state import create_default_state, ImportGraphState
from app.import_process.agent import state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 开始节点输出
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行节点,状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
        state['md_path'] = md_path
        state['local_dir'] = local_dir_obj

        with open(state["md_path"], "r", encoding="utf-8") as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.error(f">>>[{function_name}]使用mineru解析发生异常: {e}")
        raise
    finally:
        logger.info(f">>> [{function_name}] 执行完毕,状态为: {state}")
        add_done_task(state['task_id'], function_name)

    return state


def step_1_validate_paths(state: ImportGraphState):
    """
    步骤1：校验PDF文件路径和输出目录
    核心职责：参数非空校验 | PDF文件有效性校验 | 输出目录自动创建
    返回：合法的PDF文件Path对象、输出目录Path对象
    异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
    """
    log_prefix = "[step_1_validate_paths] "
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()

    # 参数非空校验
    if not pdf_path:
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：pdf_path，当前值：{repr(pdf_path)}")
    if not local_dir:
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：local_dir，当前值：{repr(local_dir)}")

    # 转换为Path对象统一处理路径
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    # PDF文件有效性校验（存在且为文件，非目录）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix}PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix}指定路径非文件（是目录），绝对路径：{pdf_path_obj.absolute()}")

    # 确保输出目录存在，不存在则递归创建
    if not output_dir_obj.exists():
        logger.info(f"{log_prefix}输出目录不存在，自动创建：{output_dir_obj.absolute()}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, output_dir_obj

def step_2_upload_and_poll(pdf_path_obj):
    """
       步骤2：上传PDF至MinerU并轮询解析任务状态
       核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
       参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
       返回：解析结果ZIP包下载链接full_zip_url
       异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
       """

    # 前置配置校验，拦截无效配置
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("MinerU配置缺失：请在.env中正确配置MINERU_BASE_URL和MINERU_API_TOKEN")
    logger.info(f"[配置校验] MinerU基础配置加载成功，开始处理文件：{pdf_path_obj.name}")

    # 构造请求头（符合HTTP规范，Bearer鉴权）
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key}"
    }

    # 1. 调用批量接口，获取上传Signed URL和任务batch_id
    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"  # 官方推荐解析模型
    }
    logger.debug(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)
    if resp.status_code != 200 or resp.json()['code'] != 0:
        raise RuntimeError(f'请求mineru解析失败')
    upload_url = resp.json()['data']['file_urls'][0]
    batch_id = resp.json()['data']['batch_id']


    http_session = requests.Session()
    http_session.trust_env = False

    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(url=upload_url, data=file_data)

        if upload_response.status_code != 200:
            logger.error("上传文件到mineru失败")
            raise RuntimeError(f"上传文件到mineru失败")
    except Exception as e:
        logger.error(e)
    finally:
        http_session.close()

    # 3. 根据batch_id轮询任务状态，直至完成/失败/超时
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600  # 最大超时时间10分钟（适配600页内PDF）
    poll_interval = 3  # 轮询间隔3秒（平衡查询频率和服务端压力）
    logger.info(f"[任务轮询] 开始监控任务状态，batch_id：{batch_id}，最大超时：{timeout_seconds}s")

    while True:
        #3.1 超时判断
        if time.time() - start_time > timeout_seconds:
            logger
            raise TimeoutError("请求超时")
        poll_resp = requests.get(url=poll_url, headers=request_headers)
        # 处理HTTP响应错误：5xx服务端繁忙则重试，其他错误直接抛出
        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(f"[任务轮询] 服务端繁忙（状态码：{poll_resp.status_code}），{poll_interval}秒后重试")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(
                    f"[任务轮询] HTTP请求失败，状态码：{poll_resp.status_code}，响应内容：{poll_resp.text}")


        json_data = poll_resp.json()
        if json_data['code'] != 0:
            raise RuntimeError(f"[任务轮询] API业务错误，返回数据：{json_data}")

        if json_data['data']['extract_result'][0]['state'] == 'done':
            return json_data['data']['extract_result'][0]['full_zip_url']
        else:
            time.sleep(poll_interval)

def step_3_download_and_extract(zip_url, local_dir_obj, pdf_stem):
    """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
        """
    logger.info(f"===== 开始处理[{pdf_stem}]的MinerU解析结果 =====")

    # 1. 下载解析结果ZIP包，120秒超时适配大文件
    logger.info(f"[步骤1/4] 开始下载ZIP包，链接：{zip_url}...")
    resp = requests.get(zip_url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"[步骤1/4] ZIP包下载失败，HTTP状态码：{resp.status_code}")

    # 拼接ZIP包保存路径，按PDF名称唯一命名
    zip_save_path = local_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(resp.content)
    logger.info(f"[步骤1/4] ZIP包下载成功，保存路径：{zip_save_path}")

    extract_target_dir = local_dir_obj / pdf_stem

    # 清理旧目录，异常则警告不终止
    if extract_target_dir.exists():
        try:
            # 递归删除整个目录树，包括目录本身及其所有子目录和文件。
            shutil.rmtree(extract_target_dir)
            logger.info(f"[步骤2/4] 已清理旧的解压目录：{extract_target_dir}")
        except Exception as e:
            logger.warning(f"[步骤2/4] 清理旧目录失败，可能不影响新文件解压：{str(e)}")


    extract_target_dir.mkdir(parents=True, exist_ok=True)




    #解压zip 使用zipfile解压和压缩


    with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
        zip_ref.extractall(extract_target_dir)

    #返回md文件的地址
    logger.info(f"[步骤3/4] 开始查找解压目录中的MD文件...")
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"[步骤3/4] 解压目录中未找到任何.md格式文件：{extract_target_dir}")
    logger.info(f"[步骤3/4] 共找到{len(md_file_list)}个MD文件，按优先级匹配目标文件")

    # 4. 按优先级匹配目标MD文件（同名→full.md→第一个，兜底避免流程中断）
    target_md_file = None
    # 优先级1：与PDF纯名称完全同名的MD文件
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[步骤4/4] 匹配到优先级1目标：与PDF同名的MD文件 {target_md_file.name}")
            break
    # 优先级2：MinerU默认生成的full.md（不区分大小写）
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[步骤4/4] 匹配到优先级2目标：MinerU默认文件 {target_md_file.name}")
                break
    # 优先级3：兜底取第一个MD文件
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[步骤4/4] 未匹配到前两级目标，兜底取第一个MD文件 {target_md_file.name}")

    # 重命名MD文件：统一为PDF纯名称，便于后续流程处理（仅不同名时执行）
    if target_md_file.stem != pdf_stem:
        logger.info(f"[步骤4/4] 开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            # 将磁盘上的文件进行重命名
            target_md_file.rename(new_md_path)
            # 更新变量引用
            target_md_file = new_md_path
            logger.info(f"[步骤4/4] MD文件重命名成功：{pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[步骤4/4] MD文件重命名失败，将使用原文件名继续流程：{str(e)}")

    # 转换为字符串绝对路径返回，适配后续仅支持字符串路径的函数
    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== [{pdf_stem}]解析结果处理完成，最终MD文件路径：{final_md_path} =====")
    return final_md_path


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
