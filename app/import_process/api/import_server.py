import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from starlette.background import BackgroundTask

# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


#访问页面
@app.get("/")
async def index():
    return RedirectResponse(url="/import")


@app.get("/import", response_class=FileResponse)
async def get_import_page():
    html_abs_path = PROJECT_ROOT / "app" / "import_process" / "page"/ "import.html"
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

        # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )

#上传文件
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_file(background_tasks:BackgroundTasks, files: List[UploadFile] = File(...)):# ...表示必传

    # 日期文件夹
    today_str =datetime.today().strftime('%Y-%m-%d')
    base_out_path = PROJECT_ROOT / "output" /today_str
    task_ids = [] #记录每个任务id

    for file in files:

        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        add_running_task(task_id, "upload_file")
        #文件的目录
        dir_path = base_out_path / task_id
        dir_path.mkdir(parents=True, exist_ok=True)
        #文件的地址
        local_file_path  = dir_path / file.filename

        with open(local_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        add_done_task(task_id, "upload_file")


#文件已经保存好了，接下来让 FastAPI 在后台执行 run_import_graph，去跑知识库导入流程，不要让上传接口一直卡着等待。
        background_tasks.add_task(run_import_graph,  task_id, str(local_file_path),str(dir_path))#最重要的
        logger.info(f"{task_id}:完成文件上传,返回了结果")

    return {
        "code": 200,
        "message": "文件上传成功，后台导入任务已启动",
        "task_ids": task_ids,
    }


def run_import_graph(task_id:str, local_file_path:str ,local_dir_path:str):


    add_done_task(task_id, "upload_file")
    add_running_task(task_id, "run_import_graph")
    try:
        update_task_status(task_id,"processing")

        init_state = get_default_state()

        init_state["task_id"] = task_id
        init_state["local_file_path"] = local_file_path
        init_state["local_dir"] = local_dir_path

        for event in kb_import_app.invoke(init_state):
            for node_name,result in event.items():
                logger.info(f"{node_name}:完成,结果为{result}")
                add_done_task(task_id, node_name)

        add_done_task(task_id, "run_import_graph")
        update_task_status(task_id,"completed")

        logger.info(f"{task_id}图状态已经执行完毕")

    except Exception as e:
        logger.exception("图执行失败")
        update_task_status(task_id, "failed")

# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8010)

