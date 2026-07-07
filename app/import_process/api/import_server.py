import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)


app = FastAPI(
    title="File Import Service",
    description="Upload files to knowledge base: parse, split, embed and import to Milvus.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return RedirectResponse(url="/import")


@app.get("/import", response_class=FileResponse)
async def get_import_page():
    html_abs_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not html_abs_path.exists():
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")
    return FileResponse(path=html_abs_path, media_type="text/html")


@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_file(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    today_str = datetime.today().strftime("%Y-%m-%d")
    base_out_path = PROJECT_ROOT / "output" / today_str
    task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        add_running_task(task_id, "upload_file")

        dir_path = base_out_path / task_id
        dir_path.mkdir(parents=True, exist_ok=True)
        local_file_path = dir_path / file.filename

        with open(local_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_import_graph, task_id, str(local_file_path), str(dir_path))
        logger.info(f"{task_id}:文件上传完成，后台导入任务已启动")

    return {
        "code": 200,
        "message": "文件上传成功，后台导入任务已启动",
        "task_ids": task_ids,
    }


def run_import_graph(task_id: str, local_file_path: str, local_dir_path: str):
    add_done_task(task_id, "upload_file")
    add_running_task(task_id, "run_import_graph")
    try:
        update_task_status(task_id, "processing")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_file_path"] = local_file_path
        init_state["local_dir"] = local_dir_path

        final_state = kb_import_app.invoke(init_state)
        logger.info(f"{task_id}:图执行完成，最终状态字段：{list(final_state.keys())}")

        add_done_task(task_id, "run_import_graph")
        update_task_status(task_id, "completed")
        logger.info(f"{task_id}:导入流程执行完成")

    except Exception:
        logger.exception("图执行失败")
        update_task_status(task_id, "failed")


@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询处理进度")
async def get_task_progress(task_id: str):
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，"
        f"已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8009)
