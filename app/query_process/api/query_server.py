from pathlib import Path
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from loguru import logger

from app.clients.mongo_history_utils import get_recent_messages, clear_history
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import create_sse_queue, sse_generator
from app.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)


app = FastAPI(title="query service", description="掌柜智库查询服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str | None = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


@app.get("/")
async def index():
    return RedirectResponse(url="/chat.html")


@app.get("/health")
async def health():
    logger.info("触发后台健康请求接口")
    return {"status": "ok"}


@app.get("/chat.html")
async def chat():
    chat_html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat html not exist")
    return FileResponse(chat_html_path, media_type="text/html")


def _task_payload(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "status": get_task_status(session_id) or "pending",
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
        "answer": get_task_result(session_id, "answer", ""),
        "error": get_task_result(session_id, "error", ""),
        "image_urls": get_task_result(session_id, "image_urls", []),
    }


def run_query_graph(original_query: str, session_id: str, is_stream: bool):
    update_task_status(session_id, "processing", is_stream)
    try:
        state = create_query_default_state(
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream,
        )
        final_state = query_app.invoke(state)
        answer = final_state.get("answer", "")
        image_urls = final_state.get("image_urls") or get_task_result(session_id, "image_urls", [])
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "image_urls", image_urls)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"session_id:{session_id} 查询出现异常: {e}")
        set_task_result(session_id, "error", str(e))
        update_task_status(session_id, "failed", is_stream)


@app.get("/status/{session_id}")
async def status(session_id: str):
    return _task_payload(session_id)


@app.get("/history/{session_id}")
async def history(session_id: str):
    try:
        from app.clients.mongo_history_utils import get_recent_messages

        items = get_recent_messages(session_id, limit=50)
        for item in items:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        return {"items": items}
    except Exception as e:
        logger.warning(f"获取历史记录失败 session_id={session_id}: {e}")
        return {"items": []}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    try:
        from app.clients.mongo_history_utils import clear_history

        return {"deleted": clear_history(session_id)}
    except Exception as e:
        logger.warning(f"清空历史记录失败 session_id={session_id}: {e}")
        return {"deleted": 0, "error": str(e)}


@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    query_text = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, query_text, session_id, is_stream)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id,
        }

    run_query_graph(query_text, session_id, is_stream)
    return {
        **_task_payload(session_id),
        "message": "处理完成",
    }


@app.get("/stream/{session_id}")
async def stream(request: Request, session_id: str):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
    )


@app.get('/history/{session_id}')
async def history(session_id: str, limit: int = 10):
    chats = get_recent_messages(session_id, limit=limit)

    return {
        "session_id": session_id,
        "items": chats
    }

@app.delete('/history/{session_id}')
async def delete_history(session_id: str):
    delete_count = clear_history(session_id)
    return {
        "message": f"{session_id}已删除",
        "deleted": delete_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
