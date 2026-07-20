import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..graph.builder import build_graph
from ..graph.state import AgentState
from ..memory.store import init_db, save_message, get_history, get_summary
from ..projects.manager import create_project, list_projects, get_project, delete_project, update_timestamp
from ..preferences.manager import get_preferences, save_preferences, apply_feedback
from ..preferences.models import PreferencesConfig
from ..tools import file_rag
from ..export.report import generate_report
from .models import (
    ChatRequest, CreateProjectRequest,
    UpdatePreferencesRequest, FeedbackRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    async with AsyncSqliteSaver.from_conn_string("data/research.db") as ck:
        app.state.checkpointer = ck
        app.state.graph = build_graph(checkpointer=ck)
        yield


app = FastAPI(title="Research Assistant Agent", lifespan=lifespan)

# 确保 plots 目录存在
os.makedirs(os.path.join("data", "plots"), exist_ok=True)
app.mount("/plots", StaticFiles(directory=os.path.join("data", "plots")), name="plots")


# ============================================================
# 项目 API
# ============================================================

@app.post("/projects")
def api_create_project(req: CreateProjectRequest):
    proj = create_project(req.name)
    # 为新项目初始化空偏好
    save_preferences(proj["id"], PreferencesConfig())
    return proj


@app.get("/projects")
def api_list_projects():
    return list_projects()


@app.get("/projects/{project_id}")
def api_get_project(project_id: str):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    return proj


@app.delete("/projects/{project_id}")
def api_delete_project(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    delete_project(project_id)
    file_rag.delete_project_index(project_id)
    return {"status": "deleted"}


@app.get("/projects/{project_id}/history")
def api_get_history(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    messages = get_history(project_id)
    return {"project_id": project_id, "messages": messages}


# ============================================================
# 偏好 API
# ============================================================

@app.get("/projects/{project_id}/preferences")
def api_get_preferences(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return get_preferences(project_id)


@app.put("/projects/{project_id}/preferences")
def api_update_preferences(project_id: str, req: UpdatePreferencesRequest):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    # 合并：只更新传入的非空字段
    current = get_preferences(project_id)
    if req.literature is not None:
        current.literature = req.literature
    if req.writing is not None:
        current.writing = req.writing
    if req.experiment is not None:
        current.experiment = req.experiment
    if req.tool is not None:
        current.tool = req.tool
    save_preferences(project_id, current)
    return current


# ============================================================
# 反馈 API
# ============================================================

@app.post("/feedback")
def api_feedback(req: FeedbackRequest):
    if not get_project(req.project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    if req.tag:
        updated = apply_feedback(req.project_id, req.tag)
        if updated is not None:
            return {
                "status": "applied",
                "tag": req.tag,
                "preferences": updated,
            }
        return {
            "status": "ignored",
            "tag": req.tag,
            "reason": "该标签无匹配的偏好规则",
        }

    # 仅有 type/comment 的反馈，暂存供后续分析（不做偏好调整）
    return {
        "status": "noted",
        "type": req.type,
        "comment": req.comment,
    }


# ============================================================
# 文件上传 + RAG API
# ============================================================

UPLOAD_BASE = os.path.join("data", "uploads")


@app.post("/projects/{project_id}/upload")
async def api_upload_file(project_id: str, file: UploadFile = File(...)):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，仅支持 PDF 和 Word 文档")

    # 保存文件
    proj_dir = os.path.join(UPLOAD_BASE, project_id)
    os.makedirs(proj_dir, exist_ok=True)
    file_path = os.path.join(proj_dir, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 索引文档
    try:
        chunk_count = file_rag.index_document(project_id, file_path, file.filename)
    except Exception as e:
        # 索引失败时删除已保存的文件
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档索引失败: {e}")

    return {
        "status": "indexed",
        "filename": file.filename,
        "size": len(content),
        "chunks": chunk_count,
    }


@app.get("/projects/{project_id}/files")
def api_list_files(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return file_rag.get_project_files(project_id)


@app.delete("/projects/{project_id}/files/{filename}")
def api_delete_file(project_id: str, filename: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    file_path = os.path.join(UPLOAD_BASE, project_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(file_path)

    # 重建索引（移除该文件的所有 chunks）
    file_rag.delete_project_index(project_id)
    # 重新索引剩余文件
    proj_dir = os.path.join(UPLOAD_BASE, project_id)
    if os.path.exists(proj_dir):
        for f in os.listdir(proj_dir):
            fpath = os.path.join(proj_dir, f)
            if os.path.isfile(fpath):
                try:
                    file_rag.index_document(project_id, fpath, f)
                except Exception:
                    pass

    return {"status": "deleted", "filename": filename}


# ============================================================
# 报告导出 API
# ============================================================

@app.post("/projects/{project_id}/export")
async def api_export_report(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    report = await generate_report(project_id)
    return {"project_id": project_id, "report": report}


# ============================================================
# 对话 API
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    if not get_project(request.project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    save_message(request.project_id, "user", request.message)
    update_timestamp(request.project_id)

    existing_summary = get_summary(request.project_id)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "project_id": request.project_id,
        "summary": existing_summary,
        "system_prompt": "",
        "search_results": [],
        "retrieved_docs": [],
        "agent_outputs": {},
        "next_agent": "",
        "supervisor_log": [],
        "required_tools": request.tools,
    }

    config = {"configurable": {"thread_id": request.project_id}}
    graph = req.app.state.graph

    AGENT_NODES = {"researcher", "analyst", "reviewer", "generate_response"}

    async def event_stream():
        full_response = ""
        current_agent = None

        async for event in graph.astream_events(initial_state, config, version="v2"):
            kind = event["event"]

            if kind == "on_chain_start":
                name = event.get("name", "")
                if name in AGENT_NODES:
                    current_agent = name
                    yield f"data: {json.dumps({'type': 'agent_phase', 'agent': name, 'status': 'start'}, ensure_ascii=False)}\n\n"

            elif kind == "on_chain_end":
                name = event.get("name", "")
                if name in AGENT_NODES:
                    yield f"data: {json.dumps({'type': 'agent_phase', 'agent': name, 'status': 'end'}, ensure_ascii=False)}\n\n"
                    current_agent = None

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and current_agent == "generate_response":
                    full_response += chunk.content
                    yield f"data: {json.dumps({'type': 'response', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name'], 'status': 'start'}, ensure_ascii=False)}\n\n"
            elif kind == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name'], 'status': 'end'}, ensure_ascii=False)}\n\n"

        save_message(request.project_id, "assistant", full_response)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
