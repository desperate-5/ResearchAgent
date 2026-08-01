import json
import os
import re
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from ..graph.builder import build_graph
from ..graph.state import AgentState
from ..memory.store import init_db, save_message, get_history, get_summary, save_project_sources, get_project_sources, save_project_plan, get_latest_plan
from ..projects.manager import create_project, list_projects, get_project, delete_project, rename_project, update_timestamp
from ..preferences.manager import get_preferences, save_preferences, apply_feedback
from ..preferences.models import PreferencesConfig
from ..tools import file_rag
from ..export.report import generate_report
from .models import (
    ChatRequest, CreateProjectRequest, RenameProjectRequest,
    UpdatePreferencesRequest, FeedbackRequest, PlanResumeRequest,
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


@app.put("/projects/{project_id}")
def api_rename_project(project_id: str, req: RenameProjectRequest):
    proj = rename_project(project_id, req.name)
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    return proj


@app.get("/projects/{project_id}/sources")
def api_get_sources(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"project_id": project_id, "sources": get_project_sources(project_id)}


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

def _parse_web_search_sources(text: str) -> list[dict]:
    """解析 web_search 工具输出中的结构化来源信息。"""
    sources = []
    # 匹配格式: N. title\n   来源: site | URL: url\n   summary
    pattern = re.compile(
        r'(\d+)\.\s*(.+?)\n\s+来源:\s*(.+?)\s*\|\s*URL:\s*(.+?)\n\s+(.+?)(?=\n\n\d+\.|\n\d+\.|\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        url = m.group(4).strip()
        title = m.group(2).strip()
        sid = hashlib.md5(f"{url}|{title}".encode()).hexdigest()[:12]
        sources.append({
            "id": sid,
            "title": title,
            "url": url,
            "summary": m.group(5).strip()[:300],
            "source_type": "web",
        })
    return sources


def _parse_paper_sources(text: str) -> list[dict]:
    """解析 aminer_search_papers 工具输出中的结构化来源信息。"""
    sources = []
    # 匹配格式: N. **title**\n   第一作者: ...\n   来源: ...\n   DOI: ...
    pattern = re.compile(
        r'(\d+)\.\s+\*\*(.+?)\*\*\n((?:(?!\n\d+\.\s).)*)',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        title = m.group(2).strip()
        meta = m.group(3)
        doi = ""
        doi_match = re.search(r'DOI:\s*(\S+)', meta)
        if doi_match:
            doi = doi_match.group(1).strip()
        url = f"https://doi.org/{doi}" if doi else ""
        sid = hashlib.md5(f"{url}|{title}".encode()).hexdigest()[:12]
        sources.append({
            "id": sid,
            "title": title,
            "url": url,
            "summary": meta.strip()[:300],
            "source_type": "paper",
        })
    return sources


def _parse_rag_sources(text: str) -> list[dict]:
    """解析 search_uploaded_docs 工具输出中的文档来源信息。"""
    sources = []
    # 匹配格式: [N] 来源: filename\ncontent
    pattern = re.compile(r'\[(\d+)\]\s*来源:\s*(.+?)\n(.+?)(?=\[\d+\]|\Z)', re.DOTALL)
    for m in pattern.finditer(text):
        title = m.group(2).strip()
        sid = hashlib.md5(title.encode()).hexdigest()[:12]
        sources.append({
            "id": sid,
            "title": title,
            "url": "",
            "summary": m.group(3).strip()[:300],
            "source_type": "document",
        })
    return sources


def _parse_tool_sources(tool_name: str, output_text: str) -> list[dict]:
    """根据工具名称解析工具输出文本中的结构化来源信息。"""
    if tool_name == "web_search":
        return _parse_web_search_sources(output_text)
    elif tool_name == "aminer_search_papers":
        return _parse_paper_sources(output_text)
    elif tool_name == "search_uploaded_docs":
        return _parse_rag_sources(output_text)
    return []


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
        "reference_sources": [],
        "plan_options": [],
        "chosen_plan_id": "",
        "chosen_plan_detail": {},
        "custom_plan_text": "",
    }

    config = {"configurable": {"thread_id": request.project_id}}
    graph = req.app.state.graph

    AGENT_NODES = {"researcher", "analyst", "planner", "reviewer", "generate_response"}

    async def event_stream():
        full_response = ""
        current_agent = None
        message_index = len(get_history(request.project_id))
        source_number_map: dict[str, int] = {}
        source_counter = [0]  # list for mutability in closure
        all_sources: list[dict] = []  # 收集本轮所有来源，流结束后持久化
        interrupted = False

        try:
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
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'status': 'end'}, ensure_ascii=False)}\n\n"
                    # 解析工具输出中的结构化来源信息
                    output = event["data"].get("output")
                    if output is not None and tool_name in ("web_search", "aminer_search_papers", "search_uploaded_docs"):
                        output_text = str(output.content) if hasattr(output, "content") else str(output)
                        sources = _parse_tool_sources(tool_name, output_text)
                        if sources:
                            # 分配编号、去重
                            numbered = []
                            for s in sources:
                                sid = s["id"]
                                if sid not in source_number_map:
                                    source_counter[0] += 1
                                    source_number_map[sid] = source_counter[0]
                                s["source_number"] = source_number_map[sid]
                                numbered.append(s)
                                # 收集到本轮来源列表（去重）
                                if not any(x["id"] == s["id"] for x in all_sources):
                                    s["message_index"] = message_index
                                    all_sources.append(dict(s))
                            yield f"data: {json.dumps({'type': 'source', 'sources': numbered, 'message_index': message_index}, ensure_ascii=False)}\n\n"
        except GraphInterrupt:
            interrupted = True

        # LangGraph 1.x 中 astream_events 不会抛出 GraphInterrupt，流会静默结束。
        # 因此需要在流结束后主动检查 state 是否被 interrupt 暂停。
        # 注意：必须用 aget_state，因为 AsyncSqliteSaver 在主线程不允许同步调用。
        if not interrupted:
            graph_state = await graph.aget_state(config)
            if graph_state.interrupts:
                interrupted = True
                # 检查 interrupt 是否是我们关注的 plan_options 类型
                is_plan = any(
                    isinstance(it.value, dict) and it.value.get("type") == "plan_options"
                    for it in graph_state.interrupts
                )
                if not is_plan:
                    interrupted = False  # 非 planner 中断，按正常流程处理

        if interrupted:
            graph_state = await graph.aget_state(config)
            plan_options = []
            for it in graph_state.interrupts:
                val = it.value
                if isinstance(val, dict) and val.get("type") == "plan_options":
                    plan_options = val.get("options", [])
                    break
            # 持久化本轮已收集的来源，防止中断导致来源丢失
            if all_sources:
                save_project_sources(request.project_id, all_sources)
            if plan_options:
                yield f"data: {json.dumps({'type': 'plan_options', 'options': plan_options, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return  # 不调 save_message，等用户选择后 resume

        save_message(request.project_id, "assistant", full_response)
        # 持久化本轮来源
        if all_sources:
            save_project_sources(request.project_id, all_sources)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# 方案选择恢复 API（人机协同）
# ============================================================

@app.post("/chat/resume")
async def chat_resume(request: PlanResumeRequest, req: Request):
    """恢复被 planner interrupt 暂停的图执行。"""
    if not get_project(request.project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    config = {"configurable": {"thread_id": request.project_id}}
    graph = req.app.state.graph

    # 构建用户选择
    user_choice = {
        "chosen_plan_id": request.chosen_plan_id,
        "custom_plan_text": request.custom_plan_text,
    }

    # 持久化方案选择
    plan_title = ""
    plan_detail = ""
    is_custom = False
    if request.chosen_plan_id:
        plan_title = f"方案 {request.chosen_plan_id}"
        plan_detail = f"用户选择了方案 {request.chosen_plan_id}"
        is_custom = False
    elif request.custom_plan_text:
        plan_title = "自定义方案"
        plan_detail = request.custom_plan_text
        is_custom = True
    if plan_detail:
        save_project_plan(request.project_id, request.chosen_plan_id, plan_title, plan_detail, is_custom)

    AGENT_NODES = {"researcher", "analyst", "planner", "reviewer", "generate_response"}

    async def event_stream():
        full_response = ""
        current_agent = None
        message_index = len(get_history(request.project_id))
        source_number_map: dict[str, int] = {}
        source_counter = [0]
        all_sources: list[dict] = []
        interrupted = False

        try:
            async for event in graph.astream_events(Command(resume=user_choice), config, version="v2"):
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
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'status': 'end'}, ensure_ascii=False)}\n\n"
                    output = event["data"].get("output")
                    if output is not None and tool_name in ("web_search", "aminer_search_papers", "search_uploaded_docs"):
                        output_text = str(output.content) if hasattr(output, "content") else str(output)
                        sources = _parse_tool_sources(tool_name, output_text)
                        if sources:
                            numbered = []
                            for s in sources:
                                sid = s["id"]
                                if sid not in source_number_map:
                                    source_counter[0] += 1
                                    source_number_map[sid] = source_counter[0]
                                s["source_number"] = source_number_map[sid]
                                numbered.append(s)
                                if not any(x["id"] == s["id"] for x in all_sources):
                                    s["message_index"] = message_index
                                    all_sources.append(dict(s))
                            yield f"data: {json.dumps({'type': 'source', 'sources': numbered, 'message_index': message_index}, ensure_ascii=False)}\n\n"
        except GraphInterrupt:
            interrupted = True

        # 防御：如果 resume 后又触发了 interrupt（如 supervisor 再次调用 planner）
        if not interrupted:
            graph_state = await graph.aget_state(config)
            if graph_state.interrupts:
                is_plan = any(
                    isinstance(it.value, dict) and it.value.get("type") == "plan_options"
                    for it in graph_state.interrupts
                )
                if is_plan:
                    interrupted = True

        if interrupted:
            graph_state = await graph.aget_state(config)
            plan_options = []
            for it in graph_state.interrupts:
                val = it.value
                if isinstance(val, dict) and val.get("type") == "plan_options":
                    plan_options = val.get("options", [])
                    break
            if plan_options:
                yield f"data: {json.dumps({'type': 'plan_options', 'options': plan_options, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        save_message(request.project_id, "assistant", full_response)
        if all_sources:
            save_project_sources(request.project_id, all_sources)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
