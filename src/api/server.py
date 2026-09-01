import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from ..graph.builder import build_graph
from ..graph.state import AgentState
from ..context.compression import compress_conversation
from ..storage.db import init_db
from ..storage.records import save_message, get_history, get_summary, save_project_sources, get_project_sources, save_project_plan, get_latest_plan
from ..storage.projects import create_project, list_projects, get_project, delete_project, rename_project, update_timestamp
from ..preferences.manager import get_preferences, save_preferences, apply_feedback, get_raw_preferences, save_raw_preferences
from ..rag.store import get_project_files, index_document, delete_project_index
from ..export.report import generate_report
from ..interaction.events import iter_interrupt_events
from ..interaction.resume import parse_resume
from .models import (
    ChatRequest, CreateProjectRequest, RenameProjectRequest,
    UpdatePreferencesRequest, FeedbackRequest, ResumeRequest,
    RawPreferencesRequest,
)


async def _background_compress(project_id: str, graph_state):
    """后台压缩历史消息，保存摘要供下次请求使用。不阻塞当前响应。"""
    state_values = graph_state.values if hasattr(graph_state, "values") else {}
    await compress_conversation(
        project_id,
        list(state_values.get("messages", [])),
        state_values.get("summary", ""),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    async with AsyncSqliteSaver.from_conn_string("data/research.db") as ck:
        app.state.checkpointer = ck
        app.state.graph = build_graph(checkpointer=ck)
        yield


app = FastAPI(title="Research Assistant Agent", lifespan=lifespan)


# ============================================================
# 项目 API
# ============================================================

@app.post("/projects")
def api_create_project(req: CreateProjectRequest):
    return create_project(req.name)


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
async def api_delete_project(project_id: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    delete_project(project_id)
    delete_project_index(project_id)
    try:
        await app.state.checkpointer.delete_thread(project_id)
    except Exception:
        pass
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

@app.get("/preferences")
def api_get_preferences():
    return get_preferences()


@app.put("/preferences")
def api_update_preferences(req: UpdatePreferencesRequest):
    # 合并：只更新传入的非空字段
    current = get_preferences()
    if req.literature is not None:
        current.literature = req.literature
    if req.writing is not None:
        current.writing = req.writing
    if req.experiment is not None:
        current.experiment = req.experiment
    if req.tool is not None:
        current.tool = req.tool
    save_preferences(current)
    return current


# ============================================================
# 原始偏好文件 API（用于设置编辑器）
# ============================================================

@app.get("/preferences/raw")
def api_get_raw_preferences():
    """返回 preferences.md 的完整原始内容。"""
    return {"content": get_raw_preferences()}


@app.put("/preferences/raw")
def api_update_raw_preferences(req: RawPreferencesRequest):
    """保存原始 markdown 内容到 preferences.md，同时解析 YAML 返回结构化结果。"""
    parsed = save_raw_preferences(req.content)
    return {
        "status": "saved",
        "preferences": parsed.model_dump() if parsed else None,
    }


# ============================================================
# 反馈 API
# ============================================================

@app.post("/feedback")
def api_feedback(req: FeedbackRequest):
    if req.tag:
        updated = apply_feedback(req.tag)
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
        chunk_count = index_document(project_id, file_path, file.filename)
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
    return get_project_files(project_id)


@app.delete("/projects/{project_id}/files/{filename}")
def api_delete_file(project_id: str, filename: str):
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    file_path = os.path.join(UPLOAD_BASE, project_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(file_path)

    # 重建索引（移除该文件的所有 chunks）
    delete_project_index(project_id)
    # 重新索引剩余文件
    proj_dir = os.path.join(UPLOAD_BASE, project_id)
    if os.path.exists(proj_dir):
        for f in os.listdir(proj_dir):
            fpath = os.path.join(proj_dir, f)
            if os.path.isfile(fpath):
                try:
                    index_document(project_id, fpath, f)
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

def _merge_ratings_into_sources(all_sources, ratings):
    """将可信度评价合并到 sources 中，确保持久化后刷新页面不丢失。

    未被评级覆盖的来源（中途死链丢弃 / 未纳入最终评审）标记为"未评级"，
    让用户区分"低可信"与"未评估"，而不是无标签。
    """
    if not ratings:
        for s in all_sources:
            s["credibility"] = "未评级"
        return
    rating_map = {}
    for r in ratings:
        sn = r.get("source_number")
        if sn is not None:
            rating_map[sn] = {"credibility": r.get("credibility", "中"), "reason": r.get("reason", "")}
    for s in all_sources:
        sn = s.get("source_number")
        if sn in rating_map:
            s["credibility"] = rating_map[sn]["credibility"]
            s["reason"] = rating_map[sn]["reason"]
        else:
            s["credibility"] = "未评级"


def _merge_state_sources(graph_state, message_index, all_sources):
    """从 graph state 的 reference_sources 中读取来源，合并到 all_sources。

    沿用图内 source_number（与 _collect_researcher_sources 一致），保证编号与
    reviewer 评级的 source_number 对齐，避免评级匹配失败/错配。
    """
    state_values = graph_state.values if hasattr(graph_state, "values") else {}
    ref_sources = state_values.get("reference_sources", [])
    new_sources: list[dict] = []
    if not ref_sources:
        return new_sources
    for s in ref_sources:
        sid = s.get("id", "")
        if not sid:
            continue
        if not any(x.get("id") == sid for x in all_sources):
            s = dict(s)
            s["message_index"] = message_index
            all_sources.append(s)
            new_sources.append(s)
    return new_sources


def _collect_researcher_sources(output, message_index, all_sources):
    """从 researcher 节点输出（已过滤死链）中提取来源，追加到 all_sources，返回新增列表。

    沿用图内 source_number，保证前端展示编号与 reviewer 评级编号一致。
    """
    ref_sources = (output or {}).get("reference_sources", [])
    if not ref_sources:
        return []
    new_sources = []
    for s in ref_sources:
        sid = s.get("id", "")
        if not sid:
            continue
        if not any(x.get("id") == sid for x in all_sources):
            s = dict(s)
            s["message_index"] = message_index
            all_sources.append(s)
            new_sources.append(s)
    return new_sources


def _persisted_sources_event(project_id: str, message_index: int) -> dict | None:
    """本轮未检索但项目已有历史来源（方案直通 / 澄清等场景）时，
    构造 source 事件 payload，让前端来源栏展示与回答引用一致的历史来源。
    无持久化来源时返回 None。
    """
    persisted = get_project_sources(project_id)
    if not persisted:
        return None
    return {
        "type": "source",
        "sources": [dict(s) | {"message_index": message_index} for s in persisted],
        "message_index": message_index,
    }


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
        "agent_outputs": {},
        "next_agent": "",
        "supervisor_log": [],
        "required_tools": request.tools,
        "reference_sources": [],
        "source_ratings": [],
        "source_assessments": [],
        "retrieval_gaps": [],
        "needs_refetch": False,
        "search_round": 0,
        "plan_options": [],
        "chosen_plan_id": "",
        "custom_plan_text": "",
        "effective_query": "",
        "was_clarified": False,
        "has_prior_research": False,
        "query_invalid": False,
    }

    config = {"configurable": {"thread_id": request.project_id}}
    graph = req.app.state.graph

    AGENT_NODES = {"supervisor", "researcher", "planner", "reviewer", "generate_response"}


    async def event_stream():
        full_response = ""
        current_agent = None
        message_index = len(get_history(request.project_id))
        all_sources: list[dict] = []  # 收集本轮所有来源，流结束后持久化
        interrupted = False

        try:
            async for event in graph.astream_events(initial_state, config, version="v2"):
                kind = event["event"]

                # DEBUG 打印已注释，需要时取消注释以分析 LangGraph 事件结构
                # if kind in ("on_chain_start", "on_chain_end"):
                #     name = event.get("name", "")
                #     tag = event.get("tags", [])
                #     meta = event.get("metadata", {})
                #     print(f"[DEBUG] {kind}: name={name!r}, tags={tag!r}, meta_keys={list(meta.keys())!r}", file=sys.stderr, flush=True)

                if kind == "on_chain_start":
                    name = event.get("name", "")
                    meta = event.get("metadata", {})
                    node = meta.get("langgraph_node", "") or name
                    if not node or node not in AGENT_NODES:
                        if name in AGENT_NODES:
                            node = name
                        elif name.endswith("_node") and name[:-5] in AGENT_NODES:
                            node = name[:-5]
                    if node in AGENT_NODES:
                        current_agent = node
                        yield f"data: {json.dumps({'type': 'agent_phase', 'agent': node, 'status': 'start'}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    name = event.get("name", "")
                    meta = event.get("metadata", {})
                    node = meta.get("langgraph_node", "") or name
                    if not node or node not in AGENT_NODES:
                        if name in AGENT_NODES:
                            node = name
                        elif name.endswith("_node") and name[:-5] in AGENT_NODES:
                            node = name[:-5]
                    if node in AGENT_NODES:
                        yield f"data: {json.dumps({'type': 'agent_phase', 'agent': node, 'status': 'end'}, ensure_ascii=False)}\n\n"
                        current_agent = None
                        # researcher 完成后推送图内已过滤（无死链）的来源
                        if node == "researcher":
                            output = event["data"].get("output") or {}
                            node_sources = _collect_researcher_sources(output, message_index, all_sources)
                            if node_sources:
                                yield f"data: {json.dumps({'type': 'source', 'sources': node_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        if current_agent == "generate_response":
                            full_response += chunk.content
                            yield f"data: {json.dumps({'type': 'response', 'content': chunk.content, 'agent': current_agent}, ensure_ascii=False)}\n\n"
                        # 其他 agent 的 LLM 输出不推送到前端，避免泄露内部决策文本

                elif kind == "on_tool_start":
                    meta = event.get("metadata", {})
                    agent = meta.get("langgraph_node", "") or current_agent or ""
                    print(f"[DEBUG] on_tool_start: name={event['name']!r}, meta_langgraph_node={meta.get('langgraph_node')!r}, current_agent={current_agent!r}", file=sys.stderr, flush=True)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name'], 'status': 'start', 'agent': agent}, ensure_ascii=False)}\n\n"
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    meta = event.get("metadata", {})
                    agent = meta.get("langgraph_node", "") or current_agent or ""
                    print(f"[DEBUG] on_tool_end: name={tool_name!r}, meta_langgraph_node={meta.get('langgraph_node')!r}, current_agent={current_agent!r}", file=sys.stderr, flush=True)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'status': 'end', 'agent': agent}, ensure_ascii=False)}\n\n"
        except GraphInterrupt:
            interrupted = True
        except Exception as e:
            # 流内异常也视为中断，确保前端能收到 done 事件
            import traceback
            print(f"[ERROR] 图执行异常: {e}")
            traceback.print_exc()
            interrupted = True

        try:
            # LangGraph 1.x 中 astream_events 不会抛出 GraphInterrupt，流会静默结束。
            # 因此需要在流结束后主动检查 state 是否被 interrupt 暂停。
            if not interrupted:
                graph_state = await graph.aget_state(config)
                if graph_state.interrupts:
                    interrupted = True

            if interrupted:
                graph_state = await graph.aget_state(config)
                interaction_events = iter_interrupt_events(graph_state.interrupts, message_index)
                state_sources = _merge_state_sources(graph_state, message_index, all_sources)
                if state_sources:
                    yield f"data: {json.dumps({'type': 'source', 'sources': state_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"
                elif not all_sources:
                    # 本轮未检索（方案直通 / 澄清等）但有历史来源：推送持久化来源，保证引用与面板一致
                    _ev = _persisted_sources_event(request.project_id, message_index)
                    if _ev:
                        yield f"data: {json.dumps(_ev, ensure_ascii=False)}\n\n"
                state_values = graph_state.values if hasattr(graph_state, "values") else {}
                ratings = state_values.get("source_ratings", [])
                _merge_ratings_into_sources(all_sources, ratings)
                if all_sources:
                    save_project_sources(request.project_id, all_sources)
                if ratings:
                    yield f"data: {json.dumps({'type': 'source_ratings', 'ratings': ratings, 'message_index': message_index}, ensure_ascii=False)}\n\n"
                for ev in interaction_events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                asyncio.create_task(_background_compress(request.project_id, graph_state))
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # 如果流式未捕获到回复内容，从 state 中读取（流式事件可能未触发）
            if not full_response:
                state_values = graph_state.values if hasattr(graph_state, "values") else {}
                msgs = state_values.get("messages", [])
                for m in reversed(msgs):
                    if hasattr(m, "type") and m.type == "ai" and hasattr(m, "content") and m.content:
                        full_response = m.content
                        yield f"data: {json.dumps({'type': 'response', 'content': full_response}, ensure_ascii=False)}\n\n"
                        break

            save_message(request.project_id, "assistant", full_response)
            state_sources = _merge_state_sources(graph_state, message_index, all_sources)
            if state_sources:
                yield f"data: {json.dumps({'type': 'source', 'sources': state_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            elif not all_sources:
                # 本轮未检索（方案直通 / 澄清等）但有历史来源：推送持久化来源，保证引用与面板一致
                _ev = _persisted_sources_event(request.project_id, message_index)
                if _ev:
                    yield f"data: {json.dumps(_ev, ensure_ascii=False)}\n\n"
            state_values = graph_state.values if hasattr(graph_state, "values") else {}
            ratings = state_values.get("source_ratings", [])
            if ratings:
                yield f"data: {json.dumps({'type': 'source_ratings', 'ratings': ratings, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            _merge_ratings_into_sources(all_sources, ratings)
            if all_sources:
                save_project_sources(request.project_id, all_sources)
            asyncio.create_task(_background_compress(request.project_id, graph_state))
            # 本轮对话已正常完成，无需再保留快照：清理该线程的所有 checkpoint，
            # 防止每个 super-step 的完整 state 快照在 checkpoints/writes 表中无限累积。
            # 注意：interrupt 等待期间不清理（resume 需要），仅在此"未中断完成"分支删除。
            try:
                await app.state.checkpointer.delete_thread(request.project_id)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            # 确保无论如何都发送 done，防止前端永久显示"停止"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# 方案选择恢复 API（人机协同）
# ============================================================

@app.post("/chat/resume")
async def chat_resume(request: ResumeRequest, req: Request):
    """恢复被交互节点 interrupt 暂停的图执行。"""
    if not get_project(request.project_id):
        raise HTTPException(status_code=404, detail="项目不存在")

    config = {"configurable": {"thread_id": request.project_id}}
    graph = req.app.state.graph

    # 通用解析：把请求体解析为 interrupt() 的返回值
    user_choice = parse_resume(request)

    # 方案选择持久化（仅 plan_options 交互）
    if (request.type or "plan_options") == "plan_options":
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

    AGENT_NODES = {"supervisor", "researcher", "planner", "reviewer", "generate_response"}


    async def event_stream():
        full_response = ""
        current_agent = None
        message_index = len(get_history(request.project_id))
        all_sources: list[dict] = []
        interrupted = False

        try:
            async for event in graph.astream_events(Command(resume=user_choice), config, version="v2"):
                kind = event["event"]

                if kind in ("on_chain_start", "on_chain_end"):
                    name = event.get("name", "")
                    tag = event.get("tags", [])
                    meta = event.get("metadata", {})
                    print(f"[DEBUG] resume {kind}: name={name!r}, tags={tag!r}, meta_keys={list(meta.keys())!r}", file=sys.stderr, flush=True)

                if kind == "on_chain_start":
                    name = event.get("name", "")
                    meta = event.get("metadata", {})
                    node = meta.get("langgraph_node", "") or name
                    if not node or node not in AGENT_NODES:
                        if name in AGENT_NODES:
                            node = name
                        elif name.endswith("_node") and name[:-5] in AGENT_NODES:
                            node = name[:-5]
                    if node in AGENT_NODES:
                        current_agent = node
                        yield f"data: {json.dumps({'type': 'agent_phase', 'agent': node, 'status': 'start'}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    name = event.get("name", "")
                    meta = event.get("metadata", {})
                    node = meta.get("langgraph_node", "") or name
                    if not node or node not in AGENT_NODES:
                        if name in AGENT_NODES:
                            node = name
                        elif name.endswith("_node") and name[:-5] in AGENT_NODES:
                            node = name[:-5]
                    if node in AGENT_NODES:
                        yield f"data: {json.dumps({'type': 'agent_phase', 'agent': node, 'status': 'end'}, ensure_ascii=False)}\n\n"
                        current_agent = None
                        # researcher 完成后推送图内已过滤（无死链）的来源
                        if node == "researcher":
                            output = event["data"].get("output") or {}
                            node_sources = _collect_researcher_sources(output, message_index, all_sources)
                            if node_sources:
                                yield f"data: {json.dumps({'type': 'source', 'sources': node_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        if current_agent == "generate_response":
                            full_response += chunk.content
                            yield f"data: {json.dumps({'type': 'response', 'content': chunk.content, 'agent': current_agent}, ensure_ascii=False)}\n\n"
                        # 其他 agent 的 LLM 输出不推送到前端，避免泄露内部决策文本

                elif kind == "on_tool_start":
                    meta = event.get("metadata", {})
                    agent = meta.get("langgraph_node", "") or current_agent or ""
                    print(f"[DEBUG] resume on_tool_start: name={event['name']!r}, meta_langgraph_node={meta.get('langgraph_node')!r}, current_agent={current_agent!r}", file=sys.stderr, flush=True)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': event['name'], 'status': 'start', 'agent': agent}, ensure_ascii=False)}\n\n"
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    meta = event.get("metadata", {})
                    agent = meta.get("langgraph_node", "") or current_agent or ""
                    print(f"[DEBUG] resume on_tool_end: name={tool_name!r}, meta_langgraph_node={meta.get('langgraph_node')!r}, current_agent={current_agent!r}", file=sys.stderr, flush=True)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'status': 'end', 'agent': agent}, ensure_ascii=False)}\n\n"
        except GraphInterrupt:
            interrupted = True
        except Exception as e:
            import traceback
            print(f"[ERROR] resume 图执行异常: {e}")
            traceback.print_exc()
            interrupted = True

        try:
            if not interrupted:
                graph_state = await graph.aget_state(config)
                if graph_state.interrupts:
                    interrupted = True

            if interrupted:
                graph_state = await graph.aget_state(config)
                interaction_events = iter_interrupt_events(graph_state.interrupts, message_index)
                state_sources = _merge_state_sources(graph_state, message_index, all_sources)
                if state_sources:
                    yield f"data: {json.dumps({'type': 'source', 'sources': state_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"
                elif not all_sources:
                    # 本轮未检索（方案直通 / 澄清等）但有历史来源：推送持久化来源，保证引用与面板一致
                    _ev = _persisted_sources_event(request.project_id, message_index)
                    if _ev:
                        yield f"data: {json.dumps(_ev, ensure_ascii=False)}\n\n"
                for ev in interaction_events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                asyncio.create_task(_background_compress(request.project_id, graph_state))
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # 如果流式未捕获到回复内容，从 state 中读取（流式事件可能未触发）
            if not full_response:
                state_values = graph_state.values if hasattr(graph_state, "values") else {}
                msgs = state_values.get("messages", [])
                for m in reversed(msgs):
                    if hasattr(m, "type") and m.type == "ai" and hasattr(m, "content") and m.content:
                        full_response = m.content
                        yield f"data: {json.dumps({'type': 'response', 'content': full_response}, ensure_ascii=False)}\n\n"
                        break

            save_message(request.project_id, "assistant", full_response)
            state_sources = _merge_state_sources(graph_state, message_index, all_sources)
            if state_sources:
                yield f"data: {json.dumps({'type': 'source', 'sources': state_sources, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            elif not all_sources:
                # 本轮未检索（方案直通 / 澄清等）但有历史来源：推送持久化来源，保证引用与面板一致
                _ev = _persisted_sources_event(request.project_id, message_index)
                if _ev:
                    yield f"data: {json.dumps(_ev, ensure_ascii=False)}\n\n"
            state_values = graph_state.values if hasattr(graph_state, "values") else {}
            ratings = state_values.get("source_ratings", [])
            if ratings:
                yield f"data: {json.dumps({'type': 'source_ratings', 'ratings': ratings, 'message_index': message_index}, ensure_ascii=False)}\n\n"
            _merge_ratings_into_sources(all_sources, ratings)
            if all_sources:
                save_project_sources(request.project_id, all_sources)
            asyncio.create_task(_background_compress(request.project_id, graph_state))
            try:
                await app.state.checkpointer.delete_thread(request.project_id)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
