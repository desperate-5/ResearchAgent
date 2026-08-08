import os
import sys
import json
import re
import asyncio
import hashlib
import httpx

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from .state import AgentState

from ..tools.web_search import web_search
from ..tools.aminer_search import aminer_search_papers
from ..tools.file_rag import search_chunks
from ..memory.store import get_latest_plan
from ..preferences.manager import get_preferences
from ..preferences.prompt_builder import build_preference_prompt

from .prompts import (
    MAX_TOOL_ITERATIONS,
    RESEARCHER_PROMPT,
    PLANNER_PROMPT,
    PLAN_KEYWORDS,
)
from .context import (
    get_recent_messages,
    build_supervisor_context,
    build_reviewer_context,
    build_planner_context,
    build_generate_context,
)


def get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.7,
        streaming=streaming,
        request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


# ============================================================
# 工具集
# ============================================================

RESEARCHER_TOOLS = [web_search, aminer_search_papers]
REVIEWER_TOOLS = []

AVAILABLE_AGENTS = {"researcher", "planner"}


# ============================================================
# 图节点
# ============================================================

async def load_context_node(state: AgentState) -> dict:
    """加载上下文：偏好配置、历史摘要、项目文件列表。不做自动 RAG 注入（由 researcher 按需检索）。"""
    project_id = state["project_id"]

    prefs = get_preferences()
    prefs_text = build_preference_prompt(prefs)

    context_parts = []
    summary = state.get("summary", "")
    if summary:
        context_parts.append(f"## 历史对话摘要\n{summary}")
    if prefs_text:
        context_parts.append(prefs_text)

    # 列出项目已上传的文件，帮助 researcher 判断是否需要调用 search_uploaded_docs
    from ..tools.file_rag import get_project_files
    project_files = get_project_files(project_id)
    if project_files:
        file_names = [f["filename"] for f in project_files]
        context_parts.append(f"## 项目已上传文件\n{', '.join(file_names)}")

    latest_plan = get_latest_plan(project_id)
    if latest_plan:
        plan_prefix = "自定义" if latest_plan.get("is_custom") else "已选定"
        plan_info = f"## 当前项目的研究方案\n{plan_prefix}方案：{latest_plan.get('plan_title', '')}\n{latest_plan.get('plan_detail', '')}"
        context_parts.append(plan_info)

    return {
        "system_prompt": "\n\n".join(context_parts),
        "retrieved_docs": [],
    }


async def supervisor_node(state: AgentState) -> dict:
    """调度者节点：LLM 驱动的智能调度。

    由 LLM 根据用户问题、已有 agent 输出、工具约束等上下文，自主决定
    调用 researcher / planner / reviewer 或 FINISH。
    """
    log = state.get("supervisor_log", [])
    agent_outputs = state.get("agent_outputs", {})

    # 防止无限循环：最多调度 5 次
    if len(log) >= 5:
        return {
            "next_agent": "FINISH",
            "supervisor_log": log + [{"next": "FINISH", "reason": "达到最大调度次数"}],
        }

    # planner 已完成方案设计，不再调用
    if "planner" in agent_outputs:
        return {
            "next_agent": "FINISH",
            "supervisor_log": log + [{"next": "FINISH", "reason": "planner 已完成方案设计，结束调度"}],
        }

    # 关键词启发式：用户明确要求设计研究方案，且 researcher 已有输出 → 强制路由到 planner
    if "researcher" in agent_outputs:
        user_query = _extract_user_query(state)
        matched_kw = [kw for kw in PLAN_KEYWORDS if kw in user_query]
        print(f"[DEBUG supervisor] agent_outputs keys={list(agent_outputs.keys())}", file=sys.stderr, flush=True)
        print(f"[DEBUG supervisor] user_query={user_query[:200]!r}", file=sys.stderr, flush=True)
        print(f"[DEBUG supervisor] PLAN_KEYWORDS matched={matched_kw}", file=sys.stderr, flush=True)
        if matched_kw:
            print("[DEBUG supervisor] Forcing route to planner (keyword match)", file=sys.stderr, flush=True)
            return {
                "next_agent": "planner",
                "supervisor_log": log + [{"next": "planner", "reason": "用户明确要求设计研究方案，强制路由到 planner"}],
            }
        else:
            print("[DEBUG supervisor] No PLAN_KEYWORDS match, falling through to LLM", file=sys.stderr, flush=True)
    else:
        print(f"[DEBUG supervisor] agent_outputs keys={list(agent_outputs.keys())} (researcher not yet run)", file=sys.stderr, flush=True)

    llm = get_llm()
    msgs = build_supervisor_context(state)
    response = await llm.ainvoke(msgs)
    decision = _parse_decision(response.content)

    next_agent = decision.get("next", "FINISH")
    reason = decision.get("reason", "")

    # 校验 next_agent 合法性
    if next_agent not in AVAILABLE_AGENTS and next_agent != "FINISH":
        print(f"[DEBUG supervisor] LLM returned unknown agent {decision.get('next', '')!r}, falling back to FINISH", file=sys.stderr, flush=True)
        next_agent = "FINISH"
        reason = f"LLM 返回了未知 agent '{decision.get('next', '')}'，回退为 FINISH"

    print(f"[DEBUG supervisor] LLM decision: next={next_agent}, reason={reason!r}", file=sys.stderr, flush=True)
    return {
        "next_agent": next_agent,
        "supervisor_log": log + [{"next": next_agent, "reason": reason}],
    }


def _parse_ratings(text: str) -> list[dict]:
    """从 reviewer 输出中提取 JSON 评级数组（用括号计数匹配完整的 JSON 数组）。"""
    idx = text.find('[')
    if idx == -1:
        return []
    depth = 0
    end = -1
    for i in range(idx, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return []
    try:
        ratings = json.loads(text[idx:end + 1])
        for r in ratings:
            r["credibility"] = r.get("credibility", "中")
            r["source_number"] = r.get("source_number", 0)
        return ratings
    except json.JSONDecodeError:
        return []


async def _verify_source_urls(sources: list[dict], timeout: int = 5) -> list[dict]:
    """异步 HEAD 请求检查来源 URL 可达性，返回仅保留存活来源的新列表。"""
    urls = [s["url"] for s in sources if s.get("url")]
    if not urls:
        return sources

    async def _check_one(client: httpx.AsyncClient, url: str) -> tuple[str, bool]:
        try:
            r = await client.head(url, timeout=timeout)
            if r.status_code < 400:
                return url, True
            if r.status_code in (405, 400, 403):
                r2 = await client.get(url, timeout=timeout)
                return url, r2.status_code < 400
            return url, False
        except Exception:
            return url, False

    async with httpx.AsyncClient(
        headers={"User-Agent": "ResearchAssistant/1.0"},
        follow_redirects=True,
    ) as client:
        tasks = [_check_one(client, u) for u in urls]
        results = await asyncio.gather(*tasks)

    alive = {url for url, ok in results if ok}
    dead = [url for url, ok in results if not ok]

    if dead:
        print(f"[researcher] 过滤掉 {len(dead)} 个死链: {dead}", file=sys.stderr, flush=True)

    return [s for s in sources if not s.get("url") or s["url"] in alive]


async def researcher_node(state: AgentState) -> dict:
    """文献检索 agent。LLM 决定调用哪些工具，并行执行后直接返回原始结果（不做二次总结）。"""
    llm = get_llm()
    user_msgs = get_recent_messages(state)
    context = state.get("system_prompt", "")
    project_id = state["project_id"]

    full_prompt = RESEARCHER_PROMPT
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    tools = [web_search, aminer_search_papers, _make_rag_tool(project_id)]

    all_sources: list[dict] = []
    seen_ids: set[str] = set()
    source_counter = 0

    def _collect_sources(tool_name: str, result: str) -> None:
        nonlocal source_counter
        parsed = _parse_tool_source(tool_name, result)
        for s in parsed:
            if s["id"] not in seen_ids:
                seen_ids.add(s["id"])
                source_counter += 1
                s["source_number"] = source_counter
                all_sources.append(s)

    async def _invoke_one(tc: dict, tool_obj):
        try:
            result = str(await tool_obj.ainvoke(tc.get("args", {})))
        except Exception as e:
            result = f"工具执行失败: {e}"
        return tc, result, tool_obj.name

    _query = _extract_user_query(state)

    # 检查是否有上传文件需要检索
    from ..tools.file_rag import get_project_files
    _has_uploads = len(get_project_files(project_id)) > 0
    _mentions_files = any(kw in _query for kw in ("上传", "文档", "文件", "这篇", "这份", "PDF"))

    # 用户已明确指定检索工具：直接并行执行，跳过 LLM 工具决策
    user_specified = [t for t in state.get("required_tools", [])
                      if t in ("web_search", "aminer_search_papers", "search_uploaded_docs")]
    if user_specified:
        query = _extract_user_query(state)[:200]
        tool_by_name = {t.name: t for t in tools}
        coros = []
        for name in user_specified:
            tool_obj = tool_by_name.get(name)
            if tool_obj is None:
                continue
            args = {"query": query}
            if name == "aminer_search_papers":
                args["count"] = 5
            coros.append(_invoke_one({"id": f"user_{name}", "name": name, "args": args}, tool_obj))

        output_parts = []
        if coros:
            results = await asyncio.gather(*coros)
            for tc, result, tool_name in results:
                output_parts.append(f"## {tool_name} 结果\n{result}")
                _collect_sources(tool_name, result)

        all_sources = await _verify_source_urls(all_sources)

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _truncate_output("\n\n".join(output_parts))},
            "reference_sources": all_sources,
        }

    # 用户提及文件且有上传文件 → 优先文档检索 + 网络搜索补充
    if _mentions_files and _has_uploads:
        print("[DEBUG researcher] direct search (search_uploaded_docs + web_search)", file=sys.stderr, flush=True)
        query = _extract_user_query(state)[:200]
        tool_by_name = {t.name: t for t in tools}
        coros = []
        for name in ("search_uploaded_docs", "web_search"):
            tool_obj = tool_by_name.get(name)
            if tool_obj:
                coros.append(_invoke_one({"id": f"file_{name}", "name": name, "args": {"query": query}}, tool_obj))

        output_parts = []
        if coros:
            results = await asyncio.gather(*coros)
            for tc, result, tool_name in results:
                output_parts.append(f"## {tool_name} 结果\n{result}")
                _collect_sources(tool_name, result)

        all_sources = await _verify_source_urls(all_sources)

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _truncate_output("\n\n".join(output_parts))},
            "reference_sources": all_sources,
        }

    # LLM 工具决策：由 LLM 决定调用哪些工具
    llm_with_tools = llm.bind_tools(tools)

    msgs = [SystemMessage(content=full_prompt)] + user_msgs
    response = await llm_with_tools.ainvoke(msgs)

    if not response.tool_calls:
        print("[DEBUG researcher] LLM returned NO tool calls, returning text only (no sources)", file=sys.stderr, flush=True)
        all_sources = await _verify_source_urls(all_sources)
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": response.content},
            "reference_sources": all_sources,
        }

    # 并行调用所有工具（单轮，不循环，不做 LLM 二次总结）
    coros = []
    for tc in response.tool_calls[:3]:
        tool_name = tc.get("name", "")
        for tool in tools:
            if tool.name == tool_name:
                coros.append(_invoke_one(tc, tool))
                break

    output_parts = []
    if coros:
        results = await asyncio.gather(*coros)
        for tc, result, tool_name in results:
            output_parts.append(f"## {tool_name} 结果\n{result}")
            _collect_sources(tool_name, result)

    print(f"[DEBUG researcher] tool_calls={len(response.tool_calls)}, coros={len(coros)}, sources={len(all_sources)}, output_len={len(_truncate_output(chr(10).join(output_parts) if output_parts else ''))}", file=sys.stderr, flush=True)
    all_sources = await _verify_source_urls(all_sources)
    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _truncate_output("\n\n".join(output_parts))},
        "reference_sources": all_sources,
    }



async def reviewer_node(state: AgentState) -> dict:
    """学术评审 agent：无工具，纯推理批判性评估。输出 JSON 评级 + 简短文字。"""
    import time
    t0 = time.time()
    llm = get_llm()
    msgs = build_reviewer_context(state)
    t1 = time.time()
    total_chars = sum(len(str(m.content)) for m in msgs)
    print(f"[DEBUG reviewer] context built in {t1 - t0:.1f}s, {len(msgs)} msgs, {total_chars} chars total", file=sys.stderr, flush=True)
    response = await llm.ainvoke(msgs)
    t2 = time.time()
    print(f"[DEBUG reviewer] LLM call took {t2 - t1:.1f}s, output {len(response.content)} chars", file=sys.stderr, flush=True)

    ratings = _parse_ratings(response.content)
    print(f"[DEBUG reviewer] parsed {len(ratings)} ratings", file=sys.stderr, flush=True)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "reviewer": response.content},
        "source_ratings": ratings,
    }


async def planner_node(state: AgentState) -> dict:
    """研究方案设计 agent：生成候选方案，通过 interrupt() 暂停等待用户选择。"""
    llm = get_llm()
    msgs = build_planner_context(state)
    response = await llm.ainvoke(msgs)
    plan_options = _parse_plan_options(response.content)

    # 暂停图执行，将候选方案抛给前端
    user_choice = interrupt({
        "type": "plan_options",
        "options": plan_options,
    })
    # ─── 图在此暂停，用户选择后通过 /chat/resume 恢复 ───

    # 解析用户选择：可能是预制方案 ID 或自定义文本
    chosen_plan_id = ""
    custom_plan_text = ""
    plan_summary = ""

    if isinstance(user_choice, dict):
        chosen_plan_id = user_choice.get("chosen_plan_id", "")
        custom_plan_text = user_choice.get("custom_plan_text", "")

    if chosen_plan_id:
        # 从 plan_options 中找到用户选中的方案
        for opt in plan_options:
            if opt.get("id") == chosen_plan_id:
                plan_summary = (
                    f"用户选择了方案 {chosen_plan_id}：「{opt.get('title', '')}」\n"
                    f"思路：{opt.get('description', '')}\n"
                    f"优势：{'、'.join(opt.get('pros', []))}\n"
                    f"风险：{'、'.join(opt.get('cons', []))}"
                )
                break
    elif custom_plan_text:
        plan_summary = f"用户自定义方案：\n{custom_plan_text}"

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "planner": plan_summary},
        "plan_options": plan_options,
        "chosen_plan_id": chosen_plan_id,
        "custom_plan_text": custom_plan_text,
    }


async def generate_response_node(state: AgentState) -> dict:
    """综合所有 agent 输出，生成最终用户回复。"""
    import time
    t0 = time.time()
    llm = get_llm(streaming=True)
    msgs = build_generate_context(state)
    t1 = time.time()
    total_chars = sum(len(str(m.content)) for m in msgs)
    print(f"[DEBUG generate] context built in {t1 - t0:.1f}s, {len(msgs)} msgs, {total_chars} chars total", file=sys.stderr, flush=True)
    response = await llm.ainvoke(msgs)
    t2 = time.time()
    print(f"[DEBUG generate] LLM call took {t2 - t1:.1f}s, output {len(response.content)} chars", file=sys.stderr, flush=True)

    return {"messages": [response]}


# ============================================================
# 内部函数
# ============================================================

MAX_OUTPUT_CHARS = 3500

# RAG 片段注入生成 prompt 的最大字符数：与 file_rag 的 CHUNK_SIZE(300) 对齐，保证每块完整注入且 prompt 体量可控
RAG_CHUNK_MAX_CHARS = 300


def _truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """截断过长输出以降低下游 LLM 的首 token 延迟。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[结果过长已截断]"


def _parse_decision(text: str) -> dict:
    """从 LLM 回复中解析 JSON 决策。"""
    # 尝试直接匹配 JSON
    match = re.search(r'\{[^{}]*"next"\s*:\s*"[^"]*"\s*,\s*"reason"\s*:\s*"[^"]*"\s*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 回退：尝试提取 next 字段
    next_match = re.search(r'"next"\s*:\s*"([^"]+)"', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    if next_match:
        return {
            "next": next_match.group(1),
            "reason": reason_match.group(1) if reason_match else "",
        }
    # 最终回退
    return {"next": "FINISH", "reason": "无法解析决策，默认结束"}


def _extract_user_query(state: AgentState) -> str:
    """从消息列表中提取最新的用户消息作为 RAG 查询。"""
    all_messages = list(state["messages"])
    for m in reversed(all_messages):
        if hasattr(m, "type") and m.type == "human":
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def _make_rag_tool(project_id: str):
    """创建一个绑定到特定项目的 RAG 检索工具。每次请求动态生成，确保 project_id 正确。"""

    @tool
    def search_uploaded_docs(query: str) -> str:
        """搜索用户已上传的 PDF/Word 文档内容。当用户提到「我上传的文件」「这篇论文」「文档里」「文件里」时使用此工具。"""
        docs = search_chunks(project_id, query)
        if not docs:
            return "未在已上传的文档中找到相关内容。"

        lines = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("filename", "未知文件")
            content = doc.get("content", "")
            page = doc.get("page", 1)
            para = doc.get("paragraph", 0)
            ci = doc.get("chunk_index", i - 1)
            section = doc.get("section", "")
            # 与 CHUNK_SIZE(300) 对齐：块在 300 以内时完整注入，超出才截断
            if len(content) > RAG_CHUNK_MAX_CHARS:
                content = content[:RAG_CHUNK_MAX_CHARS] + "…（片段已截断）"
            header = f"[{i}] 来源: {src}\n分块: {ci}"
            if section:
                header += f" | 章节: {section}"
            if page and para:
                header += f" | 第{page}页 第{para}段"
            lines.append(f"{header}\n{content}")

        return "\n\n".join(lines)

    return search_uploaded_docs


def _format_rag_context(docs: list[dict]) -> str:
    """将检索到的文档片段格式化为上下文字符串。"""
    lines = []
    for i, doc in enumerate(docs, 1):
        src = doc.get("filename", "未知文件")
        content = doc.get("content", "")
        lines.append(f"[{i}] 来源: {src}\n{content}")
    return "\n\n".join(lines)


def _parse_tool_source(tool_name: str, output_text: str) -> list[dict]:
    """从原始工具输出中解析结构化来源信息。返回 [{id, title, url, summary, source_type}, ...]"""
    sources: list[dict] = []
    if tool_name == "web_search":
        pattern = re.compile(
            r'(\d+)\.\s*(.+?)\n\s+来源:\s*(.+?)\s*\|\s*URL:\s*(.+?)\n\s+(.+?)(?=\n\n\d+\.|\n\d+\.|\Z)',
            re.DOTALL,
        )
        for m in pattern.finditer(output_text):
            url = m.group(4).strip()
            title = m.group(2).strip()
            sid = hashlib.md5(f"{url}|{title}".encode()).hexdigest()[:12]
            sources.append({
                "id": sid, "title": title, "url": url,
                "summary": m.group(5).strip()[:300], "source_type": "web",
            })
    elif tool_name == "aminer_search_papers":
        pattern = re.compile(
            r'(\d+)\.\s+\*\*(.+?)\*\*\n((?:(?!\n\d+\.\s).)*)',
            re.DOTALL,
        )
        for m in pattern.finditer(output_text):
            title = m.group(2).strip()
            meta = m.group(3)
            doi_match = re.search(r'DOI:\s*(\S+)', meta)
            url = f"https://doi.org/{doi_match.group(1).strip()}" if doi_match else ""
            sid = hashlib.md5(f"{url}|{title}".encode()).hexdigest()[:12]
            sources.append({
                "id": sid, "title": title, "url": url,
                "summary": meta.strip()[:300], "source_type": "paper",
            })
    elif tool_name == "search_uploaded_docs":
        pattern = re.compile(
            r'\[(\d+)\]\s*来源:\s*(\S+)\n分块:\s*(\d+)(?:\s*\|\s*章节:\s*(.+?))?(?:\s*\|\s*第(\d+)页\s+第(\d+)段)?\n(.+?)(?=\[\d+\]|\Z)',
            re.DOTALL,
        )
        for m in pattern.finditer(output_text):
            filename = m.group(2).strip()
            chunk_index = int(m.group(3))
            section = (m.group(4) or "").strip()
            page = int(m.group(5)) if m.group(5) else 1
            paragraph = int(m.group(6)) if m.group(6) else 0
            content = m.group(7).strip()
            sid = hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()[:12]
            pos_parts = []
            if section:
                pos_parts.append(section)
            if paragraph:
                pos_parts.append(f"第{paragraph}段")
            sources.append({
                "id": sid,
                "title": filename,
                "url": "",
                "summary": content[:50].replace("\n", " "),
                "source_type": "document",
                "page": page,
                "position": " | ".join(pos_parts),
                "chunk_index": chunk_index,
                "section": section,
            })
    return sources


def _parse_plan_options(text: str) -> list[dict]:
    """从 LLM 回复中解析候选方案 JSON 数组。解析失败时返回默认方案。"""
    # 尝试匹配 JSON 数组
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            options = json.loads(match.group())
            if isinstance(options, list) and len(options) > 0:
                return options
        except json.JSONDecodeError:
            pass
    # 回退：生成一个默认方案
    return [
        {
            "id": "plan_default",
            "title": "综合方案",
            "description": "基于已有信息的综合研究方案",
            "pros": ["全面覆盖已有信息", "风险较低"],
            "cons": ["缺乏针对性", "创新性有限"],
        }
    ]


async def _run_tool_loop(llm, system_prompt: str, user_msgs: list, tools: list) -> tuple[str, list[dict]]:
    """在 agent 内部执行工具调用循环，返回 (最终文本输出, 从工具输出中解析的来源列表)。"""
    if not tools:
        msgs = [SystemMessage(content=system_prompt)] + user_msgs
        response = await llm.ainvoke(msgs)
        return response.content, []

    llm_with_tools = llm.bind_tools(tools)
    msgs = [SystemMessage(content=system_prompt)] + user_msgs

    response = await llm_with_tools.ainvoke(msgs)
    iterations = 0

    all_sources: list[dict] = []
    seen_ids: set[str] = set()
    source_counter = 0

    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        # 并行调用工具（使用 ainvoke 保持 LangGraph 追踪上下文）
        async def _invoke_one(tc: dict, tool_obj):
            try:
                result = str(await tool_obj.ainvoke(tc.get("args", {})))
            except Exception as e:
                result = f"工具执行失败: {e}"
            return tc, result, tool_obj.name

        coros = []
        for tc in response.tool_calls[:3]:
            tool_name = tc.get("name", "")
            for tool in tools:
                if tool.name == tool_name:
                    coros.append(_invoke_one(tc, tool))
                    break

        if not coros:
            break

        results = await asyncio.gather(*coros)

        tool_msgs = []
        for tc, result, tool_name in results:
            tool_msgs.append(ToolMessage(content=result, tool_call_id=tc.get("id", "")))

            if tool_name in ("web_search", "aminer_search_papers", "search_uploaded_docs"):
                parsed = _parse_tool_source(tool_name, result)
                for s in parsed:
                    if s["id"] not in seen_ids:
                        seen_ids.add(s["id"])
                        source_counter += 1
                        s["source_number"] = source_counter
                        all_sources.append(s)

        msgs = msgs + [response] + tool_msgs
        response = await llm_with_tools.ainvoke(msgs)
        iterations += 1

    return response.content, all_sources
