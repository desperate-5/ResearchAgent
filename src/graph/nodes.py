import os
import sys
import json
import re
import asyncio
import httpx

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from .state import AgentState

from ..tools.web_search import web_search
from ..tools.aminer_search import aminer_search_papers
from ..tools.rag_tool import make_rag_tool
from ..sources.parser import parse_tool_sources
from ..storage.records import get_latest_plan, get_project_sources
from ..preferences.prompt_builder import build_preference_prompt
from ..preferences.store import compute_effective
from ..context.builders import (
    build_supervisor_context,
    build_planner_context,
    build_generate_context,
    build_researcher_messages,
)
from ..context.compression import build_summary_injection
from ..context.tool_compression import truncate_output
from ..context.windowing import extract_user_query
from ..interaction.types import PlanOptionsPayload

from .prompts import (
    PLAN_KEYWORDS,
    MAX_SEARCH_ROUNDS,
)
from .reviewer import assess_sources


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

    profile = compute_effective(project_id)
    prefs_parts = []
    base_prefs = build_preference_prompt(profile.config)
    if base_prefs:
        prefs_parts.append(base_prefs)
    if profile.domain:
        prefs_parts.append(f"用户关注的研究领域：{profile.domain}")
    if profile.method:
        prefs_parts.append(f"用户偏好的研究方法：{profile.method}")
    prefs_text = "\n".join(prefs_parts)

    context_parts = []
    summary_injection = build_summary_injection(project_id)
    if summary_injection:
        context_parts.append(summary_injection)
    if prefs_text:
        context_parts.append(prefs_text)

    # 列出项目已上传的文件，帮助 researcher 判断是否需要调用 search_uploaded_docs
    from ..rag.store import get_project_files
    project_files = get_project_files(project_id)
    if project_files:
        file_names = [f["filename"] for f in project_files]
        context_parts.append(f"## 项目已上传文件\n{', '.join(file_names)}")

    latest_plan = get_latest_plan(project_id)
    if latest_plan:
        plan_prefix = "自定义" if latest_plan.get("is_custom") else "已选定"
        plan_info = f"## 当前项目的研究方案\n{plan_prefix}方案：{latest_plan.get('plan_title', '')}\n{latest_plan.get('plan_detail', '')}"
        context_parts.append(plan_info)

    # 是否已有历史研究内容（摘要或持久化来源），供 supervisor 判断能否跳过重检索直达 planner
    has_prior_research = bool(summary_injection) or bool(get_project_sources(project_id))

    return {
        "system_prompt": "\n\n".join(context_parts),
        "has_prior_research": has_prior_research,
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

    # 检索前澄清守门（确定性规则，绕过 LLM）：用户已通过澄清选定专业方向，
    # 本轮必须先执行检索，禁止 supervisor 直接 FINISH 凭自身知识回答。
    if state.get("was_clarified") and "researcher" not in agent_outputs:
        print("[DEBUG supervisor] 澄清已选定方向，强制先执行 researcher", file=sys.stderr, flush=True)
        return {
            "next_agent": "researcher",
            "supervisor_log": log + [{"next": "researcher", "reason": "检索前澄清已选定方向，强制先执行检索"}],
        }

    # 方案选择触发（确定性门 + 护栏）：用户明确表达方案意图（关键词）且已有可靠内容 → 强制 planner
    has_content = bool(agent_outputs.get("researcher")) or bool(state.get("has_prior_research", False))
    user_query = extract_user_query(state)
    matched_kw = [kw for kw in PLAN_KEYWORDS if kw in user_query]
    if matched_kw and has_content:
        print("[DEBUG supervisor] Forcing route to planner (keyword + content)", file=sys.stderr, flush=True)
        return {
            "next_agent": "planner",
            "supervisor_log": log + [{"next": "planner", "reason": "用户明确要求设计方案且已有研究内容，路由到 planner"}],
        }

    # 缺口补搜门控（确定性规则，绕过 LLM）：仅当 reviewer 明确判定来源明显不足
    # （needs_refetch=true）且补搜次数未达上限时才定向补搜；"还能补充更多细节"
    # 这类可选项不触发补搜，避免每问必二次检索。
    search_round = state.get("search_round", 0)
    if "researcher" in agent_outputs and state.get("needs_refetch") and search_round < MAX_SEARCH_ROUNDS:
        gaps = state.get("retrieval_gaps", [])
        print(f"[DEBUG supervisor] 触发第 {search_round + 1} 次补搜（needs_refetch），缺口 {len(gaps)} 个", file=sys.stderr, flush=True)
        return {
            "next_agent": "researcher",
            "search_round": search_round + 1,
            "supervisor_log": log + [{"next": "researcher", "reason": f"reviewer 判定来源明显不足，触发第 {search_round + 1} 次补搜"}],
        }

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

    # planner 需有内容支撑：LLM 想直接方案设计但无内容 → 回退检索
    if next_agent == "planner" and not has_content:
        print("[DEBUG supervisor] LLM 返回 planner 但无研究内容，回退为 researcher", file=sys.stderr, flush=True)
        next_agent = "researcher"
        reason = "缺乏研究内容，先检索再设计方案"

    print(f"[DEBUG supervisor] LLM decision: next={next_agent}, reason={reason!r}", file=sys.stderr, flush=True)
    return {
        "next_agent": next_agent,
        "supervisor_log": log + [{"next": next_agent, "reason": reason}],
    }


async def _verify_source_urls(sources: list[dict], timeout: int = 5) -> list[dict]:
    """异步 HEAD 请求检查来源 URL 可达性，返回仅保留存活来源的新列表。

    AMiner 论文页（https://www.aminer.cn/pub/...）是规范 ID 链接，
    可能被反爬拦截导致探测失败，但并非死链，不参与过滤。
    """
    urls = [s["url"] for s in sources if s.get("url")]
    if not urls:
        return sources

    AMINER_PUB_PREFIX = "https://www.aminer.cn/pub/"

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
    # AMiner 论文页：探测失败（反爬/超时）不视为死链，保底保留
    alive |= {u for u in urls if u.startswith(AMINER_PUB_PREFIX)}
    dead = [url for url, ok in results if not ok and url not in alive]

    if dead:
        print(f"[researcher] 过滤掉 {len(dead)} 个死链: {dead}", file=sys.stderr, flush=True)

    return [s for s in sources if not s.get("url") or s["url"] in alive]


async def researcher_node(state: AgentState) -> dict:
    """文献检索 agent。LLM 决定调用哪些工具，并行执行后直接返回原始结果（不做二次总结）。"""
    llm = get_llm()
    project_id = state["project_id"]

    # 补搜轮：search_round > 0 即视为补搜，用缺口作为检索 query
    is_refetch = state.get("search_round", 0) > 0
    gaps = state.get("retrieval_gaps", []) if is_refetch else []
    _effective = _effective_query(state)
    if is_refetch:
        _query = " ".join(gaps) or _effective
    else:
        _query = _effective

    # 消息列表组装统一走 context 模块
    msgs = build_researcher_messages(state, is_refetch=is_refetch, gaps=gaps)

    tools = [web_search, aminer_search_papers, make_rag_tool(project_id)]

    # 补搜轮：继承上一轮来源，继续编号，避免 source_number 冲突
    existing_sources = state.get("reference_sources", [])
    all_sources: list[dict] = list(existing_sources)
    seen_ids: set[str] = {s["id"] for s in existing_sources}
    source_counter = max((s.get("source_number", 0) for s in existing_sources), default=0)

    def _collect_sources(tool_name: str, result: str) -> None:
        nonlocal source_counter
        parsed = parse_tool_sources(tool_name, result)
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

    prev_output = state.get("agent_outputs", {}).get("researcher", "")

    def _merge_output(new_text: str) -> str:
        new_text = (new_text or "").strip()
        if not new_text:
            return truncate_output(prev_output)
        merged = (prev_output + "\n\n" + new_text) if prev_output else new_text
        return truncate_output(merged)

    # 检查是否有上传文件需要检索
    from ..rag.store import get_project_files
    _has_uploads = len(get_project_files(project_id)) > 0
    _mentions_files = any(kw in _query for kw in ("上传", "文档", "文件", "这篇", "这份", "PDF"))

    # 用户已明确指定检索工具：直接并行执行，跳过 LLM 工具决策
    user_specified = [t for t in state.get("required_tools", [])
                      if t in ("web_search", "aminer_search_papers", "search_uploaded_docs")]
    if user_specified:
        query = _query[:200]
        tool_by_name = {t.name: t for t in tools}
        coros = []
        for name in user_specified:
            tool_obj = tool_by_name.get(name)
            if tool_obj is None:
                continue
            args = {"query": query}
            if name == "aminer_search_papers":
                args["count"] = 10
            coros.append(_invoke_one({"id": f"user_{name}", "name": name, "args": args}, tool_obj))

        output_parts = []
        if coros:
            results = await asyncio.gather(*coros)
            for tc, result, tool_name in results:
                output_parts.append(f"## {tool_name} 结果\n{result}")
                _collect_sources(tool_name, result)

        all_sources = await _verify_source_urls(all_sources)

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _merge_output("\n\n".join(output_parts))},
            "reference_sources": all_sources,
        }

    # 用户提及文件且有上传文件 → 优先文档检索 + 网络搜索补充
    if _mentions_files and _has_uploads:
        print("[DEBUG researcher] direct search (search_uploaded_docs + web_search)", file=sys.stderr, flush=True)
        query = _query[:200]
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
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _merge_output("\n\n".join(output_parts))},
            "reference_sources": all_sources,
        }

    # LLM 工具决策：由 LLM 决定调用哪些工具
    llm_with_tools = llm.bind_tools(tools)

    response = await llm_with_tools.ainvoke(msgs)

    if not response.tool_calls:
        print("[DEBUG researcher] LLM returned NO tool calls, returning text only (no sources)", file=sys.stderr, flush=True)
        all_sources = await _verify_source_urls(all_sources)
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _merge_output(response.content)},
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

    print(f"[DEBUG researcher] tool_calls={len(response.tool_calls)}, coros={len(coros)}, sources={len(all_sources)}, output_len={len(truncate_output(chr(10).join(output_parts) if output_parts else ''))}", file=sys.stderr, flush=True)
    all_sources = await _verify_source_urls(all_sources)
    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "researcher": _merge_output("\n\n".join(output_parts))},
        "reference_sources": all_sources,
    }



async def reviewer_node(state: AgentState) -> dict:
    """学术评审 agent：规则信号 + LLM 两阶段评估，输出评分卡 + 小结 + 缺口。"""
    import time
    t0 = time.time()
    user_query = _effective_query(state)
    result = await assess_sources(state, user_query=user_query)
    t1 = time.time()
    print(f"[DEBUG reviewer] assess_sources took {t1 - t0:.1f}s, {len(result.assessments)} assessments, {len(result.gaps)} gaps", file=sys.stderr, flush=True)

    assessments = [a.model_dump() for a in result.assessments]

    # 兼容旧 source_ratings 结构（source_number/credibility/reason），供 server.py 与前端沿用
    source_ratings = [
        {
            "source_number": a.source_number,
            "credibility": a.credibility,
            "reason": a.evidence or a.credibility,
        }
        for a in result.assessments
    ]
    reviewer_text = _format_reviewer_text(result)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "reviewer": reviewer_text},
        "source_ratings": source_ratings,
        "source_assessments": assessments,
        "retrieval_gaps": result.gaps,
        "needs_refetch": bool(result.needs_refetch),
    }


async def planner_node(state: AgentState) -> dict:
    """研究方案设计 agent：生成候选方案，通过 interrupt() 暂停等待用户选择。"""
    llm = get_llm()
    msgs = build_planner_context(state)
    response = await llm.ainvoke(msgs)
    plan_options = _parse_plan_options(response.content)

    # 暂停图执行，将候选方案抛给前端
    user_choice = interrupt(PlanOptionsPayload(options=plan_options).to_dict())
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

def _format_reviewer_text(result) -> str:
    """把 reviewer 结构化结果组装成人类可读文字，供 generate_response 消费。"""
    lines = []
    if result.summary:
        lines.append(f"整体质量小结：{result.summary}")
    for a in result.assessments:
        lines.append(f"来源[{a.source_number}] 可信度「{a.credibility}」(综合 {a.score})：{a.evidence}")
    if result.gaps:
        lines.append("信息缺口：" + "；".join(result.gaps))
    return "\n".join(lines) if lines else "无评估结果"


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


def _effective_query(state: AgentState) -> str:
    """澄清后的有效查询，未澄清时回退到原始用户输入。"""
    return state.get("effective_query", "") or extract_user_query(state)


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



