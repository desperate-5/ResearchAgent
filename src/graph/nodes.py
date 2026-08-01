import os
import json
import re
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from .state import AgentState

from ..tools.web_search import web_search
from ..tools.aminer_search import aminer_search_papers
from ..tools.calculator import calculator
from ..tools.python_executor import python_executor
from ..tools.file_rag import search_chunks
from ..memory.compressor import generate_summary
from ..memory.store import save_summary, get_latest_plan
from ..preferences.manager import get_preferences
from ..preferences.prompt_builder import build_preference_prompt

from .prompts import (
    MAX_CONTEXT_MESSAGES,
    COMPRESSION_THRESHOLD,
    MAX_TOOL_ITERATIONS,
    TOOL_AGENT_MAP,
    RESEARCHER_PROMPT,
    ANALYST_PROMPT,
    PLANNER_PROMPT,
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
        request_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    )


# ============================================================
# 工具集
# ============================================================

RESEARCHER_TOOLS = [web_search, aminer_search_papers]
ANALYST_TOOLS = [calculator, python_executor]
REVIEWER_TOOLS = []


# ============================================================
# 图节点
# ============================================================

async def load_context_node(state: AgentState) -> dict:
    """加载上下文：偏好配置、历史摘要。不做自动 RAG 注入（由 researcher 按需检索）。"""
    project_id = state["project_id"]

    prefs = get_preferences(project_id)
    prefs_text = build_preference_prompt(prefs)

    context_parts = []
    summary = state.get("summary", "")
    if summary:
        context_parts.append(f"## 历史对话摘要\n{summary}")
    if prefs_text:
        context_parts.append(prefs_text)

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
    """调度者节点：决定接下来调用哪个子 agent，或者结束调度。"""
    log = state.get("supervisor_log", [])
    agent_outputs = state.get("agent_outputs", {})

    # 防止无限循环：最多调度 3 次（2 次 agent + 1 次 FINISH）
    if len(log) >= 5:
        return {
            "next_agent": "FINISH",
            "supervisor_log": log + [{"next": "FINISH", "reason": "达到最大调度次数"}],
        }

    # planner 已完成方案设计，不允许再次调用，直接结束调度
    if "planner" in agent_outputs:
        return {
            "next_agent": "FINISH",
            "supervisor_log": log + [{"next": "FINISH", "reason": "planner 已完成方案设计，结束调度"}],
        }

    llm = get_llm()
    msgs = build_supervisor_context(state)
    response = await llm.ainvoke(msgs)
    decision = _parse_decision(response.content)

    log_entry = {
        "next": decision["next"],
        "reason": decision["reason"],
    }

    return {
        "next_agent": decision["next"],
        "supervisor_log": log + [log_entry],
    }


async def researcher_node(state: AgentState) -> dict:
    """文献检索 agent：动态组装工具（全局工具 + 项目级 RAG 工具）。"""
    llm = get_llm()
    user_msgs = get_recent_messages(state)
    context = state.get("system_prompt", "")

    project_id = state["project_id"]
    tools = RESEARCHER_TOOLS + [_make_rag_tool(project_id)]

    full_prompt = RESEARCHER_PROMPT
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    content, ref_sources = await _run_tool_loop(llm, full_prompt, user_msgs, tools)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "researcher": content},
        "reference_sources": ref_sources,
    }


async def analyst_node(state: AgentState) -> dict:
    """数据分析 agent：自带 calculator 工具循环，可基于检索结果进行计算分析。"""
    llm = get_llm()
    user_msgs = get_recent_messages(state)

    full_prompt = ANALYST_PROMPT
    context = state.get("system_prompt", "")
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    researcher_output = state.get("agent_outputs", {}).get("researcher", "")
    if researcher_output:
        full_prompt += f"\n\n## 文献检索结果（可作为分析依据）\n{researcher_output}"

    content, _ = await _run_tool_loop(llm, full_prompt, user_msgs, ANALYST_TOOLS)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "analyst": content},
    }


async def reviewer_node(state: AgentState) -> dict:
    """学术评审 agent：无工具，纯推理批判性评估。"""
    llm = get_llm()
    msgs = build_reviewer_context(state)
    response = await llm.ainvoke(msgs)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "reviewer": response.content},
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
    llm = get_llm(streaming=True)
    msgs = build_generate_context(state)
    response = await llm.ainvoke(msgs)

    return {"messages": [response]}


async def memory_compressor_node(state: AgentState) -> dict:
    """压缩旧对话为结构化摘要。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]

    if len(conv_msgs) <= COMPRESSION_THRESHOLD:
        return {}

    keep_recent = 10
    old_msgs = conv_msgs[:-keep_recent]

    new_summary = await generate_summary(old_msgs, state.get("summary", ""))

    save_summary(state["project_id"], new_summary)

    return {"summary": new_summary}


# ============================================================
# 内部函数
# ============================================================

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
            lines.append(f"[{i}] 来源: {src}\n{content}")

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
        pattern = re.compile(r'\[(\d+)\]\s*来源:\s*(.+?)\n(.+?)(?=\[\d+\]|\Z)', re.DOTALL)
        for m in pattern.finditer(output_text):
            title = m.group(2).strip()
            sid = hashlib.md5(title.encode()).hexdigest()[:12]
            sources.append({
                "id": sid, "title": title, "url": "",
                "summary": m.group(3).strip()[:300], "source_type": "document",
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
        tool_msgs = []
        for tc in response.tool_calls[:3]:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            result = ""
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = str(tool.invoke(tool_args))
                    except Exception as e:
                        result = f"工具执行失败: {e}"
                    break

            tool_msgs.append(ToolMessage(content=result, tool_call_id=tool_id))

            # 从原始工具输出中解析来源，分配全局编号（与 server.py 一致的顺序）
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
