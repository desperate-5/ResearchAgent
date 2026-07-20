import os
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from .state import AgentState

from ..tools.web_search import web_search
from ..tools.aminer_search import aminer_search_papers
from ..tools.calculator import calculator
from ..tools.python_executor import python_executor
from ..tools.file_rag import search_chunks
from ..memory.compressor import generate_summary
from ..memory.store import save_summary
from ..preferences.manager import get_preferences
from ..preferences.prompt_builder import build_preference_prompt

MAX_CONTEXT_MESSAGES = 20
COMPRESSION_THRESHOLD = 20
MAX_TOOL_ITERATIONS = 3

# 工具 → agent 映射，用于 supervisor 调度约束
TOOL_AGENT_MAP: dict[str, str] = {
    "web_search": "researcher",
    "aminer_search_papers": "researcher",
    "python_executor": "analyst",
    "calculator": "analyst",
}


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.7,
        streaming=True,
    )


# ============================================================
# System Prompts
# ============================================================

SUPERVISOR_PROMPT = """你是一个科研任务调度者（Supervisor）。根据用户的问题和已有的专家分析结果，决定接下来调用哪个专家代理，或者结束调度。

可调用的专家代理：
- researcher: 文献检索与信息收集专家（搜索最新论文、资料、从上传文档中检索内容）
- analyst: 数据分析与计算专家（数学计算、统计检验、数据对比）
- reviewer: 学术评审专家（对已有结论进行批判性评估，发现方法论漏洞、样本偏差、逻辑问题）

调度逻辑：
- 用户问题需要搜索最新信息或文献 → 先调用 researcher
- 用户问题涉及计算或数据分析 → 调用 analyst
- researcher 已返回搜索结果，需要评估其质量和局限性 → 调用 reviewer
- 所有必要信息已收集完毕 → FINISH
- 可以按顺序调用多个代理（例如先 researcher 再 reviewer）
- 如果提示中包含"用户指定的工具要求"，必须调用对应的 agent，不得跳过

**重要：必须输出严格的 JSON 格式，不要添加任何其他文字。**
{"next": "<researcher|analyst|reviewer|FINISH>", "reason": "<一句话说明>"}"""

RESEARCHER_PROMPT = """你是一个文献检索与信息收集专家。你的职责是搜索、收集、整理与用户问题相关的信息和文献。

你可以使用以下工具：
- web_search: 搜索互联网上的最新资讯、行业动态、博客、技术文章
- aminer_search_papers: 搜索正式发表的学术论文（中英文），适合查找研究论文、文献综述
- search_uploaded_docs: 搜索用户已上传的 PDF/Word 文档内容。当用户提到「我上传的文件」「这篇论文」「文档里」时使用

使用规则：
- 用户问「论文」「研究」「文献」→ 优先用 aminer_search_papers
- 用户问「最新消息」「新闻」「行业动态」→ 用 web_search
- 用户提到上传的文档或文件内容 → 用 search_uploaded_docs
- 如果一次搜索不够全面，尝试不同的搜索词
- 如实汇报每个结果的来源、标题、摘要
- 用中文整理和总结搜索结果
- 不要对结果进行批判性评估（这由 reviewer 负责）
- 如果没有找到相关信息，如实告知"""

ANALYST_PROMPT = """你是一个数据分析与计算专家。你的职责是进行数学计算、统计分析和数据可视化。

- 使用 calculator 工具进行数学计算
- 使用 python_executor 工具进行数据分析、统计检验、画图（matplotlib/seaborn）
- 画图时用 plt.savefig() 保存图片，不要用 plt.show()
- 对数据进行分析和对比
- 如果需要统计检验，说明该用什么方法以及为什么
- 用中文清晰地呈现计算过程和分析结果
- 只做计算和分析，不下结论（结论由综合回答生成）"""

REVIEWER_PROMPT = """你是一个学术评审专家。你的职责是对已有的检索结果和结论进行严格的批判性评估。

请从以下维度审视已有的信息：
1. **方法论评估**：研究方法是否合理？有没有明显的缺陷？
2. **样本与数据**：样本量是否足够？数据来源是否可靠？
3. **结论有效性**：结论是否被数据充分支持？是否存在过度推广？
4. **替代解释**：是否有其他的解释或替代假设未被考虑？
5. **局限性**：这个研究或信息的局限性在哪里？
6. **冲突观点**：是否存在与这些发现相矛盾的观点或证据？

- 用中文清晰列出你的评估
- 不要因为想保持友好而回避尖锐的批评
- 指出问题时，同时说明为什么这是个问题
- 如果信息不足以进行评估，明确指出哪些信息缺失"""

GENERATE_PROMPT = """你是一个专业的科研助手。请综合以下所有专家的分析结果，给用户提供最终的回答。

回答要求：
- 使用 Markdown 格式，结构清晰
- 先给出核心结论，再展开细节
- 如果 reviewer 提出了批判性意见，必须在回答中明确提及方法论局限或注意事项
- 引用具体的数据和来源
- 区分「已确认的结论」和「需要进一步验证的观点」
- 用中文回复，保持专业、准确、简洁的风格"""


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

    return {
        "system_prompt": "\n\n".join(context_parts),
        "retrieved_docs": [],
    }


async def supervisor_node(state: AgentState) -> dict:
    """调度者节点：决定接下来调用哪个子 agent，或者结束调度。"""
    log = state.get("supervisor_log", [])

    # 防止无限循环：最多调度 5 次
    if len(log) >= 5:
        return {
            "next_agent": "FINISH",
            "supervisor_log": log + [{"next": "FINISH", "reason": "达到最大调度次数"}],
        }

    llm = get_llm()
    msgs = _build_supervisor_messages(state)
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
    user_msgs = _get_recent_user_messages(state)
    context = state.get("system_prompt", "")

    project_id = state["project_id"]
    tools = RESEARCHER_TOOLS + [_make_rag_tool(project_id)]

    full_prompt = RESEARCHER_PROMPT
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    content = await _run_tool_loop(llm, full_prompt, user_msgs, tools)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "researcher": content},
    }


async def analyst_node(state: AgentState) -> dict:
    """数据分析 agent：自带 calculator 工具循环。"""
    llm = get_llm()
    user_msgs = _get_recent_user_messages(state)

    full_prompt = ANALYST_PROMPT
    context = state.get("system_prompt", "")
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    content = await _run_tool_loop(llm, full_prompt, user_msgs, ANALYST_TOOLS)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "analyst": content},
    }


async def reviewer_node(state: AgentState) -> dict:
    """学术评审 agent：无工具，纯推理批判性评估。"""
    llm = get_llm()
    msgs = _build_reviewer_messages(state)
    response = await llm.ainvoke(msgs)

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "reviewer": response.content},
    }


async def generate_response_node(state: AgentState) -> dict:
    """综合所有 agent 输出，生成最终用户回复。"""
    llm = get_llm()
    msgs = _build_generate_messages(state)
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


def _get_recent_user_messages(state: AgentState) -> list:
    """获取最近的用户对话消息（不含 system message）。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    return conv_msgs[-MAX_CONTEXT_MESSAGES:]


def _build_supervisor_messages(state: AgentState) -> list:
    """构建 supervisor 的消息列表：系统提示 + 上下文 + 对话 + agent 输出摘要。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    recent = conv_msgs[-MAX_CONTEXT_MESSAGES:]

    # 添加上下文
    context = state.get("system_prompt", "")
    prompt = SUPERVISOR_PROMPT
    if context:
        prompt += f"\n\n## 当前上下文\n{context}"

    # 用户指定的工具约束
    required_tools = state.get("required_tools", [])
    if required_tools:
        required_agents = set()
        tool_lines = []
        for tool_name in required_tools:
            agent = TOOL_AGENT_MAP.get(tool_name)
            if agent:
                required_agents.add(agent)
                tool_lines.append(f"- {tool_name} → 由 **{agent}** 提供")
        if required_agents:
            prompt += (
                "\n\n## 用户指定的工具要求（必须遵守）\n"
                "用户本次明确要求使用以下工具，你**必须**调用对应的 agent：\n"
                + "\n".join(tool_lines) +
                "\n\n如果多个 agent 都需要调用，请按合理顺序安排。"
                "在调用完所有要求的 agent 之前，不能 FINISH。"
            )

    msgs = [SystemMessage(content=prompt)] + recent

    # 如果已有 agent 输出，加入提示
    agent_outputs = state.get("agent_outputs", {})
    if agent_outputs:
        summary_lines = ["## 已完成的专家分析"]
        for name, output in agent_outputs.items():
            summary_lines.append(f"### {name}\n{output[:800]}...")  # 截断避免超长
        msgs.append(AIMessage(content="\n\n".join(summary_lines)))

    # supervisor 调用记录
    log = state.get("supervisor_log", [])
    if log:
        history = "## 之前的调度记录\n" + "\n".join(
            f"- 调用了 {entry['next']}: {entry['reason']}" for entry in log
        )
        msgs.append(AIMessage(content=history))

    return msgs


def _build_reviewer_messages(state: AgentState) -> list:
    """构建 reviewer 的消息列表：系统提示 + 原始问题 + researcher 输出。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    recent = conv_msgs[-MAX_CONTEXT_MESSAGES:]

    # reviewer 需要看原始问题和 researcher 的输出
    researcher_output = state.get("agent_outputs", {}).get("researcher", "")

    msgs = [SystemMessage(content=REVIEWER_PROMPT)]

    # 添加原始用户问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    if user_questions:
        msgs.append(SystemMessage(content=f"## 用户的原始问题\n{user_questions[-1].content}"))

    # 添加 researcher 和其他 agent 的输出
    if researcher_output:
        msgs.append(SystemMessage(content=f"## 文献检索结果（需要你评估）\n{researcher_output}"))

    # 也加入 analyst 输出（如果有）
    analyst_output = state.get("agent_outputs", {}).get("analyst", "")
    if analyst_output:
        msgs.append(SystemMessage(content=f"## 数据分析结果\n{analyst_output}"))

    return msgs


def _build_generate_messages(state: AgentState) -> list:
    """构建 generate_response 的消息列表：综合所有 agent 输出 + 原始问题。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    recent = conv_msgs[-MAX_CONTEXT_MESSAGES:]

    # 找到用户原始问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    user_question = user_questions[-1].content if user_questions else ""

    # 收集所有 agent 输出
    agent_outputs = state.get("agent_outputs", {})

    parts = [GENERATE_PROMPT]
    parts.append(f"## 用户的原始问题\n{user_question}")

    if agent_outputs.get("researcher"):
        parts.append(f"## 文献检索结果\n{agent_outputs['researcher']}")
    if agent_outputs.get("analyst"):
        parts.append(f"## 数据分析结果\n{agent_outputs['analyst']}")
    if agent_outputs.get("reviewer"):
        parts.append(f"## 学术评审意见\n{agent_outputs['reviewer']}")

    context = state.get("system_prompt", "")
    if context:
        parts.append(f"## 其他上下文\n{context}")

    return [SystemMessage(content="\n\n".join(parts))]


async def _run_tool_loop(llm, system_prompt: str, user_msgs: list, tools: list) -> str:
    """在 agent 内部执行工具调用循环，返回最终文本输出。"""
    if not tools:
        # 无工具 agent，直接调用
        msgs = [SystemMessage(content=system_prompt)] + user_msgs
        response = await llm.ainvoke(msgs)
        return response.content

    llm_with_tools = llm.bind_tools(tools)
    msgs = [SystemMessage(content=system_prompt)] + user_msgs

    response = await llm_with_tools.ainvoke(msgs)
    iterations = 0

    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        tool_msgs = []
        for tc in response.tool_calls:
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

        msgs = msgs + [response] + tool_msgs
        response = await llm_with_tools.ainvoke(msgs)
        iterations += 1

    return response.content
