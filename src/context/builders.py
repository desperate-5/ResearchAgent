"""各 agent 调用上下文组装：把系统提示、历史对话、agent 输出、调度记录拼成消息列表。"""

from langchain_core.messages import SystemMessage, AIMessage

from ..graph.prompts import (
    SUPERVISOR_PROMPT,
    SUPERVISOR_PROMPT_MINIMAL,
    RESEARCHER_PROMPT,
    PLANNER_PROMPT,
    GENERATE_PROMPT,
    TOOL_AGENT_MAP,
)
from ..sources.rerank import rerank_sources
from ..storage.records import get_project_sources, get_summary
from .windowing import get_recent_messages


def build_supervisor_context(state: dict) -> list:
    """构建 supervisor 的消息列表：系统提示 + 上下文 + 最近对话 + agent 输出摘要 + 调度记录。"""
    recent = get_recent_messages(state)

    # 拼接系统提示 + 动态上下文（摘要、偏好等）
    context = state.get("system_prompt", "")
    agent_outputs = state.get("agent_outputs", {})

    # 有 agent 输出时已是第二轮调度：主要路由已被上方确定性守门短路，完整版中相关规则
    # 对 LLM 已属冗余，继续灌输会干扰其判断，故改用精简 prompt（同时顺带减少 token）
    if agent_outputs:
        prompt = SUPERVISOR_PROMPT_MINIMAL
    else:
        prompt = SUPERVISOR_PROMPT
    if context:
        prompt += f"\n\n## 当前上下文\n{context}"

    # 检索前澄清选定的方向：让 supervisor 看到"用户已确认的专业方向"而不是模糊的原始问题
    if state.get("was_clarified") and state.get("effective_query"):
        prompt += f"\n\n## 用户提问（已通过澄清选定方向）\n{state['effective_query']}\n请按此方向调度检索，不要凭自身知识直接回答。"

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
                + "\n".join(tool_lines)
                + "\n\n如果多个 agent 都需要调用，请按合理顺序安排。"
                "在调用完所有要求的 agent 之前，不能 FINISH。"
            )

    msgs = [SystemMessage(content=prompt)] + recent

    # 如果已有 agent 输出，加入提示
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


def build_planner_context(state: dict) -> list:
    """构建 planner 的消息列表：系统提示 + 检索内容（本轮输出或持久化研究内容）。"""
    recent = get_recent_messages(state)

    researcher_output = state.get("agent_outputs", {}).get("researcher", "")

    msgs = [SystemMessage(content=PLANNER_PROMPT)]

    # 添加原始用户问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    if user_questions:
        msgs.append(SystemMessage(content=f"## 用户的原始问题\n{user_questions[-1].content}"))

    # 检索内容：本轮 researcher 输出；后续轮次跳过重检索时用持久化来源 + 摘要
    if researcher_output:
        msgs.append(SystemMessage(content=f"## 文献检索结果\n{researcher_output}"))
    else:
        persisted = _load_persisted_research(state)
        if persisted:
            msgs.append(SystemMessage(content=f"## 已有研究内容（历史）\n{persisted}"))

    return msgs


def _load_persisted_research(state: dict) -> str:
    """从持久化存储加载历史研究内容（摘要 + 来源），供 planner 跳过重检索时使用。"""
    project_id = state.get("project_id", "")
    if not project_id:
        return ""

    parts: list[str] = []
    summary = get_summary(project_id)
    if summary:
        parts.append(f"历史摘要：{summary}")

    sources = get_project_sources(project_id)
    if sources:
        src_lines = []
        for s in sources[:20]:
            sn = s.get("source_number", "?")
            title = s.get("title", "")
            src_lines.append(f"[{sn}] {title} - {s.get('url', '')}")
        parts.append("历史来源：\n" + "\n".join(src_lines))

    return "\n".join(parts)


def build_generate_context(state: dict) -> list:
    """构建 generate_response 的消息列表：综合所有 agent 输出 + 原始用户问题 + 参考文献编号对照。"""
    recent = get_recent_messages(state)

    # 找到用户原始问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    user_question = user_questions[-1].content if user_questions else ""

    agent_outputs = state.get("agent_outputs", {})

    parts = [GENERATE_PROMPT]
    parts.append(f"## 用户的原始问题\n{user_question}")

    if state.get("was_clarified") and state.get("effective_query"):
        parts.append(f"## 用户澄清后选定的检索方向（回答应围绕该方向的检索结果展开）\n{state['effective_query']}")

    if agent_outputs.get("researcher"):
        researcher_output = agent_outputs["researcher"]
        raw_sources = state.get("reference_sources", [])
        assessments = state.get("source_assessments", [])
        ref_sources = rerank_sources(raw_sources, assessments)
        parts.append(f"## 文献检索结果\n{researcher_output}")
        if ref_sources:
            by_num = {a.get("source_number"): a for a in assessments}
            ref_lines = []
            for s in ref_sources:
                sn = s["source_number"]
                title = s['title']
                a = by_num.get(sn)
                if a and float(a.get("score", 5.0)) < 2.5:
                    title += "（待验证）"
                extra = ""
                if s.get("section"):
                    extra = f"「{s['section']}」"
                if s.get("page") and s.get("position"):
                    extra += f" 第{s['page']}页 {s['position']}"
                if extra:
                    title += f" ({extra.strip()})"
                if s.get("url"):
                    ref_lines.append(f"[{sn}] {title} - {s['url']}")
                else:
                    ref_lines.append(f"[{sn}] {title}")
            parts.append(f"## 参考文献编号对照\n" + "\n".join(ref_lines))
            parts.append("**引用来源时请使用「参考文献编号对照」中的全局编号 [N]。不要使用各工具输出中的原始序号。**")
    else:
        # 本轮未检索（方案直通 / 澄清直通）：用项目持久化历史来源构建引用编号对照，
        # 避免 LLM 编造不存在的编号，保证回答引用与来源面板一致
        project_id = state.get("project_id", "")
        persisted = get_project_sources(project_id) if project_id else []
        if persisted:
            ref_lines = []
            for s in persisted[:20]:
                sn = s.get("source_number", "?")
                title = s.get("title", "")
                cred = s.get("credibility", "")
                extra = f"（{cred}）" if cred and cred != "未评级" else ""
                if s.get("url"):
                    ref_lines.append(f"[{sn}] {title}{extra} - {s['url']}")
                else:
                    ref_lines.append(f"[{sn}] {title}{extra}")
            parts.append("## 参考文献编号对照（项目历史来源）\n" + "\n".join(ref_lines))
            parts.append("**引用来源时只能使用上面列出的编号 [N]，不得编造列表中不存在的编号。**")
    if agent_outputs.get("reviewer"):
        out = agent_outputs["reviewer"]
        if out:
            parts.append(f"## 来源可信度评估\n{out}")
    if agent_outputs.get("planner"):
        out = agent_outputs["planner"]
        chosen_plan_id = state.get("chosen_plan_id", "")
        custom_plan_text = state.get("custom_plan_text", "")
        if chosen_plan_id:
            parts.append(f"## 用户选定的研究方案（{chosen_plan_id}）\n{out}")
        elif custom_plan_text:
            parts.append(f"## 用户自定义的研究方案\n{out}")
        else:
            parts.append(f"## 研究方案\n{out}")

    context = state.get("system_prompt", "")
    if context:
        parts.append(f"## 其他上下文\n{context}")

    return [SystemMessage(content="\n\n".join(parts))] + recent[-1:]


def build_researcher_messages(state: dict, *, is_refetch: bool, gaps: list[str] | None = None) -> list:
    """构建 researcher 的消息列表：检索提示词 + 补搜指令 + 上下文 + 最近对话。"""
    user_msgs = get_recent_messages(state)

    full_prompt = RESEARCHER_PROMPT
    if is_refetch:
        gaps = gaps or []
        full_prompt += (
            "\n\n## 补搜指令\n"
            "这是补搜轮。上一轮检索遗漏了以下信息缺口，请针对这些缺口定向补搜：\n"
            + "\n".join(f"- {g}" for g in gaps)
        )
    # 检索前澄清选定的方向：让 researcher 围绕用户确认的方向检索，忽略原始问题的歧义部分
    if state.get("was_clarified") and state.get("effective_query"):
        full_prompt += f"\n\n## 用户澄清后选定的检索方向（必须围绕它检索）\n{state['effective_query']}"
    context = state.get("system_prompt", "")
    if context:
        full_prompt += f"\n\n## 上下文信息\n{context}"

    return [SystemMessage(content=full_prompt)] + user_msgs
