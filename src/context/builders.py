"""各 agent 调用上下文组装：把系统提示、历史对话、agent 输出、调度记录拼成消息列表。"""

from langchain_core.messages import SystemMessage, AIMessage

from ..graph.prompts import (
    SUPERVISOR_PROMPT,
    SUPERVISOR_PROMPT_MINIMAL,
    PLANNER_PROMPT,
    GENERATE_PROMPT,
    TOOL_AGENT_MAP,
)
from ..sources.rerank import rerank_sources
from .windowing import get_recent_messages


def build_supervisor_context(state: dict) -> list:
    """构建 supervisor 的消息列表：系统提示 + 上下文 + 最近对话 + agent 输出摘要 + 调度记录。"""
    recent = get_recent_messages(state)

    # 拼接系统提示 + 动态上下文（摘要、偏好等）
    context = state.get("system_prompt", "")
    agent_outputs = state.get("agent_outputs", {})

    # 有 agent 输出时已经是第二轮调度，用精简 prompt 省 token（减少 LLM 处理时间 ~0.5s）
    if agent_outputs:
        prompt = SUPERVISOR_PROMPT_MINIMAL
    else:
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
    """构建 planner 的消息列表：系统提示 + researcher 输出。"""
    recent = get_recent_messages(state)

    researcher_output = state.get("agent_outputs", {}).get("researcher", "")

    msgs = [SystemMessage(content=PLANNER_PROMPT)]

    # 添加原始用户问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    if user_questions:
        msgs.append(SystemMessage(content=f"## 用户的原始问题\n{user_questions[-1].content}"))

    # 添加 researcher 输出
    if researcher_output:
        msgs.append(SystemMessage(content=f"## 文献检索结果\n{researcher_output}"))

    return msgs


def build_generate_context(state: dict) -> list:
    """构建 generate_response 的消息列表：综合所有 agent 输出 + 原始用户问题 + 参考文献编号对照。"""
    recent = get_recent_messages(state)

    # 找到用户原始问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    user_question = user_questions[-1].content if user_questions else ""

    agent_outputs = state.get("agent_outputs", {})

    parts = [GENERATE_PROMPT]
    parts.append(f"## 用户的原始问题\n{user_question}")

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
