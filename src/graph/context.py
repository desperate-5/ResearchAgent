"""上下文构建函数。负责组装各 agent 的消息列表（system prompt + 最近对话 + agent 输出摘要等）。"""

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from .state import AgentState
from .prompts import (
    SUPERVISOR_PROMPT,
    SUPERVISOR_PROMPT_MINIMAL,
    REVIEWER_PROMPT,
    PLANNER_PROMPT,
    GENERATE_PROMPT,
    MAX_CONTEXT_TURNS,
    TOOL_AGENT_MAP,
)


def count_turns(messages: list) -> int:
    """统计对话轮数：一轮 = 一条用户消息。"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def last_n_turns(messages: list, n: int) -> list:
    """取最近 n 轮的完整消息（按用户消息边界切分，保证轮次完整）。"""
    user_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(user_indices) <= n:
        return messages
    return messages[user_indices[-n]:]


def get_recent_messages(state: AgentState) -> list:
    """获取最近的用户对话消息（不含 system message），最多 MAX_CONTEXT_TURNS 轮。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    return last_n_turns(conv_msgs, MAX_CONTEXT_TURNS)


def build_supervisor_context(state: AgentState) -> list:
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


def build_reviewer_context(state: AgentState) -> list:
    """构建 reviewer 的消息列表：系统提示 + 原始用户问题 + researcher 输出 + 结构化来源列表。"""
    recent = get_recent_messages(state)

    researcher_output = state.get("agent_outputs", {}).get("researcher", "")
    ref_sources = state.get("reference_sources", [])

    msgs = [SystemMessage(content=REVIEWER_PROMPT)]

    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    if user_questions:
        msgs.append(SystemMessage(content=f"## 用户的原始问题\n{user_questions[-1].content}"))

    if researcher_output:
        msgs.append(SystemMessage(content=f"## 文献检索结果（需要你评估）\n{researcher_output}"))

    if ref_sources:
        source_lines = []
        for s in ref_sources:
            num = s.get("source_number", "?")
            title = s.get("title", "")
            url = s.get("url", "")
            stype = s.get("source_type", "")
            domain = ""
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).hostname or ""
                except Exception:
                    pass
            line = f"[{num}] {title} | 类型:{stype}"
            if domain:
                line += f" | 域名:{domain}"
            if url:
                line += f" | {url}"
            source_lines.append(line)
        msgs.append(SystemMessage(content="## 来源列表（结构化）\n" + "\n".join(source_lines)))

    return msgs


def build_planner_context(state: AgentState) -> list:
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


def build_generate_context(state: AgentState) -> list:
    """构建 generate_response 的消息列表：综合所有 agent 输出 + 原始用户问题 + 参考文献编号对照。"""
    recent = get_recent_messages(state)

    # 找到用户原始问题
    user_questions = [m for m in recent if hasattr(m, "type") and m.type == "human"]
    user_question = user_questions[-1].content if user_questions else ""

    agent_outputs = state.get("agent_outputs", {})

    parts = [GENERATE_PROMPT]
    parts.append(f"## 用户的原始问题\n{user_question}")

    MAX_AGENT_OUTPUT = 3000

    if agent_outputs.get("researcher"):
        researcher_output = agent_outputs["researcher"]
        ref_sources = state.get("reference_sources", [])
        # 截断过长检索结果以降低首 token 延迟（TTFT 主要由 prompt 处理时间决定）
        if len(researcher_output) > MAX_AGENT_OUTPUT:
            researcher_output = researcher_output[:MAX_AGENT_OUTPUT] + "\n\n[检索结果过长已截断，完整信息见参考文献列表]"
        parts.append(f"## 文献检索结果\n{researcher_output}")
        if ref_sources:
            ref_lines = []
            for s in ref_sources:
                sn = s["source_number"]
                title = s['title']
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
        # 去掉开头的 JSON 评级数组，只保留文字评估（用括号计数找匹配的 ]）
        idx = out.lstrip().find('[')
        if idx != -1 and idx < 5:
            depth = 0
            end = -1
            for i in range(idx, len(out)):
                if out[i] == '[':
                    depth += 1
                elif out[i] == ']':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                out = out[end + 1:].strip()
        if out:
            if len(out) > MAX_AGENT_OUTPUT:
                out = out[:MAX_AGENT_OUTPUT] + "\n\n[评审意见过长已截断]"
            parts.append(f"## 来源可信度评估\n{out}")
    if agent_outputs.get("planner"):
        out = agent_outputs["planner"]
        if len(out) > MAX_AGENT_OUTPUT:
            out = out[:MAX_AGENT_OUTPUT] + "\n\n[方案过长已截断]"
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
