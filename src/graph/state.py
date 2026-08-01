from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    project_id: str
    summary: str
    system_prompt: str
    search_results: list[dict]
    retrieved_docs: list[dict]

    # 多 Agent 新增字段
    agent_outputs: dict         # {agent_name: output_text}  各子 agent 的输出
    next_agent: str             # supervisor 决定的下一个 agent 或 "FINISH"
    supervisor_log: list[dict]  # supervisor 的调度记录 [{next, reason}, ...]
    required_tools: list[str]   # 用户指定的工具列表，supervisor 需确保调度对应 agent
    reference_sources: list[dict]  # 本轮从工具输出中解析的来源（含全局编号）

    # planner 人机协同字段
    plan_options: list[dict]      # planner 生成的候选方案 [{id, title, description, pros, cons}, ...]
    chosen_plan_id: str            # 用户选择的预制方案 ID（空字符串表示未选预制方案）
    chosen_plan_detail: dict       # 用户选中方案的完整信息（从 plan_options 提取）
    custom_plan_text: str          # 用户自定义的方案文本（未选预制方案时填写）
