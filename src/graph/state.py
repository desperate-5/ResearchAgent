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
