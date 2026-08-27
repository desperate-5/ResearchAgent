from langgraph.graph import END
from .state import AgentState
from .prompts import PLAN_KEYWORDS


def _has_plan_intent(state: AgentState) -> bool:
    """检查用户是否明确要求设计研究方案。"""
    all_messages = list(state["messages"])
    for m in reversed(all_messages):
        if hasattr(m, "type") and m.type == "human":
            content = m.content if isinstance(m.content, str) else ""
            return any(kw in content for kw in PLAN_KEYWORDS)
    return False


def route_from_supervisor(state: AgentState) -> str:
    """根据 supervisor 的决策路由到对应子 agent 或生成回复。"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "generate_response"
    if next_agent in ("researcher", "planner"):
        return next_agent
    return "generate_response"