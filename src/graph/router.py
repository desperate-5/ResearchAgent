from langgraph.graph import END
from .state import AgentState
from .nodes import COMPRESSION_TURN_THRESHOLD
from .context import count_turns


def route_from_supervisor(state: AgentState) -> str:
    """根据 supervisor 的决策路由到对应子 agent 或生成回复。"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "generate_response"
    if next_agent in ("researcher", "analyst", "planner", "reviewer"):
        return next_agent
    # 无法识别的目标，回退
    return "generate_response"


def route_from_supervisor_minimal(state: AgentState) -> str:
    """最小测试模式：仅支持 researcher 或 FINISH。"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "generate_response"
    if next_agent == "researcher":
        return "researcher"
    return "generate_response"


def route_after_generate(state: AgentState) -> str:
    """生成回复后，检查是否需要压缩（按对话轮数判断）。"""
    if count_turns(state["messages"]) > COMPRESSION_TURN_THRESHOLD:
        return "memory_compressor"
    return END


def route_after_researcher(state: AgentState) -> str:
    """researcher 执行完毕后，若用户未指定工具约束则直接短路到 generate_response，
    省掉一次 supervisor LLM 调用。有工具约束时回 supervisor 确保全部执行。"""
    if state.get("required_tools"):
        return "supervisor"
    return "generate_response"