from langgraph.graph import END
from langchain_core.messages import SystemMessage
from .state import AgentState
from .nodes import COMPRESSION_THRESHOLD


def route_from_supervisor(state: AgentState) -> str:
    """根据 supervisor 的决策路由到对应子 agent 或生成回复。"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "generate_response"
    if next_agent in ("researcher", "analyst", "reviewer"):
        return next_agent
    # 无法识别的目标，回退
    return "generate_response"


def route_after_generate(state: AgentState) -> str:
    """生成回复后，检查是否需要压缩。"""
    messages = state["messages"]
    conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(conv_msgs) > COMPRESSION_THRESHOLD:
        return "memory_compressor"
    return END
