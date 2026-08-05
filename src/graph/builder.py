from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    load_context_node,
    supervisor_node,
    researcher_node,
    # analyst_node,      # TODO: 恢复全量时取消注释
    # planner_node,      # TODO: 恢复全量时取消注释
    # reviewer_node,     # TODO: 恢复全量时取消注释
    generate_response_node,
    # memory_compressor_node,  # 压缩已改为 server.py 后台任务，不再作为图节点
)
from .router import route_from_supervisor_minimal as route_from_supervisor, route_after_researcher


def build_graph(checkpointer=None):
    """构建并编译多 Agent LangGraph 图（当前为最小测试模式: supervisor + researcher）。

    恢复全量模式时:
    1. 取消 nodes 和 router 中标注 TODO 的注释
    2. 把下面被注释的 add_node 和 add_edge 恢复
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    # graph.add_node("analyst", analyst_node)       # TODO: 恢复全量
    # graph.add_node("planner", planner_node)       # TODO: 恢复全量
    # graph.add_node("reviewer", reviewer_node)     # TODO: 恢复全量
    graph.add_node("generate_response", generate_response_node)
    # graph.add_node("memory_compressor", memory_compressor_node)  # 压缩已改为 server.py 后台任务

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            # "analyst": "analyst",         # TODO: 恢复全量
            # "planner": "planner",         # TODO: 恢复全量
            # "reviewer": "reviewer",       # TODO: 恢复全量
            "generate_response": "generate_response",
        },
    )

    graph.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "supervisor": "supervisor",
            "generate_response": "generate_response",
        },
    )
    # graph.add_edge("analyst", "supervisor")      # TODO: 恢复全量
    # graph.add_edge("planner", "supervisor")      # TODO: 恢复全量
    # graph.add_edge("reviewer", "supervisor")     # TODO: 恢复全量

    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=checkpointer)
