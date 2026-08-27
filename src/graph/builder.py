from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    load_context_node,
    supervisor_node,
    researcher_node,
    planner_node,
    reviewer_node,
    generate_response_node,
)
from .router import route_from_supervisor


def build_graph(checkpointer=None):
    """构建并编译多 Agent LangGraph 图（全量模式: supervisor + researcher + planner + reviewer）。"""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("generate_response", generate_response_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "planner": "planner",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("researcher", "reviewer")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=checkpointer)
