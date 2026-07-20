from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    load_context_node,
    supervisor_node,
    researcher_node,
    analyst_node,
    reviewer_node,
    generate_response_node,
    memory_compressor_node,
)
from .router import route_from_supervisor, route_after_generate


def build_graph(checkpointer=None):
    """构建并编译多 Agent LangGraph 图。

    图结构:
        START
          │
    load_context
          │
    supervisor ──────────────────────────────┐
      │         │          │                 │
    researcher analyst  reviewer   generate_response
      │         │          │                 │
      └─────────┴──────────┘                 │
          │                          memory_compressor
          ▼                                  │
    supervisor (循环)                        END
    """
    graph = StateGraph(AgentState)

    # 添加所有节点
    graph.add_node("load_context", load_context_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("memory_compressor", memory_compressor_node)

    # 入口
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "supervisor")

    # supervisor → 条件路由到子 agent 或生成回复
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "reviewer": "reviewer",
            "generate_response": "generate_response",
        },
    )

    # 每个子 agent 完成后 → 回到 supervisor 继续调度
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    # generate_response → 压缩检查或结束
    graph.add_conditional_edges(
        "generate_response",
        route_after_generate,
        {
            "memory_compressor": "memory_compressor",
            END: END,
        },
    )
    graph.add_edge("memory_compressor", END)

    return graph.compile(checkpointer=checkpointer)
