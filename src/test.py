"""性能测试脚本：运行多 Agent 图，输出每个步骤的耗时明细，用于回答速度优化。

用法（在项目根目录下运行）：
    python src/test.py "你的测试问题"
    python src/test.py "帮我查最新的AI Agent综述" --tools web_search aminer_search_papers
    python src/test.py "..." --project <已有项目ID>
"""

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.memory.store import init_db, get_summary
from src.projects.manager import create_project, get_project

NODES = {
    "load_context": "加载上下文",
    "supervisor": "调度决策",
    "researcher": "文献检索",
    "analyst": "数据分析",
    "planner": "方案设计",
    "reviewer": "学术评审",
    "generate_response": "生成回答",
    "memory_compressor": "记忆压缩",
}


def _extract_node(event: dict) -> str:
    """从事件中提取 LangGraph 节点名（兼容多种事件结构）。"""
    meta = event.get("metadata") or {}
    node = meta.get("langgraph_node", "")
    if node in NODES:
        return node
    name = event.get("name", "") or ""
    if name in NODES:
        return name
    for key in NODES:
        if name == key + "_node":
            return key
    return ""


class TimeTracker:
    """收集图中各阶段的时间数据。"""

    def __init__(self):
        self.total_start = time.perf_counter()
        self.node_total = defaultdict(float)
        self.node_runs = defaultdict(list)
        self._node_depth = defaultdict(int)
        self._node_first_start = {}
        self._printed_start = set()
        self._printed_end = set()
        self.llm_by_node = defaultdict(list)
        self._llm_start = defaultdict(list)
        self.tool_by_node = defaultdict(list)
        self._tool_start = defaultdict(list)
        self.current_node = None
        self.gen_started_at = None
        self.gen_first_chunk = None
        self.stream_chars = 0
        self.timeline = []

    def _now(self):
        return time.perf_counter() - self.total_start

    def on_node_start(self, node: str):
        depth = self._node_depth.get(node, 0)
        if depth == 0:
            now = time.perf_counter()
            self._node_first_start[node] = now
            self.current_node = node
            if node == "generate_response":
                self.gen_started_at = now
            self.timeline.append((self._now(), node, "开始"))
            print(f">>> [{self._now():7.2f}s] {NODES[node]} 开始", flush=True)
        self._node_depth[node] = depth + 1

    def on_node_end(self, node: str):
        depth = self._node_depth.get(node, 0)
        if depth <= 0:
            return
        self._node_depth[node] = depth - 1
        if depth == 1:
            dur = time.perf_counter() - self._node_first_start.get(node, 0.0)
            self.node_runs[node].append(dur)
            self.node_total[node] += dur
            self.timeline.append((self._now(), node, f"结束 ({dur:.2f}s)"))
            print(f"    [{self._now():7.2f}s] {NODES[node]} 结束, 耗时 {dur:.2f}s", flush=True)
            self.current_node = None

    def on_llm_start(self):
        self._llm_start[self.current_node or "?"].append(time.perf_counter())

    def on_llm_end(self):
        node = self.current_node or "?"
        starts = self._llm_start.get(node, [])
        if starts:
            self.llm_by_node[node].append(time.perf_counter() - starts.pop())

    def on_tool_start(self, name: str):
        self._tool_start[name].append(time.perf_counter())

    def on_tool_end(self, name: str):
        starts = self._tool_start.get(name, [])
        if starts:
            dur = time.perf_counter() - starts.pop()
            self.tool_by_node[self.current_node or "?"].append(dur)
            self.timeline.append((self._now(), f"工具:{name}", f"结束 ({dur:.2f}s)"))
            print(f"    [{self._now():7.2f}s] 工具 {name} 结束, 耗时 {dur:.2f}s", flush=True)

    def on_stream_chunk(self, text: str):
        if self.current_node == "generate_response":
            if self.gen_first_chunk is None and self.gen_started_at is not None:
                self.gen_first_chunk = time.perf_counter() - self.gen_started_at
            self.stream_chars += len(text)


async def run_question(question: str, project_id: str, tools: list[str]) -> TimeTracker:
    graph = build_graph(checkpointer=MemorySaver())

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "project_id": project_id,
        "summary": get_summary(project_id),
        "system_prompt": "",
        "search_results": [],
        "retrieved_docs": [],
        "agent_outputs": {},
        "next_agent": "",
        "supervisor_log": [],
        "required_tools": tools,
        "reference_sources": [],
        "plan_options": [],
        "chosen_plan_id": "",
        "chosen_plan_detail": {},
        "custom_plan_text": "",
    }

    tracker = TimeTracker()
    config = {"configurable": {"thread_id": project_id}}

    async for event in graph.astream_events(initial_state, config, version="v2"):
        kind = event["event"]
        node = _extract_node(event)

        if kind == "on_chain_start" and node:
            tracker.on_node_start(node)
        elif kind == "on_chain_end" and node:
            tracker.on_node_end(node)
        elif kind == "on_chat_model_start":
            tracker.on_llm_start()
        elif kind == "on_chat_model_end":
            tracker.on_llm_end()
        elif kind == "on_tool_start":
            tracker.on_tool_start(event["name"])
        elif kind == "on_tool_end":
            tracker.on_tool_end(event["name"])
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if getattr(chunk, "content", None):
                tracker.on_stream_chunk(str(chunk.content))

    return tracker


def print_report(tracker: TimeTracker, question: str):
    total = time.perf_counter() - tracker.total_start
    print("\n" + "=" * 62)
    print(f"问题: {question}")
    print("=" * 62)

    print("\n【节点耗时】（含节点内部的 LLM 调用与工具调用）")
    for node, label in NODES.items():
        if node in tracker.node_total:
            runs = tracker.node_runs[node]
            bar = "|" * int(tracker.node_total[node] * 2)
            extra = ""
            if len(runs) > 1:
                extra = f"  单次: " + ", ".join(f"{d:.2f}s" for d in runs)
            print(f"  {label:<5} {tracker.node_total[node]:>6.2f}s  {bar}{extra}")
        else:
            print(f"  {label:<5}    --    (本轮未运行)")

    print("\n【LLM 调用耗时】（外部模型，受网络 + 模型速度影响）")
    total_llm = 0.0
    for node in ["supervisor", "researcher", "generate_response"]:
        for dur in tracker.llm_by_node.get(node, []):
            total_llm += dur
            print(f"  {NODES[node]:<5} LLM 调用: {dur:.2f}s")

    print("\n【工具调用耗时】（外部 API，同一节点内多个工具为并行执行）")
    total_tool = 0.0
    for node in tracker.tool_by_node:
        for dur in tracker.tool_by_node[node]:
            total_tool += dur
            print(f"  {NODES.get(node, node):<5} 工具调用: {dur:.2f}s")
    print("  (注: 工具并行执行，实际等待时间 = 其中最慢的一个)")

    print("\n【节点开销构成】外部等待(LLM+工具) vs 本地开销")
    for node, label in NODES.items():
        if node not in tracker.node_total:
            continue
        llm_t = sum(tracker.llm_by_node.get(node, []))
        tool_t = sum(tracker.tool_by_node.get(node, []))
        local = tracker.node_total[node] - llm_t - tool_t
        print(f"  {label:<5} 外部等待 {llm_t + tool_t:>6.2f}s | 本地开销 {local:>6.2f}s")

    print("\n【回答生成】")
    if tracker.gen_first_chunk is not None:
        print(f"  首 token 延迟(用户感知的关键): {tracker.gen_first_chunk:.2f}s")
        gen_duration = time.perf_counter() - tracker.gen_started_at
        rate = tracker.stream_chars / gen_duration if gen_duration > 0 else 0
        print(f"  流式输出 {tracker.stream_chars} 字符, 约 {rate:.0f} 字符/秒")
    else:
        print("  未捕获到流式输出")

    print("\n【汇总】")
    if total > 0:
        print(f"  总耗时          {total:.2f}s")
        print(f"  LLM 调用总耗时  {total_llm:.2f}s ({total_llm / total * 100:.0f}%)")
        print(f"  工具调用总耗时  {total_tool:.2f}s ({total_tool / total * 100:.0f}%)")
        print(f"  本地/调度开销   {total - total_llm - total_tool:.2f}s")
    print("\n" + "=" * 62)


def main():
    parser = argparse.ArgumentParser(description="多 Agent 性能测试脚本")
    parser.add_argument("question", nargs="?", default="你好", help="要测试的问题")
    parser.add_argument("--tools", nargs="*", default=[],
                        help="用户指定的工具约束，如 web_search aminer_search_papers")
    parser.add_argument("--project", default="", help="复用已有项目 ID（默认每次新建临时项目）")
    args = parser.parse_args()

    init_db()

    if args.project and get_project(args.project):
        project_id = args.project
        print(f"复用项目: {project_id}")
    else:
        project_id = create_project("性能测试临时项目")["id"]
        print(f"新建临时项目: {project_id}")

    try:
        tracker = asyncio.run(run_question(args.question, project_id, args.tools))
    except KeyboardInterrupt:
        print("\n已中断。")
        return

    print_report(tracker, args.question)


if __name__ == "__main__":
    main()