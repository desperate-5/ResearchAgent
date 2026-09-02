"""完整执行流程测试脚本：从用户问题跑通
load_context → query_triage → supervisor → researcher → reviewer → planner → generate_response。

与 test.py --mode graph 的区别：
- test.py 裸跑 astream_events，遇到 interrupt（query_triage 澄清 / planner 选方案）会停在中途，跑不完；
- 本脚本自动处理 interrupt：澄清自动选第一个方向（或 --use-original 沿用原问题），
  方案自动选第一个，继续执行到 END，并输出各节点耗时与总耗时。

用法（在项目根目录下运行）：
    python src/test/test_full_flow.py "帮我查最新的AI Agent综述"
    python src/test/test_full_flow.py "帮我查最新的AI Agent综述" --tools web_search aminer_search_papers
    python src/test/test_full_flow.py "帮我查最新的AI Agent综述" --use-original
"""

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

# 强制 UTF-8 输出：Windows 控制台默认 GBK，web 标题含 emoji 时 print 会崩
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.builder import build_graph
from src.storage.db import init_db
from src.storage.records import get_summary
from src.storage.projects import create_project, get_project

NODES = {
    "load_context": "加载上下文",
    "query_triage": "检索前澄清",
    "supervisor": "调度决策",
    "researcher": "文献检索",
    "reviewer": "学术评审",
    "planner": "方案设计",
    "generate_response": "生成回答",
}


def _extract_node(event: dict) -> str:
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
    """记录各节点每次进入的耗时（start→end，跨 interrupt 的节点按整体计）。"""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.node_runs = defaultdict(list)  # node -> [dur, ...]
        self.node_total = defaultdict(float)
        self._depth = defaultdict(int)
        self._first_start = {}

    def _now(self):
        return time.perf_counter() - self.t0

    def on_node_start(self, node: str):
        d = self._depth.get(node, 0)
        if d == 0:
            self._first_start[node] = time.perf_counter()
            print(f">>> [{self._now():7.2f}s] {NODES[node]} 开始", flush=True)
        self._depth[node] = d + 1

    def on_node_end(self, node: str):
        d = self._depth.get(node, 0)
        if d <= 0:
            return
        self._depth[node] = d - 1
        if d == 1:
            dur = time.perf_counter() - self._first_start.get(node, 0.0)
            self.node_runs[node].append(dur)
            self.node_total[node] += dur
            print(f"    [{self._now():7.2f}s] {NODES[node]} 结束, 耗时 {dur:.2f}s", flush=True)


async def _drive(graph, payload, config, tracker: TimeTracker):
    """用 astream_events 驱动一轮执行，收集节点耗时。interrupt 时事件流自然结束。"""
    async for event in graph.astream_events(payload, config, version="v2"):
        kind = event["event"]
        node = _extract_node(event)
        if kind == "on_chain_start" and node:
            tracker.on_node_start(node)
        elif kind == "on_chain_end" and node:
            tracker.on_node_end(node)


def _build_resume(snap, use_original: bool) -> dict | None:
    """根据中断 payload 构造 resume 值；无中断返回 None。"""
    if not snap.next:
        return None
    task = snap.tasks[0]
    interrupts = getattr(task, "interrupts", None) or []
    payload = interrupts[0].value if interrupts else {}
    if not isinstance(payload, dict):
        payload = {}

    ptype = payload.get("type", "")
    if ptype == "query_clarification" or "directions" in payload:
        directions = payload.get("directions") or []
        print(f"[中断] 检索前澄清: {len(directions)} 个方向 -> 自动选择", flush=True)
        for i, d in enumerate(directions, 1):
            print(f"        方向{i}: {d}", flush=True)
        if use_original or not directions:
            print(f"[resume] 沿用原问题 (use_original=True)", flush=True)
            return {"selected_direction": "", "use_original": True}
        print(f"[resume] 选择方向: {directions[0]}", flush=True)
        return {"selected_direction": directions[0], "use_original": False}

    # 默认按 plan_options 处理
    options = payload.get("options") or []
    print(f"[中断] 方案设计: {len(options)} 个候选方案 -> 自动选择第一个", flush=True)
    for i, o in enumerate(options, 1):
        print(f"        方案{i}: {o.get('id')} | {o.get('title', '')}", flush=True)
    chosen_id = ""
    if options:
        chosen_id = options[0].get("id") or f"plan_{0}"
    return {"chosen_plan_id": chosen_id, "custom_plan_text": ""}


def _build_state(question: str, project_id: str, tools: list[str]) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "project_id": project_id,
        "summary": get_summary(project_id),
        "system_prompt": "",
        "agent_outputs": {},
        "next_agent": "",
        "supervisor_log": [],
        "required_tools": tools,
        "reference_sources": [],
        "plan_options": [],
        "chosen_plan_id": "",
        "custom_plan_text": "",
    }


def _print_report(tracker: TimeTracker, final_state: dict, question: str, out_lines: list[str]):
    total = time.perf_counter() - tracker.t0
    print("\n" + "=" * 62)
    print(f"问题: {question}")
    print("=" * 62)

    print("\n【节点耗时】（按执行顺序，同一节点多次进入分别列出）")
    for node, label in NODES.items():
        runs = tracker.node_runs.get(node, [])
        if not runs:
            print(f"  {label:<5}    --    (本轮未运行)")
            continue
        detail = ", ".join(f"{d:.2f}s" for d in runs)
        print(f"  {label:<5} 合计 {sum(runs):>6.2f}s  (单次: {detail})")

    print("\n【调度记录】")
    log = final_state.get("supervisor_log", [])
    if not log:
        print("  (无)")
    for entry in log:
        print(f"  next={entry.get('next')} | {entry.get('reason', '')}")

    sources = final_state.get("reference_sources", [])
    print(f"\n【来源】共 {len(sources)} 条")
    for s in sources:
        print(f"  [{s.get('source_number')}] {s.get('source_type', '')} | {s.get('title', '')}")

    print("\n【最终回答】")
    messages = final_state.get("messages", [])
    if messages:
        content = str(messages[-1].content)
        print(content[:3000] + ("\n……(截断显示，完整回答见输出文件)" if len(content) > 3000 else ""))
    else:
        print("  (无回答)")

    print("\n【汇总】")
    print(f"  总耗时          {total:.2f}s")
    for node in ["supervisor", "researcher", "reviewer", "planner", "generate_response"]:
        if node in tracker.node_total:
            print(f"  {NODES[node]:<5}            {tracker.node_total[node]:.2f}s")
    print("\n" + "=" * 62)

    out_lines.append(f"\n[总耗时] {total:.2f}s")


async def run_full_flow(question: str, project_id: str, tools: list[str], use_original: bool) -> None:
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": project_id}}
    tracker = TimeTracker()

    initial_state = _build_state(question, project_id, tools)
    await _drive(graph, initial_state, config, tracker)

    # 处理所有中断点，直到图结束
    while True:
        snap = graph.get_state(config)
        resume = _build_resume(snap, use_original)
        if resume is None:
            break
        await _drive(graph, Command(resume=resume), config, tracker)

    final_state = graph.get_state(config).values

    out_dir = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"full_flow_{ts}.txt")
    out_lines = [f"问题: {question}", "=" * 62]
    _print_report(tracker, final_state, question, out_lines)
    messages = final_state.get("messages", [])
    if messages:
        out_lines.append("")
        out_lines.append("【完整回答】")
        out_lines.append(str(messages[-1].content))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n完整输出已写入: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="完整执行流程测试：自动处理中断跑通全图")
    parser.add_argument("question", nargs="?", default="帮我查最新的AI Agent综述")
    parser.add_argument("--tools", nargs="*", default=[],
                        help="用户指定的工具约束（默认空=LLM 自主决策）")
    parser.add_argument("--project", default="", help="复用已有项目 ID（默认新建临时项目）")
    parser.add_argument("--use-original", action="store_true",
                        help="澄清中断时沿用原问题，不选方向")
    args = parser.parse_args()

    init_db()

    if args.project and get_project(args.project):
        project_id = args.project
        print(f"复用项目: {project_id}")
    else:
        project_id = create_project("完整流程测试临时项目")["id"]
        print(f"新建临时项目: {project_id}")

    asyncio.run(run_full_flow(args.question, project_id, args.tools, args.use_original))


if __name__ == "__main__":
    main()
