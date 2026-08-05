"""性能测试脚本：运行多 Agent 图 + 文档检索专项基准，输出耗时明细，用于回答速度优化。

用法（在项目根目录下运行）：
    # 全链路图测试：定位慢在哪个节点（调度/检索/生成）
    python src/test.py "你的测试问题"
    python src/test.py "帮我查最新的AI Agent综述" --tools web_search aminer_search_papers
    python src/test.py "..." --project <已有项目ID>

    # 文档检索专项基准：单独测 RAG 的索引与查询速度（排查文档检索慢）
    python src/test.py "某论文里的结论是什么" --mode rag --project <项目ID> --runs 3
    python src/test.py "某论文里的结论是什么" --mode rag --index path/to/file.pdf --runs 3
"""

import argparse
import asyncio
import os
import statistics
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
from src.tools.file_rag import (
    APIEmbeddingFunction,
    TOP_K,
    _get_collection,
    _get_embedding_fn,
    get_project_files,
    index_document,
    search_chunks,
)

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
            self.tool_by_node[self.current_node or "?"].append((name, dur))
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
    for node, calls in tracker.tool_by_node.items():
        for name, dur in calls:
            total_tool += dur
            print(f"  {NODES.get(node, node):<5} {name:<26} {dur:.2f}s")
    rag_embed = sum(d for _, d in EMBED_CALLS)
    if rag_embed:
        print(f"  (其中嵌入 API 调用 {len(EMBED_CALLS)} 次, 合计 {rag_embed:.2f}s, "
              f"文档检索耗时的大头通常在嵌入 API)")
    print("  (注: 工具并行执行，实际等待时间 = 其中最慢的一个)")

    print("\n【节点开销构成】外部等待(LLM+工具) vs 本地开销")
    for node, label in NODES.items():
        if node not in tracker.node_total:
            continue
        llm_t = sum(tracker.llm_by_node.get(node, []))
        tool_t = sum(dur for _, dur in tracker.tool_by_node.get(node, []))
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


# ============================================================
# 文档检索专项基准（排查 RAG 慢的问题）
# ============================================================

EMBED_CALLS: list[tuple[int, float]] = []


def _install_embedding_timer():
    """给嵌入函数装上计时包装：记录每次嵌入 API 调用的条数与耗时（只补丁一次）。"""
    if getattr(APIEmbeddingFunction, "__timed", False):
        return
    original = APIEmbeddingFunction.__call__

    def timed(self, input):
        t0 = time.perf_counter()
        try:
            return original(self, input)
        finally:
            EMBED_CALLS.append((len(input), time.perf_counter() - t0))

    APIEmbeddingFunction.__call__ = timed
    APIEmbeddingFunction.__timed = True


def _bench_index_file(project_id: str, file_path: str):
    """对单个文件做索引基准：解析 + 分块 + 嵌入 + 写入向量库。"""
    filename = os.path.basename(file_path)
    print(f"\n── 索引基准: {filename} ──")
    EMBED_CALLS.clear()
    t0 = time.perf_counter()
    n_chunks = index_document(project_id, file_path, filename)
    total = time.perf_counter() - t0

    embed_time = sum(d for _, d in EMBED_CALLS)
    embed_calls = len(EMBED_CALLS)
    local_time = total - embed_time
    print(f"  生成片段: {n_chunks} 个")
    print(f"  总耗时       {total:6.2f}s")
    if embed_calls:
        print(f"    嵌入调用   {embed_time:6.2f}s  ({embed_calls} 次, 平均 {embed_time / embed_calls:.2f}s/次, 受嵌入模型 API 影响)")
        print(f"    解析+分块   {local_time:6.2f}s  (本地 CPU/IO)")
    else:
        print(f"    本地处理   {total:6.2f}s  (解析+分块)")


def _bench_search(project_id: str, query: str, k: int, runs: int):
    """文档检索查询基准：分解 嵌入 / 向量检索 / 格式化 三段时间。"""
    collection = _get_collection(project_id)
    count = collection.count()
    if count == 0:
        print("\n集合为空，无法进行检索基准。请先通过 --index 索引文件，或使用已有 --project。")
        return

    print(f"\n── 检索查询基准: 查询词「{query[:60]}」 k={k} runs={runs} ──")
    emb_fn = _get_embedding_fn()
    emb_fn([query])  # 预热一次嵌入连接

    stats = {"total": [], "embed": [], "query": [], "format": []}
    for i in range(runs):
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        query_emb = emb_fn([query])[0]
        t_embed = time.perf_counter() - t1

        t2 = time.perf_counter()
        result = collection.query(query_embeddings=[query_emb], n_results=k)
        t_query = time.perf_counter() - t2

        t3 = time.perf_counter()
        docs = []
        if result["documents"] and result["documents"][0]:
            for di, doc_text in enumerate(result["documents"][0]):
                meta = result["metadatas"][0][di] if result["metadatas"] else {}
                docs.append({
                    "content": doc_text,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "page": meta.get("page", 1),
                    "paragraph": meta.get("paragraph", 0),
                    "section": meta.get("section", ""),
                })
        t_format = time.perf_counter() - t3

        total = time.perf_counter() - t0
        stats["total"].append(total)
        stats["embed"].append(t_embed)
        stats["query"].append(t_query)
        stats["format"].append(t_format)
        print(f"  第 {i + 1} 次: 总 {total:6.2f}s | 嵌入 {t_embed:6.2f}s | 向量检索 {t_query:6.2f}s | 格式化 {t_format:5.3f}s | 命中 {len(docs)} 个")

    print("  [汇总]")
    for key, label in [("total", "总耗时   "), ("embed", "嵌入耗时 "), ("query", "检索耗时 "), ("format", "格式化耗时")]:
        vals = stats[key]
        if not vals:
            continue
        print(f"    {label}: 平均 {statistics.mean(vals):.2f}s | 最小 {min(vals):.2f}s | 最大 {max(vals):.2f}s")

    real_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        search_chunks(project_id, query, k)
        real_times.append(time.perf_counter() - t0)
    if real_times:
        print(f"\n  实际 search_chunks 调用 {len(real_times)} 次: "
              f"平均 {statistics.mean(real_times):.2f}s | 最小 {min(real_times):.2f}s | 最大 {max(real_times):.2f}s")


def run_rag_mode(args):
    """--mode rag：文档检索专项基准（索引速度 + 查询速度分解）。"""
    _install_embedding_timer()

    if args.project and get_project(args.project):
        project_id = args.project
        print(f"复用项目: {project_id}")
    else:
        project_id = create_project("检索性能测试临时项目")["id"]
        print(f"新建临时项目: {project_id}")

    for fp in args.index:
        if not os.path.isfile(fp):
            print(f"跳过不存在的文件: {fp}")
            continue
        try:
            _bench_index_file(project_id, fp)
        except Exception as e:
            print(f"索引失败 {fp}: {e}")

    files = get_project_files(project_id)
    if files:
        print(f"\n项目上传文件 ({len(files)} 个):")
        for f in files:
            print(f"  - {f['filename']} ({f['size'] / 1024:.0f} KB)")

    _bench_search(project_id, args.question, args.k, args.runs)


def main():
    parser = argparse.ArgumentParser(description="多 Agent 性能测试脚本")
    parser.add_argument("question", nargs="?", default="你好", help="要测试的问题 / rag 模式下的检索查询词")
    parser.add_argument("--mode", choices=["graph", "rag"], default="graph",
                        help="graph=全链路图测试; rag=文档检索专项基准")
    parser.add_argument("--tools", nargs="*", default=[],
                        help="用户指定的工具约束，如 web_search aminer_search_papers search_uploaded_docs")
    parser.add_argument("--project", default="", help="复用已有项目 ID（默认每次新建临时项目）")
    parser.add_argument("--k", type=int, default=TOP_K, help="检索返回片段数（仅 rag 模式）")
    parser.add_argument("--runs", type=int, default=3, help="检索重复次数（仅 rag 模式）")
    parser.add_argument("--index", nargs="*", default=[],
                        help="索引文件路径（仅 rag 模式），用于测解析+嵌入速度；已索引过的文件会报错")
    args = parser.parse_args()

    init_db()
    _install_embedding_timer()

    if args.mode == "rag":
        run_rag_mode(args)
        return

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