"""检索质量评估测试脚本：跑通 researcher → reviewer 评估链路，打印完整评分卡。

用法（在项目根目录下运行）：
    # 真实检索 + 质量评估
    python src/test/test_reviewer.py "帮我查最新的AI Agent综述"
    python src/test/test_reviewer.py "xxx" --tools web_search aminer_search_papers

    # 用内置示例来源快速验证评估流程（不发起真实检索，仅需要 LLM 的 API Key）
    python src/test/test_reviewer.py --demo
"""

import argparse
import asyncio
import os
import sys

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

from src.graph.nodes import researcher_node
from src.graph.reviewer import assess_sources
from src.storage.db import init_db
from src.storage.records import get_summary
from src.storage.projects import create_project, get_project

# 内置示例来源：覆盖 高/中/低 三种可信度，便于快速观察评估效果
DEMO_SOURCES = [
    {
        "source_number": 1,
        "title": "A Survey on Large Language Model Based Autonomous Agents",
        "url": "https://doi.org/10.1007/s11704-024-40231-1",
        "source_type": "paper",
        "published": "2024",
        "summary": "A comprehensive survey of LLM-based autonomous agents, proposing a unified framework covering construction, applications and evaluation strategies.",
    },
    {
        "source_number": 2,
        "title": "Large Language Model Based Autonomous Agents for Industrial Maintenance",
        "url": "https://www.mdpi.com/2075-1702/13/9/831",
        "source_type": "paper",
        "published": "2025",
        "summary": "Proposes an autonomous agent powered by LLMs for predictive maintenance, comparing GPT-4o and Gemini on accuracy and cost.",
    },
    {
        "source_number": 3,
        "title": "智能体(AI Agent)全解析:从起源到2025年爆发",
        "url": "https://zhuanlan.zhihu.com/p/example",
        "source_type": "web",
        "summary": "知乎专栏文章，科普 AI Agent 的发展历程与应用场景，无实验数据支撑。",
    },
    {
        "source_number": 4,
        "title": "Why Every Team Needs an AI Agent",
        "url": "https://medium.com/@example/why-every-team-needs-an-ai-agent",
        "source_type": "web",
        "summary": "Medium 上的观点性博客，论述企业引入 AI Agent 的必要性，无引用来源。",
    },
]

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

def _print_sources(sources: list[dict]) -> None:
    print(f"\n【解析出的来源】共 {len(sources)} 条")
    for s in sources:
        print(f"  [{s.get('source_number')}] {s.get('source_type', '')} | {s.get('title', '')}")
        print(f"      url: {s.get('url', '')}")

def _print_assessment(out, question: str) -> None:
    print("\n" + "=" * 62)
    print(f"问题: {question}")
    print("=" * 62)

    print("\n【逐条来源评分卡】")
    if not out.assessments:
        print("  (无评估结果)")
    for a in out.assessments:
        d = a.dimension_scores
        print(f"  [{a.source_number}] 可信度: {a.credibility} | 综合分: {a.score}")
        print(f"      权威 {d.authority} | 时效 {d.timeliness} | 相关 {d.relevance} | 一致 {d.consistency}")
        print(f"      证据: {a.evidence}")

    print("\n【整体质量小结】")
    print(out.summary or "  (无)")

    print("\n【信息缺口】")
    if not out.gaps:
        print("  (无缺口)")
    for g in out.gaps:
        print(f"  - {g}")
    print("\n" + "=" * 62)

async def run_demo(question: str) -> None:
    """--demo：直接对内置示例来源做质量评估，不发起真实检索。"""
    state = {"reference_sources": DEMO_SOURCES}
    out = await assess_sources(state, user_query=question)
    _print_assessment(out, question)

async def run_real(question: str, tools: list[str], project_id: str) -> None:
    """真实链路：researcher 检索 → reviewer 评估（带分段计时）。"""
    import time
    t0 = time.perf_counter()
    state = _build_state(question, project_id, tools)
    result = await researcher_node(state)
    t1 = time.perf_counter()
    print(f"\n[耗时] researcher 检索: {t1 - t0:.2f}s")
    sources = result.get("reference_sources", [])
    _print_sources(sources)

    if not sources:
        print("\n(无来源可评估，请检查检索结果或工具配置)")
        print(f"[耗时] 总计: {time.perf_counter() - t0:.2f}s")
        return

    eval_state = {"reference_sources": sources}
    out = await assess_sources(eval_state, user_query=question)
    t2 = time.perf_counter()
    print(f"[耗时] reviewer 评审: {t2 - t1:.2f}s")
    print(f"[耗时] 总计: {t2 - t0:.2f}s")
    _print_assessment(out, question)

def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评估测试：researcher 检索 → reviewer 评估")
    parser.add_argument("question", nargs="?", default="帮我查最新的AI Agent综述", help="要检索并评估的问题")
    parser.add_argument("--demo", action="store_true", help="使用内置示例来源评估（不发起真实检索）")
    parser.add_argument("--tools", nargs="*", default=["aminer_search_papers", "web_search"],
                        help="检索工具（默认 aminer_search_papers web_search）")
    parser.add_argument("--project", default="", help="复用已有项目 ID（默认新建临时项目）")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_demo(args.question))
        return

    init_db()

    if args.project and get_project(args.project):
        project_id = args.project
        print(f"复用项目: {project_id}")
    else:
        project_id = create_project("质量评估测试临时项目")["id"]
        print(f"新建临时项目: {project_id}")

    asyncio.run(run_real(args.question, args.tools, project_id))

if __name__ == "__main__":
    main()