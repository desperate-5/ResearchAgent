"""回归测试：澄清 interrupt -> 选择方向 -> resume 后，supervisor 必须执行 researcher。

关键验证点：即使 supervisor LLM（假实现）返回 FINISH，确定性守门
（was_clarified 且尚未检索）也必须覆盖为 researcher —— 修复"澄清后跳过检索"的回归。
运行：python src/test/test_resume_flow.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage


# ── 确定性假 LLM：supervisor 故意返回 FINISH（模拟不听话），其余 agent 正常输出 ──
class FakeLLM:
    def __init__(self, streaming: bool = False):
        self.streaming = streaming

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        joined = " ".join(str(getattr(m, "content", m)) for m in msgs)
        if "任务调度者" in joined:
            # supervisor：故意 FINISH —— 守门逻辑必须覆盖它
            return AIMessage(content='{"next": "FINISH", "reason": "测试：LLM 想直接回答"}')
        if "文献检索与信息收集专家" in joined:
            return AIMessage(content="检索结果：\n1. [web] 测试标题\n   来源: 测试来源\n   链接: https://example.com/a\n   内容: 测试内容")
        if "研究方案设计专家" in joined:
            return AIMessage(content='[{"id": "plan_a", "title": "方案A", "description": "思路", "pros": ["p1"], "cons": ["c1"]}]')
        if "研究分析AI" in joined:
            return AIMessage(content="最终回答：测试答案")
        return AIMessage(content="默认回复")


# 把 get_llm 换成 FakeLLM（nodes / query_triage 都用 get_llm 拿 LLM）
import src.graph.nodes as nodes_mod
import src.graph.query_triage as triage_mod


def _fake_get_llm(**kwargs):
    return FakeLLM(streaming=kwargs.get("streaming", False))


nodes_mod.get_llm = _fake_get_llm
triage_mod._get_classify_llm = lambda: FakeLLM()
triage_mod._classify = lambda raw, broad_hint=False: asyncio.ensure_future(_classify_fake(raw, broad_hint))


async def _classify_fake(raw, broad_hint=False):
    return {"ambiguous": True, "directions": ["领域A：方向1", "领域B：方向2", "领域C：方向3"], "invalid": False}


def _initial_state():
    return {
        "messages": [HumanMessage(content="人机交互怎么做")],
        "project_id": "test-proj",
        "summary": "",
        "system_prompt": "",
        "agent_outputs": {},
        "next_agent": "",
        "supervisor_log": [],
        "required_tools": [],
        "reference_sources": [],
        "source_ratings": [],
        "source_assessments": [],
        "retrieval_gaps": [],
        "search_round": 0,
        "plan_options": [],
        "chosen_plan_id": "",
        "custom_plan_text": "",
        "effective_query": "",
        "was_clarified": False,
        "has_prior_research": False,
        "query_invalid": False,
    }


async def main():
    from src.graph.builder import build_graph

    ck = MemorySaver()
    graph = build_graph(checkpointer=ck)
    config = {"configurable": {"thread_id": "test-thread"}}

    print("=== 第一轮：触发澄清 ===")
    try:
        async for _ in graph.astream_events(_initial_state(), config, version="v2"):
            pass
    except Exception as e:
        print("  异常:", type(e).__name__, e)

    st = await graph.aget_state(config)
    assert st.interrupts, "第一轮应产生澄清 interrupt"
    print("  [OK] 第一轮中断:", [(getattr(i, "value", i)) for i in st.interrupts])

    print("=== 第二轮：Command(resume=选择方向)，supervisor 假 LLM 返回 FINISH ===")
    seen2 = []
    try:
        async for ev in graph.astream_events(Command(resume={"selected_direction": "领域A：方向1"}), config, version="v2"):
            kind = ev.get("event")
            if kind in ("on_chain_start", "on_chain_end"):
                name = ev.get("name", "")
                meta = ev.get("metadata", {})
                node = meta.get("langgraph_node", "") or name
                seen2.append((kind, node, name))
    except Exception as e:
        import traceback
        print("  resume 异常:", type(e).__name__, e)
        traceback.print_exc()

    st2 = await graph.aget_state(config)
    print("  resume 后有效查询:", repr(st2.values.get("effective_query", "")))
    print("  resume 后 was_clarified:", st2.values.get("was_clarified"))
    print("  resume 后 agent_outputs keys:", list(st2.values.get("agent_outputs", {}).keys()))

    nodes_run = [n for k, n, _ in seen2 if k == "on_chain_start"]
    has_researcher = "researcher" in nodes_run
    has_generate = "generate_response" in nodes_run
    print("  resume 轮启动过的节点:", nodes_run)

    assert st2.values.get("was_clarified") is True, "was_clarified 应为 True"
    assert has_researcher, "[FAIL] 澄清后 supervisor 未执行 researcher —— 守门逻辑失效"
    assert has_generate, "[FAIL] 未执行 generate_response"
    print("  [OK] PASS：澄清后确定性执行了 researcher（守门覆盖了 LLM 的 FINISH），最终 generate_response")


if __name__ == "__main__":
    asyncio.run(main())

