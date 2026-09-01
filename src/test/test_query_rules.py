"""规则层自测：验证乱输入识别与词素放行，不依赖 LLM。

运行：python src/test/test_query_rules.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.graph.query_rules import rule_reject, has_any_morpheme, is_repetitive_pattern
from src.graph.query_triage import _has_broad_intent, _has_explicit_plan_intent

# (输入, 期望 rule_reject 结果, 说明)
CASES = [
    # ── 应拦截的乱输入 ──
    ("ndbajdkla566", True, "随机字母+数字（用户原例）"),
    ("ndbajdkla", True, "纯随机字母"),
    ("xnisjdl", True, "含2字母词 is/an 但无 >=4 词素"),
    ("gfdgfdgfd", True, "重复单元（原规则只查整串单字符）"),
    ("123123123", True, "数字重复单元"),
    ("锟斤拷锟斤拷", True, "中文乱码重复单元"),
    ("啊啊啊啊", True, "重复单字符"),
    ("!!!@@@", True, "纯符号"),
    ("12345", True, "纯数字"),
    ("abc", True, "无意义短串"),
    ("", True, "空输入"),
    ("a", True, "单字符"),
    ("   ", True, "纯空白"),
    ("fjdkslaqwerty", True, "长随机串（无词素子串）"),
    # ── 应放行的合法输入 ──
    ("LLM", False, "全大写术语"),
    ("RAG", False, "全大写缩写"),
    ("AI", False, "全大写缩写"),
    ("test123", False, "含真实单词"),
    ("hello2026", False, "含真实单词"),
    ("the", False, "短词整词命中"),
    ("llm agent 综述", False, "术语+中文"),
    ("如何评价深度学习在医疗影像中的应用", False, "中文问题"),
    ("我想了解一下人机协同", False, "中文概览提问"),
    ("你好", False, "中文寒暄"),
    ("2026年AI进展", False, "中文+术语"),
    ("What is the best survey on RAG?", False, "英文问题"),
    ("langgraph checkpoint 清理", False, "领域术语+中文"),
]


def main():
    failures = 0
    for text, expect, note in CASES:
        got = rule_reject(text)
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            failures += 1
        print(f"[{status}] rule_reject({text!r}) = {got} (期望 {expect})  # {note}")
    print("-" * 60)

    # 单元级抽查
    checks = [
        (has_any_morpheme("ndbajdkla566"), False, "无词素"),
        (has_any_morpheme("test123"), True, "有词素(test)"),
        (has_any_morpheme("LLM"), True, "全大写缩写"),
        (has_any_morpheme("如何评价深度学习"), True, "中文词素"),
        (is_repetitive_pattern("gfdgfdgfd"), True, "重复单元"),
        (is_repetitive_pattern("锟斤拷锟斤拷"), True, "中文重复单元"),
        (is_repetitive_pattern("hello"), False, "正常词"),
    ]
    for got, expect, note in checks:
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            failures += 1
        print(f"[{status}] {note}: got={got}, 期望={expect}")

    print("-" * 60)
    # How-to / 概览型确定性澄清信号（不依赖 LLM 的部分）
    broad_cases = [
        ("人机交互怎么做", True, "用户反馈的案例：必须触发澄清"),
        ("怎么做人机交互", True, "How-to 变体"),
        ("怎么实现一个 RAG 系统", True, "怎么实现"),
        ("如何开展实验研究", True, "如何开展"),
        ("怎样构建知识图谱", True, "怎样构建"),
        ("我想了解一下人机协同", True, "原有概览型"),
        ("最近 AI Agent 有什么进展", True, "原有概览型"),
        ("Python list 怎么排序", False, "具体小问题，不误触发"),
        ("这个 API 怎么用", False, "怎么用不在模式内"),
        ("帮我查 2024 年 LLM Agent 记忆机制的综述论文", False, "具体检索需求"),
    ]
    for text, expect, note in broad_cases:
        got = _has_broad_intent(text)
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            failures += 1
        print(f"[{status}] _has_broad_intent({text!r}) = {got} (期望 {expect})  # {note}")

    print("-" * 60)
    # 明确方案设计意图：应跳过检索前澄清，直达 supervisor 的 planner 路由
    plan_cases = [
        ("我要实现一个上下文模块要怎么进行方案设计", True, "用户反馈案例：含方案设计，必须跳过澄清"),
        ("帮我设计一个研究方案", True, "设计一个 + 研究方案"),
        ("制定技术路线", True, "技术路线"),
        ("实验设计怎么做", True, "实验设计"),
        ("人机交互怎么做", False, "无明确方案词，仍需澄清选专业方向"),
        ("怎么实现一个 RAG 系统", False, "泛化 How-to，不视为明确方案意图"),
    ]
    for text, expect, note in plan_cases:
        got = _has_explicit_plan_intent(text)
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            failures += 1
        print(f"[{status}] _has_explicit_plan_intent({text!r}) = {got} (期望 {expect})  # {note}")

    print("-" * 60)
    total = len(CASES) + len(checks) + len(broad_cases) + len(plan_cases)
    print(f"共 {total} 项断言，失败 {failures} 项")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
