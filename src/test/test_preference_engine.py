"""偏好进化引擎单测：先验 → 累积 → 衰减 → margin → 手动层 全可断言（不依赖 LLM / DB）。

运行：python src/test/test_preference_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.preferences.engine import (
    apply_evidence, arbitrate, decayed_ab, effective, effective_mu,
    manual_locked_dimensions, mu,
)
from src.preferences.models import ProfileItemRow, PreferencesConfig
from src.preferences.extract import rule_precheck

NOW = "2026-01-01T00:00:00+00:00"
OLD = "2025-01-01T00:00:00+00:00"  # 一年前（半衰期 30 天 → 衰减到几乎先验）

TOTAL = 0
FAILURES = []


def check(label, got, expect, tol=1e-6):
    global TOTAL
    TOTAL += 1
    if isinstance(got, float) and isinstance(expect, float):
        ok = abs(got - expect) < tol
    else:
        ok = got == expect
    if not ok:
        FAILURES.append(label)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got}, expect={expect}")


def check_true(label, cond):
    global TOTAL
    TOTAL += 1
    ok = bool(cond)
    if not ok:
        FAILURES.append(label)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={cond}, expect=True")


def main():
    # 1. Beta μ
    check("mu(4,1)=0.8", mu(4.0, 1.0), 0.8)

    # 2. 陈述式（explicit +3）1 次生效
    items = apply_evidence([], dimension="writing.sentence_style", value="concise",
                           source="explicit", evidence="尽量简短", now=NOW)
    check("explicit 1次 a", items[0].a, 4.0)
    check("explicit 1次 μ", items[0].mu, 0.8)

    # 3. 选择式（choice +2）2 次生效
    items = []
    for _ in range(2):
        items = apply_evidence(items, dimension="domain", value="软件工程领域：Agent 框架",
                               source="choice", evidence="软件工程领域：Agent 框架",
                               scope="project", project_id="p1", now=NOW)
    check("choice 2次 a", items[0].a, 5.0)
    check("choice 2次 μ", items[0].mu, 5 / 6)

    # 4. 观察式（observed +1）3 次生效
    items = []
    for _ in range(3):
        items = apply_evidence(items, dimension="method", value="实验对比",
                               source="observed", evidence="要多做对比实验",
                               scope="project", project_id="p1", now=NOW)
    check("observed 3次 a", items[0].a, 4.0)
    check("observed 3次 μ", items[0].mu, 0.8)

    # 5. 改口回落：先简洁后详细 → 旧值 +3 反证据
    items = apply_evidence([], dimension="writing.sentence_style", value="concise",
                           source="explicit", evidence="简洁点", now=NOW)
    items = apply_evidence(items, dimension="writing.sentence_style", value="elaborate",
                           source="explicit", evidence="详细点", now=NOW)
    concise = next(i for i in items if i.value == "concise")
    elaborate = next(i for i in items if i.value == "elaborate")
    check("改口后 concise b", concise.b, 4.0)
    check("改口后 concise μ", concise.mu, 0.5)
    check("改口后 elaborate μ", elaborate.mu, 0.8)

    # 6. 时间衰减
    a, b = decayed_ab(4.0, 1.0, OLD, NOW, 30.0)
    check_true("衰减后 a 接近先验", abs(a - 1.0) < 0.01)
    it = ProfileItemRow(dimension="writing.sentence_style", value="concise", a=4.0, b=1.0, last_seen=OLD)
    check_true("衰减后 μ<0.8", effective_mu(it, NOW) < 0.8)

    # 7. margin 仲裁
    # 7a 赢家（margin 0.3 ≥ 0.15）
    items = [
        ProfileItemRow(dimension="writing.sentence_style", value="concise", a=4.0, b=1.0, last_seen=NOW),
        ProfileItemRow(dimension="writing.sentence_style", value="elaborate", a=1.0, b=1.0, last_seen=NOW),
    ]
    items = arbitrate(items, NOW)
    applied = [i for i in items if i.applied == 1]
    check("margin 赢家数量", len(applied), 1)
    check("margin 赢家值", applied[0].value, "concise")

    # 7b 悬置（margin 不足 0.15）
    items = [
        ProfileItemRow(dimension="writing.sentence_style", value="concise", a=4.0, b=1.0, last_seen=NOW),    # μ0.8
        ProfileItemRow(dimension="writing.sentence_style", value="elaborate", a=4.0, b=1.5, last_seen=NOW),  # μ0.727
    ]
    items = arbitrate(items, NOW)
    check("margin 悬置", [i.applied for i in items], [0, 0])

    # 7c 低置信不生效（μ=2/3 < 0.8）
    items = [ProfileItemRow(dimension="writing.sentence_style", value="concise", a=2.0, b=1.0, last_seen=NOW)]
    items = arbitrate(items, NOW)
    check("低置信不生效", items[0].applied, 0)

    # 8. 手动层合并
    manual = PreferencesConfig()  # sentence_style 默认空
    applied = [ProfileItemRow(dimension="writing.sentence_style", value="concise", a=4.0, b=1.0, applied=1, last_seen=NOW)]
    eff = effective(manual, applied)
    check("学习层填入空字段", eff.config.writing.sentence_style, "concise")

    # 手动层已设 ref_format → 学习层不改（手动层永不被动）
    manual2 = PreferencesConfig()  # ref_format 默认 GB/T 7714
    applied2 = [ProfileItemRow(dimension="writing.ref_format", value="APA", a=4.0, b=1.0, applied=1, last_seen=NOW)]
    check("手动层锁定 ref_format", "writing.ref_format" in manual_locked_dimensions(manual2), True)
    eff2 = effective(manual2, applied2)
    check("手动层优先 ref_format", eff2.config.writing.ref_format, "GB/T 7714")

    # 9. 提取预检（零 LLM 成本）
    check("预检命中", rule_precheck("以后回答简洁点"), True)
    check("预检未命中", rule_precheck("帮我查论文"), False)

    print("-" * 60)
    print(f"共 {TOTAL} 项断言，失败 {len(FAILURES)} 项" + (f": {FAILURES}" if FAILURES else ""))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
