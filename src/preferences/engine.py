"""偏好进化引擎：纯函数（无 I/O），可独立单测。

对应设计文档三步流程中的"判断"与"更新"：
- 证据折算（+3/+2/+1）→ Beta 账本 a/b（先验 a=b=1，μ = a/(a+b)）
- 时间衰减 → 证据强度随时间回落先验
- margin 仲裁（μ≥0.8 且与第二名差距 ≥0.15）→ applied 置位 / 失效
- 手动层合并（effective）：手动层优先，冲突维度锁定学习条目
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .models import PreferencesConfig, ProfileItemRow

# ── 信号三分法权重 ──
WEIGHTS: dict[str, float] = {
    "explicit": 3.0,   # ① 陈述式显式（"以后回答简洁点"）
    "choice": 2.0,     # ② 选择式显式（澄清方向 / 方案选择）
    "observed": 1.0,   # ③ 观察式隐式（会话蒸馏）
    "manual": 0.0,     # 手动层（不参与学习层账本）
}

RECANT_WEIGHT = 3.0        # 改口/撤销：给旧值 +3 反证据
MU_THRESHOLD = 0.8         # Beta 生效线
MARGIN = 0.15              # 同维度多候选 margin 仲裁阈值
PRIOR_A = 1.0              # 先验
PRIOR_B = 1.0
DECAY_HALF_LIFE_DAYS = 30.0
MAX_EVIDENCE = 10          # 每条目保留的最近证据句数

# 学习层维度 → PreferencesConfig 字段映射（手动层冲突时锁定这些维度）
DIMENSION_CONFIG_MAP: dict[str, tuple[str, str]] = {
    "writing.sentence_style": ("writing", "sentence_style"),
    "writing.figure_norm": ("writing", "figure_norm"),
    "writing.abstract_style": ("writing", "abstract_style"),
    "writing.ref_format": ("writing", "ref_format"),
    "writing.lang": ("writing", "lang"),
    "literature.source_type": ("literature", "source_type"),
    "literature.paper_type": ("literature", "paper_type"),
    "literature.preferred_language": ("literature", "preferred_language"),
}

# 情境层维度（不映射到 PreferencesConfig，单独注入；scope=project）
CONTEXT_DIMENSIONS = {"domain", "method"}


def mu(a: float, b: float) -> float:
    """Beta 分布均值 μ = a/(a+b)。"""
    s = a + b
    return a / s if s > 0 else 0.0


def decayed_ab(a: float, b: float, last_seen: str, now: str,
               half_life_days: float = DECAY_HALF_LIFE_DAYS) -> tuple[float, float]:
    """时间衰减：证据强度按半衰期指数回落到先验 (a=1, b=1)。"""
    if not last_seen or not now:
        return a, b
    try:
        last = datetime.fromisoformat(last_seen)
        cur = datetime.fromisoformat(now)
    except ValueError:
        return a, b
    elapsed = max(0.0, (cur - last).total_seconds())
    if elapsed <= 0 or half_life_days <= 0:
        return a, b
    factor = 0.5 ** (elapsed / 86400.0 / half_life_days)
    a2 = PRIOR_A + (a - PRIOR_A) * factor
    b2 = PRIOR_B + (b - PRIOR_B) * factor
    return max(PRIOR_A, a2), max(PRIOR_B, b2)


def effective_mu(item: ProfileItemRow, now: str,
                 half_life_days: float = DECAY_HALF_LIFE_DAYS) -> float:
    """衰减后的有效置信度 μ。"""
    a, b = decayed_ab(item.a, item.b, item.last_seen, now, half_life_days)
    return mu(a, b)


def _find(items, scope: str, project_id: str, dimension: str, value: str):
    for it in items:
        if (it.scope == scope and it.project_id == project_id
                and it.dimension == dimension and it.value == value):
            return it
    return None


def _dump_evidence(evidences) -> str:
    return json.dumps(list(evidences)[-MAX_EVIDENCE:], ensure_ascii=False)


def apply_evidence(items, *, dimension: str, value: str, source: str, evidence: str,
                   scope: str = "global", project_id: str = "", now: str = ""):
    """把一条证据折算进 Beta 账本，返回（可能新增条目后的）items 列表。

    目标条目 a += 权重；同维度不同值的条目在 source=explicit（改口）时 b += RECANT_WEIGHT。
    条目对象就地更新（items 为浅拷贝，共享条目对象）。
    """
    weight = WEIGHTS.get(source, 1.0)
    items = list(items)
    target = _find(items, scope, project_id, dimension, value)
    if target is None:
        target = ProfileItemRow(
            scope=scope, project_id=project_id, dimension=dimension, value=value,
            a=PRIOR_A, b=PRIOR_B, source=source,
            applied=0, user_locked=0, last_seen=now or "", evidence_json="[]",
        )
        items.append(target)

    target.a += weight
    target.source = source
    if now:
        target.last_seen = now
    if evidence:
        evs = target.evidence_list
        evs.append(str(evidence))
        target.evidence_json = _dump_evidence(evs)

    # 改口/撤销：同维度不同值 → 反证据
    if source == "explicit":
        for it in items:
            if it is target:
                continue
            if (it.scope == scope and it.project_id == project_id
                    and it.dimension == dimension and it.value != value):
                it.b += RECANT_WEIGHT

    return items


def arbitrate(items, now: str, half_life_days: float = DECAY_HALF_LIFE_DAYS):
    """按 (scope, project_id, dimension) 分组做 margin 仲裁，返回置位 applied 后的 items。

    每组按衰减后 μ 降序；赢家需 μ≥MU_THRESHOLD 且与第二名 margin≥MARGIN；
    否则该维度悬置（全 applied=0）。user_locked 条目永不置位。
    """
    items = list(items)
    groups: dict[tuple, list] = {}
    for it in items:
        groups.setdefault((it.scope, it.project_id, it.dimension), []).append(it)

    for group in groups.values():
        scored = sorted(group, key=lambda it: effective_mu(it, now, half_life_days), reverse=True)
        winner = None
        top = scored[0]
        top_mu = effective_mu(top, now, half_life_days)
        if top_mu >= MU_THRESHOLD:
            if len(scored) == 1:
                winner = top
            else:
                second_mu = effective_mu(scored[1], now, half_life_days)
                if top_mu - second_mu >= MARGIN:
                    winner = top
        for it in group:
            it.applied = 1 if (it is winner and not it.user_locked) else 0

    return items


def _is_manually_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def manual_locked_dimensions(manual: PreferencesConfig) -> set[str]:
    """手动层已设置（非空）的维度：学习层不得覆盖。"""
    locked = set()
    for dim, (category, field) in DIMENSION_CONFIG_MAP.items():
        cat = getattr(manual, category, None)
        if cat is None:
            continue
        if _is_manually_set(getattr(cat, field, None)):
            locked.add(dim)
    return locked


def sync_manual_locks(manual: PreferencesConfig, items) -> list:
    """把与手动层冲突的学习条目标记 user_locked=1（挂起，系统不再更新/生效）。"""
    locked = manual_locked_dimensions(manual)
    for it in items:
        if it.dimension in locked:
            it.user_locked = 1
    return items


@dataclass
class EffectiveProfile:
    """effective() 的合并结果：手动层优先 + 已生效学习条目。"""
    config: PreferencesConfig
    domain: str = ""
    method: str = ""
    locked_dimensions: set = field(default_factory=set)


def effective(manual: PreferencesConfig, applied_items, project_id: str = "") -> EffectiveProfile:
    """合并手动层 + 学习层（已生效条目）。手动层优先；冲突维度锁定学习条目。"""
    merged = manual.model_copy(deep=True)
    locked = manual_locked_dimensions(manual)
    domain = ""
    method = ""

    for it in applied_items:
        if it.user_locked or it.dimension in locked:
            continue
        if it.dimension in DIMENSION_CONFIG_MAP:
            category, field = DIMENSION_CONFIG_MAP[it.dimension]
            cat = getattr(merged, category, None)
            if cat is None:
                continue
            if not _is_manually_set(getattr(cat, field, None)):
                try:
                    setattr(cat, field, it.value)
                except Exception:
                    pass
        elif it.dimension == "domain":
            domain = it.value
        elif it.dimension == "method":
            method = it.value

    return EffectiveProfile(config=merged, domain=domain, method=method, locked_dimensions=locked)
