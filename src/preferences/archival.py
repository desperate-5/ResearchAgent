"""学习层后台编排：done 后每轮异步执行，主流程零阻塞。

三步流程落地：
1 提取：① 陈述式（extract）→ ② 选择式（interaction_events）→ ③ 观察式（distill）
2 判断：engine.apply_evidence（证据折算）→ engine.arbitrate（衰减 + μ + margin）
3 更新：持久化 profile_items、标记事件已处理、置位 applied / 失效
"""

from __future__ import annotations

import json
import os
import traceback

from ..storage.records import get_history, get_summary
from . import engine, extract, distill, store
from .models import PreferenceCandidate, ObservedCandidate

DISTILL_EVERY_TURNS = int(os.getenv("PREF_DISTILL_EVERY_TURNS", "10"))


def _scope_for(dimension: str) -> str:
    return "project" if dimension in engine.CONTEXT_DIMENSIONS else "global"


def _extract_domain(direction: str) -> str:
    """把澄清方向归一为学科/领域前缀："软件工程领域：Agent 框架" → "软件工程领域"。"""
    for sep in ("：", ":"):
        if sep in direction:
            return direction.split(sep, 1)[0].strip()
    return direction


def _event_to_record(ev: dict) -> dict | None:
    """把 interaction_events 一行折算为证据记录（dimension/value/evidence/scope）。"""
    kind = ev.get("kind")
    payload = ev.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if kind == "clarification_choice":
        direction = str(payload.get("selected_direction") or "").strip()
        if not direction:
            return None
        domain = _extract_domain(direction)
        return {"dimension": "domain", "value": domain, "evidence": direction, "scope": "project"}

    if kind == "plan_choice":
        if payload.get("is_custom"):
            value = "自定义方案"
        else:
            value = str(payload.get("plan_title") or payload.get("plan_id") or "自定义方案").strip()
        if not value:
            return None
        return {"dimension": "method", "value": value, "evidence": value, "scope": "project"}

    return None


def _explicit_records(candidates: list[PreferenceCandidate]):
    for c in candidates:
        yield {"dimension": c.dimension, "value": c.value, "evidence": c.evidence,
               "scope": _scope_for(c.dimension), "source": "explicit"}


def _observed_records(candidates: list[ObservedCandidate]):
    for c in candidates:
        for quote in c.evidence:
            yield {"dimension": c.dimension, "value": c.value, "evidence": quote,
                   "scope": _scope_for(c.dimension), "source": "observed"}


async def run_archival(project_id: str) -> None:
    """每轮 done 后执行的学习管线（异步、吞异常，不阻塞主流程）。"""
    try:
        now = store.now_iso()
        history = get_history(project_id)
        from .manager import get_preferences

        manual = get_preferences()
        items = store.list_profile_items(project_id)
        locked = engine.manual_locked_dimensions(manual)
        items = engine.sync_manual_locks(manual, items)

        user_turns = [m for m in history if m.get("role") == "user"]
        turn_count = len(user_turns)

        # ① 陈述式：最新用户消息（每个用户轮只提取一次）
        if user_turns:
            explicit_key = f"explicit_turns:{project_id}"
            stored = int(store.get_meta(explicit_key, "0") or "0")
            if turn_count > stored:
                candidates = await extract.extract_explicit(user_turns[-1]["content"])
                for rec in _explicit_records(candidates):
                    if rec["dimension"] in locked:
                        continue
                    items = engine.apply_evidence(
                        items, dimension=rec["dimension"], value=rec["value"],
                        source="explicit", evidence=rec["evidence"],
                        scope=rec["scope"], project_id=project_id, now=now,
                    )
                store.set_meta(explicit_key, turn_count)

        # ② 选择式：未处理的 interaction_events
        pending = store.list_pending_events(project_id)
        if pending:
            processed_ids = []
            for ev in pending:
                processed_ids.append(ev["id"])
                rec = _event_to_record(ev)
                if rec is None or rec["dimension"] in locked:
                    continue
                items = engine.apply_evidence(
                    items, dimension=rec["dimension"], value=rec["value"],
                    source="choice", evidence=rec["evidence"],
                    scope=rec["scope"], project_id=project_id, now=now,
                )
            store.mark_events_processed(processed_ids)

        # ③ 观察式：每 N 轮后台一次（压缩顺带触发点见 compression 模块，这里用轮次阈值兜底）
        if user_turns:
            distill_key = f"distill_turns:{project_id}"
            last_distill = int(store.get_meta(distill_key, "0") or "0")
            if turn_count - last_distill >= DISTILL_EVERY_TURNS:
                observed = await distill.distill(history[-40:], get_summary(project_id))
                for rec in _observed_records(observed):
                    if rec["dimension"] in locked:
                        continue
                    items = engine.apply_evidence(
                        items, dimension=rec["dimension"], value=rec["value"],
                        source="observed", evidence=rec["evidence"],
                        scope=rec["scope"], project_id=project_id, now=now,
                    )
                store.set_meta(distill_key, turn_count)

        # ② 判断 + ③ 更新：仲裁（衰减 + μ + margin）后持久化
        items = engine.arbitrate(items, now)
        for it in items:
            store.upsert_profile_item(it)
    except Exception:
        # 学习管线失败不影响主流程
        traceback.print_exc()
