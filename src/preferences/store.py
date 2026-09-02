"""学习层 / 证据层物理存储：profile_items + interaction_events + profile_meta。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..storage.db import _get_conn
from . import engine
from .models import ProfileItemRow


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_item(row) -> ProfileItemRow:
    return ProfileItemRow(**dict(row))


# ── 证据层 interaction_events ──

def save_interaction_event(project_id: str, message_index: int | None, kind: str, payload) -> int:
    """写入一条交互证据（澄清方向 / 方案选择）。"""
    conn = _get_conn()
    if isinstance(payload, (dict, list)):
        payload_text = json.dumps(payload, ensure_ascii=False)
    else:
        payload_text = str(payload)
    cur = conn.execute(
        "INSERT INTO interaction_events (project_id, message_index, kind, payload, created_at, processed) VALUES (?,?,?,?,?,0)",
        (project_id, message_index, kind, payload_text, now_iso()),
    )
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    return eid


def list_pending_events(project_id: str) -> list[dict]:
    """未折算进学习层的证据事件。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, message_index, kind, payload, created_at FROM interaction_events WHERE project_id=? AND processed=0 ORDER BY id ASC",
        (project_id,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d["payload"])
        except Exception:
            pass
        out.append(d)
    return out


def mark_events_processed(ids: list[int]) -> None:
    if not ids:
        return
    conn = _get_conn()
    conn.execute(
        f"UPDATE interaction_events SET processed=1 WHERE id IN ({','.join('?' * len(ids))})",
        ids,
    )
    conn.commit()
    conn.close()


def list_interaction_events(project_id: str | None = None, limit: int = 200) -> list[dict]:
    conn = _get_conn()
    if project_id:
        rows = conn.execute(
            "SELECT * FROM interaction_events WHERE project_id=? ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM interaction_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d["payload"])
        except Exception:
            pass
        out.append(d)
    return out


# ── 学习层 profile_items ──

def list_profile_items(project_id: str | None = None) -> list[ProfileItemRow]:
    """列出学习层条目：给定项目 → global + 该项目；None → 全部。"""
    conn = _get_conn()
    if project_id:
        rows = conn.execute(
            "SELECT * FROM profile_items WHERE scope='global' OR (scope='project' AND project_id=?) ORDER BY id",
            (project_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM profile_items ORDER BY id").fetchall()
    conn.close()
    return [_row_to_item(dict(r)) for r in rows]


def upsert_profile_item(item: ProfileItemRow) -> ProfileItemRow:
    conn = _get_conn()
    try:
        if item.id:
            conn.execute(
                "UPDATE profile_items SET a=?, b=?, source=?, applied=?, user_locked=?, last_seen=?, evidence_json=? WHERE id=?",
                (item.a, item.b, item.source, item.applied, item.user_locked, item.last_seen, item.evidence_json, item.id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO profile_items (scope, project_id, dimension, value, a, b, source, applied, user_locked, last_seen, evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item.scope, item.project_id, item.dimension, item.value, item.a, item.b, item.source, item.applied, item.user_locked, item.last_seen, item.evidence_json),
            )
            item.id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return item


# ── meta 键值（显式提取 / 蒸馏的轮次水位去重） ──

def get_meta(key: str, default: str = "") -> str:
    conn = _get_conn()
    row = conn.execute("SELECT value FROM profile_meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_meta(key: str, value) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO profile_meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


# ── 读侧合并（手动层 + 学习层） ──

def compute_effective(project_id: str) -> engine.EffectiveProfile:
    """每轮开头读取：手动层 + 已生效学习层合并（冲突手动赢）。"""
    from .manager import get_preferences

    manual = get_preferences()
    items = list_profile_items(project_id)
    applied = [i for i in items if i.applied == 1 and i.user_locked == 0]
    return engine.effective(manual, applied, project_id)


def get_applied_hints(project_id: str) -> dict:
    """读取已生效的 domain/method 情境层条目（供 query_triage / planner 使用）。"""
    domain = ""
    method = ""
    for i in list_profile_items(project_id):
        if i.applied != 1 or i.user_locked == 1:
            continue
        if i.dimension == "domain":
            domain = i.value
        elif i.dimension == "method":
            method = i.value
    return {"domain": domain, "method": method}
