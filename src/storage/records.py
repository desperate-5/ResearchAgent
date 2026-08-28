import json
from datetime import datetime, timezone

from .db import _get_conn


def save_summary(project_id: str, summary: str):
    """保存对话摘要。每次压缩生成的新摘要以追加方式写入 summaries 表。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO summaries (project_id, content, created_at) VALUES (?, ?, ?)",
        (project_id, summary, now),
    )
    conn.commit()
    conn.close()


def get_summary(project_id: str) -> str:
    """获取项目最新的对话摘要。按创建时间倒序取第一条，如果没有则返回空字符串。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT content FROM summaries WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    conn.close()
    return row["content"] if row else ""


def save_message(project_id: str, role: str, content: str):
    """保存单条对话消息到 messages 表。role 为 'user' 或 'assistant'。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO messages (project_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (project_id, role, content, now),
    )
    conn.commit()
    conn.close()


def save_project_sources(project_id: str, sources: list[dict]):
    """保存项目的参考文献来源列表。使用 INSERT OR REPLACE 覆盖更新。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO project_sources (project_id, source_data, updated_at) VALUES (?, ?, ?)",
        (project_id, json.dumps(sources, ensure_ascii=False), now),
    )
    conn.commit()
    conn.close()


def get_project_sources(project_id: str) -> list[dict]:
    """获取项目保存的参考文献来源列表。如果 JSON 解析失败则返回空列表。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT source_data FROM project_sources WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row and row["source_data"]:
        try:
            return json.loads(row["source_data"])
        except json.JSONDecodeError:
            return []
    return []


def get_history(project_id: str) -> list[dict]:
    """获取项目的完整对话历史，按消息 id 升序排列（从最早到最新）。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE project_id = ? ORDER BY id ASC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_project_plan(project_id: str, plan_id: str, plan_title: str, plan_detail: str, is_custom: bool = False):
    """保存用户选定的研究方案。追加写入，保留历史版本。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO project_plans (project_id, plan_id, plan_title, plan_detail, is_custom, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, plan_id if plan_id else None, plan_title, plan_detail, 1 if is_custom else 0, now),
    )
    conn.commit()
    conn.close()


def get_latest_plan(project_id: str) -> dict | None:
    """获取项目最新的研究方案。按创建时间倒序取第一条，没有则返回 None。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT plan_id, plan_title, plan_detail, is_custom, created_at FROM project_plans WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
