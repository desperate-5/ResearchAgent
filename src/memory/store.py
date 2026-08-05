import json
import sqlite3
import os
from datetime import datetime, timezone

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "research.db")


def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 数据库连接。自动创建 data 目录，启用 WAL 模式提高并发性能。"""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库，创建所有必要的表（projects、messages、summaries、project_sources、project_plans）和索引。
    如果存在旧的 user_preferences 表，则迁移数据到 preferences.md 并删除旧表。"""
    conn = _get_conn()

    # ── 迁移：旧版 user_preferences 表 → data/preferences.md ──
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
    )
    if cursor.fetchone():
        row = conn.execute(
            "SELECT literature, writing, experiment, tool FROM user_preferences WHERE id = 1"
        ).fetchone()
        if row and not os.path.exists(os.path.join("data", "preferences.md")):
            lit = json.loads(row["literature"] or "{}")
            wrt = json.loads(row["writing"] or "{}")
            exp = json.loads(row["experiment"] or "{}")
            tool = json.loads(row["tool"] or "{}")
            yaml_dict = {"literature": lit, "writing": wrt, "experiment": exp, "tool": tool}
            import yaml as _yaml
            yaml_block = _yaml.dump(yaml_dict, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "preferences.md"), "w", encoding="utf-8") as f:
                f.write(f"---\n{yaml_block}\n---\n\n# 科研助手偏好配置\n\n此文件已从旧版数据库中自动迁移。\n")
        conn.execute("DROP TABLE IF EXISTS user_preferences")
        conn.execute("DROP TABLE IF EXISTS project_preferences")
        conn.commit()

    conn.executescript('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS project_sources (
            project_id TEXT PRIMARY KEY,
            source_data TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS project_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            plan_id TEXT,
            plan_title TEXT,
            plan_detail TEXT NOT NULL,
            is_custom INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id, id);
        CREATE INDEX IF NOT EXISTS idx_summaries_project ON summaries(project_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_plans_project ON project_plans(project_id, created_at);
    ''')
    conn.commit()
    conn.close()


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
