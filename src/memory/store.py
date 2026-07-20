import sqlite3
import os
from datetime import datetime, timezone

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "research.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_conn()
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
        CREATE TABLE IF NOT EXISTS project_preferences (
            project_id TEXT PRIMARY KEY,
            literature TEXT NOT NULL DEFAULT '{}',
            writing TEXT NOT NULL DEFAULT '{}',
            experiment TEXT NOT NULL DEFAULT '{}',
            tool TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id, id);
        CREATE INDEX IF NOT EXISTS idx_summaries_project ON summaries(project_id, created_at);
    ''')
    conn.commit()
    conn.close()


def save_summary(project_id: str, summary: str):
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO summaries (project_id, content, created_at) VALUES (?, ?, ?)",
        (project_id, summary, now),
    )
    conn.commit()
    conn.close()


def get_summary(project_id: str) -> str:
    conn = _get_conn()
    row = conn.execute(
        "SELECT content FROM summaries WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    conn.close()
    return row["content"] if row else ""


def save_message(project_id: str, role: str, content: str):
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO messages (project_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (project_id, role, content, now),
    )
    conn.commit()
    conn.close()


def get_history(project_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE project_id = ? ORDER BY id ASC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
