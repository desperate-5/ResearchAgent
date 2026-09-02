import json
import os
import sqlite3

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
            "SELECT literature, writing, experiment FROM user_preferences WHERE id = 1"
        ).fetchone()
        if row and not os.path.exists(os.path.join("data", "preferences.md")):
            lit = json.loads(row["literature"] or "{}")
            wrt = json.loads(row["writing"] or "{}")
            exp = json.loads(row["experiment"] or "{}")
            yaml_dict = {"literature": lit, "writing": wrt, "experiment": exp}
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
        CREATE TABLE IF NOT EXISTS profile_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL DEFAULT 'global',
            project_id TEXT NOT NULL DEFAULT '',
            dimension TEXT NOT NULL,
            value TEXT NOT NULL,
            a REAL NOT NULL DEFAULT 1.0,
            b REAL NOT NULL DEFAULT 1.0,
            source TEXT NOT NULL DEFAULT 'explicit',
            applied INTEGER NOT NULL DEFAULT 0,
            user_locked INTEGER NOT NULL DEFAULT 0,
            last_seen TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS interaction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            message_index INTEGER,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS profile_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_profile_items_dim ON profile_items(scope, project_id, dimension);
        CREATE INDEX IF NOT EXISTS idx_events_project ON interaction_events(project_id, processed);
    ''')
    conn.commit()
    conn.close()
