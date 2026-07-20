import uuid
from datetime import datetime, timezone
from ..memory.store import _get_conn


def create_project(name: str) -> dict:
    conn = _get_conn()
    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (project_id, name, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": project_id, "name": name, "created_at": now, "updated_at": now}


def list_projects() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM projects ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_project(project_id: str) -> bool:
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return True


def update_timestamp(project_id: str):
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE projects SET updated_at = ? WHERE id = ?",
        (now, project_id),
    )
    conn.commit()
    conn.close()
