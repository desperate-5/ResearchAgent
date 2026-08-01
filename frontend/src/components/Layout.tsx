import { useState, useEffect, useCallback } from "react";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import { listProjects, deleteProject, renameProject } from "../api/client";
import type { Project } from "../api/client";
import ThemeToggle from "./ThemeToggle";

export default function Layout() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const navigate = useNavigate();
  const { projectId } = useParams<{ projectId?: string }>();

  const refresh = useCallback(() => {
    listProjects().then(setProjects).catch(console.error);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("确定删除此项目？对话历史和文件将被一并删除。")) return;
    try {
      await deleteProject(id);
      if (projectId === id) navigate("/");
      refresh();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "删除失败";
      alert(msg);
    }
  };

  const startRename = (e: React.MouseEvent, p: Project) => {
    e.stopPropagation();
    setEditingId(p.id);
    setEditName(p.name);
  };

  const saveRename = async () => {
    const trimmed = editName.trim();
    if (!trimmed || !editingId) {
      setEditingId(null);
      return;
    }
    try {
      await renameProject(editingId, trimmed);
      refresh();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "重命名失败";
      alert(msg);
    }
    setEditingId(null);
  };

  const handleRenameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") saveRename();
    if (e.key === "Escape") setEditingId(null);
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ marginBottom: 0 }}>多智能体科研系统</h2>
          <ThemeToggle />
        </div>
        <button className="new-btn" onClick={() => navigate("/")}>
          + 新建项目
        </button>
        <div className="project-list">
          {projects.map((p) => (
            <div
              key={p.id}
              className={`project-item ${p.id === projectId ? "active" : ""}`}
              onClick={() => navigate(`/chat/${p.id}`)}
            >
              {editingId === p.id ? (
                <input
                  className="project-name-input"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={handleRenameKeyDown}
                  onBlur={saveRename}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                  onFocus={(e) => e.target.select()}
                />
              ) : (
                <span
                  className="project-name"
                  onDoubleClick={(e) => startRename(e, p)}
                  title="双击重命名"
                >
                  {p.name}
                </span>
              )}
              <button className="del-btn" onClick={(e) => handleDelete(e, p.id)}>
                &times;
              </button>
            </div>
          ))}
        </div>
      </aside>
      <Outlet />
    </div>
  );
}
