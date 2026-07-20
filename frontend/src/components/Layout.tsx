import { useState, useEffect, useCallback } from "react";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import { listProjects, deleteProject } from "../api/client";
import type { Project } from "../api/client";
import ThemeToggle from "./ThemeToggle";

export default function Layout() {
  const [projects, setProjects] = useState<Project[]>([]);
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

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ marginBottom: 0 }}>科研助手</h2>
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
              <span className="project-name">{p.name}</span>
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
