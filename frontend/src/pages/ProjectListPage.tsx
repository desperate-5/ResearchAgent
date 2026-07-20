import { useState } from "react";
import { createProject } from "../api/client";
import { useNavigate } from "react-router-dom";

export default function ProjectListPage() {
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    try {
      const proj = await createProject(name.trim());
      navigate(`/chat/${proj.id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "创建失败";
      alert(msg);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="main">
      <div className="project-create">
        <h2>新建研究项目</h2>
        <form onSubmit={handleCreate}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="输入项目名称..."
            autoFocus
          />
          <button type="submit" disabled={!name.trim() || creating}>
            {creating ? "创建中..." : "创建"}
          </button>
        </form>
      </div>
    </div>
  );
}
