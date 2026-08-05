import { useState } from "react";

interface Props {
  tool: string;
  status: "start" | "end";
  agent?: string;
}

const TOOL_LABELS: Record<string, string> = {
  web_search: "网络搜索",
  aminer_search_papers: "学术论文检索",
  search_uploaded_docs: "文档检索",
  calculator: "数学计算",
  python_executor: "数据分析",
};

const AGENT_LABELS: Record<string, string> = {
  supervisor: "分析调度",
  researcher: "检索文献",
  analyst: "分析数据",
  planner: "方案设计",
  reviewer: "评审结果",
  generate_response: "生成回答",
};

export default function ToolCallCard({ tool, status, agent }: Props) {
  const [open, setOpen] = useState(status === "start");

  return (
    <div className="tool-call-card">
      <div className="tool-header" onClick={() => setOpen(!open)}>
        <span className={`dot ${status === "start" ? "running" : "done"}`} />
        <span>{agent ? `${AGENT_LABELS[agent] || agent} · ` : ""}{status === "start" ? "正在调用" : "调用完成"}: {TOOL_LABELS[tool] || tool}</span>
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#94a3b8" }}>
          {open ? "收起" : "展开"}
        </span>
      </div>
      {open && (
        <div className="tool-body">
          {status === "start" ? "等待工具返回结果..." : "工具已返回结果。"}
        </div>
      )}
    </div>
  );
}
