import { useState } from "react";

interface Props {
  tool: string;
  status: "start" | "end";
}

export default function ToolCallCard({ tool, status }: Props) {
  const [open, setOpen] = useState(status === "start");

  return (
    <div className="tool-call-card">
      <div className="tool-header" onClick={() => setOpen(!open)}>
        <span className={`dot ${status === "start" ? "running" : "done"}`} />
        <span>{status === "start" ? "正在调用" : "调用完成"}: {tool}</span>
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
