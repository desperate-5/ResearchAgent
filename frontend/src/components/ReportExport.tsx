import { useState } from "react";
import { exportReport } from "../api/client";

interface Props {
  projectId: string;
}

export default function ReportExport({ projectId }: Props) {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await exportReport(projectId);
      setReport(r);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "导出失败";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button onClick={handleExport} disabled={loading}>
        {loading ? "生成中..." : "导出报告"}
      </button>

      {/* 加载弹窗 — 点击后立即显示，解决长时间 LLM 调用无反馈的问题 */}
      {loading && (
        <div className="modal-overlay">
          <div className="modal loading-modal">
            <div className="spinner" />
            <p className="loading-text">
              正在生成研究报告...
              <br />
              <small>LLM 正在基于对话历史、文献来源和研究方案撰写学术文章，请耐心等待（约 1 分钟）</small>
            </p>
          </div>
        </div>
      )}

      {/* 错误弹窗 */}
      {error && (
        <div className="modal-overlay" onClick={() => setError(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>导出失败</h3>
            <p style={{ color: "var(--danger)", marginBottom: 16 }}>{error}</p>
            <div className="modal-buttons">
              <button className="primary" onClick={() => setError(null)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 报告弹窗 */}
      {report && (
        <div className="modal-overlay" onClick={() => setReport(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>研究报告</h3>
            <div
              className="report-body"
              dangerouslySetInnerHTML={{ __html: markdownToHtml(report) }}
            />
            <div className="modal-buttons">
              <button
                className="primary"
                onClick={() => {
                  const blob = new Blob([report], { type: "text/markdown" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "research_report.md";
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                下载 .md
              </button>
              <button className="secondary" onClick={() => setReport(null)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function markdownToHtml(md: string): string {
  return md
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>")
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // 表格 — 必须在段落包裹之前处理
    .replace(/(^\|.+\|\n^\|[-: |]+\|\n(?:^\|.+\|\n?)+)/gm, (match) => {
      const lines = match.trim().split("\n");
      if (lines.length < 2) return match;
      const parseRow = (line: string, tag: "th" | "td") => {
        const cells = line.replace(/^\||\|$/g, "").split("|");
        return `<tr>${cells.map((c) => `<${tag}>${c.trim()}</${tag}>`).join("")}</tr>`;
      };
      const thead = `<thead>${parseRow(lines[0], "th")}</thead>`;
      const tbody = `<tbody>${lines.slice(2).map((l) => parseRow(l, "td")).join("")}</tbody>`;
      return `<table>${thead}${tbody}</table>`;
    })
    .replace(/\n\n/g, "</p><p>")
    .replace(/^(?!<[hultb])(.+)$/gm, "<p>$1</p>");
}
