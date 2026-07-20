import { useState } from "react";
import { exportReport } from "../api/client";

interface Props {
  projectId: string;
}

export default function ReportExport({ projectId }: Props) {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<string | null>(null);

  const handleExport = async () => {
    setLoading(true);
    try {
      const r = await exportReport(projectId);
      setReport(r);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Export failed";
      alert(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button onClick={handleExport} disabled={loading}>
        {loading ? "生成中..." : "导出报告"}
      </button>

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
    .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/^(?!<[hulb])(.+)$/gm, "<p>$1</p>");
}
