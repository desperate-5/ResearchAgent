import { useEffect, useRef } from "react";
import type { SourceItem } from "../api/client";

export interface SourceWithIndex extends SourceItem {
  message_index: number;
}

interface SourcesPanelProps {
  sources: SourceWithIndex[];
  onJumpToMessage: (index: number, sourceNum: number) => void;
  highlightIndex?: number;
}

const TYPE_LABELS: Record<string, string> = {
  web: "网页",
  paper: "论文",
  document: "文档",
};

function getDomain(url: string): string {
  try {
    return new URL(url).hostname.replace("www.", "");
  } catch {
    return url;
  }
}

const CREDIBILITY_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  "高": { bg: "#dcfce7", color: "#166534", label: "高可信" },
  "中": { bg: "#fef9c3", color: "#854d0e", label: "中可信" },
  "低": { bg: "#ffedd5", color: "#9a3412", label: "低可信" },
  "未评级": { bg: "#f3f4f6", color: "#6b7280", label: "未评级" },
};

export default function SourcesPanel({ sources, onJumpToMessage, highlightIndex }: SourcesPanelProps) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (highlightIndex === undefined) return;
    const el = listRef.current?.querySelector(`[data-source-num="${highlightIndex}"]`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [highlightIndex]);

  return (
    <div className="sources-panel">
      <div className="sources-header">
        <h3>信息来源</h3>
        {sources.length > 0 && <span className="sources-count">{sources.length}</span>}
      </div>

      <div className="sources-list" ref={listRef}>
        {sources.length === 0 ? (
          <div className="sources-empty">暂无信息来源</div>
        ) : (
          sources.map((s, i) => {
            const displayNum = s.source_number || i + 1;
            return (
            <div
              key={s.id || `${s.url}_${s.title}_${i}`}
              className={`source-item${highlightIndex === displayNum ? " highlighted" : ""}`}
              data-source-num={displayNum}
            >
              <div className="source-header-row">
                <span className="source-number">{displayNum}</span>
                <span className="source-type-badge">{TYPE_LABELS[s.source_type] || s.source_type}</span>
                {s.credibility && CREDIBILITY_STYLE[s.credibility] && (
                  <span
                    className="credibility-badge"
                    style={{
                      background: CREDIBILITY_STYLE[s.credibility].bg,
                      color: CREDIBILITY_STYLE[s.credibility].color,
                    }}
                  >
                    {CREDIBILITY_STYLE[s.credibility].label}
                  </span>
                )}
              </div>
              <div className="source-title" title={s.title}>
                {s.title}
              </div>
              {s.source_type === "document" && (s.page != null || s.position) && (
                <div className="source-location">
                  {s.position ? `${s.position} | ` : ""}第{s.page}页
                </div>
              )}
              {s.summary && (
                <div className="source-summary">{s.summary}</div>
              )}
              {s.url && (
                <div className="source-domain">{getDomain(s.url)}</div>
              )}
              <div className="source-actions">
                {s.url ? (
                  <button
                    className="source-btn source-btn-open"
                    onClick={() => window.open(s.url, "_blank", "noopener")}
                    title="打开原文"
                  >
                    打开原文
                  </button>
                ) : (
                  <span
                    className="source-no-link"
                    title="该来源没有可打开的外部链接（如论文未提供 DOI）"
                  >
                    无原文链接
                  </span>
                )}
                <button
                  className="source-btn source-btn-jump"
                  onClick={() => onJumpToMessage(s.message_index, displayNum)}
                  title="定位对话"
                >
                  定位对话
                </button>
              </div>
            </div>
            );
          })
        )}
      </div>
    </div>
  );
}
