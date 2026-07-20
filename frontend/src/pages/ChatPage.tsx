import { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "react-router-dom";
import { streamChat, getHistory, listProjects } from "../api/client";
import type { SSEEvent, Project } from "../api/client";
import ToolCallCard from "../components/ToolCallCard";
import FeedbackButtons from "../components/FeedbackButtons";
import FileUpload from "../components/FileUpload";
import ReportExport from "../components/ReportExport";
import { renderMarkdown } from "../utils/markdown";

interface ToolCallEvent {
  tool: string;
  status: "start" | "end";
  id: number;
}

const AGENT_LABELS: Record<string, string> = {
  researcher: "检索文献",
  analyst: "分析数据",
  reviewer: "评审结果",
  generate_response: "生成回答",
};

const TOOL_OPTIONS = [
  { id: "web_search", label: "网络搜索" },
  { id: "aminer_search_papers", label: "学术论文" },
  { id: "python_executor", label: "数据画图" },
];

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCallEvent[];
}

export default function ChatPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [projectName, setProjectName] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const messagesEnd = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // load project name and history
  useEffect(() => {
    if (!projectId) return;
    setLoadingHistory(true);
    setMessages([]);

    Promise.all([
      listProjects(),
      getHistory(projectId),
    ]).then(([projects, history]) => {
      const p = projects.find((pp: Project) => pp.id === projectId);
      setProjectName(p?.name ?? projectId);

      const msgs: ChatMessage[] = [];
      for (const h of history) {
        if (h.role === "user" || h.role === "assistant") {
          msgs.push({
            role: h.role as "user" | "assistant",
            content: h.content,
            toolCalls: [],
          });
        }
      }
      setMessages(msgs);
      setLoadingHistory(false);
    }).catch(() => setLoadingHistory(false));
  }, [projectId]);

  const scrollToBottom = (smooth: boolean) => {
    messagesEnd.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto" });
  };

  useEffect(() => {
    if (!loadingHistory) {
      scrollToBottom(false);
    }
  }, [loadingHistory]);

  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
    if (atBottom) {
      scrollToBottom(true);
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || streaming || !projectId) return;

    const userMsg: ChatMessage = { role: "user", content: input, toolCalls: [] };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStreaming(true);

    const assistantMsg: ChatMessage = { role: "assistant", content: "", toolCalls: [] };
    setMessages((prev) => [...prev, assistantMsg]);

    const abort = new AbortController();
    abortRef.current = abort;

    const pendingTools: Map<string, ToolCallEvent> = new Map();
    let toolIdCounter = 0;

    try {
      const tools = Array.from(selectedTools);
      await streamChat(
        projectId,
        userMsg.content,
        (event: SSEEvent) => {
          if (event.type === "response") {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last && last.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + event.content,
                };
              }
              return updated;
            });
          } else if (event.type === "tool_call") {
            if (event.status === "start") {
              const tc: ToolCallEvent = { tool: event.tool, status: "start", id: ++toolIdCounter };
              pendingTools.set(event.tool, tc);
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role === "assistant") {
                  updated[updated.length - 1] = {
                    ...last,
                    toolCalls: [...last.toolCalls, tc],
                  };
                }
                return updated;
              });
            } else {
              const existing = pendingTools.get(event.tool);
              if (existing) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === "assistant") {
                    updated[updated.length - 1] = {
                      ...last,
                      toolCalls: last.toolCalls.map((tc) =>
                        tc.id === existing.id ? { ...tc, status: "end" as const } : tc
                      ),
                    };
                  }
                  return updated;
                });
                pendingTools.delete(event.tool);
              }
            }
          } else if (event.type === "agent_phase") {
            setActiveAgents((prev) => {
              const next = new Set(prev);
              if (event.status === "start") {
                next.add(event.agent);
              } else {
                next.delete(event.agent);
              }
              return next;
            });
          }
        },
        abort.signal,
        tools,
      );
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      const msg = e instanceof Error ? e.message : "Stream error";
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === "assistant") {
          updated[updated.length - 1] = { ...last, content: last.content + `\n\n[错误: ${msg}]` };
        }
        return updated;
      });
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, streaming, projectId]);

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const toggleTool = (toolId: string) => {
    setSelectedTools((prev) => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (loadingHistory) {
    return (
      <div className="main">
        <div className="empty-state"><p>加载中...</p></div>
      </div>
    );
  }

  return (
    <div className="main">
      <div className="chat-container">
        <div className="chat-header">
          <h2>{projectName}</h2>
          <div className="header-actions">
            <FileUpload projectId={projectId!} />
            <ReportExport projectId={projectId!} />
          </div>
        </div>

        <div className="messages" ref={messagesRef}>
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="role">{msg.role === "user" ? "你" : "助手"}</div>

              {msg.toolCalls.map((tc) => (
                <ToolCallCard key={tc.id} tool={tc.tool} status={tc.status} />
              ))}

              {msg.content && (
                <div
                  className="bubble"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                />
              )}

              {msg.role === "assistant" && msg.content && !streaming && (
                <FeedbackButtons projectId={projectId!} />
              )}
            </div>
          ))}
          <div ref={messagesEnd} />
        </div>

        {activeAgents.size > 0 && (
          <div className="agent-status">
            {Array.from(activeAgents).map((agent) => (
              <span key={agent} className="agent-badge">
                <span className="dot running" />
                {AGENT_LABELS[agent] || agent}
              </span>
            ))}
          </div>
        )}

        <div className="chat-input-area">
          <div style={{ display: "flex", flexDirection: "column", flex: 1, gap: 6 }}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题..."
              rows={2}
              disabled={streaming}
              style={{ width: "100%" }}
            />
            <div className="tool-selector">
              {TOOL_OPTIONS.map((opt) => (
                <span
                  key={opt.id}
                  className={`tool-toggle${selectedTools.has(opt.id) ? " active" : ""}`}
                  onClick={() => toggleTool(opt.id)}
                >
                  {opt.label}
                </span>
              ))}
            </div>
          </div>
          {streaming ? (
            <button onClick={handleStop} style={{ background: "var(--danger)" }}>
              停止
            </button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim()}>
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
