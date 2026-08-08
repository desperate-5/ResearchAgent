import { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "react-router-dom";
import { streamChat, resumeChat, getHistory, listProjects, getSources } from "../api/client";
import type { SSEEvent, PlanOption, Project } from "../api/client";
import ToolCallCard from "../components/ToolCallCard";
import PlanCard from "../components/PlanCard";
import FeedbackButtons from "../components/FeedbackButtons";
import FileUpload from "../components/FileUpload";
import ReportExport from "../components/ReportExport";
import SourcesPanel from "../components/SourcesPanel";
import type { SourceWithIndex } from "../components/SourcesPanel";
import { renderMarkdown } from "../utils/markdown";

interface ToolCallEvent {
  tool: string;
  status: "start" | "end";
  id: number;
  agent?: string;
}

const AGENT_LABELS: Record<string, string> = {
  supervisor: "分析调度",
  researcher: "检索文献",
  planner: "方案设计",
  reviewer: "评审结果",
  generate_response: "生成回答",
};

const TOOL_OPTIONS = [
  { id: "web_search", label: "网络搜索" },
  { id: "aminer_search_papers", label: "学术论文" },
  { id: "search_uploaded_docs", label: "上传文档" },
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
  const [sources, setSources] = useState<SourceWithIndex[]>([]);
  const [highlightedSourceNum, setHighlightedSourceNum] = useState<number | undefined>();
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [rightPanelWidth, setRightPanelWidth] = useState(320);
  const [isDragging, setIsDragging] = useState(false);
  const [planOptions, setPlanOptions] = useState<PlanOption[] | null>(null);
  const [, setPlanMessageIndex] = useState(0);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false);
  const messageIndexRef = useRef(0);

  // load project name and history
  useEffect(() => {
    if (!projectId) return;
    setLoadingHistory(true);
    setMessages([]);
    setSources([]);

    Promise.all([
      listProjects(),
      getHistory(projectId),
      getSources(projectId),
    ]).then(([projects, history, savedSources]) => {
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
      messageIndexRef.current = msgs.length;

      // 恢复持久化的来源
      if (savedSources && savedSources.length > 0) {
        setSources(savedSources as SourceWithIndex[]);
      }

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

  // citation marker click handler (delegated)
  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    const handler = (e: MouseEvent) => {
      const target = (e.target as HTMLElement).closest(".citation-marker") as HTMLElement | null;
      if (!target) return;
      const numStr = target.getAttribute("data-source-num");
      if (!numStr) return;
      const num = parseInt(numStr, 10);
      if (!isNaN(num)) {
        setHighlightedSourceNum(num);
      }
    };
    container.addEventListener("click", handler);
    return () => container.removeEventListener("click", handler);
  }, [loadingHistory]);

  // scroll listener for showScrollBtn
  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    const handleScroll = () => {
      if (!container) return;
      const dist = container.scrollHeight - container.scrollTop - container.clientHeight;
      setShowScrollBtn(dist > 200);
    };
    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, [loadingHistory]);

  const scrollToMessage = useCallback((index: number, sourceNum?: number) => {
    if (sourceNum !== undefined) {
      // 查找引用标记：支持精确匹配 [2] 和多项引用 [2,3] 中的任一数字
      const markers = document.querySelectorAll(".citation-marker");
      let found: Element | null = null;
      for (const marker of markers) {
        const nums = (marker.getAttribute("data-source-num") || "")
          .split(",")
          .map((s) => parseInt(s.trim(), 10));
        if (nums.includes(sourceNum)) {
          found = marker;
          break;
        }
      }
      if (found) {
        found.scrollIntoView({ behavior: "smooth", block: "center" });
        const parentMsg = found.closest("[data-msg-index]") as HTMLElement | null;
        if (parentMsg) {
          parentMsg.classList.add("msg-flash");
          setTimeout(() => parentMsg.classList.remove("msg-flash"), 1500);
        }
        return;
      }
    }
    // 回退到消息级别定位
    const el = document.querySelector(`[data-msg-index="${index}"]`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("msg-flash");
      setTimeout(() => el.classList.remove("msg-flash"), 1500);
    }
  }, []);

  // resize divider
  useEffect(() => {
    if (!isDragging) return;
    const handleMouseMove = (e: MouseEvent) => {
      const w = window.innerWidth - e.clientX;
      setRightPanelWidth(Math.max(200, Math.min(600, w)));
    };
    const handleMouseUp = () => setIsDragging(false);
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [isDragging]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || sendingRef.current || !projectId) return;
    sendingRef.current = true;

    const userMsg: ChatMessage = { role: "user", content: input, toolCalls: [] };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStreaming(true);
    // 每轮清除旧的来源和方案弹窗
    setSources([]);
    setHighlightedSourceNum(undefined);
    setPlanOptions(null);

    const assistantMsg: ChatMessage = { role: "assistant", content: "", toolCalls: [] };
    setMessages((prev) => [...prev, assistantMsg]);

    const abort = new AbortController();
    abortRef.current = abort;

    const seenTools = new Set<string>();
    let toolIdCounter = 0;
    const currentMsgIndex = messageIndexRef.current;

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
              if (!seenTools.has(event.tool)) {
                seenTools.add(event.tool);
                const tc: ToolCallEvent = { tool: event.tool, status: "start", id: ++toolIdCounter, agent: event.agent };
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
              }
            } else {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role === "assistant") {
                  updated[updated.length - 1] = {
                    ...last,
                    toolCalls: last.toolCalls.map((tc) =>
                      tc.tool === event.tool ? { ...tc, status: "end" as const } : tc
                    ),
                  };
                }
                return updated;
              });
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
          } else if (event.type === "source") {
            const mi = event.message_index;
            setSources((prev) => {
              const seen = new Set(prev.map((s) => s.id || `${s.url}__${s.title}`));
              const newSources = event.sources
                .filter((s) => !seen.has(s.id || `${s.url}__${s.title}`))
                .map((s) => ({ ...s, message_index: mi }));
              return newSources.length > 0 ? [...prev, ...newSources] : prev;
            });
          } else if (event.type === "source_ratings") {
            setSources((prev) =>
              prev.map((s) => {
                const rating = event.ratings.find(
                  (r) => r.source_number === s.source_number,
                );
                return rating ? { ...s, credibility: rating.credibility } : s;
              }),
            );
          } else if (event.type === "plan_options") {
            setPlanOptions(event.options);
            setPlanMessageIndex(event.message_index);
            // planner 已暂停等待用户选择，清除其运行状态
            setActiveAgents((prev) => {
              const next = new Set(prev);
              next.delete("planner");
              return next;
            });
          } else if (event.type === "done") {
            setActiveAgents(new Set());
            setStreaming(false);
            // 异步任务（如记忆压缩）可能在 done 后继续，但 UI 已完成
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
      sendingRef.current = false;
      abortRef.current = null;
      messageIndexRef.current = currentMsgIndex + 1;
    }
  }, [input, projectId, selectedTools]);

  const handlePlanSelect = useCallback(async (chosenPlanId: string, customPlanText: string) => {
    if (!projectId || sendingRef.current) return;
    sendingRef.current = true;

    // 清除弹窗，进入等待状态
    setPlanOptions(null);
    setStreaming(true);

    const abort = new AbortController();
    abortRef.current = abort;

    const seenTools = new Set<string>();
    let toolIdCounter = 0;
    const currentMsgIndex = messageIndexRef.current;

    try {
      await resumeChat(
        projectId,
        chosenPlanId,
        customPlanText,
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
              if (!seenTools.has(event.tool)) {
                seenTools.add(event.tool);
                const tc: ToolCallEvent = { tool: event.tool, status: "start", id: ++toolIdCounter, agent: event.agent };
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
              }
            } else {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role === "assistant") {
                  updated[updated.length - 1] = {
                    ...last,
                    toolCalls: last.toolCalls.map((tc) =>
                      tc.tool === event.tool ? { ...tc, status: "end" as const } : tc
                    ),
                  };
                }
                return updated;
              });
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
          } else if (event.type === "source") {
            const mi = event.message_index;
            setSources((prev) => {
              const seen = new Set(prev.map((s) => s.id || `${s.url}__${s.title}`));
              const newSources = event.sources
                .filter((s) => !seen.has(s.id || `${s.url}__${s.title}`))
                .map((s) => ({ ...s, message_index: mi }));
              return newSources.length > 0 ? [...prev, ...newSources] : prev;
            });
          } else if (event.type === "source_ratings") {
            setSources((prev) =>
              prev.map((s) => {
                const rating = event.ratings.find(
                  (r) => r.source_number === s.source_number,
                );
                return rating ? { ...s, credibility: rating.credibility } : s;
              }),
            );
          } else if (event.type === "plan_options") {
            // 防御：如果 resume 后又触发了 planner interrupt
            setPlanOptions(event.options);
            setPlanMessageIndex(event.message_index);
            setActiveAgents((prev) => {
              const next = new Set(prev);
              next.delete("planner");
              return next;
            });
          } else if (event.type === "done") {
            setActiveAgents(new Set());
            setStreaming(false);
          }
        },
        abort.signal,
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
      sendingRef.current = false;
      abortRef.current = null;
      messageIndexRef.current = currentMsgIndex + 1;
    }
  }, [projectId]);

  const handleStop = () => {
    abortRef.current?.abort();
    sendingRef.current = false;
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
      <div className="chat-area">
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
              <div key={i} className={`message ${msg.role}`} data-msg-index={i}>
                <div className="role">{msg.role === "user" ? "你" : "助手"}</div>

                {msg.toolCalls.map((tc) => (
                  <ToolCallCard key={tc.id} tool={tc.tool} status={tc.status} agent={tc.agent} />
                ))}

                {msg.content && (
                  <div
                    className="bubble"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                  />
                )}

                {msg.role === "assistant" && msg.content && !streaming && (
                  <FeedbackButtons />
                )}
              </div>
            ))}
            <div ref={messagesEnd} />

            {planOptions && (
              <PlanCard
                options={planOptions}
                onSelect={handlePlanSelect}
                disabled={false}
              />
            )}
          </div>

          {showScrollBtn && (
            <button
              className="scroll-to-bottom"
              onClick={() => scrollToBottom(true)}
              title="回到底部"
            >
              ↓
            </button>
          )}

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

          {streaming && activeAgents.size === 0 && (
            <div className="agent-status">
              <span className="agent-badge">
                <span className="dot running" />
                正在生成回答...
              </span>
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

      <div
        className="resize-divider"
        onMouseDown={(e) => { e.preventDefault(); setIsDragging(true); }}
      />

      <div className="sources-wrapper" style={{ width: rightPanelWidth }}>
        <SourcesPanel
          sources={sources}
          onJumpToMessage={scrollToMessage}
          highlightIndex={highlightedSourceNum}
        />
      </div>
    </div>
  );
}
