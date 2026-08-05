const BASE = "/api";

// ---- types ----

export interface Project {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface HistoryMessage {
  role: string;
  content: string;
  created_at: string;
}

export interface PreferencesConfig {
  literature: Record<string, unknown>;
  writing: Record<string, unknown>;
  experiment: Record<string, unknown>;
  tool: Record<string, unknown>;
}

// ---- SSE stream event types ----

export interface SourceItem {
  id: string;
  title: string;
  url: string;
  summary: string;
  source_type: "web" | "paper" | "document";
  source_number: number;
  page?: number;
  position?: string;
  chunk_index?: number;
}

export type SSEEvent =
  | { type: "response"; content: string; agent?: string }
  | { type: "tool_call"; tool: string; status: "start" | "end"; agent?: string }
  | { type: "agent_phase"; agent: string; status: "start" | "end" }
  | { type: "source"; sources: SourceItem[]; message_index: number }
  | { type: "plan_options"; options: PlanOption[]; message_index: number }
  | { type: "done" };

export interface PlanOption {
  id: string;
  title: string;
  description: string;
  pros: string[];
  cons: string[];
}

// ---- projects ----

export async function listProjects(): Promise<Project[]> {
  const res = await fetch(`${BASE}/projects`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createProject(name: string): Promise<Project> {
  const res = await fetch(`${BASE}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteProject(id: string): Promise<void> {
  const res = await fetch(`${BASE}/projects/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function renameProject(id: string, name: string): Promise<Project> {
  const res = await fetch(`${BASE}/projects/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getHistory(projectId: string): Promise<HistoryMessage[]> {
  const res = await fetch(`${BASE}/projects/${projectId}/history`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.messages;
}

// ---- chat (SSE streaming) ----

export function streamChat(
  projectId: string,
  message: string,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal,
  tools: string[] = [],
): Promise<void> {
  return fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId, message, tools }),
    signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          const json = trimmed.slice(6);
          if (json) {
            try {
              onEvent(JSON.parse(json));
            } catch {
              // skip unparseable chunks
            }
          }
        }
      }
    }

    // flush remaining buffer
    const remaining = buffer.trim();
    if (remaining.startsWith("data: ")) {
      const json = remaining.slice(6);
      if (json) {
        try {
          onEvent(JSON.parse(json));
        } catch {
          // skip
        }
      }
    }
  });
}

export async function getSources(projectId: string): Promise<SourceItem[]> {
  const res = await fetch(`${BASE}/projects/${projectId}/sources`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.sources;
}

// ---- preferences ----

export async function getPreferences(): Promise<PreferencesConfig> {
  const res = await fetch(`${BASE}/preferences`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updatePreferences(
  prefs: Partial<PreferencesConfig>
): Promise<PreferencesConfig> {
  const res = await fetch(`${BASE}/preferences`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(prefs),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---- raw preferences (settings editor) ----

export async function getRawPreferences(): Promise<string> {
  const res = await fetch(`${BASE}/preferences/raw`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.content;
}

export async function updateRawPreferences(content: string): Promise<unknown> {
  const res = await fetch(`${BASE}/preferences/raw`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---- feedback ----

export async function sendFeedback(
  type: "like" | "dislike",
  tag: string = "",
  comment: string = ""
): Promise<unknown> {
  const res = await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, tag, comment }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---- file upload ----

export async function uploadFile(projectId: string, file: File): Promise<unknown> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/projects/${projectId}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listFiles(projectId: string): Promise<{ filename: string; size: number }[]> {
  const res = await fetch(`${BASE}/projects/${projectId}/files`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteFile(projectId: string, filename: string): Promise<void> {
  const res = await fetch(`${BASE}/projects/${projectId}/files/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

// ---- export ----

export async function exportReport(projectId: string): Promise<string> {
  const res = await fetch(`${BASE}/projects/${projectId}/export`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.report;
}

// ---- plan resume (人机协同) ----

export function resumeChat(
  projectId: string,
  chosenPlanId: string,
  customPlanText: string,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return fetch(`${BASE}/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: projectId,
      chosen_plan_id: chosenPlanId,
      custom_plan_text: customPlanText,
    }),
    signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          const json = trimmed.slice(6);
          if (json) {
            try {
              onEvent(JSON.parse(json));
            } catch {
              // skip unparseable chunks
            }
          }
        }
      }
    }

    const remaining = buffer.trim();
    if (remaining.startsWith("data: ")) {
      const json = remaining.slice(6);
      if (json) {
        try {
          onEvent(JSON.parse(json));
        } catch {
          // skip
        }
      }
    }
  });
}
