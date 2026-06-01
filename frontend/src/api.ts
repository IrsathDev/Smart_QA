import type { Conversation, ConversationDetail, Document, Message, User } from "./types";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function cookie(name: string) {
  return document.cookie.split("; ").find((part) => part.startsWith(`${name}=`))?.split("=")[1] ?? "";
}

function endpoint(path: string) {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const csrf = decodeURIComponent(cookie("smartqa_csrf"));
  if (csrf) headers.set("X-CSRF-Token", csrf);
  const response = await fetch(endpoint(path), { ...options, headers, credentials: "include" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || "Request failed");
  }
  return response.status === 204 ? undefined as T : response.json();
}

export const client = {
  me: () => api<User>("/api/auth/me"),
  login: (email: string, password: string) => api("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  register: (email: string, password: string) => api("/api/auth/register", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => api("/api/auth/logout", { method: "POST" }),
  conversations: (query = "") => api<Conversation[]>(query ? `/api/conversations/search?q=${encodeURIComponent(query)}` : "/api/conversations"),
  createConversation: () => api<Conversation>("/api/conversations", { method: "POST", body: JSON.stringify({ title: "New chat" }) }),
  conversation: (id: string) => api<ConversationDetail>(`/api/conversations/${id}`),
  rename: (id: string, title: string) => api<Conversation>(`/api/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  removeConversation: (id: string) => api(`/api/conversations/${id}`, { method: "DELETE" }),
  branches: (id: string) => api<Message[]>(`/api/conversations/${id}/branches`),
  selectBranch: (id: string, leaf_message_id: string) => api<ConversationDetail>(`/api/conversations/${id}/branch`, { method: "POST", body: JSON.stringify({ leaf_message_id }) }),
  edit: (id: string, messageId: string, content: string) => api<Message>(`/api/conversations/${id}/messages/${messageId}/edit`, { method: "POST", body: JSON.stringify({ content }) }),
  documents: () => api<Document[]>("/api/documents"),
  upload: (files: FileList) => {
    const body = new FormData();
    Array.from(files).forEach((file) => body.append("files", file));
    return api<Document[]>("/api/documents", { method: "POST", body });
  },
  removeDocument: (id: string) => api(`/api/documents/${id}`, { method: "DELETE" }),
  attach: (id: string, document_ids: string[]) => api<Document[]>(`/api/conversations/${id}/documents`, { method: "PUT", body: JSON.stringify({ document_ids }) }),
};

export async function stream(
  path: string,
  body: unknown,
  signal: AbortSignal,
  onEvent: (event: string, data: any) => void,
) {
  const response = await fetch(endpoint(path), {
    method: "POST",
    body: JSON.stringify(body),
    signal,
    credentials: "include",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": decodeURIComponent(cookie("smartqa_csrf")) },
  });
  if (!response.ok || !response.body) throw new Error("Unable to start response stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = block.match(/^event: (.+)$/m)?.[1];
      const data = block.match(/^data: (.+)$/m)?.[1];
      if (event && data) onEvent(event, JSON.parse(data));
    }
  }
}
