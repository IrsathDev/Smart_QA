import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_BASE_URL, client, stream } from "./api";
import type { Citation, Conversation, ConversationDetail, Document, Message, User } from "./types";

function App() {
  const [user, setUser] = useState<User | null>();
  useEffect(() => { client.me().then(setUser).catch(() => setUser(null)); }, []);
  if (user === undefined) return <div className="center">Loading Smart Q&A...</div>;
  if (!user) return <Auth onAuth={() => client.me().then(setUser)} />;
  return <Workspace user={user} onLogout={() => client.logout().finally(() => setUser(null))} />;
}

function Auth({ onAuth }: { onAuth: () => void }) {
  const [register, setRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    try {
      setError("");
      await (register ? client.register(email, password) : client.login(email, password));
      onAuth();
    } catch (reason) { setError((reason as Error).message); }
  }
  return <main className="auth-page">
    <form className="auth-card" onSubmit={submit}>
      <div className="brand-mark">SQ</div>
      <h1>Smart Q&A</h1>
      <p>Your private AI workspace for conversations and documents.</p>
      <input type="email" placeholder="Email address" value={email} onChange={(e) => setEmail(e.target.value)} required />
      <input type="password" placeholder="Password (8+ characters)" value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} required />
      {error && <div className="error">{error}</div>}
      <button className="primary">{register ? "Create account" : "Sign in"}</button>
      <button type="button" className="link" onClick={() => setRegister(!register)}>
        {register ? "Already registered? Sign in" : "New here? Create an account"}
      </button>
    </form>
  </main>;
}

function Workspace({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [active, setActive] = useState<ConversationDetail | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [branches, setBranches] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [prompt, setPrompt] = useState("");
  const [liveText, setLiveText] = useState("");
  const [liveCitations, setLiveCitations] = useState<Citation[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [library, setLibrary] = useState(false);
  const [dark, setDark] = useState(localStorage.getItem("theme") === "dark");
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController>();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { refreshSidebar(); refreshDocuments(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active, liveText]);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; localStorage.setItem("theme", dark ? "dark" : "light"); }, [dark]);

  async function refreshSidebar(search = query) { setConversations(await client.conversations(search)); }
  async function refreshDocuments() { setDocuments(await client.documents()); }
  async function openConversation(id: string) {
    setActive(await client.conversation(id));
    setBranches(await client.branches(id));
    setLibrary(false);
  }
  async function createConversation() {
    const conversation = await client.createConversation();
    await refreshSidebar("");
    await openConversation(conversation.id);
  }
  async function removeConversation(id: string) {
    if (!confirm("Delete this conversation?")) return;
    await client.removeConversation(id);
    if (active?.id === id) setActive(null);
    await refreshSidebar();
  }
  async function rename(conversation: Conversation) {
    const title = window.prompt("Conversation title", conversation.title)?.trim();
    if (!title) return;
    await client.rename(conversation.id, title);
    await refreshSidebar();
    if (active?.id === conversation.id) setActive({ ...active, title });
  }
  function handleEvent(event: string, data: any) {
    if (event === "message.started") setActive((current) => current ? { ...current, messages: [...current.messages, data.user_message] } : current);
    if (event === "response.delta") setLiveText((current) => current + data.delta);
    if (event === "citation") setLiveCitations((current) => [...current, data]);
    if (event === "response.error") setError(data.error);
  }
  async function runStream(path: string, body: object) {
    if (!active) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true); setLiveText(""); setLiveCitations([]); setError("");
    try {
      await stream(path, body, controller.signal, handleEvent);
      await openConversation(active.id);
      await refreshSidebar();
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") setError((reason as Error).message);
    } finally { setStreaming(false); setLiveText(""); setLiveCitations([]); }
  }
  async function send(event: FormEvent) {
    event.preventDefault();
    const content = prompt.trim();
    if (!active || !content || streaming) return;
    setPrompt("");
    await runStream(`/api/conversations/${active.id}/messages:stream`, { content });
  }
  async function regenerate(message: Message) {
    if (active && !streaming) await runStream(`/api/conversations/${active.id}/messages/${message.id}/regenerate:stream`, {});
  }
  async function edit(message: Message) {
    const content = window.prompt("Edit your message", message.content)?.trim();
    if (active && content && !streaming) await runStream(`/api/conversations/${active.id}/messages/${message.id}/edit:stream`, { content });
  }
  async function upload(files: FileList | null) {
    if (!files?.length) return;
    try { await client.upload(files); await refreshDocuments(); } catch (reason) { setError((reason as Error).message); }
  }
  async function toggleAttachment(documentId: string) {
    if (!active) return;
    const selected = active.document_ids.includes(documentId)
      ? active.document_ids.filter((id) => id !== documentId) : [...active.document_ids, documentId];
    await client.attach(active.id, selected);
    setActive({ ...active, document_ids: selected });
  }
  async function selectBranch(leaf_message_id: string) {
    if (!active) return;
    setActive(await client.selectBranch(active.id, leaf_message_id));
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark small">SQ</span><strong>Smart Q&A</strong></div>
      <button className="new-chat" onClick={createConversation}>+ New chat</button>
      <input className="search" placeholder="Search chats" value={query} onChange={(e) => { setQuery(e.target.value); refreshSidebar(e.target.value); }} />
      <nav className="chat-list">
        {conversations.map((conversation) => <div className={`chat-row ${active?.id === conversation.id ? "selected" : ""}`} key={conversation.id}>
          <button onClick={() => openConversation(conversation.id)}>{conversation.title}</button>
          <span><button title="Rename" onClick={() => rename(conversation)}>E</button><button title="Delete" onClick={() => removeConversation(conversation.id)}>X</button></span>
        </div>)}
      </nav>
      <div className="sidebar-footer">
        <button onClick={() => setDark(!dark)}>{dark ? "Light mode" : "Dark mode"}</button>
        <button onClick={() => setLibrary(true)}>Document library</button>
        <small title={user.email}>{user.email}</small>
        <button onClick={onLogout}>Sign out</button>
      </div>
    </aside>
    <section className="main">
      {library ? <Library documents={documents} upload={upload} remove={async (id) => { await client.removeDocument(id); refreshDocuments(); }} />
        : active ? <>
          <header className="topbar">
            <div><strong>{active.title}</strong><small>{active.document_ids.length} document{active.document_ids.length === 1 ? "" : "s"} attached</small></div>
            <div>
              <button onClick={() => setLibrary(true)}>Documents</button>
              {branches.length > 1 && <select value={active.active_leaf_id ?? ""} onChange={(e) => selectBranch(e.target.value)}>
                {branches.map((branch, index) => <option key={branch.id} value={branch.id}>Branch {index + 1}</option>)}
              </select>}
              <a className="button" href={`${API_BASE_URL}/api/conversations/${active.id}/export`}>Export</a>
            </div>
          </header>
          <div className="messages">
            {active.messages.map((message) => <MessageBubble key={message.id} message={message} onEdit={edit} onRegenerate={regenerate} />)}
            {streaming && <MessageBubble message={{ id: "live", conversation_id: active.id, parent_message_id: null, role: "assistant", content: liveText || "Thinking...", citations: liveCitations, created_at: "" }} />}
            <div ref={bottomRef} />
          </div>
          {error && <div className="error banner">{error}</div>}
          <div className="composer-wrap">
            <div className="attachments">
              {documents.map((doc) => <label key={doc.id} className={active.document_ids.includes(doc.id) ? "attached" : ""}>
                <input type="checkbox" checked={active.document_ids.includes(doc.id)} disabled={doc.status !== "ready"} onChange={() => toggleAttachment(doc.id)} />
                {doc.name} <small>{doc.status}</small>
              </label>)}
            </div>
            <form className="composer" onSubmit={send}>
              <textarea placeholder="Message Smart Q&A" value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); }
              }} />
              {streaming ? <button type="button" className="stop" onClick={() => abortRef.current?.abort()}>Stop</button> : <button className="primary">Send</button>}
            </form>
          </div>
        </> : <div className="empty"><div className="brand-mark">SQ</div><h1>How can I help?</h1><p>Start a conversation or open your private document library.</p><button className="primary" onClick={createConversation}>Start a new chat</button></div>}
    </section>
  </div>;
}

function MessageBubble({ message, onEdit, onRegenerate }: { message: Message; onEdit?: (message: Message) => void; onRegenerate?: (message: Message) => void }) {
  return <article className={`message ${message.role}`}>
    <div className="avatar">{message.role === "user" ? "You" : "SQ"}</div>
    <div className="message-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
        code: ({ children }) => <code>{children}</code>,
      }}>{message.content}</ReactMarkdown>
      {message.citations.length > 0 && <div className="citations">{message.citations.map((citation) =>
        <details key={citation.chunk_id}><summary>{citation.document_name}{citation.page ? `, page ${citation.page}` : ""}</summary><p>{citation.excerpt}</p></details>)}</div>}
      {message.id !== "live" && <div className="message-actions">
        <button onClick={() => navigator.clipboard.writeText(message.content)}>Copy</button>
        {message.role === "user" && onEdit && <button onClick={() => onEdit(message)}>Edit</button>}
        {message.role === "assistant" && onRegenerate && <button onClick={() => onRegenerate(message)}>Regenerate</button>}
      </div>}
    </div>
  </article>;
}

function Library({ documents, upload, remove }: { documents: Document[]; upload: (files: FileList | null) => void; remove: (id: string) => void }) {
  return <section className="library">
    <header><div><h1>Document library</h1><p>Upload private knowledge and attach ready files from any conversation.</p></div>
      <label className="button primary">Upload files<input type="file" multiple accept=".pdf,.txt,.md,.docx" onChange={(e) => upload(e.target.files)} /></label></header>
    <div className="document-grid">{documents.map((document) => <article className="document-card" key={document.id}>
      <div className="file-icon">{document.name.split(".").pop()?.toUpperCase()}</div>
      <strong>{document.name}</strong>
      <small>{Math.ceil(document.size_bytes / 1024)} KB</small>
      <span className={`status ${document.status}`}>{document.status}</span>
      {document.error && <p className="error">{document.error}</p>}
      <button onClick={() => remove(document.id)}>Delete</button>
    </article>)}</div>
    {documents.length === 0 && <div className="empty subtle">No documents uploaded yet.</div>}
  </section>;
}

export default App;
