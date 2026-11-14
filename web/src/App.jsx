import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_TABLE_AGENT_API_KEY || "";
const SUPPORTED_EXTENSIONS = [".xlsx", ".xlsm", ".xltx", ".xltm"];

function Icon({ name, size = 18 }) {
  const paths = {
    arrow: <path d="m5 12 7-7 7 7M12 5v14" />,
    file: <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8M8 17h6" />,
    spark: <path d="m12 3 1.4 4.6L18 9l-4.6 1.4L12 15l-1.4-4.6L6 9l4.6-1.4L12 3ZM5 16l.7 2.3L8 19l-2.3.7L5 22l-.7-2.3L2 19l2.3-.7L5 16Z" />,
    table: <path d="M4 5h16v14H4zM4 10h16M9 5v14M15 5v14" />,
    trash: <path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6" />,
    upload: <path d="M12 16V4m0 0L7 9m5-5 5 5M5 15v4h14v-4" />,
  };

  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.8"
    >
      {paths[name]}
    </svg>
  );
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRuntime(milliseconds) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(milliseconds < 10000 ? 1 : 0)} s`;
}

function isSupported(file) {
  const lowerName = file.name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
}

function Runtime({ value }) {
  return <span className="runtime">Runtime {formatRuntime(value)}</span>;
}

function MarkdownMessage({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} rel="noreferrer" target="_blank" />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  function chooseFile(incoming) {
    const accepted = Array.from(incoming).find(isSupported);
    if (!accepted) return;
    setFile(accepted);
    setMessages([]);
  }

  function addMessage(message) {
    setMessages((current) => [...current, { id: `${Date.now()}-${current.length}`, ...message }]);
  }

  async function askQuestion(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text || !file || busy) return;
    setQuestion("");
    addMessage({ role: "user", kind: "text", text });
    setBusy(true);
    const started = performance.now();
    const formData = new FormData();
    formData.append("payload", JSON.stringify({ questions: [text] }));
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/v1/jobs/upload`, {
        method: "POST",
        headers: API_KEY ? { "X-API-Key": API_KEY } : undefined,
        body: formData,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Request failed with HTTP ${response.status}`);
      const answer = body.answers?.[0]?.answer;
      if (!answer) throw new Error("The service did not return an answer.");
      addMessage({ role: "assistant", kind: "answer", text: answer, runtime: performance.now() - started });
    } catch (error) {
      addMessage({ role: "assistant", kind: "error", text: error.message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="TableAgent home">
          <span className="brand-mark"><Icon name="table" size={20} /></span>
          <span>TableAgent</span>
        </a>
        <div className={`status ${busy ? "status-busy" : ""}`}>
          <span className="status-dot" />
          {busy ? "Working" : file ? "Ready for questions" : "Ready"}
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <span>Workbook</span>
            <span className="file-count">{file ? 1 : 0}</span>
          </div>
          <button
            className={`dropzone ${dragging ? "dropzone-active" : ""}`}
            disabled={busy}
            onClick={() => fileInputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files);
            }}
            type="button"
          >
            <span className="dropzone-icon"><Icon name="upload" size={21} /></span>
            <strong>Drop a workbook here</strong>
            <span>or choose one from your computer</span>
          </button>
          <input
            ref={fileInputRef}
            className="sr-only"
            type="file"
            accept={SUPPORTED_EXTENSIONS.join(",")}
            onChange={(event) => {
              chooseFile(event.target.files);
              event.target.value = "";
            }}
          />
          {file && (
            <div className="file-list">
              <div className="file-row">
                <span className="file-icon"><Icon name="file" size={17} /></span>
                <span className="file-copy">
                  <strong title={file.name}>{file.name}</strong>
                  <span>{formatBytes(file.size)}</span>
                </span>
                <button
                  aria-label={`Remove ${file.name}`}
                  className="icon-button"
                  disabled={busy}
                  onClick={() => {
                    setFile(null);
                    setMessages([]);
                  }}
                  type="button"
                >
                  <Icon name="trash" size={16} />
                </button>
              </div>
            </div>
          )}
        </aside>

        <section className="chat-panel">
          <div className="chat-scroll" ref={chatRef}>
            {!messages.length && (
              <div className="empty-state">
                <span className="empty-icon"><Icon name="table" size={30} /></span>
                <h1>Ask your tables.</h1>
                <p>Upload a workbook, then ask a question about its data.</p>
              </div>
            )}
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="avatar">
                    {message.role === "assistant" ? <Icon name="spark" size={16} /> : "You"}
                  </div>
                  <div className="message-body">
                    <div className="message-name">{message.role === "assistant" ? "TableAgent" : "You"}</div>
                    <div className={`bubble ${message.kind === "error" ? "bubble-error" : ""}`}>
                      {message.kind === "answer" ? <MarkdownMessage>{message.text}</MarkdownMessage> : <p>{message.text}</p>}
                      {message.runtime !== undefined && <Runtime value={message.runtime} />}
                    </div>
                  </div>
                </article>
              ))}
              {busy && (
                <article className="message message-assistant">
                  <div className="avatar"><Icon name="spark" size={16} /></div>
                  <div className="message-body">
                    <div className="message-name">TableAgent</div>
                    <div className="thinking"><span /><span /><span /></div>
                  </div>
                </article>
              )}
            </div>
          </div>

          <form className="composer" onSubmit={askQuestion}>
            <div className="composer-box">
              <textarea
                aria-label="Ask a question about the workbook"
                disabled={!file || busy}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder={file ? "Ask a question about your workbook..." : "Upload a workbook before asking a question"}
                rows="1"
                value={question}
              />
              <button aria-label="Send question" className="send-button" disabled={!question.trim() || !file || busy} type="submit">
                <Icon name="arrow" size={18} />
              </button>
            </div>
            <span className="composer-hint">Enter to send · Shift + Enter for a new line</span>
          </form>
        </section>
      </div>
    </main>
  );
}

export default App;
