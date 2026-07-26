import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_TABLE_AGENT_API_KEY || "";
const SUPPORTED_EXTENSIONS = [".xls", ".xlsx", ".xlsm", ".xltx", ".xltm"];

function Icon({ name, size = 18 }) {
  const paths = {
    arrow: <path d="m5 12 7-7 7 7M12 5v14" />,
    check: <path d="m5 12 4 4L19 6" />,
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

function fileKey(file) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function isSupported(file) {
  const lowerName = file.name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
}

function YamlBlock({ value }) {
  const lines = value.trim().split("\n");

  return (
    <pre className="yaml-block" aria-label="schema.yaml content">
      {lines.map((line, index) => {
        const keyMatch = line.match(/^(\s*(?:-\s+)?)([^:#][^:]*:)(.*)$/);
        return (
          <span className="yaml-line" key={`${index}-${line}`}>
            <span className="yaml-number">{index + 1}</span>
            <span className="yaml-code">
              {keyMatch ? (
                <>
                  {keyMatch[1]}
                  <span className="yaml-key">{keyMatch[2]}</span>
                  <span className="yaml-value">{keyMatch[3]}</span>
                </>
              ) : (
                line
              )}
            </span>
          </span>
        );
      })}
    </pre>
  );
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
  const [files, setFiles] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [processState, setProcessState] = useState("idle");
  const [qaState, setQaState] = useState("idle");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);
  const chatRef = useRef(null);

  const busy = processState === "running" || qaState === "running";
  const processed = artifacts.length > 0;

  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, processState, qaState]);

  function resetProcessedState() {
    setArtifacts([]);
    setMessages([]);
    setProcessState("idle");
    setQaState("idle");
  }

  function addFiles(incoming) {
    const accepted = Array.from(incoming).filter(isSupported);
    if (!accepted.length) return;

    const known = new Set(files.map(fileKey));
    const newFiles = accepted.filter((file) => !known.has(fileKey(file)));
    if (!newFiles.length) return;

    setFiles((current) => [...current, ...newFiles]);
    resetProcessedState();
  }

  function removeFile(target) {
    setFiles((current) => current.filter((file) => fileKey(file) !== fileKey(target)));
    resetProcessedState();
  }

  async function postJob(payload) {
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    files.forEach((file) => formData.append("files", file));

    const headers = API_KEY ? { "X-API-Key": API_KEY } : undefined;
    let response;
    try {
      response = await fetch(`${API_BASE}/v1/jobs/upload`, {
        method: "POST",
        headers,
        body: formData,
      });
    } catch {
      throw new Error(
        "Cannot reach the TableAgent API. Make sure it is running on port 3636, then restart the web server.",
      );
    }
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join(", ")
        : data.detail;
      throw new Error(detail || `Request failed with status ${response.status}`);
    }

    return data;
  }

  async function processFiles() {
    if (!files.length || busy) return;

    const startedAt = performance.now();
    setProcessState("running");
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "user",
        kind: "request",
        text: `Process ${files.length === 1 ? files[0].name : `${files.length} workbooks`}`,
      },
    ]);

    try {
      const result = await postJob({ stage: "structure" });
      const runtime = performance.now() - startedAt;
      const schemas = result.schema_artifacts || [];
      const structuresBySheet = new Map(
        (result.structures || []).map((structure) => [
          `${structure.workbook}::${structure.sheet}`,
          structure.structure,
        ]),
      );
      const nextArtifacts = (result.retrieval_records || [])
        .map((record) => ({
          ...record,
          structure_yaml:
            record.structure_yaml ||
            structuresBySheet.get(`${record.workbook || record.document_name}::${record.sheet}`) ||
            "",
        }))
        .filter((record) => record.structure_yaml);

      if (!schemas.length) throw new Error("The process stage did not return schema.yaml.");
      if (!nextArtifacts.length) throw new Error("The process stage did not return query artifacts.");

      setArtifacts(nextArtifacts);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "schema",
          schemas,
          runtime,
        },
      ]);
      setProcessState("done");
    } catch (error) {
      setProcessState("error");
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "error",
          text: error.message,
          runtime: performance.now() - startedAt,
        },
      ]);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery || !processed || busy) return;

    const startedAt = performance.now();
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      kind: "request",
      text: normalizedQuery,
    };
    setMessages((current) => [...current, userMessage]);
    setQuery("");
    setQaState("running");

    try {
      const result = await postJob({
        stage: "qa",
        queries: [normalizedQuery],
        artifacts,
        mode: "thinking",
      });
      const predicted = result.answers?.[0]?.answer;
      if (!predicted) throw new Error("The QA stage did not return a predicted answer.");

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "answer",
          text: predicted,
          runtime: performance.now() - startedAt,
        },
      ]);
      setQaState("done");
    } catch (error) {
      setQaState("error");
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "error",
          text: error.message,
          runtime: performance.now() - startedAt,
        },
      ]);
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
          {busy ? "Working" : processed ? "Ready for questions" : "Ready"}
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <span>Workbooks</span>
            <span className="file-count">{files.length}</span>
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
              addFiles(event.dataTransfer.files);
            }}
            type="button"
          >
            <span className="dropzone-icon"><Icon name="upload" size={21} /></span>
            <strong>Drop files here</strong>
            <span>or choose from your computer</span>
          </button>
          <input
            ref={fileInputRef}
            className="sr-only"
            type="file"
            multiple
            accept={SUPPORTED_EXTENSIONS.join(",")}
            onChange={(event) => {
              addFiles(event.target.files);
              event.target.value = "";
            }}
          />

          <div className="file-list">
            {files.map((file) => (
              <div className="file-row" key={fileKey(file)}>
                <span className="file-icon"><Icon name="file" size={17} /></span>
                <span className="file-copy">
                  <strong title={file.name}>{file.name}</strong>
                  <span>{formatBytes(file.size)}</span>
                </span>
                <button
                  aria-label={`Remove ${file.name}`}
                  className="icon-button"
                  disabled={busy}
                  onClick={() => removeFile(file)}
                  type="button"
                >
                  <Icon name="trash" size={16} />
                </button>
              </div>
            ))}
          </div>

          <button
            className="process-button"
            disabled={!files.length || busy}
            onClick={processFiles}
            type="button"
          >
            {processState === "running" ? <span className="spinner" /> : <Icon name="spark" size={18} />}
            {processState === "running" ? "Processing" : processed ? "Process again" : "Process files"}
          </button>

          <div className="pipeline">
            <div className={`pipeline-step ${processState === "done" ? "complete" : ""}`}>
              <span>{processState === "done" ? <Icon name="check" size={14} /> : "1"}</span>
              Process schema
            </div>
            <div className={`pipeline-line ${processed ? "complete" : ""}`} />
            <div className={`pipeline-step ${qaState === "done" ? "complete" : ""}`}>
              <span>{qaState === "done" ? <Icon name="check" size={14} /> : "2"}</span>
              Ask questions
            </div>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-scroll" ref={chatRef}>
            {!messages.length && (
              <div className="empty-state">
                <span className="empty-icon"><Icon name="table" size={30} /></span>
                <h1>Ask your tables.</h1>
                <p>Upload a spreadsheet, process its schema, then query the data.</p>
              </div>
            )}

            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="avatar">
                    {message.role === "assistant" ? <Icon name="spark" size={16} /> : "You"}
                  </div>
                  <div className="message-body">
                    <div className="message-name">
                      {message.role === "assistant" ? "TableAgent" : "You"}
                    </div>

                    {message.kind === "schema" ? (
                      <div className="schema-response">
                        <div className="response-heading">
                          <div>
                            <span className="eyebrow">Process complete</span>
                            <h2>schema.yaml</h2>
                          </div>
                          <Runtime value={message.runtime} />
                        </div>
                        {message.schemas.map((schema, index) => (
                          <div className="schema-card" key={`${schema.workbook}-${index}`}>
                            <div className="schema-toolbar">
                              <span><Icon name="file" size={15} />{schema.workbook}</span>
                              <span>YAML</span>
                            </div>
                            <YamlBlock value={schema.schema || ""} />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className={`bubble ${message.kind === "error" ? "bubble-error" : ""}`}>
                        {message.kind === "answer" ? (
                          <MarkdownMessage>{message.text}</MarkdownMessage>
                        ) : (
                          <p>{message.text}</p>
                        )}
                        {message.runtime !== undefined && <Runtime value={message.runtime} />}
                      </div>
                    )}
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
                aria-label="Ask a question about the processed files"
                disabled={!processed || busy}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder={processed ? "Ask a question about your files..." : "Process files before asking a question"}
                rows="1"
                value={query}
              />
              <button
                aria-label="Send question"
                className="send-button"
                disabled={!query.trim() || !processed || busy}
                type="submit"
              >
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
