import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import NewSubject from "./components/NewSubject.jsx";
import SyllabusView from "./components/SyllabusView.jsx";
import QuestionGenerator from "./components/QuestionGenerator.jsx";
import PaperBuilder from "./components/PaperBuilder.jsx";

const TABS = [
  { id: "syllabus", label: "Syllabus" },
  { id: "questions", label: "Questions" },
  { id: "paper", label: "Full paper" },
];

export default function App() {
  const [subjects, setSubjects] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [adding, setAdding] = useState(false);
  const [tab, setTab] = useState("syllabus");
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const list = await api.listSubjects();
      setSubjects(list);
      setError("");
      return list;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch((err) => setError(err.message));
    refresh().then((list) => {
      if (list.length) setSelectedId(list[0].id);
      else setAdding(true);
    });
  }, [refresh]);

  // While any subject is still being indexed, poll so its status updates.
  const indexing = subjects.some((s) => s.status === "ingesting");
  useEffect(() => {
    if (!indexing) return undefined;
    const timer = setInterval(refresh, 1500);
    return () => clearInterval(timer);
  }, [indexing, refresh]);

  const selected = subjects.find((s) => s.id === selectedId) || null;

  async function handleCreated(subject) {
    setAdding(false);
    setSelectedId(subject.id);
    setTab("syllabus");
    await refresh();
  }

  async function handleDelete(subject) {
    try {
      await api.deleteSubject(subject.id);
      const list = await refresh();
      setSelectedId(list.length ? list[0].id : null);
      if (!list.length) setAdding(true);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">Question Paper Generator</h1>
        <button className="btn primary full" onClick={() => setAdding(true)}>
          New subject
        </button>
        <ul className="subject-list">
          {subjects.map((subject) => (
            <li key={subject.id}>
              <button
                className={`subject ${subject.id === selectedId && !adding ? "active" : ""}`}
                onClick={() => {
                  setSelectedId(subject.id);
                  setAdding(false);
                }}
              >
                <span className="subject-name">{subject.name}</span>
                <span className={`pill ${subject.status}`}>{subject.status}</span>
              </button>
            </li>
          ))}
        </ul>
        {health && (
          <p className={`health ${health.llm_available ? "ok" : "down"}`}>
            {health.llm_available
              ? `Model ${health.llm_model} is ready`
              : `Ollama is not reachable. Start it, then run: ollama pull ${health.llm_model}`}
          </p>
        )}
      </aside>

      <main className="main">
        {error && <p className="notice error">{error}</p>}

        {adding && <NewSubject onCreated={handleCreated} onCancel={subjects.length ? () => setAdding(false) : null} />}

        {!adding && selected && (
          <>
            <header className="subject-header">
              <div>
                <h2>{selected.name}</h2>
                <p className="muted">
                  {selected.sources.join(", ") || "No files yet"}
                  {selected.n_chunks ? ` · ${selected.n_chunks} passages indexed` : ""}
                </p>
              </div>
              <button className="btn danger" onClick={() => handleDelete(selected)}>
                Delete subject
              </button>
            </header>

            {selected.status === "ingesting" && (
              <p className="notice">Indexing: {selected.message || "starting"}…</p>
            )}
            {selected.status === "failed" && (
              <p className="notice error">Indexing failed: {selected.message}</p>
            )}

            {selected.status === "ready" && (
              <>
                <nav className="tabs">
                  {TABS.map((t) => (
                    <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
                      {t.label}
                    </button>
                  ))}
                </nav>
                {tab === "syllabus" && <SyllabusView subject={selected} />}
                {tab === "questions" && <QuestionGenerator key={selected.id} subject={selected} />}
                {tab === "paper" && <PaperBuilder key={selected.id} subject={selected} />}
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
