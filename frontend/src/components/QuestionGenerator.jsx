import { useState } from "react";
import { api } from "../api.js";
import QuestionCard from "./QuestionCard.jsx";

const MARKS = [2, 4, 5, 8, 10, 16];

export default function QuestionGenerator({ subject }) {
  const units = subject.syllabus.units;
  const [mode, setMode] = useState("unit");
  const [unit, setUnit] = useState("all");
  const [topic, setTopic] = useState("");
  const [marks, setMarks] = useState(2);
  const [count, setCount] = useState(3);
  const [questions, setQuestions] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function generate(event) {
    event.preventDefault();
    const safeCount = Math.min(20, Math.max(1, Number(count) || 1));
    const payload = { marks, count: safeCount };
    if (mode === "topic") {
      if (!topic.trim()) return setError("Type a topic or keyword.");
      payload.topic = topic.trim();
    } else if (unit !== "all") {
      payload.unit = Number(unit);
    }
    setBusy(true);
    setError("");
    try {
      setQuestions(await api.generateQuestions(subject.id, payload));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
    return undefined;
  }

  return (
    <section>
      <form className="card form" onSubmit={generate}>
        <div className="row">
          <label className="radio">
            <input type="radio" checked={mode === "unit"} onChange={() => setMode("unit")} /> By unit
          </label>
          <label className="radio">
            <input type="radio" checked={mode === "topic"} onChange={() => setMode("topic")} /> By topic or keyword
          </label>
        </div>

        {mode === "unit" ? (
          <label className="field">
            <span>Unit</span>
            <select value={unit} onChange={(e) => setUnit(e.target.value)}>
              <option value="all">All units</option>
              {units.map((u) => (
                <option key={u.number} value={u.number}>
                  Unit {u.number}: {u.title}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <label className="field">
            <span>Topic or keyword</span>
            <input
              type="text"
              list="topic-options"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Any phrase from the material"
            />
            <datalist id="topic-options">
              {units.flatMap((u) => u.topics.map((t) => <option key={`${u.number}-${t.name}`} value={t.name} />))}
            </datalist>
          </label>
        )}

        <div className="row">
          <label className="field">
            <span>Marks per question</span>
            <select value={marks} onChange={(e) => setMarks(Number(e.target.value))}>
              {MARKS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Number of questions</span>
            <input type="number" min="1" max="20" value={count} onChange={(e) => setCount(e.target.value)} />
          </label>
        </div>

        <button className="btn primary" type="submit" disabled={busy}>
          {busy ? "Generating…" : "Generate questions"}
        </button>
        {error && <p className="notice error">{error}</p>}
      </form>

      {questions.length > 0 && (
        <div className="card">
          {questions.map((q, i) => (
            <QuestionCard key={`${i}-${q.text}`} question={q} label={`${i + 1}.`} />
          ))}
        </div>
      )}
    </section>
  );
}
