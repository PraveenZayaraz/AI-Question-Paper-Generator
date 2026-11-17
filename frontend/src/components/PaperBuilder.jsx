import { useEffect, useState } from "react";
import { api } from "../api.js";
import QuestionCard from "./QuestionCard.jsx";

export default function PaperBuilder({ subject }) {
  const [blueprint, setBlueprint] = useState(null);
  const [paper, setPaper] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.getBlueprint(subject.id).then(setBlueprint).catch((err) => setError(err.message));
  }, [subject.id]);

  // Poll the paper while the server is still writing it.
  const generating = paper && paper.status === "generating";
  const paperId = paper ? paper.id : null;
  useEffect(() => {
    if (!generating) return undefined;
    const timer = setInterval(async () => {
      try {
        setPaper(await api.getPaper(paperId));
      } catch (err) {
        setError(err.message);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [generating, paperId]);

  if (!blueprint) return error ? <p className="notice error">{error}</p> : <p className="muted">Loading…</p>;

  function updateSection(index, patch) {
    setBlueprint({
      ...blueprint,
      sections: blueprint.sections.map((s, i) => (i === index ? { ...s, ...patch } : s)),
    });
  }

  function addSection() {
    setBlueprint({
      ...blueprint,
      sections: [
        ...blueprint.sections,
        { name: `Part ${String.fromCharCode(65 + blueprint.sections.length)}`, marks: 5, count: 2, either_or: false, split: false, bloom: [] },
      ],
    });
  }

  function removeSection(index) {
    setBlueprint({ ...blueprint, sections: blueprint.sections.filter((_, i) => i !== index) });
  }

  const total = blueprint.sections.reduce((sum, s) => sum + (Number(s.marks) || 0) * (Number(s.count) || 0), 0);

  async function generate() {
    setError("");
    const clean = {
      ...blueprint,
      sections: blueprint.sections.map((s) => ({ ...s, marks: Number(s.marks) || 1, count: Number(s.count) || 1 })),
    };
    try {
      setPaper(await api.createPaper(subject.id, clean));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section>
      <div className="card form">
        <label className="field">
          <span>Paper title</span>
          <input type="text" value={blueprint.title} onChange={(e) => setBlueprint({ ...blueprint, title: e.target.value })} />
        </label>

        <table className="blueprint">
          <thead>
            <tr>
              <th>Section</th>
              <th>Marks each</th>
              <th>Questions</th>
              <th>Either / or</th>
              <th>Allow sub-parts</th>
              <th aria-label="Remove" />
            </tr>
          </thead>
          <tbody>
            {blueprint.sections.map((s, i) => (
              <tr key={i}>
                <td>
                  <input type="text" value={s.name} onChange={(e) => updateSection(i, { name: e.target.value })} />
                </td>
                <td>
                  <input type="number" min="1" max="100" value={s.marks} onChange={(e) => updateSection(i, { marks: e.target.value })} />
                </td>
                <td>
                  <input type="number" min="1" max="50" value={s.count} onChange={(e) => updateSection(i, { count: e.target.value })} />
                </td>
                <td>
                  <input type="checkbox" checked={s.either_or} onChange={(e) => updateSection(i, { either_or: e.target.checked })} />
                </td>
                <td>
                  <input type="checkbox" checked={s.split} onChange={(e) => updateSection(i, { split: e.target.checked })} />
                </td>
                <td>
                  {blueprint.sections.length > 1 && (
                    <button className="btn small" type="button" onClick={() => removeSection(i)}>
                      Remove
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="actions">
          <button className="btn" type="button" onClick={addSection}>
            Add section
          </button>
          <span className="muted">Total: {total} marks</span>
          <button className="btn primary" type="button" onClick={generate} disabled={generating}>
            {generating ? "Generating…" : "Generate paper"}
          </button>
        </div>
        {error && <p className="notice error">{error}</p>}
      </div>

      {paper && paper.status === "generating" && (
        <div className="card">
          <p>
            Writing questions: {paper.progress} of {paper.total}
          </p>
          <progress value={paper.progress} max={paper.total || 1} />
        </div>
      )}

      {paper && paper.status === "failed" && <p className="notice error">Paper failed: {paper.message}</p>}

      {paper && paper.status === "ready" && (
        <div className="card">
          <div className="actions">
            <h3>{paper.blueprint.title}</h3>
            <a className="btn primary" href={api.downloadUrl(paper.id)}>
              Download Word file
            </a>
          </div>
          {paper.sections.map((section) => (
            <div key={section.name}>
              <h4>
                {section.name} ({section.items.length} × {section.marks} marks)
              </h4>
              {section.items.map((item) =>
                item.options.map((option, o) => (
                  <div key={`${item.number}-${o}`}>
                    {o > 0 && <p className="or">(OR)</p>}
                    {option.map((q, i) => (
                      <QuestionCard
                        key={`${item.number}-${o}-${i}`}
                        question={q}
                        label={
                          (i === 0 ? `${item.number}${item.options.length > 1 ? (o === 0 ? "a" : "b") : ""}. ` : "") +
                          (option.length > 1 ? `(${"i".repeat(i + 1)})` : "")
                        }
                      />
                    ))}
                  </div>
                ))
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
