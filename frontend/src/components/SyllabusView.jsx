import { useState } from "react";
import { api } from "../api.js";

export default function SyllabusView({ subject }) {
  const [topic, setTopic] = useState(null);
  const [passages, setPassages] = useState([]);
  const [error, setError] = useState("");

  const units = subject.syllabus ? subject.syllabus.units : [];
  const topics = units.flatMap((u) => u.topics);
  const covered = topics.filter((t) => t.support > 0).length;

  async function showPassages(name) {
    setTopic(name);
    setPassages([]);
    setError("");
    try {
      setPassages(await api.search(subject.id, name));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section>
      <p className="muted">
        {units.length} units and {topics.length} topics were found. {covered} of {topics.length} topics are covered by
        the uploaded material. Select a topic to see the passages it retrieves.
      </p>

      <div className="units">
        {units.map((unit) => (
          <div className="card" key={unit.number}>
            <h3>
              Unit {unit.number}: {unit.title}
            </h3>
            <div className="chips">
              {unit.topics.map((t) => (
                <button
                  key={t.name}
                  className={`chip ${t.support > 0 ? "" : "uncovered"} ${topic === t.name ? "active" : ""}`}
                  title={t.support > 0 ? `${t.support} supporting passages` : "Not found in the uploaded material"}
                  onClick={() => showPassages(t.name)}
                >
                  {t.name}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {topic && (
        <div className="card">
          <h3>Passages retrieved for “{topic}”</h3>
          {error && <p className="notice error">{error}</p>}
          {passages.map((p) => (
            <blockquote key={p.rank} className="passage">
              <p>{p.text}</p>
              <footer>
                {p.source}, page {p.page_start}
                {p.page_end !== p.page_start ? `-${p.page_end}` : ""} · similarity {p.similarity.toFixed(2)}
              </footer>
            </blockquote>
          ))}
        </div>
      )}
    </section>
  );
}
