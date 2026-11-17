import { useState } from "react";
import { api } from "../api.js";

export default function NewSubject({ onCreated, onCancel }) {
  const [name, setName] = useState("");
  const [syllabusFile, setSyllabusFile] = useState(null);
  const [syllabusText, setSyllabusText] = useState("");
  const [textbooks, setTextbooks] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (!name.trim()) return setError("Give the subject a name.");
    if (!syllabusFile && !syllabusText.trim()) return setError("Upload the syllabus or paste its text.");
    if (!textbooks.length) return setError("Add at least one textbook or notes file.");
    setBusy(true);
    setError("");
    try {
      const subject = await api.createSubject({ name: name.trim(), syllabusFile, syllabusText, textbooks });
      onCreated(subject);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
    return undefined;
  }

  return (
    <form className="card form" onSubmit={submit}>
      <h2>New subject</h2>
      <p className="muted">
        Upload a syllabus and the material it should be examined from. Units and topics are read from the syllabus
        automatically.
      </p>

      <label className="field">
        <span>Subject name</span>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Operating Systems" />
      </label>

      <label className="field">
        <span>Syllabus file (PDF, DOCX or TXT)</span>
        <input type="file" accept=".pdf,.docx,.txt,.md" onChange={(e) => setSyllabusFile(e.target.files[0] || null)} />
      </label>

      <label className="field">
        <span>Or paste the syllabus text</span>
        <textarea
          rows={5}
          value={syllabusText}
          onChange={(e) => setSyllabusText(e.target.value)}
          placeholder={"Unit 1: Processes - Threads - CPU scheduling\nUnit 2: Paging - Segmentation - Virtual memory"}
        />
      </label>

      <label className="field">
        <span>Textbooks or notes (one or more files)</span>
        <input
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.md"
          onChange={(e) => setTextbooks(Array.from(e.target.files))}
        />
      </label>

      {error && <p className="notice error">{error}</p>}

      <div className="actions">
        <button className="btn primary" type="submit" disabled={busy}>
          {busy ? "Uploading…" : "Create and index"}
        </button>
        {onCancel && (
          <button className="btn" type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
