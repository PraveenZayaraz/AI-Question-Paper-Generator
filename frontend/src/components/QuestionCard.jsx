export default function QuestionCard({ question, label }) {
  const pages = question.sources
    .map((s) => `${s.source} p.${s.page_start}${s.page_end !== s.page_start ? `-${s.page_end}` : ""}`)
    .join("; ");
  return (
    <div className="question">
      <p className="question-text">
        {label && <strong>{label} </strong>}
        {question.text} <span className="marks">[{question.marks}]</span>
      </p>
      <p className="question-meta">
        {question.unit > 0 ? `Unit ${question.unit} · ` : ""}
        {question.topic} · {question.bloom} · sources: {pages} · grounding {question.grounding.toFixed(2)}
      </p>
    </div>
  );
}
