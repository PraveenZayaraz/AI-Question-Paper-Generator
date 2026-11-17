// Thin wrapper around fetch: every call returns parsed JSON or throws an
// Error whose message is the server's explanation.
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch {
    throw new Error("Cannot reach the server. Is the backend running on port 8000?");
  }
  if (response.status === 204) return null;
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

const json = (method, payload) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

export const api = {
  health: () => request("/health"),
  listSubjects: () => request("/subjects"),
  getSubject: (id) => request(`/subjects/${id}`),
  deleteSubject: (id) => request(`/subjects/${id}`, { method: "DELETE" }),
  createSubject: ({ name, syllabusFile, syllabusText, textbooks }) => {
    const form = new FormData();
    form.append("name", name);
    if (syllabusFile) form.append("syllabus", syllabusFile);
    if (syllabusText) form.append("syllabus_text", syllabusText);
    textbooks.forEach((file) => form.append("textbooks", file));
    return request("/subjects", { method: "POST", body: form });
  },
  search: (id, q) => request(`/subjects/${id}/search?q=${encodeURIComponent(q)}&k=3`),
  generateQuestions: (id, payload) => request(`/subjects/${id}/questions`, json("POST", payload)),
  getBlueprint: (id) => request(`/subjects/${id}/blueprint`),
  createPaper: (id, blueprint) => request(`/subjects/${id}/papers`, json("POST", { blueprint })),
  getPaper: (id) => request(`/papers/${id}`),
  downloadUrl: (id) => `/api/papers/${id}/download`,
};
