# Walkthrough

A plain-language tour of the code, in the order data flows through it. Read this with the files open.

## 1. Ingestion: `extract.py` → `chunking.py` → `embeddings.py` → `ingest.py`

**Problem.** A language model cannot read a 700-page textbook per question, so the book is cut into small passages that can be searched.

- `extract.py` returns `(page number, text)` pairs. `strip_repeated_lines` removes running headers and footers: a line that appears at the top or bottom of 40% of pages is page furniture.
- `chunking.is_back_matter` drops end-of-chapter exercise lists and bibliographies. Without it, the generator copied the textbook's own exercises as "new" questions.
- `chunking.py` packs whole sentences into chunks of at most 900 characters. The last sentence of each chunk is repeated at the start of the next (overlap), so an idea split across a boundary is complete in at least one chunk. Each chunk remembers its page range, which is how a question can cite pages later.
- `embeddings.py` turns each chunk into a 384-number vector with MiniLM. Texts with similar meaning get vectors that point in similar directions. Vectors are normalised to length 1, so the dot product of two vectors is their cosine similarity.

**Why 900 characters?** Small enough that one chunk is about one idea, large enough to hold a full explanation. It is a setting (`QPG_CHUNK_CHARS`).

## 2. Reading the syllabus: `syllabus.py`

Rules run first: a regular expression finds headers such as `Unit I`, `Module 2:` or `Chapter 3`, and the text under each header is split into topics on dashes, commas, semicolons and bullets. Sections such as "Text Books" stop the collection.

If the rules find fewer than two units or under three topics per unit, the syllabus is sent to Llama 3 with a request for JSON. Each topic it returns is kept only if at least 60% of its words appear in the syllabus text. That check stops the model from inventing topics.

**Why rules first?** They are instant, free, and cannot make things up. The model is a fallback for messy layouts.

## 3. Coverage: `ingest.score_coverage`

Each topic is searched against the material. `support` is the number of retrieved chunks with cosine similarity at or above 0.35. A topic with `support = 0` is not in the book, so the generator skips it rather than letting the model answer from its own memory.

## 4. Retrieval: `index.py`

Two searches run for every query:

- **Dense** (`VectorIndex`): FAISS finds the chunks whose vectors are closest to the query vector. Good at paraphrases.
- **Sparse** (`BM25`): classic keyword ranking. Good at exact terms such as `CRC` or `IPv4` that embeddings can blur.

The unit title is added to the search only when the topic alone finds no confident match (best similarity under 0.5) or is a single word. "Design Issues" needs its unit to land in the right chapter. "CSMA Protocols" does not, and adding the unit dragged it to the chapter introduction.

`reciprocal_rank_fusion` merges them. A chunk scores `1 / (60 + rank)` in each list it appears in, and the scores are added. A chunk ranked well by both searches wins. Fusing ranks avoids having to put two unrelated score scales on a common footing.

## 5. Generation: `generate.py`

1. Retrieve passages for the topic. If none clears the threshold, raise `NoSourceMaterial` before calling the model.
   The best passage is the anchor; the others must start within 8 pages of it, topped up with the text that follows the anchor. Passages from different chapters made questions drift and produced factual mix-ups.
2. Pick a Bloom level from the marks (`bloom.py`). The model chooses whichever verb at that level fits the material.
3. Send the prompt to Ollama over HTTP with `format: "json"`.
4. Clean and validate the reply. Reject it if it is too short, over 50 words, refers to "the passage" or "this unit", contains an answer, or is within 0.88 cosine similarity of a question already produced. On rejection, retry up to three times, telling the model what was wrong.
5. Compute `grounding`: the highest cosine similarity between the question and its source chunks. Below 0.45, the question is retried, because a low score suggests the model wrote from its own memory.

Asking for the same topic again moves the anchor to the next-best passage, so repeat questions draw on a different section.

Every one of these rules came from reading real output and finding a fault. Grounding measures similarity, not correctness: the model can still state something false in a well-grounded question, which is why the output is a draft for a teacher to review.

## 6. Papers: `paper.py`

A `Blueprint` is a list of sections, each with marks, a count, and flags for either/or and sub-parts. `build_paper` walks the sections, rotating through units so every unit is examined, and `TopicPicker` prefers topics not yet used, weighted by `support`. `export_docx` writes the paper and an examiner's key listing each question's topic, Bloom level and source pages.

## 7. Serving: `service.py`, `api.py`, `store.py`

- `Service` loads the embedding model once and caches each subject's index in memory. The first version of this project started a new Python process per request and reloaded the model every time.
- Ingestion and paper generation run in background threads. The client polls for status.
- Papers are downloaded by id. `store._check_id` only accepts lowercase letters, digits and hyphens, so a request can never name a path outside the data folder.
- Uploads are limited by size and file type, and the client's file name is reduced to its base name.

## 8. Tests

`HashingEmbedder` and `FakeLLM` replace the real models. Because the pipeline only depends on two small interfaces (`encode(texts)` and `generate_json(prompt)`), everything else can be tested quickly and deterministically.

## Questions you should be able to answer

- Why hybrid retrieval instead of only embeddings?
- What does reciprocal rank fusion do, and why fuse ranks instead of scores?
- How do you stop the model from writing questions that are not in the textbook?
- What does the grounding score measure, and what does it not measure?
- Why does the syllabus parser use rules before the language model?
- What was the security problem with downloading by file path, and how is it fixed?
- What would you change to support scanned PDFs, or a hundred users at once?
