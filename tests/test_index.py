import numpy as np

from qpg.embeddings import HashingEmbedder
from qpg.index import BM25, HybridRetriever, VectorIndex, reciprocal_rank_fusion, tokenize
from qpg.models import Chunk


def test_embeddings_are_normalised_and_deterministic():
    e = HashingEmbedder()
    a, b = e.encode(["sliding window protocol", "sliding window protocol"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    other = e.encode(["volcanic rock formation"])[0]
    assert float(a @ b) > float(a @ other)


def test_vector_index_returns_nearest_first():
    vectors = np.eye(4, dtype=np.float32)
    scores, ids = VectorIndex(vectors).search(np.array([0, 0, 1, 0], dtype=np.float32), k=2)
    assert ids[0] == 2 and scores[0] == 1.0


def test_bm25_prefers_documents_with_rare_query_terms():
    docs = [tokenize(t) for t in [
        "routers forward packets between networks",
        "the checksum detects errors in a frame",
        "packets and frames and packets again",
    ]]
    _, ids = BM25(docs).search(tokenize("checksum errors"), k=3)
    assert ids[0] == 1


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])[:2] == [1, 2]
    assert reciprocal_rank_fusion([[5, 9], [9, 7]])[0] == 9


def test_hybrid_retriever_finds_the_right_chunk():
    texts = [
        "A cyclic redundancy check appends a remainder so the receiver can detect damaged frames.",
        "Distance vector routing shares a table of distances with neighbouring routers.",
        "Optical fibre carries light pulses and is immune to electromagnetic interference.",
    ]
    chunks = [Chunk(id=i, text=t, source="n.txt", page_start=1, page_end=1) for i, t in enumerate(texts)]
    embedder = HashingEmbedder()
    retriever = HybridRetriever(chunks, embedder.encode(texts), embedder)
    hits = retriever.search("distance vector routing", k=2)
    assert hits[0].chunk.id == 1 and hits[0].rank == 1
    assert hits[0].similarity > hits[1].similarity


def test_unit_context_is_used_only_when_the_topic_alone_is_not_confident():
    texts = [
        "Carrier sense multiple access protocols listen to the channel before a station transmits a frame.",
        "The data link layer has several design issues including framing and the services it offers upward.",
        "The network layer has its own design issues such as store and forward packet switching.",
    ]
    chunks = [Chunk(id=i, text=t, source="n.txt", page_start=i + 1, page_end=i + 1) for i, t in enumerate(texts)]
    embedder = HashingEmbedder()
    retriever = HybridRetriever(chunks, embedder.encode(texts), embedder)

    # Vague topic: the unit title decides between the two "design issues" passages.
    vague = retriever.search("design issues", k=1, context_query="design issues (network layer)")
    assert vague[0].chunk.id == 2
    # Confident topic: with the bar set low, the unit title is ignored.
    plain = retriever.search("carrier sense multiple access protocols", k=1,
                             context_query="carrier sense multiple access protocols (network layer design issues)",
                             context_below=0.05)
    assert plain[0].chunk.id == 0
