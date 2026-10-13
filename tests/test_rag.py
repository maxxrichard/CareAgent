import pytest

from careagent.data.policies import load_policy_documents
from careagent.rag.chunking import chunk_documents
from careagent.rag.embeddings import HashingEmbedder
from careagent.rag.index import HybridIndex
from careagent.rag.reranker import OverlapReranker


def test_policy_documents_have_front_matter():
    docs = load_policy_documents()
    ids = {d.doc_id for d in docs}
    assert {
        "return_policy",
        "shipping_policy",
        "cancellation_policy",
        "payment_policy",
        "account_policy",
    } <= ids
    assert all(d.title and d.text for d in docs)


def test_chunking_is_section_aware():
    chunks = chunk_documents(load_policy_documents(), chunk_size=400, overlap=50)
    assert len(chunks) > 20
    assert all(c.chunk_id.startswith(c.doc_id + "#") for c in chunks)
    assert all(len(c.text) <= 400 + 80 for c in chunks)  # title prefix adds a little
    sections = {c.section for c in chunks if c.doc_id == "return_policy"}
    assert "Return window" in sections and "Refund timing" in sections


@pytest.mark.parametrize(
    "query, doc, section_kw",
    [
        ("how many days do I have to return an item", "return_policy", "window"),
        ("refund timing paypal", "return_policy", "Refund timing"),
        ("cancel order before dispatch", "cancellation_policy", "Cancelling"),
        ("express delivery cost next day", "shipping_policy", "Delivery options"),
        ("parcel marked delivered but missing", "shipping_policy", "Missing"),
        ("payment declined order held", "payment_policy", "declined"),
        ("customer service opening hours", "account_policy", "Contact hours"),
    ],
)
def test_retrieval_finds_gold_section(retriever, query, doc, section_kw):
    hits = retriever.retrieve(query, top_n=4)
    assert any(h.doc_id == doc and section_kw.lower() in (h.section or "").lower() for h in hits), [
        (h.chunk_id, h.section) for h in hits
    ]


def test_hybrid_index_round_trip(tmp_path):
    chunks = chunk_documents(load_policy_documents())
    emb = HashingEmbedder(dim=256)
    idx = HybridIndex(chunks, emb)
    idx.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, emb)
    assert len(loaded) == len(idx)
    assert [c.chunk_id for c in loaded.search("return window", k=3)] == [
        c.chunk_id for c in idx.search("return window", k=3)
    ]
    other = HashingEmbedder(dim=256)
    other.name = "different-embedder"
    with pytest.raises(ValueError):
        HybridIndex.load(tmp_path, other)


def test_overlap_reranker_prefers_section_title_match():
    from careagent.models import PolicyChunk

    a = PolicyChunk(
        chunk_id="a", doc_id="d", title="t", section="Refund timing", text="Refunds take time."
    )
    b = PolicyChunk(
        chunk_id="b", doc_id="d", title="t", section="Exchanges", text="We do not offer exchanges."
    )
    top = OverlapReranker().rerank("refund timing", [b, a], top_n=1)
    assert top[0].chunk_id == "a"
