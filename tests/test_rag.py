import pytest
from app.rag.vectorstore import get_embeddings, get_collection_stats
from app.rag.retriever import (
    retrieve_relevant_factchecks,
    retrieve_with_scores,
    format_retrieved_context,
)


class TestEmbeddings:

    def test_model_loads(self):
        embeddings = get_embeddings()
        assert embeddings is not None

    def test_vector_dimensions(self):
        embeddings = get_embeddings()
        vec = embeddings.embed_query("COVID-19 vaccine safety")
        assert len(vec) == 384

    def test_vector_is_normalized(self):
        embeddings = get_embeddings()
        vec = embeddings.embed_query("test query")
        magnitude = sum(v ** 2 for v in vec) ** 0.5
        assert abs(magnitude - 1.0) < 0.01


class TestVectorstore:

    def test_collection_stats(self):
        stats = get_collection_stats()
        assert "total_documents" in stats
        assert "embedding_model" in stats
        assert "ready" in stats

    @pytest.mark.skipif(
        not get_collection_stats().get("ready", False),
        reason="Pinecone not ingested yet"
    )
    def test_has_documents(self):
        stats = get_collection_stats()
        assert stats["total_documents"] > 0


@pytest.mark.skipif(
    not get_collection_stats().get("ready", False),
    reason="ChromaDB not ingested yet"
)
class TestRetrieval:

    def test_retrieval_returns_results(self):
        docs = retrieve_relevant_factchecks("vaccines cause autism", top_k=3)
        assert len(docs) > 0

    def test_retrieval_with_scores(self):
        results = retrieve_with_scores("COVID vaccine microchip", top_k=3)
        assert len(results) > 0
        doc, score = results[0]
        assert 0.0 <= score <= 1.0

    def test_formatted_context_not_empty(self):
        docs = retrieve_relevant_factchecks("election fraud", top_k=2)
        formatted = format_retrieved_context(docs)
        assert len(formatted) > 0
        assert "Fact-Check #1" in formatted

    def test_empty_docs_formatting(self):
        formatted = format_retrieved_context([])
        assert "No relevant fact-checks" in formatted
