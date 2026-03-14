import logging
from typing import List
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve_relevant_factchecks(
    query:         str,
    top_k:         int  = 5,
    use_mmr:       bool = True,
) -> List[Document]:
    """
    Retrieve the most relevant fact-checks for a given query.

    Args:
        query:      The claim or article text to search for
        top_k:      Number of results to return
        use_mmr:    If True, use MMR for diverse results;
                    if False, use pure cosine similarity

    Returns:
        List of LangChain Documents with metadata
    """
    vectorstore = get_vectorstore()

    if use_mmr:
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=min(20, top_k * 4),
            lambda_mult=0.7,   # 0=max diversity, 1=max relevance
        )
    else:
        docs = vectorstore.similarity_search(query, k=top_k)

    logger.info("Retrieved %d fact-checks for query: '%s...'", len(docs), query[:60])
    return docs


def retrieve_with_scores(
    query:       str,
    top_k:       int = 5,
) -> List[tuple[Document, float]]:
    """
    Retrieve results with their similarity scores (0.0-1.0).
    Useful for debugging and the evaluation notebook.
    """
    vectorstore = get_vectorstore()
    results     = vectorstore.similarity_search_with_score(query, k=top_k)
    logger.info("Top score: %.3f", results[0][1]) if results else logger.info("No results")
    return results


def format_retrieved_context(docs: List[Document]) -> str:
    """
    Format retrieved fact-checks into a clean block
    that gets injected into the LLM's CoT prompt.
    The LLM sees this as grounding evidence.
    """
    if not docs:
        return "No relevant fact-checks found in the database."

    sections = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        verdict = meta.get("verdict", "unknown").upper()
        source  = meta.get("source", "Unknown")
        claim   = meta.get("claim", "")[:200]
        url     = meta.get("url", "")

        section = f"""[Fact-Check #{i}]
Verdict : {verdict}
Source  : {source}
Claim   : {claim}
Evidence: {doc.page_content[:400]}"""
        if url:
            section += f"\nURL     : {url}"

        sections.append(section)

    return "\n\n" + "\n\n".join(sections)


def retrieve_for_claims(
    claims:      List[str],
    top_k_each:  int = 3,
) -> dict:
    """
    Retrieve fact-checks for a list of individual claims.
    Called by the agent after claim extraction.

    Returns:
        {
          "claim_text": {
              "docs": [...],
              "formatted": "..."
          }
        }
    """
    results = {}
    for claim in claims:
        docs    = retrieve_relevant_factchecks(claim, top_k=top_k_each)
        results[claim] = {
            "docs":      docs,
            "formatted": format_retrieved_context(docs)
        }
    return results


def get_retriever(top_k: int = 5):
    """
    Returns a LangChain-compatible retriever object.
    This is what gets passed into the LangChain RAG chain.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           top_k,
            "fetch_k":     top_k * 4,
            "lambda_mult": 0.7,
        }
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "vaccines cause autism"

    logger.info("Querying Pinecone for: '%s'", query)
    logger.info("-" * 60)

    try:
        results = retrieve_with_scores(query, top_k=3)
        for doc, score in results:
            logger.info("Score: %.3f", score)
            logger.info("   Verdict : %s", doc.metadata.get('verdict', '?').upper())
            logger.info("   Source  : %s", doc.metadata.get('source', '?'))
            logger.info("   Content : %s", doc.page_content[:200])
    except Exception as e:
        logger.error("%s", e)