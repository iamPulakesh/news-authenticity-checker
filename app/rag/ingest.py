import csv
import logging
import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.rag.vectorstore import (
    get_embeddings, ensure_index_exists, PINECONE_INDEX,
)
from app.config import settings

logger = logging.getLogger(__name__)


def load_csv_factchecks(csv_path: str) -> List[Document]:
    """
    Load a CSV of fact-checks into LangChain Documents.
    Each row becomes one Document with rich metadata.

    Supported columns (auto-detected):
      claim, verdict, explanation, source, category, url, description
    """
    docs = []
    path = Path(csv_path)

    if not path.exists():
        logger.warning("CSV not found: %s", csv_path)
        return []

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim       = row.get("claim",       row.get("statement",    row.get("title", "")))
            verdict     = row.get("verdict",      row.get("label",        "unknown"))
            explanation = row.get("explanation",  row.get("description",  row.get("summary", "")))
            source      = row.get("source",       row.get("publisher",    "Unknown"))
            url         = row.get("url",          row.get("news_url",     ""))
            category    = row.get("category",     "general")

            if not claim or len(claim.strip()) < 15:
                continue

            text = f"""FACT-CHECK RECORD
Claim: {claim.strip()}
Verdict: {verdict.upper()}
Explanation: {explanation.strip() if explanation else 'No explanation provided.'}
Source: {source}
Category: {category}"""

            metadata = {
                "claim":    claim.strip()[:500],
                "verdict":  verdict.lower().strip(),
                "source":   source,
                "url":      url,
                "category": category,
                "dataset":  path.stem,
            }

            docs.append(Document(page_content=text, metadata=metadata))

    logger.info("  %s: %d documents", path.name, len(docs))
    return docs


def load_all_datasets(raw_dir: str = "./data/raw") -> List[Document]:
    """Load all CSV files from the raw data directory."""
    raw_path = Path(raw_dir)
    all_docs = []

    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{raw_dir}'.\n"
            "Run: python scripts/download_data.py"
        )

    for csv_file in csv_files:
        docs = load_csv_factchecks(str(csv_file))
        all_docs.extend(docs)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split long documents into smaller chunks for better retrieval.
    chunk_size=500, overlap=50 is optimal for fact-check records.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    logger.info("Chunked %d documents into %d chunks", len(docs), len(chunks))
    return chunks


def deduplicate(docs: List[Document]) -> List[Document]:
    """Remove duplicate claims (same text content)."""
    seen   = set()
    unique = []

    for doc in docs:
        key = doc.page_content[:200].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    removed = len(docs) - len(unique)
    if removed:
        logger.info("Removed %d duplicates", removed)
    return unique


def ingest_fact_checks(
    raw_dir:      str = "./data/raw",
    batch_size:   int = 100,
) -> int:
    """
    Full ingestion pipeline:
      Load CSVs -> Chunk -> Deduplicate -> Embed -> Store in Pinecone

    Args:
        raw_dir:     Directory containing CSV fact-check files
        batch_size:  Number of docs to embed at once (memory management)

    Returns:
        Number of chunks stored in Pinecone
    """
    logger.info("Ingestion pipeline starting...")

    docs = load_all_datasets(raw_dir)
    if not docs:
        raise ValueError("No documents loaded. Check your data directory.")

    chunks = chunk_documents(docs)

    chunks = deduplicate(chunks)

    logger.info("Embedding %d chunks into Pinecone...", len(chunks))

    embeddings = get_embeddings()

    # Ensure Pinecone index exists
    ensure_index_exists()

    # Clear existing vectors before re-ingesting
    logger.info("Clearing existing vectors from Pinecone index...")
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        index.delete(delete_all=True)
        time.sleep(3)  # Wait for deletion to propagate
        logger.info("Existing vectors cleared.")
    except Exception as e:
        logger.warning("Could not clear existing vectors (may be empty): %s", e)

    # Create vectorstore and add documents in batches
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        vectorstore.add_documents(batch)
        logger.info("  Batch %d/%d ingested (%d docs)", batch_num, total_batches, len(batch))
        time.sleep(0.5)  # Small delay to respect rate limits on free tier

    # Wait for indexing to complete and get final count
    time.sleep(3)
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        final_count = stats.total_vector_count
    except Exception:
        final_count = len(chunks)

    logger.info("Ingestion complete -- %d chunks stored in Pinecone", final_count)

    return final_count