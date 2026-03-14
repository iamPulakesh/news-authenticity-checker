import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings

logger = logging.getLogger(__name__)

_embeddings  = None
_vectorstore = None

EMBEDDING_MODEL  = settings.EMBEDDING_MODEL
VECTORSTORE_PATH = str(settings.VECTORSTORE_DIR)
COLLECTION_NAME  = settings.CHROMA_COLLECTION_NAME


def get_embeddings():
    """
    Load the embedding model once and reuse it.
    all-MiniLM-L6-v2: 80MB, runs on CPU, 384-dimensional vectors.
    Downloads automatically on first run from HuggingFace.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        logger.info("(First run downloads ~80MB -- subsequent runs are instant)")
        model_kwargs = {"device": settings.EMBEDDING_DEVICE}
        if settings.HF_TOKEN:
            model_kwargs["token"] = settings.HF_TOKEN
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embeddings


def get_vectorstore(persist_directory: str = VECTORSTORE_PATH):
    """
    Load an existing ChromaDB vectorstore from disk.
    Raises FileNotFoundError if not ingested yet.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    path = Path(persist_directory)
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"No vectorstore found at '{persist_directory}'.\n"
            "Run: python scripts/ingest_data.py"
        )

    logger.info("Loading ChromaDB from: %s", persist_directory)
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )
    count = _vectorstore._collection.count()
    logger.info("ChromaDB loaded -- %d documents in collection.", count)
    return _vectorstore


def create_vectorstore(persist_directory: str = VECTORSTORE_PATH):
    """
    Create a brand new empty ChromaDB vectorstore.
    Called by ingest.py before adding documents.
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )
    logger.info("Created new ChromaDB at: %s", persist_directory)
    return store


def get_collection_stats(persist_directory: str = VECTORSTORE_PATH) -> dict:
    """Return stats about the vectorstore for display in the UI."""
    try:
        store = get_vectorstore(persist_directory)
        count = store._collection.count()
        return {
            "total_documents": count,
            "persist_directory": persist_directory,
            "embedding_model": EMBEDDING_MODEL,
            "ready": count > 0
        }
    except FileNotFoundError:
        return {
            "total_documents": 0,
            "persist_directory": persist_directory,
            "embedding_model": EMBEDDING_MODEL,
            "ready": False
        }