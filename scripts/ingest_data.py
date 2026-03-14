import sys
import os
import argparse
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from pathlib import Path
from app.rag.ingest import ingest_fact_checks
from app.rag.vectorstore import get_collection_stats

RAW_DIR     = "./data/raw"
PERSIST_DIR = "./data/vectorstore"


def check_data_exists() -> bool:
    raw_path = Path(RAW_DIR)
    return raw_path.exists() and any(raw_path.glob("*.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument("--redownload", action="store_true",
                        help="Re-download datasets even if they exist")
    args = parser.parse_args()

    if not check_data_exists() or args.redownload:
        logger.info("No data found -- downloading datasets...")
        from scripts.download_data import (
            create_seed_dataset,
            create_indian_seed_dataset,
            download_snopes_rss,
            download_boom_rss,
            download_liar_dataset,
            download_google_factcheck,
            download_fullfact_rss,
            download_health_science_rss,
            download_factcheckorg_rss,
        )
        create_seed_dataset()
        create_indian_seed_dataset()
        for name, func in [
            ("Snopes RSS",              download_snopes_rss),
            ("BOOM Live RSS",           download_boom_rss),
            ("FullFact RSS",            download_fullfact_rss),
            ("Health/Science Feedback", download_health_science_rss),
            ("FactCheck.org RSS",       download_factcheckorg_rss),
            ("LIAR dataset",            download_liar_dataset),
            ("Google FC API",           download_google_factcheck),
        ]:
            try:
                func()
            except Exception as e:
                logger.warning(f"{name} skipped: {e}")
    else:
        logger.info(f"Data found in {RAW_DIR}")

    start = time.time()
    count = ingest_fact_checks(raw_dir=RAW_DIR, persist_dir=PERSIST_DIR)
    elapsed = time.time() - start

    stats = get_collection_stats(PERSIST_DIR)

    logger.info(
        "Ingestion complete in %.1fs -- "
        "%d chunks stored at %s",
        elapsed, stats['total_documents'], stats['persist_directory']
    )