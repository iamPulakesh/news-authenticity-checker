import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from app.agent.runner import run_fact_check
from app.rag.ingest import ingest_fact_checks
from app.config import settings

logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


load_dotenv()

def cmd_check(args):

    raw_input = args.input
    logger.info("=" * 60)
    logger.info("  News Authenticity Checker")
    logger.info("=" * 60)
    logger.info("  Input: %s", raw_input[:80])
    logger.info("  Running fact-check pipeline...")

    verdict = run_fact_check(raw_input, verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("  %s", verdict.verdict_emoji())
    logger.info("  Confidence : %s", verdict.confidence_bar())
    logger.info("  Title      : %s", verdict.article_title)
    logger.info("  Summary    : %s", verdict.reasoning_summary)

    if verdict.claims_analyzed:
        logger.info("  Claims analyzed (%d):", len(verdict.claims_analyzed))
        for c in verdict.claims_analyzed:
            logger.info("    [%s] %s", c.status, c.claim[:70])
            logger.info("             %s", c.evidence[:80])

    if verdict.sources_consulted:
        logger.info("  Sources: %s", ", ".join(verdict.sources_consulted[:5]))
    logger.info("=" * 60)


def cmd_ui(args):
    ui_path = Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py"

    if not ui_path.exists():
        logger.error("Frontend not found at: %s", ui_path)
        sys.exit(1)

    port = args.port or 8501
    logger.info("Launching Streamlit UI on port %d...", port)
    logger.info("  Open: http://localhost:%d", port)

    os.system(f"streamlit run \"{ui_path}\" --server.port={port}")


def cmd_ingest(args):
    """Run the RAG data ingestion pipeline."""

    raw_dir = str(settings.RAW_DATA_DIR)

    logger.info("Starting RAG ingestion pipeline...")
    count = ingest_fact_checks(raw_dir=raw_dir)
    logger.info("Ingested %d chunks into Pinecone.", count)


def main():
    parser = argparse.ArgumentParser(
        description="News Authenticity Checker -- AI-powered fact-checking agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m app.main check https://apnews.com/article/...\n"
            "  python -m app.main check ./images/news_screenshot.png --verbose\n"
            "  python -m app.main ui --port 8502\n"
            "  python -m app.main ingest\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_check = sub.add_parser("check", help="Fact-check a URL or image")
    p_check.add_argument("input", help="News article URL or image file path")
    p_check.add_argument("-v", "--verbose", action="store_true",
                         help="Enable verbose logging")
    p_check.set_defaults(func=cmd_check)

    p_ui = sub.add_parser("ui", help="Launch Streamlit web interface")
    p_ui.add_argument("--port", type=int, default=8501,
                      help="Port for Streamlit (default: 8501)")
    p_ui.set_defaults(func=cmd_ui)

    p_ingest = sub.add_parser("ingest", help="Run RAG data ingestion")
    p_ingest.set_defaults(func=cmd_ingest)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(message)s")

    args.func(args)


if __name__ == "__main__":
    main()

