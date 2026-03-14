import os
import sys
import logging
from pathlib import Path
from urllib.parse import urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.multimodal.ocr import extract_text_from_image, get_image_metadata
    from app.multimodal.scraper import extract_text_from_url, is_valid_url
else:
    from .ocr import extract_text_from_image, get_image_metadata
    from .scraper import extract_text_from_url, is_valid_url

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def detect_input_type(raw_input: str) -> str:
    """
    Determine if the input is an image path or a URL.

    Returns:
        "image" | "url" | "unknown"
    """
    raw_input = raw_input.strip()

    if is_valid_url(raw_input):
        return "url"

    path = Path(raw_input)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"

    if raw_input.startswith("www.") or raw_input.startswith("http"):
        return "url"

    if len(raw_input) > 30:
        return "text"

    return "unknown"


def process_input(raw_input: str) -> dict:
    """
    Universal entry point -- routes image or URL to the right handler.

    Args:
        raw_input: Either a file path to an image OR a news article URL

    Returns:
        Standardized dict with:
          - input_type (str)    : "image" or "url"
          - text (str)          : extracted article text (ready for LLM)
          - title (str)         : headline (from meta or OCR first line)
          - source (str)        : domain name or filename
          - metadata (dict)     : extra info (author, date, image size, etc.)
          - success (bool)      : whether processing succeeded
          - error (str|None)    : error message if failed
    """
    raw_input  = raw_input.strip()
    input_type = detect_input_type(raw_input)

    logger.info("Input type detected: %s -- '%s'", input_type, raw_input[:80])

    if input_type == "image":
        result  = extract_text_from_image(raw_input)
        imgmeta = get_image_metadata(raw_input)

        if not result["success"]:
            return {
                "input_type": "image", "text": "", "title": "",
                "source": Path(raw_input).name, "metadata": imgmeta,
                "success": False, "error": result["error"]
            }

        lines = [l.strip() for l in result["text"].splitlines() if l.strip()]
        title = lines[0] if lines else "News Image"

        return {
            "input_type": "image",
            "text":       result["text"],
            "title":      title,
            "source":     Path(raw_input).name,
            "metadata": {
                "ocr_method": result["method"],
                "char_count": result["char_count"],
                **imgmeta
            },
            "success": True,
            "error":   None
        }

    elif input_type == "url":
        result = extract_text_from_url(raw_input)

        if not result["success"]:
            return {
                "input_type": "url", "text": "", "title": "",
                "source": raw_input, "metadata": {},
                "success": False, "error": result["error"]
            }

        return {
            "input_type": "url",
            "text":       result["text"],
            "title":      result["title"],
            "source":     result["domain"],
            "metadata": {
                "author":       result["author"],
                "date":         result["date"],
                "summary":      result["summary"],
                "keywords":     result["keywords"],
                "scrape_method": result["method"],
                "char_count":   result["char_count"],
            },
            "success": True,
            "error":   None
        }

    elif input_type == "text":
        text = raw_input.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = lines[0][:120] if lines else "Pasted Text"

        return {
            "input_type": "text",
            "text":       text,
            "title":      title,
            "source":     "user_input",
            "metadata": {
                "char_count":  len(text),
                "input_method": "paste",
            },
            "success": True,
            "error":   None
        }

    else:
        return {
            "input_type": "unknown", "text": "", "title": "",
            "source": raw_input, "metadata": {},
            "success": False,
            "error": (
                f"Could not determine input type for: '{raw_input}'. "
                "Please provide a valid image file path, a news article URL, or paste article text."
            )
        }


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        logger.info("Usage: python router.py <image-path-or-url>")
        sys.exit(1)

    inp = sys.argv[1]
    logger.info("Processing: %s", inp)
    logger.info("-" * 60)

    out = process_input(inp)

    logger.info("Input type : %s", out['input_type'])
    logger.info("Success    : %s", out['success'])
    logger.info("Title      : %s", out['title'])
    logger.info("Source     : %s", out['source'])
    logger.info("Metadata   : %s", json.dumps(out['metadata'], indent=2))
    if out["success"]:
        logger.info("Text preview (first 400 chars): %s", out['text'][:400])
    else:
        logger.info("Error: %s", out['error'])