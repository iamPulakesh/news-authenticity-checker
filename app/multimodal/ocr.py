import os
import re
import platform
import logging
import warnings
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import numpy as np
from app.config import settings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
logger = logging.getLogger(__name__)

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

USE_GPU       = settings.USE_GPU
MAX_FILE_SIZE = settings.MAX_FILE_SIZE

_easyocr_reader = None

def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Loading EasyOCR model..")
        _easyocr_reader = easyocr.Reader(
            ["en"],
            gpu=USE_GPU,
            verbose=False
        )
        logger.info("EasyOCR model loaded.")
    return _easyocr_reader


def validate_image(image_path: str) -> None:
    file_size = os.path.getsize(image_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"Image too large ({file_size // (1024*1024)} MB). Max allowed: 5 MB.")

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = Path(image_path).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {allowed_extensions}")


def preprocess_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")

    min_width = 1000
    if img.width < min_width:
        scale    = min_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img      = img.resize(new_size, Image.LANCZOS)
        logger.debug("Upscaled image to %s", new_size)

    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)

    return img


def clean_ocr_text(text: str) -> str:
    lines   = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if len(line) < 4:
            continue

        clean_chars = sum(1 for c in line if c.isalnum() or c.isspace())
        ratio       = clean_chars / len(line)
        if ratio < 0.6:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def ocr_with_easyocr(image_path: str) -> str:
    reader = _get_easyocr_reader()
    img    = preprocess_image(image_path)
    img_np = np.array(img)

    results = reader.readtext(
        img_np,
        detail=0,
        paragraph=True
    )

    text = "\n".join(results).strip()
    logger.info("EasyOCR extracted %d characters from %s", len(text), Path(image_path).name)
    return text


def ocr_with_tesseract(image_path: str) -> str:
    img = preprocess_image(image_path)

    custom_config = r"--oem 3 --psm 3"
    text          = pytesseract.image_to_string(img, config=custom_config).strip()

    logger.info("Tesseract extracted %d characters from %s", len(text), Path(image_path).name)
    return text


def merge_ocr_results(easy_text: str, tess_text: str) -> str:

    if not easy_text and not tess_text:
        return ""

    if not easy_text:
        logger.warning("EasyOCR returned empty so falling back to Tesseract.")
        return tess_text

    logger.info(
        "Using EasyOCR result (%d chars). Tesseract had %d chars (ignored).",
        len(easy_text), len(tess_text)
    )
    return easy_text


def extract_text_from_image(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {
            "text": "", "method": "none",
            "char_count": 0, "success": False,
            "error": f"File not found: {image_path}"
        }

    try:
        validate_image(image_path)
    except ValueError as e:
        return {
            "text": "", "method": "none",
            "char_count": 0, "success": False,
            "error": str(e)
        }

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        return {
            "text": "", "method": "none",
            "char_count": 0, "success": False,
            "error": f"Invalid image file: {e}"
        }

    easy_text = ""
    tess_text = ""
    errors    = []

    try:
        easy_text = ocr_with_easyocr(image_path)
    except Exception as e:
        errors.append(f"EasyOCR failed: {e}")
        logger.warning("EasyOCR failed: %s", e)

    try:
        tess_text = ocr_with_tesseract(image_path)
    except Exception as e:
        errors.append(f"Tesseract failed: {e}")
        logger.warning("Tesseract failed: %s", e)

    raw_text   = merge_ocr_results(easy_text, tess_text)
    final_text = clean_ocr_text(raw_text)

    if not final_text:
        return {
            "text": "", "method": "none",
            "char_count": 0, "success": False,
            "error": "; ".join(errors) or "Both OCR engines returned empty text"
        }

    method = "easyocr" if easy_text else "tesseract"

    return {
        "text":       final_text,
        "method":     method,
        "char_count": len(final_text),
        "success":    True,
        "error":      None
    }


def get_image_metadata(image_path: str) -> dict:
    try:
        with Image.open(image_path) as img:
            return {
                "format":    img.format,
                "size":      img.size,
                "mode":      img.mode,
                "file_size": os.path.getsize(image_path)
            }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        logger.info("Usage: python ocr.py <path-to-image>")
        sys.exit(1)

    path = sys.argv[1]
    logger.info("Running OCR on: %s", path)
    logger.info("-" * 50)

    result = extract_text_from_image(path)

    if result["success"]:
        logger.info("Method      : %s", result['method'])
        logger.info("Characters  : %d", result['char_count'])
        logger.info("Extracted Text:")
        logger.info(result["text"])
    else:
        logger.info("Failed: %s", result['error'])