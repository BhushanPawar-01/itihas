from pathlib import Path

from src.storage.file_store import get_raw_format, read_raw, write_processed_text
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_ocr(doc_id: str) -> Path:
    """
    Run OCR on image or image-based PDF.
    
    This is a placeholder for the full OCR pipeline
    (TrOCR -> Tesseract -> IndicOCR).
    """
    log.info("Starting OCR pipeline for: %s", doc_id)
    extension = get_raw_format(doc_id)
    data, _ = read_raw(doc_id)

    # Placeholder for actual OCR logic
    # In reality, this would use TrOCR, Tesseract, etc., based on language and format
    log.warning("OCR is currently a mock implementation. Returning placeholder text.")
    
    extracted_text = f"[OCR Extracted Text for {doc_id} ({extension})]\nThis is mock OCR output."

    path = write_processed_text(doc_id, extracted_text)
    log.info("Finished OCR pipeline for: %s", doc_id)
    return path
