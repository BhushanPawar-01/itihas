import io
import re

import pdfplumber
from pathlib import Path

from src.storage.file_store import get_raw_format, read_raw, write_processed_text


def extract_text(doc_id: str) -> Path:
    """Extract text based on the raw file format and write it to processed storage."""
    extension = get_raw_format(doc_id)
    data, _ = read_raw(doc_id)

    if extension == "txt":
        text = data.decode("utf-8", errors="replace")
        text = text.replace('\x00', '')
        text = re.sub(r'\n{3,}', '\n\n', text)
    elif extension == "pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError(f"{doc_id} appears to be image-based PDF — needs OCR, not text extraction")
    elif extension == "md":
        text = data.decode("utf-8", errors="replace")
        # Remove lines starting with #
        text = re.sub(r'(?m)^#+\s+.*$', '', text)
        # Remove ** and __ bold markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        # Remove [] and () link syntax (keep the text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    elif extension in ("jpg", "jpeg", "png"):
        raise ValueError(f"{doc_id} is an image file — needs OCR, not text extraction")
    else:
        raise ValueError(f"Unsupported format: {extension}")

    path = write_processed_text(doc_id, text)
    return path
