import io
from pathlib import Path

import pdfplumber
import pytest
from reportlab.pdfgen import canvas

from src.processing.text_extractor import extract_text


@pytest.fixture
def mock_file_store(monkeypatch):
    class MockStore:
        def __init__(self):
            self.files = {}
            self.written = {}
            self.formats = {}

        def add_file(self, doc_id, data, ext):
            self.files[doc_id] = (data, ext)
            self.formats[doc_id] = ext

        def read_raw(self, doc_id):
            return self.files[doc_id]

        def get_raw_format(self, doc_id):
            return self.formats[doc_id]

        def write_processed_text(self, doc_id, text):
            self.written[doc_id] = text
            return Path(f"mock_processed/{doc_id}.txt")

    store = MockStore()
    monkeypatch.setattr("src.processing.text_extractor.read_raw", store.read_raw)
    monkeypatch.setattr("src.processing.text_extractor.get_raw_format", store.get_raw_format)
    monkeypatch.setattr("src.processing.text_extractor.write_processed_text", store.write_processed_text)
    return store


def test_extract_txt(mock_file_store):
    doc_id = "test_txt_001"
    content = b"Line 1\n\n\n\nLine 2\x00"
    mock_file_store.add_file(doc_id, content, "txt")

    path = extract_text(doc_id)
    assert path == Path("mock_processed/test_txt_001.txt")
    assert mock_file_store.written[doc_id] == "Line 1\n\nLine 2"


def test_extract_md(mock_file_store):
    doc_id = "test_md_001"
    content = b"# Header\nSome **bold** and __strong__ text with a [link](http://example.com)."
    mock_file_store.add_file(doc_id, content, "md")

    path = extract_text(doc_id)
    assert mock_file_store.written[doc_id] == "\nSome bold and strong text with a link."


def test_extract_pdf_with_text(mock_file_store):
    doc_id = "test_pdf_001"
    # Create a minimal valid PDF with text using reportlab (in-memory)
    output = io.BytesIO()
    c = canvas.Canvas(output)
    c.drawString(100, 100, "Hello PDF World")
    c.showPage()
    c.save()
    pdf_bytes = output.getvalue()

    mock_file_store.add_file(doc_id, pdf_bytes, "pdf")
    extract_text(doc_id)
    assert "Hello PDF World" in mock_file_store.written[doc_id]


def test_extract_image_jpg(mock_file_store):
    doc_id = "test_jpg_001"
    mock_file_store.add_file(doc_id, b"fake_jpg_data", "jpg")

    with pytest.raises(ValueError) as excinfo:
        extract_text(doc_id)
    assert "needs OCR" in str(excinfo.value)


def test_extract_image_only_pdf(mock_file_store):
    doc_id = "test_pdf_image_001"
    # Create an empty PDF
    output = io.BytesIO()
    c = canvas.Canvas(output)
    c.showPage()  # Empty page
    c.save()
    pdf_bytes = output.getvalue()

    mock_file_store.add_file(doc_id, pdf_bytes, "pdf")
    with pytest.raises(ValueError) as excinfo:
        extract_text(doc_id)
    assert "needs OCR" in str(excinfo.value)
