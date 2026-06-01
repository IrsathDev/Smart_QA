import re
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def parse_document(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type")
    if suffix == ".pdf":
        pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(PdfReader(path).pages)]
    elif suffix == ".docx":
        pages = [(None, "\n".join(paragraph.text for paragraph in DocxDocument(path).paragraphs))]
    else:
        pages = [(None, path.read_text(encoding="utf-8"))]
    chunks: list[tuple[int | None, str]] = []
    for page, text in pages:
        cleaned = re.sub(r"\s+", " ", text).strip()
        for start in range(0, len(cleaned), 900):
            chunk = cleaned[max(0, start - 100):start + 900].strip()
            if chunk:
                chunks.append((page, chunk))
    if not chunks:
        raise ValueError("No extractable text found. Scanned PDFs require OCR before upload.")
    return chunks
