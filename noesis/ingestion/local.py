"""Local file ingestion.

Supports plain text, markdown, and PDF (via PyPDF2 or unstructured).
"""

import os
import warnings


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_pypdf2(path):
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        warnings.warn(f"PyPDF2 failed for {path}: {exc}")
        return ""


def _read_pdf_unstructured(path):
    try:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(path)
        return "\n".join(str(el) for el in elements)
    except Exception as exc:
        warnings.warn(f"unstructured failed for {path}: {exc}")
        return ""


def ingest_local(path):
    """Extract text from a local file path."""
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return ""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _read_pdf_unstructured(path)
        if not text:
            text = _read_pdf_pypdf2(path)
        return text

    return _read_text(path)
