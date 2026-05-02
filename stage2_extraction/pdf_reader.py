from __future__ import annotations

"""Extract plain text from an uploaded PDF file object (Streamlit UploadedFile)."""

import io
from typing import Optional


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from a PDF uploaded via Streamlit.

    Uses pypdf (pure-Python, no system dependencies). Falls back to a
    helpful error message if the PDF is image-only (scanned without OCR).

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader().

    Returns
    -------
    str
        Concatenated text of all pages, separated by newlines.

    Raises
    ------
    RuntimeError
        If the PDF cannot be read or yields no extractable text.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Add 'pypdf' to requirements.txt and reinstall."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF '{uploaded_file.name}': {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    if not pages:
        raise RuntimeError(
            f"No extractable text found in '{uploaded_file.name}'. "
            "The PDF may be image-only (scanned). Please use a version with embedded text."
        )

    return "\n\n".join(pages)


def truncate_to_token_budget(text: str, max_chars: int = 60_000) -> str:
    """
    Hard-truncate text to a character limit before sending to the API.

    60 000 chars ≈ 15 000 tokens — well within typical local LLM context
    windows while keeping inference time predictable. For very long papers the most important
    mechanistic content is nearly always in the introduction, methods, results,
    and discussion; the reference list is less important.

    We keep the first 80 % and last 20 % of the budget so we capture both the
    introduction/methods and the discussion/conclusion.
    """
    if len(text) <= max_chars:
        return text

    front = int(max_chars * 0.80)
    back = max_chars - front
    return text[:front] + "\n\n[... truncated ...]\n\n" + text[-back:]
