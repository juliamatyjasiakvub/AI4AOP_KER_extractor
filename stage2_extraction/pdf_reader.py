from __future__ import annotations

"""Extract plain text from an uploaded PDF file object (Streamlit UploadedFile)."""

import io
import re
from typing import Optional


# DOI regex per Crossref's recommendation:
#   prefix  : "10." followed by 4-9 digits
#   slash   : literal "/"
#   suffix  : at least one URL-safe character
# Trailing punctuation (., ), ;, ", ']) is stripped after match.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_DOI_TRAILING = re.compile(r"[\.\),;:\"'\]>]+$")


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


def truncate_to_token_budget(text: str, max_chars: int = 120_000) -> str:
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

# ---------------------------------------------------------------------------
# DOI extraction
# ---------------------------------------------------------------------------

def _clean_doi(candidate: str) -> str:
    """Strip surrounding whitespace and trailing punctuation from a DOI match."""
    doi = candidate.strip()
    # Iteratively strip trailing punctuation that is unlikely to belong to the DOI.
    while True:
        new = _DOI_TRAILING.sub("", doi)
        if new == doi:
            break
        doi = new
    return doi


def find_doi_in_text(text: str) -> Optional[str]:
    """
    Find the first plausible DOI in `text` and return it lowercased, or None.

    Looks for an explicit `doi:` / `DOI:` / `https://doi.org/` prefix first
    (highest precedence — usually the paper's own DOI on page 1) then falls
    back to the first bare 10.xxxx/yyyy match.
    """
    if not text:
        return None

    # 1. Prefixed forms — these almost always point at the paper itself.
    prefixed = re.search(
        r"(?:https?://(?:dx\.)?doi\.org/|doi\s*[:=]\s*)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if prefixed:
        return _clean_doi(prefixed.group(1)).lower()

    # 2. Bare DOI anywhere in the text — first match wins (typically page 1).
    bare = _DOI_RE.search(text)
    if bare:
        return _clean_doi(bare.group(0)).lower()

    return None


def extract_doi_from_pdf(uploaded_file) -> Optional[str]:
    """
    Try to extract the DOI from a Streamlit-uploaded PDF.

    Strategy (cheap → expensive):
      1. PDF metadata (`/doi` or any value matching the DOI regex).
      2. Text of the first 3 pages (where the DOI is almost always printed).
      3. Full text fallback.

    Returns the DOI in lowercase, or None if nothing plausible is found.
    Resets the file pointer so callers can read the PDF again afterwards.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Add 'pypdf' to requirements.txt and reinstall."
        ) from exc

    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)  # so subsequent extract_text_from_pdf still works
    except Exception:
        pass

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception:
        return None

    # 1. Metadata
    meta = getattr(reader, "metadata", None) or {}
    for key, value in dict(meta).items():
        if value is None:
            continue
        v = str(value)
        # Some publishers embed the DOI under a /doi key, others bury it elsewhere.
        if "doi" in str(key).lower() and v.lower().startswith("10."):
            return _clean_doi(v).lower()
        match = _DOI_RE.search(v)
        if match:
            return _clean_doi(match.group(0)).lower()

    # 2. First few pages
    head_text_parts: list[str] = []
    for page in reader.pages[:3]:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            head_text_parts.append(t)
    head_text = "\n".join(head_text_parts)
    doi = find_doi_in_text(head_text)
    if doi:
        return doi

    # 3. Full text fallback
    full_parts: list[str] = []
    for page in reader.pages[3:]:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            full_parts.append(t)
    return find_doi_in_text("\n".join(full_parts))