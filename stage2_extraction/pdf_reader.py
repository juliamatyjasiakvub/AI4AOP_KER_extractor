from __future__ import annotations

"""Extract text and handle preprocessing from uploaded PDF files (Streamlit UploadedFile)."""

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
    Extract all text and tables from a PDF uploaded via Streamlit.

    Uses pdfplumber (pure-Python, excellent layout/table handling).
    Falls back to a helpful error message if the PDF is image-only (scanned without OCR).

    Tables are converted to simple pipe-delimited format:
        col1 | col2 | col3
        val1 | val2 | val3

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader().

    Returns
    -------
    str
        Concatenated text of all pages with embedded tables, separated by newlines.

    Raises
    ------
    RuntimeError
        If the PDF cannot be read or yields no extractable text.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Add 'pdfplumber>=0.10.0' to requirements.txt and reinstall."
        ) from exc

    try:
        pdf_bytes = io.BytesIO(uploaded_file.read())
        pdf = pdfplumber.open(pdf_bytes)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF '{uploaded_file.name}': {exc}") from exc

    pages: list[str] = []
    try:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text() or ""
            if text:
                text = text.strip()
            
            # Extract tables and convert to simple text format
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Convert table (list of lists) to pipe-delimited format
                    table_text = _format_table_simple(table)
                    if text:
                        text = text + "\n\n[TABLE]\n" + table_text + "\n[/TABLE]\n"
                    else:
                        text = "[TABLE]\n" + table_text + "\n[/TABLE]\n"
            
            if text:
                pages.append(text)
    finally:
        pdf.close()

    if not pages:
        raise RuntimeError(
            f"No extractable text found in '{uploaded_file.name}'. "
            "The PDF may be image-only (scanned). Please use a version with embedded text."
        )

    return "\n\n".join(pages)


def _format_table_simple(table: list[list]) -> str:
    """Convert a table (list of lists) to simple pipe-delimited text format.
    
    Parameters
    ----------
    table : list[list]
        Table as nested list (rows of cells).
    
    Returns
    -------
    str
        Pipe-delimited table text, e.g. "col1 | col2\nval1 | val2".
    """
    if not table:
        return ""
    
    rows = []
    for row in table:
        # Convert each cell to string and join with pipes
        cells = [str(cell or "").strip() for cell in row]
        rows.append(" | ".join(cells))
    
    return "\n".join(rows)


def _strip_references(text: str) -> str:
    """Remove reference section from academic paper text.
    
    Detects common reference headers and citation patterns, then strips
    everything from the first match onwards. This conserves tokens by removing
    non-mechanistic content.
    
    Detection strategy (priority order):
    1. Section headers: "References", "Bibliography", "Works Cited", "Citations"
    2. Common citation patterns: lines starting with [1], 1., Author et al., etc.
    3. DOI/URL patterns: lines starting with http://, https://, 10.
    
    Parameters
    ----------
    text : str
        Full paper text (may include references).
    
    Returns
    -------
    str
        Text with references and everything after removed (or original if none detected).
    """
    if not text:
        return text
    
    lines = text.split('\n')
    
    # Pattern 1: Section headers (case-insensitive, at line start)
    reference_header_pattern = re.compile(
        r'^\s*(references|bibliography|works cited|citations)\s*$',
        re.IGNORECASE
    )
    
    # Pattern 2: Citation patterns
    # Numbered citations: [1], [123], 1., 123.
    citation_pattern = re.compile(r'^\s*(\[\d+\]|\d+\.)\s+')
    
    # Pattern 3: DOI/URL patterns
    doi_url_pattern = re.compile(r'^\s*(https?://|10\.)')
    
    for i, line in enumerate(lines):
        # Check for reference header
        if reference_header_pattern.match(line):
            # Return text up to this line
            return '\n'.join(lines[:i]).strip()
        
        # Check for citation pattern (but not on first few lines, allow some false positives there)
        if i > 10 and citation_pattern.match(line):
            # Additional heuristic: if this line + next lines look like citations, strip
            # Count how many of next 3 lines look like citations
            citation_count = sum(
                1 for j in range(i, min(i+3, len(lines)))
                if citation_pattern.match(lines[j]) or doi_url_pattern.match(lines[j])
            )
            if citation_count >= 1:
                # Likely the references section
                return '\n'.join(lines[:i]).strip()
        
        # Check for DOI/URL pattern (at line start, often marks references)
        if i > 10 and doi_url_pattern.match(line):
            # Similar check: if multiple following lines also start with URLs/DOIs
            url_count = sum(
                1 for j in range(i, min(i+3, len(lines)))
                if doi_url_pattern.match(lines[j])
            )
            if url_count >= 1:
                return '\n'.join(lines[:i]).strip()
    
    # No references detected, return original
    return text


def truncate_to_token_budget(text: str, max_chars: int = 120_000) -> str:
    """
    Hard-truncate text to a character limit before sending to the API.

    60 000 chars ≈ 15 000 tokens — well within typical local LLM context
    windows while keeping inference time predictable. For very long papers the most important
    mechanistic content is nearly always in the introduction, methods, results,
    and discussion; references are already removed.

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
    Try to extract the DOI from a Streamlit-uploaded PDF using pdfplumber.

    Strategy (cheap → expensive):
      1. PDF metadata (`/doi` or any value matching the DOI regex).
      2. Text of the first 3 pages (where the DOI is almost always printed).
      3. Full text fallback.

    Returns the DOI in lowercase, or None if nothing plausible is found.
    Resets the file pointer so callers can read the PDF again afterwards.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Add 'pdfplumber>=0.10.0' to requirements.txt and reinstall."
        ) from exc

    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)  # so subsequent extract_text_from_pdf still works
    except Exception:
        pass

    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
    except Exception:
        return None

    try:
        # 1. Metadata
        meta = getattr(pdf, "metadata", None) or {}
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
        for page in pdf.pages[:3]:
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
        for page in pdf.pages[3:]:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                full_parts.append(t)
        return find_doi_in_text("\n".join(full_parts))
    finally:
        pdf.close()