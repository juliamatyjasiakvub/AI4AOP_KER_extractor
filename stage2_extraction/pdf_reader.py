from __future__ import annotations

"""
Page- and section-aware reading of uploaded PDF papers.

The extractor no longer sees a paper as one undifferentiated blob of text.
`extract_document()` returns a `PaperDocument` in which every character of the
paper is addressable by page number, section heading and chunk id. That is what
makes evidence-to-KER provenance possible: when the model quotes a sentence, we
can locate that sentence and report the exact page and section it came from.

Public API
----------
    extract_document(uploaded_file, doi=None)  -> PaperDocument   (preferred)
    extract_text_from_pdf(uploaded_file)       -> str             (legacy)
    extract_doi_from_pdf(uploaded_file)        -> Optional[str]
    find_doi_in_text(text)                     -> Optional[str]
    truncate_to_token_budget(text, max_chars)  -> str
    locate_quote(quote, document)              -> Optional[EvidenceSpan-ish dict]
"""

import io
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional, Sequence

from schemas import Chunk, PageText, PaperDocument

# DOI regex per Crossref's recommendation:
#   prefix  : "10." followed by 4-9 digits
#   slash   : literal "/"
#   suffix  : at least one URL-safe character
# Trailing punctuation (., ), ;, ", ']) is stripped after match.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_DOI_TRAILING = re.compile(r"[\.\),;:\"'\]>]+$")


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

#: Canonical section kinds, and the heading words that map onto them. Order
#: matters: the first pattern that matches a heading line wins.
_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract",   re.compile(r"^\s*(?:\d+\.?\s*)?(abstract|summary|graphical abstract)\b", re.I)),
    ("intro",      re.compile(r"^\s*(?:\d+\.?\s*)?(introduction|background)\b", re.I)),
    ("methods",    re.compile(r"^\s*(?:\d+\.?\s*)?((?:materials?\s+and\s+)?methods?|methodology|experimental(?:\s+(?:section|procedures?))?|materials)\b", re.I)),
    ("results",    re.compile(r"^\s*(?:\d+\.?\s*)?(results?(?:\s+and\s+discussion)?|findings)\b", re.I)),
    ("discussion", re.compile(r"^\s*(?:\d+\.?\s*)?(discussion|general discussion)\b", re.I)),
    ("conclusion", re.compile(r"^\s*(?:\d+\.?\s*)?(conclusions?|concluding remarks|perspectives?)\b", re.I)),
    ("references", re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\b", re.I)),
    ("back",       re.compile(r"^\s*(?:\d+\.?\s*)?(acknowledge?ments?|funding|conflicts? of interest|author contributions?|supplementary|abbreviations|data availability|declaration)\b", re.I)),
)

#: A numbered heading like "3.2 Mitochondrial dysfunction".
_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{2,70}$")

#: Maximum characters a line can have and still plausibly be a heading.
_MAX_HEADING_CHARS = 90


def _classify_heading(line: str) -> Optional[tuple[str, str]]:
    """
    Decide whether `line` is a section heading.

    Returns (section_kind, cleaned_heading_text) or None.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None
    # Headings very rarely end in a sentence-terminating period followed by
    # nothing else, and never end in a comma or semicolon.
    if stripped.endswith((",", ";", ":")) and len(stripped) > 40:
        return None

    for kind, pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return kind, stripped

    # Numbered sub-headings ("3.2 Oxidative stress markers") are useful even
    # when we cannot map them onto a canonical IMRaD kind.
    if _NUMBERED_HEADING.match(stripped):
        return "other", stripped

    # All-caps short lines are typically headings in older typesetting.
    letters = [c for c in stripped if c.isalpha()]
    if 3 <= len(letters) <= 60 and letters and all(c.isupper() for c in letters):
        return "other", stripped.title()

    return None


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_table_simple(table: list[list]) -> str:
    """Convert a table (list of lists) to simple pipe-delimited text format."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [strip_control_chars(str(cell or "")).strip() for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Core document extraction
# ---------------------------------------------------------------------------

#: Control characters that survive PDF text extraction and break downstream
#: consumers. The one that actually matters is NUL: badly encoded PDFs yield
#: it, it travels through the model and into the database unnoticed, and then
#: `DataFrame.to_csv` dies on it with "need to escape, but no escapechar set"
#: — at export time, long after the run that produced it. Tab, newline and
#: carriage return are kept; everything else below 0x20, plus DEL, is noise
#: that no quotation needs.
_CONTROL_CHAR_MAP = {
    code: None for code in range(0x20) if code not in (0x09, 0x0A, 0x0D)
}
_CONTROL_CHAR_MAP[0x7F] = None


def strip_control_chars(value):
    """
    Remove control characters from `value`, passing non-strings through.

    Applied where text enters the pipeline and again where it leaves, because
    rows extracted before this existed are already carrying them.
    """
    if not isinstance(value, str):
        return value
    return value.translate(_CONTROL_CHAR_MAP)


def _read_pages(uploaded_file) -> list[PageText]:
    """Extract per-page text (with tables inlined) from a Streamlit upload."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pdfplumber is not installed. Add 'pdfplumber>=0.10.0' to "
            "requirements.txt and reinstall."
        ) from exc

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    name = getattr(uploaded_file, "name", "uploaded.pdf")
    try:
        pdf_bytes = io.BytesIO(uploaded_file.read())
        pdf = pdfplumber.open(pdf_bytes)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF '{name}': {exc}") from exc

    pages: list[PageText] = []
    try:
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                text = strip_control_chars(page.extract_text() or "").strip()
            except Exception:
                text = ""

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            if tables:
                for table in tables:
                    table_text = _format_table_simple(table)
                    if not table_text.strip():
                        continue
                    block = f"[TABLE]\n{table_text}\n[/TABLE]"
                    text = f"{text}\n\n{block}" if text else block

            if text:
                pages.append(PageText(page_number=page_index, text=text))
    finally:
        pdf.close()

    if not pages:
        raise RuntimeError(
            f"No extractable text found in '{name}'. The PDF may be image-only "
            "(scanned). Please use a version with embedded text."
        )
    return pages


def _guess_title(pages: list[PageText]) -> Optional[str]:
    """Best-effort paper title: the first substantial line of page 1."""
    if not pages:
        return None
    for line in pages[0].text.split("\n")[:12]:
        candidate = line.strip()
        if len(candidate) < 20 or len(candidate) > 250:
            continue
        if _DOI_RE.search(candidate):
            continue
        if candidate.lower().startswith(("doi", "http", "www", "received", "accepted")):
            continue
        return candidate
    return None


def _build_full_text(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Concatenate pages into one string and return a page offset map.

    The map is a list of (page_number, char_start, char_end) so any character
    offset in `full_text` can be resolved back to the page it was printed on.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        offsets.append((page.page_number, start, cursor))
        parts.append("\n\n")
        cursor += 2
    return "".join(parts), offsets


def _page_for_offset(offset: int, offsets: list[tuple[int, int, int]]) -> int:
    """Resolve a character offset in the full text back to a page number."""
    for page_number, start, end in offsets:
        if start <= offset < end:
            return page_number
    return offsets[-1][0] if offsets else 1


def _split_paragraphs(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Split a block into paragraphs, returning (paragraph, absolute_offset)."""
    out: list[tuple[str, int]] = []
    cursor = 0
    for raw in re.split(r"\n\s*\n", text):
        idx = text.find(raw, cursor)
        if idx == -1:
            idx = cursor
        cursor = idx + len(raw)
        if raw.strip():
            out.append((raw, base_offset + idx))
    return out


def build_chunks(
    full_text: str,
    page_offsets: list[tuple[int, int, int]],
    *,
    target_chars: int = 2800,
    min_chars: int = 400,
    drop_back_matter: bool = True,
) -> list[Chunk]:
    """
    Split the full document text into section-aware, page-tagged chunks.

    Chunking respects section boundaries first and paragraph boundaries second,
    so a chunk never straddles two sections and rarely cuts a sentence in half.
    Reference lists and other back matter are dropped by default because they
    contain no mechanistic content but a great many tokens.
    """
    lines = full_text.split("\n")

    # Walk the lines once, recording where each section starts.
    #   segments: list of (section_name, section_kind, char_start, char_end)
    segments: list[list] = []
    current_name = "Body"
    current_kind = "other"
    current_start = 0
    cursor = 0

    for line in lines:
        line_start = cursor
        cursor += len(line) + 1  # +1 for the newline consumed by split
        heading = _classify_heading(line)
        if heading is None:
            continue
        kind, name = heading
        # Close the previous segment.
        if line_start > current_start:
            segments.append([current_name, current_kind, current_start, line_start])
        current_name, current_kind, current_start = name, kind, line_start

    segments.append([current_name, current_kind, current_start, len(full_text)])

    if drop_back_matter:
        segments = [s for s in segments if s[1] not in ("references", "back")]

    chunks: list[Chunk] = []
    counter = 0

    for name, kind, seg_start, seg_end in segments:
        seg_text = full_text[seg_start:seg_end]
        if not seg_text.strip():
            continue

        buffer: list[str] = []
        buf_start: Optional[int] = None
        buf_len = 0

        def flush() -> None:
            nonlocal buffer, buf_start, buf_len, counter
            if not buffer or buf_start is None:
                return
            body = "\n\n".join(buffer)
            char_start = buf_start
            char_end = buf_start + len(body)
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"c{counter:03d}",
                    text=body,
                    section=name,
                    section_kind=kind,
                    page_start=_page_for_offset(char_start, page_offsets),
                    page_end=_page_for_offset(max(char_start, char_end - 1), page_offsets),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            buffer = []
            buf_start = None
            buf_len = 0

        for paragraph, abs_offset in _split_paragraphs(seg_text, seg_start):
            if buf_start is None:
                buf_start = abs_offset
            buffer.append(paragraph)
            buf_len += len(paragraph)
            if buf_len >= target_chars:
                flush()
        flush()

    # Merge away runt chunks so the scorer is not distracted by fragments.
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.text) < min_chars
            and merged[-1].section == chunk.section
        ):
            prev = merged[-1]
            prev.text = f"{prev.text}\n\n{chunk.text}"
            prev.char_end = chunk.char_end
            prev.page_end = chunk.page_end
        else:
            merged.append(chunk)

    # Re-number so ids stay contiguous after merging.
    for i, chunk in enumerate(merged, start=1):
        chunk.chunk_id = f"c{i:03d}"

    return merged


def extract_document(
    uploaded_file,
    doi: Optional[str] = None,
    *,
    target_chars: int = 2800,
) -> PaperDocument:
    """
    Read an uploaded PDF into a fully addressable `PaperDocument`.

    Parameters
    ----------
    uploaded_file
        The object returned by `st.file_uploader()`.
    doi
        Known DOI for the paper. If omitted, one is auto-detected.
    target_chars
        Approximate size of each chunk.
    """
    pages = _read_pages(uploaded_file)
    full_text, page_offsets = _build_full_text(pages)
    chunks = build_chunks(full_text, page_offsets, target_chars=target_chars)

    if doi is None:
        doi = find_doi_in_text("\n".join(p.text for p in pages[:3])) or find_doi_in_text(full_text)

    return PaperDocument(
        filename=getattr(uploaded_file, "name", "uploaded.pdf"),
        doi=doi,
        full_text=full_text,
        pages=pages,
        chunks=chunks,
        title=_guess_title(pages),
    )


# ---------------------------------------------------------------------------
# Quote localisation — the heart of evidence-to-KER provenance
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalise_for_match(text: str) -> str:
    """Collapse whitespace and ligatures so PDF quirks do not break matching."""
    text = text.replace("­", "")               # soft hyphen
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("‐", "-").replace("‑", "-")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = _WS_RE.sub(" ", text)
    return text.strip().lower()


def _build_offset_index(text: str) -> tuple[str, list[int]]:
    """
    Return (normalised_text, index_map) where index_map[i] is the offset into
    the ORIGINAL text of the i-th character of the normalised text.
    """
    normalised_chars: list[str] = []
    index_map: list[int] = []
    prev_space = True
    replacements = {
        "­": "", "ﬁ": "fi", "ﬂ": "fl",
        "‐": "-", "‑": "-", "–": "-", "—": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
    }
    for i, ch in enumerate(text):
        mapped = replacements.get(ch, ch)
        if mapped == "":
            continue
        if mapped.isspace():
            if prev_space:
                continue
            normalised_chars.append(" ")
            index_map.append(i)
            prev_space = True
            continue
        prev_space = False
        for sub in mapped.lower():
            normalised_chars.append(sub)
            index_map.append(i)
    return "".join(normalised_chars).strip(), index_map


def locate_quote(
    quote: str,
    document: PaperDocument,
    *,
    min_ratio: float = 0.72,
) -> dict:
    """
    Find `quote` inside `document` and report where it came from.

    Tries an exact (whitespace-normalised, case-folded) substring match first.
    If that fails — models often paraphrase slightly or drop a clause — falls
    back to the best fuzzy alignment against individual chunks.

    Returns a dict with keys: verified, match_ratio, chunk_id, section,
    section_kind, page_start, page_end, char_start, char_end. Every value is
    None / False when the quote cannot be located at all.
    """
    blank = {
        "verified": False,
        "match_ratio": 0.0,
        "chunk_id": None,
        "section": None,
        "section_kind": None,
        "page_start": None,
        "page_end": None,
        "char_start": None,
        "char_end": None,
    }
    if not quote or not quote.strip() or not document.full_text:
        return blank

    needle = _normalise_for_match(quote)
    if len(needle) < 15:
        return blank

    haystack, index_map = _build_offset_index(document.full_text)

    # --- Pass 1: exact substring ------------------------------------------
    pos = haystack.find(needle)
    if pos != -1:
        char_start = index_map[pos] if pos < len(index_map) else 0
        end_idx = min(pos + len(needle) - 1, len(index_map) - 1)
        char_end = index_map[end_idx] + 1
        return _describe_span(document, char_start, char_end, ratio=1.0)

    # --- Pass 1b: same text, different spacing ----------------------------
    #
    # Some PDFs lose their word gaps entirely — "cells in other brain regions"
    # arrives as "cellsinotherbrainregionsat". A model reading that writes the
    # quote back with the spaces restored, which is the right thing to do and
    # makes the quote unfindable by substring search. Every quotation in a
    # trial run came back unverified for exactly this reason, which
    # reads as fabrication when it is a text-layer artefact.
    #
    # Comparing with all spaces removed settles it. At fifteen characters or
    # more a spurious match is not a realistic risk, and the ratio is recorded
    # just below 1.0 so a spacing-only match stays distinguishable from a
    # literal one.
    squashed_needle = needle.replace(" ", "")
    if len(squashed_needle) >= 15:
        squashed_hay: list[str] = []
        squashed_map: list[int] = []
        for i, ch in enumerate(haystack):
            if ch != " ":
                squashed_hay.append(ch)
                squashed_map.append(i)
        pos = "".join(squashed_hay).find(squashed_needle)
        if pos != -1:
            start_in_hay = squashed_map[pos]
            end_in_hay = squashed_map[
                min(pos + len(squashed_needle) - 1, len(squashed_map) - 1)
            ]
            char_start = index_map[start_in_hay] if start_in_hay < len(index_map) else 0
            end_idx = min(end_in_hay, len(index_map) - 1)
            char_end = index_map[end_idx] + 1
            return _describe_span(document, char_start, char_end, ratio=0.99)

    # --- Pass 2: fuzzy alignment per chunk --------------------------------
    best_ratio = 0.0
    best_chunk: Optional[Chunk] = None
    best_bounds: Optional[tuple[int, int]] = None

    for chunk in document.chunks:
        chunk_norm = _normalise_for_match(chunk.text)
        if not chunk_norm:
            continue
        matcher = SequenceMatcher(None, needle, chunk_norm, autojunk=False)
        # quick_ratio is cheap; skip chunks that cannot possibly win.
        if matcher.quick_ratio() < best_ratio:
            continue
        blocks = matcher.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        ratio = matched / max(1, len(needle))
        if ratio > best_ratio:
            best_ratio = ratio
            best_chunk = chunk
            spans = [b for b in blocks if b.size > 3]
            if spans:
                lo = min(b.b for b in spans)
                hi = max(b.b + b.size for b in spans)
                # Map chunk-normalised offsets back to document offsets by
                # proportion — good enough to pin the sentence to a page.
                scale = len(chunk.text) / max(1, len(chunk_norm))
                best_bounds = (
                    chunk.char_start + int(lo * scale),
                    chunk.char_start + int(hi * scale),
                )

    if best_chunk is not None and best_ratio >= min_ratio:
        if best_bounds:
            start, end = best_bounds
        else:
            start, end = best_chunk.char_start, best_chunk.char_end
        return _describe_span(document, start, end, ratio=round(best_ratio, 3))

    return blank


def locate_quote_in_chunks(
    quote: str,
    chunks: Sequence[Any],
    *,
    min_ratio: float = 0.72,
) -> dict:
    """
    Find `quote` among already-stored chunks, for when the PDF is long gone.

    `locate_quote` needs a live `PaperDocument`, which only exists during a
    run. A curator entering a claim days later has the paper's text in
    `paper_chunk` and nothing else, so verification has to work from that or
    not happen at all — and "not happen at all" would mean every hand-entered
    quotation is unverified by construction, which says nothing about whether
    the paper contains it.

    `chunks` is any sequence of objects or mappings carrying `text`, and
    optionally `chunk_id`, `section`, `section_kind`, `page_start`, `page_end`.
    Returns the same dict shape as `locate_quote`, reporting the matched
    chunk's own location rather than a document offset.
    """
    blank = {
        "verified": False,
        "match_ratio": 0.0,
        "chunk_id": None,
        "section": None,
        "section_kind": None,
        "page_start": None,
        "page_end": None,
        "char_start": None,
        "char_end": None,
    }
    needle = _normalise_for_match(quote or "")
    if len(needle) < 15 or not chunks:
        return blank

    def field(chunk: Any, name: str) -> Any:
        if isinstance(chunk, dict):
            return chunk.get(name)
        return getattr(chunk, name, None)

    def described(chunk: Any, ratio: float) -> dict:
        return {
            "verified": ratio >= 0.99,
            "match_ratio": round(float(ratio), 3),
            "chunk_id": field(chunk, "chunk_id"),
            "section": field(chunk, "section"),
            "section_kind": field(chunk, "section_kind"),
            "page_start": field(chunk, "page_start"),
            "page_end": field(chunk, "page_end"),
            "char_start": field(chunk, "char_start"),
            "char_end": field(chunk, "char_end"),
        }

    squashed_needle = needle.replace(" ", "")
    best_ratio = 0.0
    best_chunk: Any = None

    for chunk in chunks:
        text = _normalise_for_match(str(field(chunk, "text") or ""))
        if not text:
            continue
        if needle in text:
            return described(chunk, 1.0)
        # Same forgiveness as `locate_quote`: PDFs that lost their word gaps
        # make a correctly-transcribed quote unfindable by substring search.
        if len(squashed_needle) >= 15 and squashed_needle in text.replace(" ", ""):
            return described(chunk, 0.99)

        matcher = SequenceMatcher(None, needle, text, autojunk=False)
        if matcher.quick_ratio() < best_ratio:
            continue
        matched = sum(b.size for b in matcher.get_matching_blocks())
        ratio = matched / max(1, len(needle))
        if ratio > best_ratio:
            best_ratio, best_chunk = ratio, chunk

    if best_chunk is not None and best_ratio >= min_ratio:
        return described(best_chunk, best_ratio)
    return blank


def _describe_span(
    document: PaperDocument,
    char_start: int,
    char_end: int,
    *,
    ratio: float,
) -> dict:
    """Attach chunk / section / page metadata to a character range."""
    char_start = max(0, min(char_start, len(document.full_text)))
    char_end = max(char_start, min(char_end, len(document.full_text)))

    owning: Optional[Chunk] = None
    for chunk in document.chunks:
        if chunk.char_start <= char_start < chunk.char_end:
            owning = chunk
            break
    if owning is None:
        # Quote may sit in dropped back matter; find the nearest chunk.
        candidates = [c for c in document.chunks if c.char_start <= char_start]
        owning = candidates[-1] if candidates else (document.chunks[0] if document.chunks else None)

    page_start = _page_number_for(document, char_start)
    page_end = _page_number_for(document, max(char_start, char_end - 1))

    return {
        "verified": ratio >= 0.99,
        "match_ratio": ratio,
        "chunk_id": owning.chunk_id if owning else None,
        "section": owning.section if owning else None,
        "section_kind": owning.section_kind if owning else None,
        "page_start": page_start,
        "page_end": page_end,
        "char_start": char_start,
        "char_end": char_end,
    }


def _page_number_for(document: PaperDocument, offset: int) -> Optional[int]:
    cursor = 0
    for page in document.pages:
        start = cursor
        cursor += len(page.text) + 2
        if start <= offset < cursor:
            return page.page_number
    return document.pages[-1].page_number if document.pages else None


# ---------------------------------------------------------------------------
# Legacy helpers (kept so existing call sites keep working)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text and tables from a PDF as a single string."""
    pages = _read_pages(uploaded_file)
    return "\n\n".join(p.text for p in pages)


def _strip_references(text: str) -> str:
    """Remove the reference section from academic paper text."""
    if not text:
        return text

    lines = text.split("\n")
    reference_header_pattern = re.compile(
        r"^\s*(references|bibliography|works cited|citations)\s*$", re.IGNORECASE
    )
    citation_pattern = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+")
    doi_url_pattern = re.compile(r"^\s*(https?://|10\.)")

    for i, line in enumerate(lines):
        if reference_header_pattern.match(line):
            return "\n".join(lines[:i]).strip()
        if i > 10 and citation_pattern.match(line):
            citation_count = sum(
                1
                for j in range(i, min(i + 3, len(lines)))
                if citation_pattern.match(lines[j]) or doi_url_pattern.match(lines[j])
            )
            if citation_count >= 1:
                return "\n".join(lines[:i]).strip()
        if i > 10 and doi_url_pattern.match(line):
            url_count = sum(
                1
                for j in range(i, min(i + 3, len(lines)))
                if doi_url_pattern.match(lines[j])
            )
            if url_count >= 1:
                return "\n".join(lines[:i]).strip()
    return text


def truncate_to_token_budget(text: str, max_chars: int = 120_000) -> str:
    """Hard-truncate text to a character limit, keeping the head and the tail."""
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
    while True:
        new = _DOI_TRAILING.sub("", doi)
        if new == doi:
            break
        doi = new
    return doi


def find_doi_in_text(text: str) -> Optional[str]:
    """Find the first plausible DOI in `text` and return it lowercased, or None."""
    if not text:
        return None
    prefixed = re.search(
        r"(?:https?://(?:dx\.)?doi\.org/|doi\s*[:=]\s*)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if prefixed:
        return _clean_doi(prefixed.group(1)).lower()
    bare = _DOI_RE.search(text)
    if bare:
        return _clean_doi(bare.group(0)).lower()
    return None


def extract_doi_from_pdf(uploaded_file) -> Optional[str]:
    """Try to extract the DOI from a Streamlit-uploaded PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pdfplumber is not installed. Add 'pdfplumber>=0.10.0' to "
            "requirements.txt and reinstall."
        ) from exc

    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
    except Exception:
        return None

    try:
        meta = getattr(pdf, "metadata", None) or {}
        for key, value in dict(meta).items():
            if value is None:
                continue
            v = str(value)
            if "doi" in str(key).lower() and v.lower().startswith("10."):
                return _clean_doi(v).lower()
            match = _DOI_RE.search(v)
            if match:
                return _clean_doi(match.group(0)).lower()

        head_parts: list[str] = []
        for page in pdf.pages[:3]:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                head_parts.append(t)
        doi = find_doi_in_text("\n".join(head_parts))
        if doi:
            return doi

        tail_parts: list[str] = []
        for page in pdf.pages[3:]:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                tail_parts.append(t)
        return find_doi_in_text("\n".join(tail_parts))
    finally:
        pdf.close()


__all__ = [
    "extract_document",
    "build_chunks",
    "locate_quote",
    "extract_text_from_pdf",
    "extract_doi_from_pdf",
    "find_doi_in_text",
    "truncate_to_token_budget",
    "_strip_references",
]
