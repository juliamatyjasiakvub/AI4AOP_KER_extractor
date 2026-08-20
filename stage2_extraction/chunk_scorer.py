from __future__ import annotations

"""
Prioritise mechanistically relevant paper sections before expensive extraction.

A typical toxicology paper is 40 000-90 000 characters, but the passages that
actually describe a causal link between two biological events usually account
for well under a third of that. Sending the whole paper to the model for each
of the ~30 per-paper calls is slow and expensive, and it dilutes the signal.

This module scores every chunk produced by `pdf_reader.build_chunks()` and
selects the most mechanistically informative ones. Two scorers are available:

    score_chunks_heuristic(chunks)          — free, instant, no network
    score_chunks_llm(chunks, cfg)           — one extra LLM call, better recall

`select_chunks()` combines a scorer with a character budget and always keeps
the abstract, because the abstract states the paper's central mechanistic claim
in almost every case.
"""

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Sequence

from schemas import Chunk
from stage2_extraction.llm_providers import LLMConfig, LLMProviderError

# ---------------------------------------------------------------------------
# Heuristic vocabulary
# ---------------------------------------------------------------------------

#: Words that signal a causal or mechanistic assertion. Weighted because
#: "leads to" is far more diagnostic of a KER than "associated".
_CAUSAL_TERMS: dict[str, float] = {
    r"lead(?:s|ing)?\s+to": 3.0,
    r"result(?:s|ed|ing)?\s+in": 2.5,
    r"cause(?:s|d)?\b": 2.5,
    r"induce(?:s|d)?\b": 2.5,
    r"trigger(?:s|ed|ing)?\b": 2.5,
    r"activat(?:e|es|ed|ion)\b": 2.0,
    r"inhibit(?:s|ed|ion)?\b": 2.0,
    r"suppress(?:es|ed|ion)?\b": 2.0,
    r"upregulat|downregulat": 2.0,
    r"mediat(?:e|es|ed|ing)\b": 2.0,
    r"promot(?:e|es|ed|ing)\b": 1.5,
    r"drives?\b": 1.5,
    r"contribut(?:e|es|ed|ing)\s+to": 1.5,
    r"associated\s+with": 1.0,
    r"correlat(?:e|es|ed|ion)\b": 1.0,
    r"dose[-\s]dependent": 2.5,
    r"time[-\s]dependent": 1.5,
    r"downstream\s+of": 2.0,
    r"upstream\s+of": 2.0,
    r"knock(?:out|down)\b": 3.0,
    r"\bsiRNA\b|\bshRNA\b|\bCRISPR\b": 2.5,
    r"antagonist|agonist|blocker|inhibitor": 2.0,
    r"rescue(?:d|s)?\b": 2.0,
    r"abolish(?:ed|es)?\b": 2.0,
    r"attenuat(?:e|es|ed|ion)\b": 2.0,
    r"mechanis(?:m|tic)": 2.5,
    r"pathway": 1.5,
    r"signal(?:l?ing|ling)\b": 1.5,
    r"adverse\s+outcome": 3.0,
    r"key\s+event": 3.0,
    r"mode\s+of\s+action": 3.0,
}

#: Biological entity vocabulary — a chunk that mentions concrete biology is
#: more likely to yield a usable Key Event than one that does not.
_BIO_TERMS: dict[str, float] = {
    r"oxidative\s+stress|reactive\s+oxygen|\bROS\b": 2.0,
    r"apoptosis|necrosis|pyroptosis|ferroptosis|cell\s+death": 2.0,
    r"inflammation|inflammatory|cytokine|chemokine": 2.0,
    r"DNA\s+damage|genotoxic|mutagen|adduct": 2.0,
    r"mitochondri\w+": 2.0,
    r"receptor\b|\bAhR\b|\bPPAR\b|\bER[αβ]?\b|nuclear\s+receptor": 2.0,
    r"transcription|mRNA|gene\s+expression": 1.5,
    r"protein\s+expression|phosphorylat\w+|enzyme\s+activity": 1.5,
    r"steatosis|fibrosis|necroinflammation|hepatotox|nephrotox|neurotox|cardiotox": 2.5,
    r"proliferation|differentiation|migration|senescence": 1.5,
    r"barrier\s+(?:function|integrity)|permeability": 1.5,
    r"lipid\s+peroxidation|glutathione|\bGSH\b|\bMDA\b|\bSOD\b": 2.0,
    r"membrane\s+potential|calcium\s+influx|ion\s+channel": 1.5,
    r"tumou?r|carcinogen|neoplas\w+": 2.0,
    r"reproducti\w+|fecundity|fertility|population\s+decline": 2.0,
    r"behavio\w*r|locomot\w+|cognit\w+": 1.5,
}

#: Statistical / quantitative reporting, which usually accompanies empirical
#: evidence for a relationship.
_QUANT_TERMS: dict[str, float] = {
    r"\bp\s*[<=>]\s*0?\.\d+": 2.0,
    r"\bEC\d{2}\b|\bIC\d{2}\b|\bLD\d{2}\b|\bLOAEL\b|\bNOAEL\b|\bBMD\b": 3.0,
    r"\bfold[-\s]change\b|\d+\s*-\s*fold": 2.0,
    r"\b95\s*%\s*CI\b|confidence\s+interval": 1.5,
    r"\bµM\b|\bμM\b|\bnM\b|\bmM\b|mg\s*/\s*kg|µg\s*/\s*[mL]": 2.0,
    r"\bR\s*2\b|\bR²\b|regression|correlation\s+coefficient": 1.0,
}

#: Prior weight by section. Results and discussion carry the mechanistic
#: argument; methods describe technique; back matter carries none.
_SECTION_PRIOR: dict[str, float] = {
    "abstract": 1.35,
    "intro": 1.05,
    "methods": 0.55,
    "results": 1.30,
    "discussion": 1.25,
    "conclusion": 1.20,
    "other": 1.0,
    "references": 0.0,
    "back": 0.1,
}

#: Text that marks a chunk as almost certainly useless regardless of keywords.
_NOISE_PATTERNS = (
    re.compile(r"^\s*\[TABLE\]", re.M),
    re.compile(r"supplementary\s+(?:table|figure|material)", re.I),
)

_COMPILED_CAUSAL = [(re.compile(p, re.I), w) for p, w in _CAUSAL_TERMS.items()]
_COMPILED_BIO = [(re.compile(p, re.I), w) for p, w in _BIO_TERMS.items()]
_COMPILED_QUANT = [(re.compile(p, re.I), w) for p, w in _QUANT_TERMS.items()]


@dataclass
class ScoringReport:
    """Summary of one scoring pass, for display in the UI."""

    method: str                     # "heuristic" | "llm" | "heuristic+llm"
    n_chunks: int
    n_selected: int
    chars_total: int
    chars_selected: int
    llm_error: Optional[str] = None

    @property
    def reduction_pct(self) -> float:
        if not self.chars_total:
            return 0.0
        return 100.0 * (1.0 - self.chars_selected / self.chars_total)


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------

def _weighted_hits(text: str, patterns: Sequence[tuple[re.Pattern[str], float]]) -> tuple[float, list[str]]:
    total = 0.0
    hits: list[str] = []
    for pattern, weight in patterns:
        found = pattern.findall(text)
        if found:
            # Diminishing returns: the tenth "induced" says little more than
            # the third, so use a log of the count.
            total += weight * (1.0 + math.log(len(found), 4))
            sample = found[0]
            if isinstance(sample, tuple):
                sample = next((s for s in sample if s), "")
            hits.append(str(sample).strip().lower())
    return total, hits


#: Words too common to distinguish one passage from another when matching a
#: target Key Event. "increased oligodendrocyte differentiation" should select
#: on "oligodendrocyte" and "differentiation", not on "increased".
_TARGET_STOPWORDS = {
    "the", "of", "in", "a", "an", "and", "to", "on", "at", "for", "with", "by",
    "from", "into", "increased", "decreased", "altered", "impaired", "reduced",
    "elevated", "loss", "level", "levels", "activity", "expression", "cells",
    "cell", "response", "function",
}


def _target_terms(*labels: str) -> list[str]:
    """Content words from the target Key Event labels, longest first."""
    terms: set[str] = set()
    for label in labels:
        for token in re.split(r"[^a-zA-Z0-9]+", (label or "").lower()):
            if len(token) > 3 and token not in _TARGET_STOPWORDS:
                terms.add(token)
    return sorted(terms, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------

#: Characters that begin or end a word in biomedical text. Substring matching
#: is not usable here: the vocabulary contains two- and three-letter acronyms,
#: and "OL" for oligodendrocyte occurs inside "control", "molecular", "role"
#: and "protocol". A vocabulary term that fires on every methods section does
#: not merely add noise — the score is a coverage ratio, so it makes irrelevant
#: passages outrank the ones carrying the evidence.
_WORD_EDGE = r"[^A-Za-z0-9]"


@lru_cache(maxsize=4096)
def compile_term(term: str) -> Optional[re.Pattern]:
    """
    Build a word-boundary matcher for one vocabulary term.

    `\\b` alone is wrong for this vocabulary. Symbols like "NaV1.6" and
    "SNAP-25" end in a digit or contain punctuation, and Python's `\\b` sits
    between a word and a non-word character, so `NaV1.6\\b` fails to match
    "NaV1.6," in some positions and `\\bTTX` would match inside "TTX-sensitive"
    only by accident. Explicit edge classes make the intent legible: a term
    matches when what surrounds it is not alphanumeric.

    Internal whitespace is relaxed to match any run of whitespace, so a term
    still matches across a line break introduced by PDF extraction.
    """
    term = (term or "").strip()
    if not term:
        return None

    escaped = re.escape(term.lower())
    # Let a single space in the term match any whitespace run, including the
    # newlines pdfplumber leaves mid-sentence.
    escaped = escaped.replace(r"\ ", r"\s+")
    # A hyphen in the term should also match a space or an en dash, since
    # "voltage-gated", "voltage gated" and "voltage–gated" all occur.
    escaped = escaped.replace(r"\-", r"[-‐-―\s]")

    # An optional trailing "s" so a singular term still matches the plural the
    # paper actually wrote. Word-boundary matching is otherwise strict enough
    # that "OPC" would miss every occurrence of "OPCs", which is how most
    # papers refer to them.
    try:
        return re.compile(
            rf"(?:^|{_WORD_EDGE})({escaped}s?)(?={_WORD_EDGE}|$)",
            re.IGNORECASE,
        )
    except re.error:
        return None


def term_matches(term: str, text_lower: str) -> bool:
    """True when `term` occurs in `text_lower` as a whole word or symbol."""
    pattern = compile_term(term)
    return bool(pattern and pattern.search(text_lower))


def score_chunks_heuristic(
    chunks: Sequence[Chunk],
    target_terms: Optional[Sequence[str]] = None,
    target_groups: Optional[Sequence[Sequence[str]]] = None,
) -> list[Chunk]:
    """
    Score every chunk in place on a 0-1 scale and return the list.

    The score blends causal language, biological entity vocabulary and
    quantitative reporting, normalised by chunk length so that a long
    methods section cannot out-score a dense results paragraph, then
    multiplied by a section prior.

    `target_terms` switches the scorer from "is this passage mechanistic?" to
    "is this passage about the relationship being asked about?". In targeted
    mode the generic ranking is close to useless: every paragraph of a
    mechanistic paper looks mechanistic, and the budget gets spent on passages
    that have nothing to do with the question.

    `target_groups` is the same idea with the vocabulary kept separated by side
    of the relationship — upstream terms in one group, downstream in another —
    so a passage can be credited for discussing both events even though it
    contains only a handful of the many words either event goes by.
    """
    raw_scores: list[float] = []
    reasons: list[str] = []

    if target_groups:
        groups = [
            [t.lower() for t in group if t and t.strip()]
            for group in target_groups
            if any(t and t.strip() for t in group)
        ]
    elif target_terms:
        groups = [[t.lower() for t in target_terms if t and t.strip()]]
    else:
        groups = []

    for chunk in chunks:
        text = chunk.text
        length_kb = max(0.5, len(text) / 1000.0)

        causal, causal_hits = _weighted_hits(text, _COMPILED_CAUSAL)
        bio, bio_hits = _weighted_hits(text, _COMPILED_BIO)
        quant, quant_hits = _weighted_hits(text, _COMPILED_QUANT)

        density = (2.0 * causal + 1.5 * bio + 1.0 * quant) / length_kb
        prior = _SECTION_PRIOR.get(chunk.section_kind, 1.0)

        target_hits: list[str] = []
        if groups:
            lowered = text.lower()
            # Multiplicative rather than additive: a passage mentioning none of
            # the target vocabulary is not merely less interesting, it is very
            # unlikely to be about the relationship at all.
            #
            # Strength is measured per side of the relationship, not across the
            # pooled vocabulary. Pooling looks reasonable until the vocabulary
            # is expanded: no passage contains forty synonyms, so dividing hits
            # by the size of the list means every added synonym drives every
            # score down, and the expansion that was supposed to find more
            # passages finds fewer. What matters is whether the passage speaks
            # about the upstream event and about the downstream event — in
            # whichever words — so each side is scored on its own and averaged.
            strengths: list[float] = []
            for group in groups:
                hits = [t for t in group if term_matches(t, lowered)]
                target_hits.extend(hits)
                # Two distinct terms from one side is as much confirmation as
                # that side can give; more is repetition, not more evidence.
                strengths.append(min(len(hits) / 2.0, 1.0) if group else 0.0)
            coverage = sum(strengths) / len(strengths) if strengths else 0.0
            density *= 0.15 + 1.85 * coverage

        for noise in _NOISE_PATTERNS:
            if noise.search(text):
                density *= 0.7
                break

        raw_scores.append(density * prior)

        bits: list[str] = []
        if target_hits:
            bits.append("target: " + ", ".join(sorted(set(target_hits))[:5]))
        elif groups:
            bits.append("no target vocabulary")
        if causal_hits:
            bits.append("causal: " + ", ".join(sorted(set(causal_hits))[:4]))
        if bio_hits:
            bits.append("biology: " + ", ".join(sorted(set(bio_hits))[:4]))
        if quant_hits:
            bits.append("quantitative evidence present")
        reasons.append("; ".join(bits) if bits else "no mechanistic vocabulary detected")

    # Normalise to 0-1 against the best-scoring chunk in this document.
    peak = max(raw_scores) if raw_scores else 0.0
    for chunk, raw, reason in zip(chunks, raw_scores, reasons):
        chunk.relevance_score = round(raw / peak, 4) if peak > 0 else 0.0
        chunk.relevance_reason = reason

    return list(chunks)


# ---------------------------------------------------------------------------
# LLM triage scorer
# ---------------------------------------------------------------------------

_TRIAGE_PREVIEW_CHARS = 420


def _triage_prompt(chunks: Sequence[Chunk]) -> str:
    lines = [
        "You are triaging sections of a toxicology paper before Key Event "
        "Relationship (KER) extraction.",
        "",
        "Score each numbered excerpt from 0 to 10 for how likely it is to "
        "describe a CAUSAL or MECHANISTIC link between an upstream biological "
        "event and a downstream biological event.",
        "",
        "  10 = explicitly states that one biological event causes, induces or "
        "prevents another",
        "   5 = reports relevant biological measurements without an explicit link",
        "   0 = methods boilerplate, funding, references, or unrelated content",
        "",
        "EXCERPTS:",
    ]
    for chunk in chunks:
        preview = re.sub(r"\s+", " ", chunk.text[:_TRIAGE_PREVIEW_CHARS]).strip()
        lines.append(f'[{chunk.chunk_id}] (section: {chunk.section}) "{preview}"')
    lines += [
        "",
        "Return ONLY JSON of the form:",
        '  {"scores": [{"id": "c001", "score": 8, "reason": "<8 words max>"}, ...]}',
        "Include every excerpt id exactly once.",
        "JSON:",
    ]
    return "\n".join(lines)


def score_chunks_llm(
    chunks: Sequence[Chunk],
    cfg: LLMConfig,
    *,
    batch_size: int = 25,
) -> tuple[list[Chunk], Optional[str]]:
    """
    Ask the model to triage chunks. Falls back silently to whatever scores are
    already present if the call fails.

    Returns (chunks, error_message_or_None).
    """
    if not chunks:
        return list(chunks), None

    by_id = {c.chunk_id: c for c in chunks}
    errors: list[str] = []
    got_any = False

    for start in range(0, len(chunks), batch_size):
        batch = list(chunks)[start : start + batch_size]
        prompt = _triage_prompt(batch)
        try:
            from dataclasses import replace

            raw = replace(cfg, max_output_tokens=1536).generate(prompt)
        except LLMProviderError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:  # defensive — triage must never be fatal
            errors.append(str(exc))
            continue

        try:
            payload = _loads_lenient(raw)
            for item in payload.get("scores", []):
                chunk = by_id.get(str(item.get("id", "")).strip())
                if chunk is None:
                    continue
                score = float(item.get("score", 0))
                chunk.relevance_score = max(0.0, min(1.0, score / 10.0))
                reason = str(item.get("reason") or "").strip()
                if reason:
                    chunk.relevance_reason = reason
                got_any = True
        except Exception as exc:
            errors.append(f"could not parse triage response: {exc}")

    error = "; ".join(errors) if errors and not got_any else None
    return list(chunks), error


def _loads_lenient(raw: str) -> dict:
    """Parse JSON out of a possibly fenced or chatty model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_chunks(
    chunks: Sequence[Chunk],
    *,
    char_budget: int = 45_000,
    min_score: float = 0.15,
    always_keep_kinds: Sequence[str] = ("abstract",),
    max_chunks: Optional[int] = None,
) -> list[Chunk]:
    """
    Mark the most relevant chunks as `selected` and return them in reading order.

    Selection is greedy by score under a character budget, with two guarantees:
    the abstract is always kept, and at least one chunk is always returned even
    if every score is below `min_score` (a low-signal paper should still be
    attempted rather than silently skipped).
    """
    for chunk in chunks:
        chunk.selected = False

    if not chunks:
        return []

    chosen: list[Chunk] = []
    used = 0

    # 1. Mandatory sections.
    for chunk in chunks:
        if chunk.section_kind in always_keep_kinds:
            chunk.selected = True
            chosen.append(chunk)
            used += len(chunk.text)

    # 2. Greedy by score.
    ranked = sorted(
        (c for c in chunks if not c.selected),
        key=lambda c: c.relevance_score,
        reverse=True,
    )
    for chunk in ranked:
        if max_chunks is not None and len(chosen) >= max_chunks:
            break
        if chunk.relevance_score < min_score:
            continue
        if used + len(chunk.text) > char_budget:
            continue
        chunk.selected = True
        chosen.append(chunk)
        used += len(chunk.text)

    # 3. Never return nothing.
    if not chosen:
        best = max(chunks, key=lambda c: c.relevance_score)
        best.selected = True
        chosen.append(best)

    return sorted(chosen, key=lambda c: c.char_start)


def build_extraction_text(selected: Sequence[Chunk]) -> str:
    """
    Assemble the selected chunks into the text handed to the extractor.

    Each chunk keeps a visible header carrying its id, section and page range.
    This is deliberate: it lets the model cite a location directly, and it
    gives us a second way to attribute a quotation if fuzzy matching struggles.
    """
    blocks: list[str] = []
    for chunk in selected:
        header = f"[{chunk.chunk_id} | section: {chunk.section} | {chunk.page_label}]"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def prepare_paper_text(
    chunks: Sequence[Chunk],
    cfg: Optional[LLMConfig] = None,
    *,
    use_llm_triage: bool = False,
    char_budget: int = 45_000,
    min_score: float = 0.15,
    target_kes: Optional[Sequence[str]] = None,
    target_terms: Optional[Sequence[str]] = None,
    target_term_groups: Optional[Sequence[Sequence[str]]] = None,
) -> tuple[str, list[Chunk], ScoringReport]:
    """
    End-to-end convenience wrapper: score, select, and assemble.

    `target_kes` is the pair of Key Event labels in targeted mode; passing it
    ranks passages by relevance to that specific relationship rather than by
    generic mechanistic content.

    `target_term_groups` supplies that vocabulary directly, one group per side
    of the relationship, and takes precedence over both. `target_terms` is the
    ungrouped equivalent.
    Deriving terms from the labels alone matches only what the AOP happens to
    call the event: a label reading "voltage-gated sodium channel" scores
    nothing against a paper written in terms of NaV1.6, VGSC or SCN8A, and the
    passages that carry the actual evidence get dropped below the threshold
    before any model sees them. Callers that have expanded the labels into the
    vocabulary papers use should pass it here.

    Returns (extraction_text, selected_chunks, report).
    """
    groups: Optional[list[list[str]]] = None
    if target_term_groups:
        groups = [
            [t for t in group if t and t.strip()]
            for group in target_term_groups
            if any(t and t.strip() for t in group)
        ]
        method_stem = "heuristic-targeted+synonyms"
    elif target_terms:
        groups = [[t for t in target_terms if t and t.strip()]]
        method_stem = "heuristic-targeted+synonyms"
    elif target_kes:
        # Each label on its own, so the two sides are scored separately even
        # when no expansion was available.
        groups = [_target_terms(label) for label in target_kes if label]
        groups = [g for g in groups if g]
        method_stem = "heuristic-targeted"
    else:
        method_stem = "heuristic"

    score_chunks_heuristic(chunks, target_groups=groups)

    method = method_stem
    llm_error: Optional[str] = None
    if use_llm_triage and cfg is not None:
        heuristic_scores = {c.chunk_id: c.relevance_score for c in chunks}
        _, llm_error = score_chunks_llm(chunks, cfg)
        if llm_error is None:
            method = f"{method_stem}+llm"
            # Blend so a confident heuristic hit is not thrown away by a
            # dismissive triage score, and vice versa.
            for chunk in chunks:
                prior = heuristic_scores.get(chunk.chunk_id, 0.0)
                chunk.relevance_score = round(0.4 * prior + 0.6 * chunk.relevance_score, 4)

    selected = select_chunks(chunks, char_budget=char_budget, min_score=min_score)

    report = ScoringReport(
        method=method,
        n_chunks=len(chunks),
        n_selected=len(selected),
        chars_total=sum(len(c.text) for c in chunks),
        chars_selected=sum(len(c.text) for c in selected),
        llm_error=llm_error,
    )
    return build_extraction_text(selected), selected, report


__all__ = [
    "ScoringReport",
    "score_chunks_heuristic",
    "score_chunks_llm",
    "select_chunks",
    "build_extraction_text",
    "prepare_paper_text",
]
