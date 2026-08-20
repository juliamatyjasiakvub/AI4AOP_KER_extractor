from __future__ import annotations

"""
Cross-paper weight-of-evidence synthesis for a single KER.

What was missing
----------------
Extraction produces one assessment per paper. Table 2 then aggregates those
rows structurally — paper counts, confidence score, joined DOIs — but it never
combined the prose. `ker_description` was taken from whichever row came first
and the other twenty-one discarded; `biological_plausibility` was four papers'
sentences concatenated with semicolons. Neither is a synthesis, and
`biological_plausibility_synthesis` was a column that only ever held None.

So a user who ran twenty-two papers against one relationship got twenty-two
separate readings and no answer.

What this does
--------------
One model call per KER, reading every per-paper assessment already extracted
and writing a single consolidated narrative in the structure the AOP Handbook
asks for: mechanistic basis, biological plausibility, empirical evidence,
essentiality, applicability domain, and — the part that matters most —
uncertainties and inconsistencies.

Two design choices worth stating:

*No paper text is resent.* The input is the structured fields already
extracted, so the call is small and cheap regardless of how many papers
contributed. Twenty-two papers cost about as much as one step of one
extraction.

*Disagreement is a required output, not an optional one.* A synthesis that
smooths over a null result is worse than no synthesis, because it looks
authoritative. The prompt demands that contradicting and qualifying findings
be stated, and the assembled input marks which rows were flagged as
contradicting so the model cannot miss them.
"""

import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

import run_manifest
from json_repair import extract_json
from stage2_extraction.llm_providers import (
    LLMAuthError,
    LLMConfig,
    LLMProviderError,
)

__all__ = [
    "KERSynthesis",
    "synthesise_ker",
    "build_synthesis_input",
    "synthesis_markdown",
    "WOE_RATINGS",
]


#: OECD weight-of-evidence ratings used in the AOP Handbook for biological
#: plausibility, empirical support and essentiality.
WOE_RATINGS = ("High", "Moderate", "Low", "Not determined")

#: Per-paper fields worth sending. Everything here was already extracted; the
#: point is to give the model the whole evidence base in one view, not to
#: re-read the papers.
_PAPER_FIELDS: list[tuple[str, str]] = [
    ("ker_description", "mechanism"),
    ("biological_plausibility", "plausibility"),
    ("empirical_evidence_summary", "empirical"),
    ("essentiality_evidence", "essentiality"),
    ("quantitative_relationships", "quantitative"),
    ("response_response_relationship", "response_response"),
    ("time_scale", "time_scale"),
    ("taxonomic_applicability", "taxa"),
    ("sex_applicability", "sex"),
    ("life_stage_applicability", "life_stage"),
    ("study_design", "design"),
    ("chemical_stressor", "stressor"),
    ("modulating_factors", "modulating"),
]


@dataclass
class KERSynthesis:
    """A consolidated assessment of one KER across every contributing paper."""

    ker_key: str = ""
    ker_name: str = ""
    #: Distinct contributing papers. Reported to the model and stored.
    n_papers: int = 0
    #: Table 1 rows behind those papers. Kept separately because the two
    #: diverge whenever the extractor splits a paper into several claims, and
    #: a synthesis whose row count moved while its paper count held is a
    #: different situation from one where new literature arrived.
    n_rows: int = 0
    record_ids: list[int] = field(default_factory=list)

    mechanistic_basis: str = ""
    biological_plausibility: str = ""
    biological_plausibility_rating: str = "Not determined"
    empirical_evidence: str = ""
    empirical_evidence_rating: str = "Not determined"
    essentiality: str = ""
    essentiality_rating: str = "Not determined"
    quantitative_understanding: str = ""
    applicability_domain: str = ""
    uncertainties: str = ""
    overall_confidence: str = "Not determined"

    generated_at: str = ""
    model: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.mechanistic_basis.strip())


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null", "not specified"):
        return None
    return text


def _ordered_for_prompt(paper_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Put the evidence blocks in an order that depends only on the evidence.

    They used to arrive in `record_id` order, which is upload order — so
    re-extracting the same corpus with the PDFs added in a different sequence
    produced a different prompt, and therefore a different narrative and
    possibly different OECD ratings, from identical evidence. Sorting by DOI
    makes the prompt a function of the papers. `record_id` is the tie-break, so
    two rows from one paper still have a fixed order between them.
    """
    if paper_rows is None or paper_rows.empty:
        return paper_rows
    frame = paper_rows.copy()
    frame["_doi_key"] = (
        frame["source_doi"].astype(str).str.strip().str.lower()
        if "source_doi" in frame.columns
        else ""
    )
    frame["_row_key"] = (
        frame["record_id"] if "record_id" in frame.columns else range(len(frame))
    )
    frame = frame.sort_values(["_doi_key", "_row_key"], kind="stable")
    return frame.drop(columns=["_doi_key", "_row_key"])


def n_contributing_papers(paper_rows: pd.DataFrame) -> int:
    """
    How many distinct papers are behind this synthesis.

    Not `len(paper_rows)`. A paper contributing two rows to one relationship is
    one paper, and how many rows a paper yields is exactly what moves between
    extraction runs — so counting rows made the reported support change without
    any change in the literature.
    """
    if paper_rows is None or paper_rows.empty:
        return 0
    if "source_doi" not in paper_rows.columns:
        return int(len(paper_rows))

    identities: set[str] = set()
    for _, row in paper_rows.iterrows():
        doi = _clean(row.get("source_doi"))
        if doi:
            identities.add(f"doi:{doi.lower()}")
            continue
        # No DOI: fall back to the filename, and failing that treat the row as
        # its own paper. Overcounting an unidentifiable row is recoverable;
        # silently merging two real papers is not.
        filename = _clean(row.get("source_filename"))
        identities.add(
            f"file:{filename.lower()}" if filename else f"record:{row.get('record_id')}"
        )
    return len(identities)


def build_synthesis_input(paper_rows: pd.DataFrame) -> str:
    """
    Render the per-paper evidence base as compact text for the model.

    Each paper is one block keyed by DOI so the model can attribute claims
    without being handed the papers themselves. Rows recorded as contradicting
    are labelled prominently — a synthesis that quietly drops the dissenting
    study is the specific failure this whole function exists to prevent.
    """
    blocks: list[str] = []
    for i, (_, row) in enumerate(_ordered_for_prompt(paper_rows).iterrows(), start=1):
        doi = _clean(row.get("source_doi")) or f"paper-{i}"
        contradicts = bool(row.get("contradicts_ker"))
        header = f"[{i}] DOI {doi}"
        if contradicts:
            header += "  *** RECORDED AS ARGUING AGAINST THIS RELATIONSHIP ***"
        confidence = _clean(row.get("extraction_confidence"))
        if confidence:
            header += f"  (extraction confidence: {confidence})"

        lines = [header]
        for column, label in _PAPER_FIELDS:
            value = _clean(row.get(column))
            if value:
                lines.append(f"  {label}: {value}")

        verified = row.get("n_verified_spans")
        spans = row.get("n_evidence_spans")
        try:
            if spans and int(spans) > 0:
                lines.append(f"  quotations located verbatim: {int(verified or 0)}/{int(spans)}")
        except (TypeError, ValueError):
            pass

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PERSONA = (
    "You are an expert in adverse outcome pathway (AOP) development writing a "
    "Key Event Relationship assessment for the AOP-Wiki, following the OECD "
    "AOP Developers' Handbook.\n\n"
    "You are given the per-paper assessments already extracted for ONE Key "
    "Event Relationship. Your job is to write the consolidated assessment that "
    "the whole evidence base supports — not to summarise each paper in turn.\n\n"
    "Rules you must follow:\n"
    "1. Synthesise. Say what the body of evidence collectively establishes. Do "
    "not produce a paper-by-paper list.\n"
    "2. Attribute. Cite the DOIs supporting each claim in brackets, e.g. "
    "[10.1002/stem.1515]. Every substantive claim needs at least one.\n"
    "3. Report disagreement explicitly. If papers conflict, if one found no "
    "effect, or if a study qualifies its own result, that belongs in the "
    "uncertainties section and must not be smoothed away. A confident-sounding "
    "assessment that hides a null result is worse than no assessment.\n"
    "4. Do not invent. Use only what the assessments below contain. If the "
    "evidence for a section is absent, say so and rate it 'Not determined'.\n"
    "5. Distinguish mechanism from correlation, and note where support comes "
    "from a single study or a single test system.\n\n"
    "Rate biological plausibility, empirical evidence and essentiality as "
    "High, Moderate, Low or Not determined, using the OECD criteria:\n"
    "- High: extensive understanding based on broad agreement across "
    "independent studies and consistency with established biology.\n"
    "- Moderate: the relationship is plausible and supported, but evidence is "
    "incomplete, from few studies, or partly inconsistent.\n"
    "- Low: empirical support is sparse, conflicting, or largely "
    "correlational.\n"
)


def _synthesis_task(ker_name: str, n_papers: int, evidence_block: str) -> str:
    return (
        f"KEY EVENT RELATIONSHIP: {ker_name}\n"
        f"Number of contributing papers: {n_papers}\n\n"
        "PER-PAPER ASSESSMENTS:\n"
        f"{evidence_block}\n\n"
        "Write the consolidated assessment. Return ONLY JSON:\n"
        "  {\n"
        '    "mechanistic_basis": "<2-4 sentences: the mechanism the evidence '
        'collectively supports, with DOI citations>",\n'
        '    "biological_plausibility": "<2-4 sentences on structural and '
        'functional plausibility, with citations>",\n'
        '    "biological_plausibility_rating": "High|Moderate|Low|Not determined",\n'
        '    "empirical_evidence": "<2-4 sentences on dose-response, temporal '
        'and incidence concordance across studies, with citations>",\n'
        '    "empirical_evidence_rating": "High|Moderate|Low|Not determined",\n'
        '    "essentiality": "<1-3 sentences on whether blocking or restoring '
        'the upstream event changes the downstream one, with citations>",\n'
        '    "essentiality_rating": "High|Moderate|Low|Not determined",\n'
        '    "quantitative_understanding": "<1-3 sentences on what is known '
        'quantitatively, or that little is>",\n'
        '    "applicability_domain": "<1-3 sentences: taxa, sex, life stages '
        'and test systems the evidence covers>",\n'
        '    "uncertainties": "<REQUIRED. Conflicting findings, null results, '
        'qualifications, gaps, over-reliance on one model system. Name the '
        'dissenting DOIs. If there genuinely are none, say what would be '
        'needed to raise confidence>",\n'
        '    "overall_confidence": "High|Moderate|Low|Not determined"\n'
        "  }\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def synthesise_ker(
    ker_name: str,
    paper_rows: pd.DataFrame,
    cfg: LLMConfig,
    *,
    ker_key: str = "",
    max_output_tokens: int = 3000,
) -> KERSynthesis:
    """
    Produce one consolidated assessment from many per-paper assessments.

    Never raises for content reasons: failures come back on `.error` so the
    caller can show them next to the evidence rather than losing the run.
    """
    result = KERSynthesis(
        ker_key=ker_key,
        ker_name=ker_name,
        n_papers=n_contributing_papers(paper_rows),
        n_rows=int(len(paper_rows)) if paper_rows is not None else 0,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        model=getattr(cfg, "model", None),
    )
    if paper_rows is None or paper_rows.empty:
        result.error = "No contributing rows to synthesise."
        return result

    if "record_id" in paper_rows.columns:
        result.record_ids = sorted(
            int(r) for r in paper_rows["record_id"].dropna().tolist()
        )

    evidence_block = build_synthesis_input(paper_rows)
    # The prompt is told the number of PAPERS, because that is the word it
    # uses, and because the OECD criteria it is asked to apply turn on
    # agreement across independent studies. Handing it a row count invited it
    # to read one paper split in two as two-fold replication.
    prompt = _synthesis_task(ker_name, result.n_papers, evidence_block)

    from dataclasses import replace as _replace

    call_cfg = _replace(cfg, max_output_tokens=max_output_tokens)
    try:
        raw = call_cfg.generate(prompt, cached_prefix=_SYNTHESIS_PERSONA)
        run_manifest.record("llm_call", step="synthesis")
    except LLMAuthError as exc:
        result.error = str(exc)
        return result
    except LLMProviderError as exc:
        run_manifest.record("provider_error")
        result.error = str(exc)
        return result

    try:
        parsed = extract_json(raw, context="synthesis response")
    except Exception as exc:
        run_manifest.record("step_failure", step="synthesis")
        result.error = f"Could not parse the synthesis reply: {exc}"
        return result

    if not isinstance(parsed, dict):
        result.error = "Synthesis reply was not a JSON object."
        return result

    def _rating(key: str) -> str:
        value = str(parsed.get(key) or "").strip()
        for allowed in WOE_RATINGS:
            if value.lower() == allowed.lower():
                return allowed
        return "Not determined"

    result.mechanistic_basis = str(parsed.get("mechanistic_basis") or "").strip()
    result.biological_plausibility = str(parsed.get("biological_plausibility") or "").strip()
    result.biological_plausibility_rating = _rating("biological_plausibility_rating")
    result.empirical_evidence = str(parsed.get("empirical_evidence") or "").strip()
    result.empirical_evidence_rating = _rating("empirical_evidence_rating")
    result.essentiality = str(parsed.get("essentiality") or "").strip()
    result.essentiality_rating = _rating("essentiality_rating")
    result.quantitative_understanding = str(parsed.get("quantitative_understanding") or "").strip()
    result.applicability_domain = str(parsed.get("applicability_domain") or "").strip()
    result.uncertainties = str(parsed.get("uncertainties") or "").strip()
    result.overall_confidence = _rating("overall_confidence")

    if not result.uncertainties:
        # The section most likely to be dropped is the one that keeps the
        # assessment honest, so its absence is recorded rather than ignored.
        result.uncertainties = (
            "The model returned no uncertainties section. Treat this "
            "assessment as incomplete rather than as evidence that none exist."
        )

    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def synthesis_markdown(s: KERSynthesis) -> str:
    """Render a synthesis as a filing-ready Markdown assessment."""
    if s.error:
        return f"# {s.ker_name}\n\nSynthesis failed: {s.error}\n"

    parts = [
        f"# {s.ker_name}\n",
        f"Consolidated from {s.n_papers} paper(s)"
        + (f" across {s.n_rows} extracted claim(s)" if s.n_rows > s.n_papers else "")
        + f" · generated {s.generated_at}"
        + (f" · {s.model}" if s.model else "")
        + "\n",
        f"**Overall confidence: {s.overall_confidence}**\n",
        "## Mechanistic basis\n", s.mechanistic_basis, "\n",
        f"## Biological plausibility — {s.biological_plausibility_rating}\n",
        s.biological_plausibility, "\n",
        f"## Empirical evidence — {s.empirical_evidence_rating}\n",
        s.empirical_evidence, "\n",
        f"## Essentiality — {s.essentiality_rating}\n", s.essentiality, "\n",
    ]
    if s.quantitative_understanding:
        parts += ["## Quantitative understanding\n", s.quantitative_understanding, "\n"]
    if s.applicability_domain:
        parts += ["## Applicability domain\n", s.applicability_domain, "\n"]
    parts += ["## Uncertainties and inconsistencies\n", s.uncertainties, "\n"]
    parts += [
        "\n---\n",
        "*Machine-generated from extracted assessments and not validated by an "
        "expert. Every claim should be checked against the cited sources "
        "before use.*\n",
    ]
    return "\n".join(parts)
