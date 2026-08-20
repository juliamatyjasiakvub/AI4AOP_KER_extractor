from __future__ import annotations

"""
Claims a curator enters by hand.

Extraction misses things. A model reads sixteen papers and returns what it
found; the relationship the developer knows is in paper four, stated in a
figure legend, is simply absent — and until now the only recourse was to
re-run and hope, or to accept an AOP with a hole in it.

The rule this module exists to hold: **a manual claim is a Table 1 row.**
Nothing in this application exists except as a consequence of one. Canonical
Key Events are rebuilt from raw labels on every normalization run, and a KER
is not stored at all — it is a group-by over Table 1. So an entry point that
wrote a canonical Key Event, or invented somewhere for a KER to live, would
create objects the rest of the pipeline cannot see and the next re-run would
delete. Entering the row instead means curation, approval, synthesis and the
final figure all treat it exactly as they treat everything else.

What must NOT be equal is how it is *labelled*. A curator-entered row carries
`origin='curator'`, and three things read that:

    the QC report      — a hand-typed row has no model output to verify, so
                         counting it in the quotation-verification rate makes
                         the model look better or worse than it was;
    the provenance UI  — "the model read this in the paper" and "the developer
                         asserts this" are different claims;
    the AOP figure     — an edge nobody's evidence produced is drawn so that
                         it is recognisable as such.

Three distinct situations arrive here, and they are not the same claim:

    1. the paper says it and extraction missed it   — quote it, verify it
    2. the curator knows it from elsewhere          — cite it, cannot verify
    3. the ends exist but nothing links them        — an assertion, no source

Case 1 is the common one and the only one that can be verified, which is why
`verify_quote` works from stored chunks rather than needing the PDF back.
"""

from typing import Any, Optional, Sequence

from schemas import (
    DIRECTION_VALUES,
    KE_LEVEL_ORDER,
    KER_ADJACENCY_VALUES,
    RELATION_KIND_VALUES,
    EvidenceSpan,
    KERExtraction,
)
from stage2_extraction import table1_store
from stage2_extraction.pdf_reader import locate_quote_in_chunks

#: The DOI stand-in for a claim with no paper behind it. A real-looking but
#: fake DOI would be worse than an obviously absent one: it would resolve to
#: nothing and be silently exported as a citation.
NO_SOURCE_DOI = "curator-asserted"

#: What a curator-entered row says about its own confidence.
#:
#: `extraction_confidence` is documented as the model's self-assessment. There
#: is no model here, and writing "High" into it would put a hand-typed row at
#: the top of every confidence-sorted view on the strength of nothing. The QC
#: report filters these out by origin; the value is here so the column is not
#: null for rows that still have to render.
CURATOR_CONFIDENCE = "Curator-entered"

#: How the claim's support was established, for the curator to declare.
#: Mirrors the model's `evidence_type` vocabulary plus the one value only a
#: person can mean.
MANUAL_EVIDENCE_TYPES = (
    "rescue",
    "perturbation",
    "correlation",
    "reverse_only",
    "expert_judgement",
    "not_stated",
)


class ValidationError(ValueError):
    """The form does not describe a claim that can be stored."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(values: dict[str, Any]) -> list[str]:
    """
    Everything wrong with a proposed claim, as sentences a curator can act on.

    Returns an empty list when the claim is storable. Deliberately checks the
    few things that make a row *meaningless* rather than merely thin — an
    unnamed event, a self-loop, a level outside the vocabulary — and leaves
    the rest to the curator. A form that refuses incomplete records is a form
    people work around.
    """
    problems: list[str] = []

    upstream = str(values.get("upstream_ke_name") or "").strip()
    downstream = str(values.get("downstream_ke_name") or "").strip()

    if not upstream:
        problems.append("The upstream Key Event needs a name.")
    if not downstream:
        problems.append("The downstream Key Event needs a name.")
    if upstream and downstream and upstream.casefold() == downstream.casefold():
        problems.append(
            "The upstream and downstream Key Events are the same event. A "
            "relationship needs two ends."
        )

    for side in ("upstream", "downstream"):
        level = values.get(f"{side}_ke_level")
        if level not in KE_LEVEL_ORDER:
            problems.append(
                f"The {side} biological level must be one of "
                f"{', '.join(KE_LEVEL_ORDER)}."
            )

    if values.get("direction") not in DIRECTION_VALUES:
        problems.append(
            f"Direction must be one of {', '.join(DIRECTION_VALUES)}."
        )
    if values.get("relation_kind") not in RELATION_KIND_VALUES:
        problems.append(
            f"Relationship kind must be one of {', '.join(RELATION_KIND_VALUES)}."
        )
    if values.get("ker_adjacency") not in KER_ADJACENCY_VALUES:
        problems.append(
            f"Adjacency must be one of {', '.join(KER_ADJACENCY_VALUES)}."
        )

    if not str(values.get("entry_rationale") or "").strip():
        problems.append(
            "Say why this claim belongs in the pathway. It is the only "
            "provenance a hand-entered row has."
        )

    return problems


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

def verify_quote(
    quote: str,
    *,
    source_doi: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Check a pasted quotation against the stored text of its paper.

    A curator who pastes the sentence out of the PDF gets the same verified
    flag, page and section a model-extracted quote would get — which is the
    point: a hand-entered claim backed by a located quotation is *better*
    evidence than a model claim whose quotation could not be found, and the
    interface should be able to say so.

    Returns the `locate_quote` dict plus `searched`, which is False when the
    paper's text was never stored (extraction predates chunk storage, or the
    source is not in the corpus at all). False is not a failed verification
    and must not be shown as one.
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
        "searched": False,
    }
    if not str(quote or "").strip():
        return blank
    if not source_doi and not source_filename:
        return blank

    chunks = table1_store.load_chunks(
        source_doi=source_doi or None,
        source_filename=source_filename or None,
        limit=2000,
    )
    if chunks.empty:
        return blank

    located = locate_quote_in_chunks(quote, chunks.to_dict("records"))
    located["searched"] = True
    return located


def _build_spans(
    quotes: Sequence[str],
    *,
    source_doi: Optional[str],
    source_filename: Optional[str],
    field: str = "curator_entry",
) -> list[EvidenceSpan]:
    """Turn pasted quotations into located spans, verified where possible."""
    spans: list[EvidenceSpan] = []
    for raw in quotes:
        quote = str(raw or "").strip()
        if not quote:
            continue
        located = verify_quote(
            quote, source_doi=source_doi, source_filename=source_filename
        )
        spans.append(
            EvidenceSpan(
                quote=quote,
                field=field,
                section=located["section"],
                section_kind=located["section_kind"],
                page_start=located["page_start"],
                page_end=located["page_end"],
                chunk_id=located["chunk_id"],
                char_start=located["char_start"],
                char_end=located["char_end"],
                verified=bool(located["verified"]),
                match_ratio=float(located["match_ratio"]),
                source_doi=source_doi,
                source_filename=source_filename,
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Building and saving
# ---------------------------------------------------------------------------

def build_extraction(values: dict[str, Any]) -> KERExtraction:
    """
    Assemble a `KERExtraction` from form values, with honest defaults.

    Optional fields left blank stay None rather than being filled with "not
    stated" or "unknown": an empty field means the curator did not say, and a
    string saying so is indistinguishable from a model that reported it.
    """
    def text(key: str) -> Optional[str]:
        value = str(values.get(key) or "").strip()
        return value or None

    upstream = str(values.get("upstream_ke_name") or "").strip()
    downstream = str(values.get("downstream_ke_name") or "").strip()

    return KERExtraction(
        upstream_ke_name=upstream,
        upstream_ke_level=str(values.get("upstream_ke_level") or "Molecular"),
        downstream_ke_name=downstream,
        downstream_ke_level=str(values.get("downstream_ke_level") or "Cellular"),
        ker_name=text("ker_name") or f"{upstream} leads to {downstream}",
        ker_description=text("ker_description") or "",
        ker_adjacency=str(values.get("ker_adjacency") or "Adjacent"),
        paper_type=str(values.get("paper_type") or "Primary study"),
        cited_evidence_dois=text("cited_evidence_dois"),
        biological_plausibility=text("biological_plausibility"),
        empirical_evidence_summary=text("empirical_evidence_summary"),
        essentiality_evidence=text("essentiality_evidence"),
        contradicts_ker=bool(values.get("contradicts_ker", False)),
        taxonomic_applicability=str(values.get("taxonomic_applicability") or ""),
        sex_applicability=str(values.get("sex_applicability") or ""),
        life_stage_applicability=str(values.get("life_stage_applicability") or ""),
        modulating_factors=text("modulating_factors"),
        quantitative_relationships=text("quantitative_relationships"),
        response_response_relationship=text("response_response_relationship"),
        time_scale=text("time_scale"),
        feedforward_feedback_loops=text("feedforward_feedback_loops"),
        study_design=str(values.get("study_design") or ""),
        exposure_route=text("exposure_route"),
        chemical_stressor=text("chemical_stressor"),
        extraction_confidence=CURATOR_CONFIDENCE,
        direction=str(values.get("direction") or "unclear"),
        upstream_change=text("upstream_change"),
        downstream_change=text("downstream_change"),
        upstream_cell_type=text("upstream_cell_type"),
        downstream_cell_type=text("downstream_cell_type"),
        relation_kind=str(values.get("relation_kind") or "causal"),
        evidence_type=str(values.get("evidence_type") or "not_stated"),
        measured_as=text("measured_as"),
        upstream_target=text("upstream_target"),
        downstream_target=text("downstream_target"),
        null_findings=text("null_findings"),
        study_context=text("study_context"),
    )


def save_manual_claim(
    values: dict[str, Any],
    *,
    curator: str,
    quotes: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Store one hand-entered claim as a Table 1 row.

    Returns {"record_id", "run_id", "n_verified", "n_quotes"}. Raises
    `ValidationError` listing every problem at once, so a curator fixes the
    form in one pass rather than one message at a time.
    """
    problems = validate(values)
    if problems:
        raise ValidationError("\n".join(problems))

    source_doi = str(values.get("source_doi") or "").strip() or NO_SOURCE_DOI
    source_filename = str(values.get("source_filename") or "").strip() or None
    source_title = str(values.get("source_title") or "").strip() or None

    extraction = build_extraction(values)
    extraction.evidence_spans = _build_spans(
        quotes,
        source_doi=None if source_doi == NO_SOURCE_DOI else source_doi,
        source_filename=source_filename,
    )

    run_id = table1_store.start_manual_run(curator)
    record_id = table1_store.insert_table1_row(
        extraction,
        source_doi,
        wiki_ids={},
        source_filename=source_filename,
        source_title=source_title,
        run_id=run_id,
        origin="curator",
        entered_by=curator or None,
        entry_rationale=str(values.get("entry_rationale") or "").strip() or None,
    )

    return {
        "record_id": record_id,
        "run_id": run_id,
        "n_quotes": len(extraction.evidence_spans),
        "n_verified": sum(1 for s in extraction.evidence_spans if s.verified),
    }


def update_manual_claim(
    record_id: int,
    values: dict[str, Any],
    *,
    curator: str,
    rationale: str = "",
) -> dict[str, Any]:
    """
    Correct an existing row, whoever originally produced it.

    Correcting a model row is not the same act as entering one, and the store
    records the difference: the row becomes `curator_edited` and the version
    that was there is archived. Without this, the only way to fix a wrong
    extraction is to add a second row saying the opposite — which is how a
    corpus ends up asserting both.
    """
    problems = validate({**values, "entry_rationale": rationale or "given"})
    if problems:
        raise ValidationError("\n".join(problems))

    changes = {k: v for k, v in values.items() if k in table1_store.EDITABLE_FIELDS}
    return table1_store.update_table1_row(
        int(record_id), changes, curator=curator, rationale=rationale
    )


# ---------------------------------------------------------------------------
# Suggestions for the form
# ---------------------------------------------------------------------------

def known_event_names() -> list[str]:
    """
    Every name the corpus already uses for a Key Event — canonical and raw.

    Offered as a picklist so a manual row joins an existing group instead of
    inventing a synonym that then has to be merged back in by hand. This is
    the difference between manual entry that costs one form and manual entry
    that costs a form plus a curation decision.
    """
    names: set[str] = set()

    canonical = table1_store.load_canonical_kes()
    if not canonical.empty:
        names.update(str(n) for n in canonical["canonical_name"].dropna())
        if "aliases" in canonical:
            for blob in canonical["aliases"].dropna():
                names.update(part.strip() for part in str(blob).split(";") if part.strip())

    table1 = table1_store.load_table1_as_dataframe()
    if not table1.empty:
        for column in ("upstream_ke_name", "downstream_ke_name"):
            names.update(str(n) for n in table1[column].dropna())

    return sorted(names, key=str.casefold)


def level_for_name(name: str) -> Optional[str]:
    """
    The biological level the corpus already gives this event, if any.

    Saves re-answering a question the corpus has already answered, and stops
    the same event arriving at two levels and being drawn in two columns.
    """
    if not str(name or "").strip():
        return None
    target = str(name).strip().casefold()

    canonical = table1_store.load_canonical_kes()
    if not canonical.empty:
        for _, row in canonical.iterrows():
            if str(row["canonical_name"]).strip().casefold() == target:
                return str(row["level"])

    table1 = table1_store.load_table1_as_dataframe()
    if table1.empty:
        return None
    for side in ("upstream", "downstream"):
        match = table1[
            table1[f"{side}_ke_name"].astype(str).str.strip().str.casefold() == target
        ]
        if not match.empty:
            return str(match.iloc[0][f"{side}_ke_level"])
    return None


def known_papers() -> list[dict[str, str]]:
    """Papers already in the corpus, for the source picker."""
    papers = table1_store.list_source_papers()
    if papers.empty:
        return []
    return [
        {
            "doi": str(row.get("source_doi") or ""),
            "filename": str(row.get("source_filename") or ""),
            "title": str(row.get("source_title") or ""),
        }
        for _, row in papers.iterrows()
    ]
