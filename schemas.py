from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Stage 1 — PubMed search & screening
# ---------------------------------------------------------------------------

@dataclass
class PubMedRecord:
    pmid: str
    doi: Optional[str]
    first_author: Optional[str]
    journal: Optional[str]
    year: Optional[int]
    title: str
    abstract: str
    query_used: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningDecision:
    decision: str  # yes | no | maybe
    rationale: str
    triggered_inclusion_rule: Optional[str]
    triggered_exclusion_rule: Optional[str]
    evidence_quote: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 2 — KER extraction (Table 1, per-paper rows)
# ---------------------------------------------------------------------------

@dataclass
class KERExtraction:
    """One KER extracted from one paper by the LLM. Maps 1-to-1 with Table 1 rows."""

    # Section B — Key events
    upstream_ke_name: str
    upstream_ke_level: str
    downstream_ke_name: str
    downstream_ke_level: str

    # Section C — KER identity
    ker_name: str
    ker_description: str
    ker_adjacency: str  # Adjacent | Non-adjacent

    # Section D — Evidence
    paper_type: str  # Primary study | Review / meta-analysis | In silico
    cited_evidence_dois: Optional[str]
    biological_plausibility: Optional[str]
    empirical_evidence_summary: Optional[str]
    essentiality_evidence: Optional[str]
    contradicts_ker: bool

    # Section E — Applicability
    taxonomic_applicability: str
    sex_applicability: str
    life_stage_applicability: str

    # Section F — Quantitative understanding
    modulating_factors: Optional[str]
    quantitative_relationships: Optional[str]
    response_response_relationship: Optional[str]
    time_scale: Optional[str]
    feedforward_feedback_loops: Optional[str]

    # Section G — Study metadata
    study_design: str
    exposure_route: Optional[str]
    chemical_stressor: Optional[str]

    # Quality flag (LLM self-assessment)
    extraction_confidence: str  # High | Medium | Low

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Table1Row:
    """KERExtraction enriched with pipeline-added fields. Stored in SQLite."""

    # Pipeline-added identity fields
    record_id: Optional[int]        # set by SQLite autoincrement
    source_doi: str                 # DOI of the uploaded paper
    extraction_date: str            # ISO date string YYYY-MM-DD
    aop_id: Optional[str]           # from AOP-Wiki API (may be semicolon-sep list)
    aop_status: Optional[str]       # existing | novel
    upstream_ke_id: Optional[int]   # from AOP-Wiki API
    downstream_ke_id: Optional[int] # from AOP-Wiki API
    ker_id: Optional[int]           # from AOP-Wiki API

    # All LLM-extracted fields (mirrored from KERExtraction)
    upstream_ke_name: str
    upstream_ke_level: str
    downstream_ke_name: str
    downstream_ke_level: str
    ker_name: str
    ker_description: str
    ker_adjacency: str
    paper_type: str
    cited_evidence_dois: Optional[str]
    biological_plausibility: Optional[str]
    empirical_evidence_summary: Optional[str]
    essentiality_evidence: Optional[str]
    contradicts_ker: bool
    taxonomic_applicability: str
    sex_applicability: str
    life_stage_applicability: str
    modulating_factors: Optional[str]
    quantitative_relationships: Optional[str]
    response_response_relationship: Optional[str]
    time_scale: Optional[str]
    feedforward_feedback_loops: Optional[str]
    study_design: str
    exposure_route: Optional[str]
    chemical_stressor: Optional[str]
    extraction_confidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
