from __future__ import annotations

"""
Dataclass definitions shared across the whole application.

Layout of this module
---------------------
    Stage 1   — PubMed search & screening
    Document  — page/section-aware representation of an uploaded paper
    Provenance— evidence spans tying every claim back to an exact quotation
    Stage 2   — KER extraction (Table 1, per-paper rows)
    Ontology  — OLS4 candidate matches and canonical Key Events
    Curation  — expert accept/reject/rename/merge decisions
    Layout    — persisted node coordinates so a curated map is never re-flowed
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

#: Biological level of organisation, ordered from the molecular initiating
#: event through to the population. The ORDER of this tuple defines the
#: left-to-right lane order in the layered AOP layout, so it must not be
#: re-sorted casually.
KE_LEVEL_ORDER: tuple[str, ...] = (
    "MIE",
    "Molecular",
    "Cellular",
    "Tissue",
    "Organ",
    "Individual",
    "Population",
)

#: Map from level name to its lane index (0 = leftmost).
KE_LEVEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(KE_LEVEL_ORDER)}

KER_ADJACENCY_VALUES = ("Adjacent", "Non-adjacent")

#: How the two events of a relationship moved together in one experiment.
#: "positive" and "negative" are claims a graph can act on; "none" is a
#: measured absence of coupling; "unclear" means the paper did not say, and
#: must never be silently rendered as either sign.
DIRECTION_VALUES = ("positive", "negative", "none", "unclear")

#: What kind of link a relationship is.
#:
#: An AOP is a chain of causal steps, but extraction produces two other kinds
#: that look identical once written as "A leads to B". A *marker* link says B
#: is how A was measured — myelin basic protein expression is a readout of
#: oligodendrocyte maturation, not an event downstream of it. A *definitional*
#: link says B is part of what A means. Both are worth keeping, and neither
#: should be drawn as a causal step or counted as an adverse outcome.
RELATION_KIND_VALUES = ("causal", "marker", "definitional")

#: What kind of experiment supports the direction of a link, strongest first.
#:
#: This is the distinction an AOP lives or dies on, and the one nothing in the
#: pipeline was capturing. "Deleting SCN2A abolished spiking" and "spiking and
#: maturation both declined" are not the same claim, but written as
#: "A leads to B" they are indistinguishable — so a graph built from them
#: shows an observation and a knockout as the same arrow.
#:
#: `reverse_only` is the case that produced a backwards edge: the paper
#: manipulated the DOWNSTREAM event and measured the upstream one. Glutamatergic
#: input triggering Nav1.2-mediated spikes is evidence that input causes
#: spiking, not that Nav1.2 loss causes more input.
#: `common_stressor` is the case a toxicology corpus is mostly made of, and
#: the one the pipeline was discarding outright. A lead-exposure study that
#: reports Nav1.6 node disruption AND oligodendrocyte loss has not shown that
#: one causes the other — but it has shown that a stressor known to act on the
#: initiating event also produces the downstream one, in the same animals, at
#: the same dose. That is the shape of nearly every regulatory AOP, and
#: returning "no pathway found" for it throws away the evidence the framework
#: exists to organise. It sits above plain correlation, because an exposure
#: was administered, and below perturbation, because the event itself was not.
EVIDENCE_TYPE_VALUES = (
    "rescue",           # upstream restored/removed AND downstream followed both ways
    "perturbation",     # upstream manipulated, downstream measured
    "common_stressor",  # one exposure, both events responded
    "correlation",      # both measured, neither manipulated
    "reverse_only",     # only the downstream event was manipulated
    "not_stated",
)

#: Evidence types strong enough to draw as a direct causal step.
CAUSAL_EVIDENCE = frozenset({"rescue", "perturbation"})
PAPER_TYPE_VALUES = ("Primary study", "Review / meta-analysis", "In silico")
STUDY_DESIGN_VALUES = (
    "In vivo",
    "In vitro",
    "In silico",
    "Ex vivo",
    "Epidemiological",
    "Review / meta-analysis",
)
SEX_VALUES = ("Male", "Female", "Mixed", "Not specified")
CONFIDENCE_VALUES = ("High", "Medium", "Low")

#: Verdicts an expert curator can record against a KE or KER.
CURATION_STATUS_VALUES = ("unreviewed", "accepted", "rejected")


def level_index(level: Optional[str]) -> int:
    """Return the lane index for a KE level, defaulting to 'Molecular'."""
    if not level:
        return KE_LEVEL_INDEX["Molecular"]
    return KE_LEVEL_INDEX.get(str(level).strip(), KE_LEVEL_INDEX["Molecular"])


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

    #: True when the record satisfied an inclusion criterion AND an exclusion
    #: criterion at the same time — e.g. an in vivo study that also reports
    #: cell-culture work when in vitro is excluded. Such records are the ones
    #: most worth a human glance, so they are flagged rather than silently
    #: resolved.
    criteria_conflict: bool = False

    #: How the conflict was resolved: 'maybe' | 'exclude' | 'include' | None.
    conflict_policy: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Document model — page- and section-aware view of one uploaded paper
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Text of a single PDF page, 1-indexed as printed in a citation."""

    page_number: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """
    A contiguous, addressable slice of a paper.

    Chunks are the unit of relevance scoring and the unit that an evidence
    quotation is resolved against, which is what allows every KER to carry a
    page number and section name rather than a bare string.
    """

    chunk_id: str            # stable within one document, e.g. "c007"
    text: str
    section: str             # detected heading, e.g. "Results"
    section_kind: str        # normalised: abstract|intro|methods|results|discussion|other
    page_start: int
    page_end: int
    char_start: int          # offset into the full document text
    char_end: int
    relevance_score: float = 0.0
    relevance_reason: Optional[str] = None
    selected: bool = False   # True if this chunk was fed to the extractor

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperDocument:
    """Everything we know about one uploaded PDF before extraction begins."""

    filename: str
    doi: Optional[str]
    full_text: str
    pages: list[PageText] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    title: Optional[str] = None

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def selected_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if c.selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "doi": self.doi,
            "title": self.title,
            "n_pages": self.n_pages,
            "n_chunks": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }


# ---------------------------------------------------------------------------
# Provenance — evidence spans
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSpan:
    """
    One verbatim quotation supporting one field of one KER.

    `verified` records whether the quoted text was actually found in the
    source document. Unverified quotes are kept (they are still informative)
    but are flagged in the UI so a curator can tell a real quotation from a
    model paraphrase.
    """

    quote: str
    field: str                       # which KER field this supports
    section: Optional[str] = None
    section_kind: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    verified: bool = False
    match_ratio: float = 0.0         # 1.0 = exact substring match
    source_doi: Optional[str] = None
    source_filename: Optional[str] = None

    @property
    def citation(self) -> str:
        """Human-readable short citation, e.g. '10.1016/x — Results, p. 7'."""
        bits: list[str] = []
        if self.source_doi:
            bits.append(self.source_doi)
        if self.section:
            bits.append(self.section)
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                bits.append(f"pp. {self.page_start}–{self.page_end}")
            else:
                bits.append(f"p. {self.page_start}")
        return " — ".join(bits) if bits else "location unknown"

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

    # Section H — Provenance (new)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)

    # Section I — Sign and context.
    #
    # A relationship without a sign is not a relationship: "reduced Nav1.2
    # causes loss of spiking" and "increased Na+ current causes activation"
    # are opposite findings that reduce to the same sentence once the
    # direction is dropped. `direction` is how the two events moved together
    # in this paper; the two `*_change` fields are what the paper did to each
    # end; the cell types keep the same event in two cell types apart.
    direction: str = "unclear"          # positive | negative | none | unclear
    upstream_change: Optional[str] = None
    downstream_change: Optional[str] = None
    upstream_cell_type: Optional[str] = None
    downstream_cell_type: Optional[str] = None
    relation_kind: str = "causal"       # causal | marker | definitional

    # How the direction was established, what the paper measured to establish
    # it, what did NOT change, and in what model. A null result is evidence
    # and was being discarded: "basal mEPSC amplitude was unchanged" belongs
    # on the record beside the changes, not left out because it is not a
    # change. Study context keeps a spinal-cord-injury finding from being
    # chained onto a developmental one purely because the labels matched.
    evidence_type: str = "not_stated"
    measured_as: Optional[str] = None
    upstream_target: Optional[str] = None
    downstream_target: Optional[str] = None
    null_findings: Optional[str] = None
    study_context: Optional[str] = None

    @property
    def n_verified_spans(self) -> int:
        return sum(1 for s in self.evidence_spans if s.verified)

    @property
    def is_signed(self) -> bool:
        """Whether this row asserts a direction that a graph can rely on."""
        return self.direction in ("positive", "negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Table1Row:
    """KERExtraction enriched with pipeline-added fields. Stored in SQLite."""

    # Pipeline-added identity fields
    record_id: Optional[int]        # set by SQLite autoincrement
    source_doi: str                 # DOI of the uploaded paper
    extraction_date: str            # ISO date string YYYY-MM-DD
    aop_id: Optional[str]           # from AOP-Wiki dump (may be semicolon-sep list)
    aop_status: Optional[str]       # existing | novel
    upstream_ke_id: Optional[int]   # from AOP-Wiki dump
    downstream_ke_id: Optional[int] # from AOP-Wiki dump
    ker_id: Optional[int]           # from AOP-Wiki dump

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

    # Provenance + normalization (new)
    source_filename: Optional[str] = None
    source_title: Optional[str] = None
    upstream_ke_canonical_id: Optional[int] = None
    downstream_ke_canonical_id: Optional[int] = None
    n_evidence_spans: int = 0
    n_verified_spans: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ontology — OLS4 matches and canonical Key Events
# ---------------------------------------------------------------------------

@dataclass
class OntologyMatch:
    """A single candidate term returned by OLS4 for a free-text KE label."""

    query: str                  # the raw label we searched for
    label: str                  # canonical ontology label
    curie: str                  # e.g. "GO:0006979"
    iri: str                    # e.g. "http://purl.obolibrary.org/obo/GO_0006979"
    ontology: str               # e.g. "go", "uberon", "cl", "hp", "mp"
    score: float                # 0-1 confidence that this is the right term
    description: Optional[str] = None
    is_exact: bool = False
    obsolete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalKE:
    """
    One merged Key Event.

    A canonical KE groups every raw label the extractor produced for what is
    biologically the same event. `canonical_name` is the display name;
    `aliases` preserves the original extracted terminology verbatim so nothing
    said by a paper is ever lost.

    `alias_basis` records, per raw label, *why* that label ended up in this
    group: a rule name and a human-readable detail. Without it the step from
    Table 1 to the canonical events is a number changing — "27 claims, 18 Key
    Events" — with nothing on screen saying which wording went where or on
    whose authority. Rule names are the ones named in the methods text:
    `aopwiki`, `ontology`, `normalised_string`, `token_order`, `lexical`,
    `own_group`, `curator`.
    """

    canonical_id: Optional[int]
    canonical_name: str
    level: str
    aliases: list[str] = field(default_factory=list)
    alias_basis: dict[str, list[str]] = field(default_factory=dict)
    ontology_curie: Optional[str] = None
    ontology_iri: Optional[str] = None
    ontology_label: Optional[str] = None
    ontology_source: Optional[str] = None
    ontology_score: float = 0.0
    aopwiki_ke_id: Optional[int] = None
    merge_method: str = "auto"       # auto | manual | ontology
    curation_status: str = "unreviewed"
    n_source_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysioMapLink:
    """A link from a tissue/organ-level KE to an external physiological map."""

    canonical_id: int
    provider: str              # e.g. "uberon", "reactome", "custom"
    entity_label: str
    entity_id: str             # CURIE or provider-native id
    url: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

@dataclass
class CurationRecord:
    """An expert decision about one canonical KE or one consolidated KER."""

    target_type: str           # "ke" | "ker"
    target_key: str            # canonical_id (as str) for KE, ker_key for KER
    status: str = "unreviewed" # unreviewed | accepted | rejected
    display_name: Optional[str] = None   # curator-supplied rename
    note: Optional[str] = None
    merged_into: Optional[str] = None    # target_key this was merged into
    curator: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------

@dataclass
class LayoutPosition:
    """A saved coordinate for one node in one named layout."""

    layout_name: str
    node_key: str              # canonical KE key
    x: float
    y: float
    lane: Optional[str] = None       # KE level lane the node sits in
    group: Optional[str] = None      # optional curator-defined cluster
    pinned: bool = False             # True = never move this node again
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "KE_LEVEL_ORDER",
    "KE_LEVEL_INDEX",
    "KER_ADJACENCY_VALUES",
    "PAPER_TYPE_VALUES",
    "STUDY_DESIGN_VALUES",
    "SEX_VALUES",
    "CONFIDENCE_VALUES",
    "CURATION_STATUS_VALUES",
    "level_index",
    "PubMedRecord",
    "ScreeningDecision",
    "PageText",
    "Chunk",
    "PaperDocument",
    "EvidenceSpan",
    "KERExtraction",
    "Table1Row",
    "OntologyMatch",
    "CanonicalKE",
    "PhysioMapLink",
    "CurationRecord",
    "LayoutPosition",
]
