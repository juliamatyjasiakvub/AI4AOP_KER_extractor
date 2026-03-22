from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DecisionLabel = Literal["yes", "no", "maybe"]


@dataclass
class PubMedRecord:
    pmid: str
    doi: str | None
    first_author: str | None
    journal: str | None
    year: int | None
    title: str
    abstract: str
    query_used: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningDecision:
    decision: DecisionLabel
    rationale: str
    triggered_inclusion_rule: str | None
    triggered_exclusion_rule: str | None
    evidence_quote: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScreenedRecord:
    pmid: str
    doi: str | None
    first_author: str | None
    journal: str | None
    year: int | None
    title: str
    abstract: str
    query_used: str
    screening_decision: DecisionLabel
    rationale: str
    triggered_inclusion_rule: str | None
    triggered_exclusion_rule: str | None
    evidence_quote: str | None

    @classmethod
    def from_parts(cls, record: PubMedRecord, decision: ScreeningDecision) -> "ScreenedRecord":
        return cls(
            pmid=record.pmid,
            doi=record.doi,
            first_author=record.first_author,
            journal=record.journal,
            year=record.year,
            title=record.title,
            abstract=record.abstract,
            query_used=record.query_used,
            screening_decision=decision.decision,
            rationale=decision.rationale,
            triggered_inclusion_rule=decision.triggered_inclusion_rule,
            triggered_exclusion_rule=decision.triggered_exclusion_rule,
            evidence_quote=decision.evidence_quote,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
