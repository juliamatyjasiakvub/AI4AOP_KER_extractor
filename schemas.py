from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


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
