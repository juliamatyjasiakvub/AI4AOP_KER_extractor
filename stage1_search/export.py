from __future__ import annotations

import io
from typing import Iterable, Optional

import pandas as pd

from schemas import PubMedRecord, ScreeningDecision


def build_export_rows(
    screened_pairs: Iterable[tuple[PubMedRecord, ScreeningDecision]],
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
) -> list[dict]:
    rows: list[dict] = []
    for record, decision in screened_pairs:
        rows.append(
            {
                "query": query,
                "inclusion_criteria": inclusion_criteria or "",
                "exclusion_criteria": exclusion_criteria or "",
                "screening_decision": decision.decision,
                "rationale": decision.rationale,
                "evidence_quote": decision.evidence_quote or "",
                "triggered_inclusion_rule": decision.triggered_inclusion_rule or "",
                "triggered_exclusion_rule": decision.triggered_exclusion_rule or "",
                "PMID": record.pmid,
                "DOI": record.doi or "",
                "first_author": record.first_author or "",
                "year": record.year or "",
                "journal": record.journal or "",
                "title": record.title,
                "abstract": record.abstract,
            }
        )
    return rows


def build_export_dataframe(
    screened_pairs: Iterable[tuple[PubMedRecord, ScreeningDecision]],
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        build_export_rows(
            screened_pairs,
            query=query,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
        )
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
