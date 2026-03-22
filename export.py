from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from schemas import ScreenedRecord

EXPORT_COLUMNS = [
    "screening_decision",
    "rationale",
    "evidence_quote",
    "triggered_inclusion_rule",
    "triggered_exclusion_rule",
    "PMID",
    "DOI",
    "first_author",
    "year",
    "journal",
    "title",
    "abstract",
    "query_used",
]


def screened_records_to_dataframe(records: list[ScreenedRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "screening_decision": record.screening_decision,
                "rationale": record.rationale,
                "evidence_quote": record.evidence_quote,
                "triggered_inclusion_rule": record.triggered_inclusion_rule,
                "triggered_exclusion_rule": record.triggered_exclusion_rule,
                "PMID": record.pmid,
                "DOI": record.doi,
                "first_author": record.first_author,
                "year": record.year,
                "journal": record.journal,
                "title": record.title,
                "abstract": record.abstract,
                "query_used": record.query_used,
            }
        )
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def write_screened_csv(records: list[ScreenedRecord], output_path: str | Path) -> Path:
    df = screened_records_to_dataframe(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
