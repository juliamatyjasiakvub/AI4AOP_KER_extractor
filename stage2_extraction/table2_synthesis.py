from __future__ import annotations

"""
Compute Table 2 (KER-level synthesis) from all Table 1 rows.

Table 2 is NOT stored persistently — it is recomputed on demand from the
current Table 1 contents. This means it is always up to date when a new
paper is added, and there is no sync problem between the two tables.

The uncertainty_level thresholds follow the AOP-Wiki Developer's Handbook:
  Low      : 0 contradicting papers
  Moderate : 1 to 25 % of papers contradict
  High     : > 25 % of papers contradict
"""

from typing import Optional

import pandas as pd


def _uncertainty_level(n_total: int, n_contra: int) -> str:
    if n_total == 0:
        return "Low"
    if n_contra == 0:
        return "Low"
    pct = n_contra / n_total
    if pct <= 0.25:
        return "Moderate"
    return "High"


def _join_unique(series: pd.Series, sep: str = "; ") -> Optional[str]:
    """Combine non-null values from a Series, deduplicated."""
    parts: list[str] = []
    seen: set[str] = set()
    for val in series:
        if pd.isna(val) or not str(val).strip():
            continue
        for item in str(val).split(";"):
            item = item.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower())
                parts.append(item)
    return sep.join(parts) if parts else None


def _first_non_null(series: pd.Series) -> Optional[str]:
    for val in series:
        if not pd.isna(val) and str(val).strip():
            return str(val).strip()
    return None


def _ker_key(row: pd.Series) -> str:
    """Canonical join key for grouping rows into the same KER."""
    uid = row.get("upstream_ke_id")
    did = row.get("downstream_ke_id")
    if pd.notna(uid) and pd.notna(did):
        return f"{int(uid)}_{int(did)}"
    # Fall back to name-based key (lowercase, stripped)
    u = str(row.get("upstream_ke_name", "")).strip().lower()
    d = str(row.get("downstream_ke_name", "")).strip().lower()
    return f"name::{u}::{d}"


def compute_table2(table1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Table 1 rows into Table 2 KER-level summary rows.

    Parameters
    ----------
    table1_df : pd.DataFrame
        Output of table1_store.load_table1_as_dataframe().

    Returns
    -------
    pd.DataFrame
        One row per unique KER, with aggregated evidence fields.
        Returns empty DataFrame if table1_df is empty.
    """
    if table1_df.empty:
        return pd.DataFrame()

    df = table1_df.copy()
    df["ker_key"] = df.apply(_ker_key, axis=1)
    df["contradicts_ker"] = df["contradicts_ker"].astype(bool)

    rows: list[dict] = []

    for ker_key, group in df.groupby("ker_key"):
        n_total = len(group)
        n_contra = int(group["contradicts_ker"].sum())
        n_support = n_total - n_contra

        row: dict = {
            # Identity
            "ker_key": ker_key,
            "ker_id": _first_non_null(group["ker_id"].astype(str)),
            "ker_name": _first_non_null(group["ker_name"]),
            "upstream_ke_name": _first_non_null(group["upstream_ke_name"]),
            "upstream_ke_id": _first_non_null(group["upstream_ke_id"].astype(str)),
            "downstream_ke_name": _first_non_null(group["downstream_ke_name"]),
            "downstream_ke_id": _first_non_null(group["downstream_ke_id"].astype(str)),
            "aop_id": _join_unique(group["aop_id"]),
            "aop_status": "existing" if any(group["aop_status"] == "existing") else "novel",

            # Evidence counts
            "n_papers_total": n_total,
            "n_papers_supporting": n_support,
            "n_papers_contradicting": n_contra,
            "uncertainty_level": _uncertainty_level(n_total, n_contra),

            # Applicability (union across all papers)
            "all_taxa": _join_unique(group["taxonomic_applicability"]),
            "sex_applicability": _join_unique(group["sex_applicability"]),
            "life_stage_applicability": _join_unique(group["life_stage_applicability"]),

            # Evidence level thresholds (0-3 Low, 4-8 Moderate, 9+ High)
            "taxonomic_evidence_level": (
                "Low" if n_total <= 3 else "Moderate" if n_total <= 8 else "High"
            ),
            "sex_evidence_level": (
                "Low" if n_total <= 3 else "Moderate" if n_total <= 8 else "High"
            ),
            "life_stage_evidence_level": (
                "Low" if n_total <= 3 else "Moderate" if n_total <= 8 else "High"
            ),

            # Quantitative (best available — take first non-null; human will refine)
            "quantitative_relationships": _first_non_null(group["quantitative_relationships"]),
            "response_response_relationship": _first_non_null(group["response_response_relationship"]),
            "time_scale": _first_non_null(group["time_scale"]),
            "modulating_factors": _join_unique(group["modulating_factors"]),
            "feedforward_feedback_loops": _join_unique(group["feedforward_feedback_loops"]),

            # Provenance
            "all_source_dois": _join_unique(group["source_doi"]),
            "all_cited_dois": _join_unique(group["cited_evidence_dois"]),
            "last_updated": group["extraction_date"].max(),

            # Human-review fields (empty until reviewer fills them in)
            "uncertainty_description": None,
            "biological_plausibility_synthesis": None,
            "review_status": "Draft",
        }
        rows.append(row)

    return pd.DataFrame(rows)
