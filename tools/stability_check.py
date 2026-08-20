#!/usr/bin/env python
"""
Determinism check for the post-extraction half of the pipeline.

Costs nothing and makes no model calls. It takes the Table 1 rows already in a
database, feeds them through the deterministic stages many times in different
input orders, and reports whether the answer changes.

Why input order. Every stage downstream of extraction is supposed to be a pure
function of the row set: the same 63 rows should produce the same Table 2, the
same per-edge paper counts and the same synthesis prompt, no matter what order
they arrive in. Row order is not stable in practice — `record_id` is assigned in
upload order, so re-uploading the same corpus in a different sequence permutes
it, and so does re-extracting after one paper failed and was retried. If any
stage is order-sensitive, that is a source of run-to-run variation with no model
in it at all, and it has to be ruled out before any variance is blamed on
sampling.

What it checks, per permutation:

    table2_raw          Table 2 grouped on literal extracted strings
    table2_normalized   Table 2 grouped on canonical KE identity
    edge_counts         per edge: supporting rows, supporting distinct DOIs,
                        confidence score, confidence band
    synthesis_input     the exact prompt block `evidence_synthesis` would build
                        for the target edge, byte for byte

Each is hashed and compared against the first permutation. Any difference is
reported with a diff.

It also reports, separately from the determinism verdict, every edge where the
row count and the distinct-DOI count disagree — those are the edges where
`n_papers` overstates the number of papers, which is stable within a run but
moves between runs as the extractor splits papers into different numbers of
rows.

Usage:
    python tools/stability_check.py --db aop_rag.db
    python tools/stability_check.py --db aop_rag.db --permutations 50
    python tools/stability_check.py --db aop_rag.db --edge "voltage-gated sodium channel"

Exit code is 0 when every stage was order-independent, 1 otherwise.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Import the real pipeline modules rather than reimplementing them — the point
# is to test what the app runs, not a copy of it that could drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2_extraction import evidence_synthesis, table2_synthesis  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_table1(db_path: Path) -> pd.DataFrame:
    """Table 1 exactly as `table1_store.load_table1_as_dataframe` returns it."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM table1_extractions ORDER BY record_id", conn
        )


def canonical_names(db_path: Path) -> dict[int, str]:
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT canonical_id, canonical_name FROM ke_canonical"
            ).fetchall()
        except sqlite3.Error:
            return {}
    return {int(a): str(b) for a, b in rows}


# ---------------------------------------------------------------------------
# The stages under test
# ---------------------------------------------------------------------------

def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _null_safe(value: Any) -> Any:
    """
    Collapse every spelling of "missing" to one value before comparing.

    pandas builds a column of all-None as object dtype in one permutation and
    float64 NaN in another purely by construction order, and `str()` renders
    those as "None" and "nan". Comparing raw would report every empty column as
    order-sensitive, which is a fact about dtype inference and not about the
    pipeline. This is the difference between a report that finds one real
    problem and one that cries wolf on thirty columns.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    return "" if text.strip().lower() in ("nan", "none", "null", "<na>") else text


def _edge_counts(table2: pd.DataFrame) -> list[dict]:
    """
    Per-edge summary, sorted so the comparison is about content not row order.

    `n_supporting_rows` is what the pipeline currently reports as
    `n_papers_supporting`. `n_supporting_dois` is what it would report if it
    counted papers. Both are recorded so the gap is visible.
    """
    if table2.empty:
        return []

    out: list[dict] = []
    for _, row in table2.iterrows():
        dois = _null_safe(row.get("supporting_dois"))
        n_dois = len([d for d in dois.split(";") if d.strip()])
        n_rows_support = row.get("n_rows_supporting")
        out.append(
            {
                "ker_key": row.get("ker_key"),
                "upstream": row.get("upstream_ke_name"),
                "downstream": row.get("downstream_ke_name"),
                "n_supporting_papers": int(row.get("n_papers_supporting") or 0),
                "n_supporting_rows": int(
                    n_rows_support
                    if n_rows_support is not None and not pd.isna(n_rows_support)
                    else row.get("n_papers_supporting") or 0
                ),
                "n_supporting_dois": n_dois,
                "n_contradicting": int(row.get("n_papers_contradicting") or 0),
                "direction": row.get("direction"),
                "evidence_type": row.get("evidence_type"),
                "confidence_score": row.get("confidence_score"),
                "confidence_band": row.get("confidence_band"),
                "taxonomic_evidence_level": row.get("taxonomic_evidence_level"),
            }
        )
    return sorted(out, key=lambda d: str(d["ker_key"]))


def _table2_signature(table2: pd.DataFrame) -> list[dict]:
    """Whole Table 2, key-sorted, for an exact content comparison."""
    if table2.empty:
        return []
    records = []
    for rec in table2.to_dict("records"):
        rec.pop("last_updated", None)  # a timestamp, not a result
        records.append({k: _null_safe(v) for k, v in rec.items()})
    return sorted(records, key=lambda r: str(r.get("ker_key")))


def _target_edge_key(table2: pd.DataFrame, needle: Optional[str]) -> Optional[str]:
    """The ker_key of the edge with the most supporting rows, or a named one."""
    if table2.empty:
        return None
    if needle:
        lowered = needle.lower()
        hits = table2[
            table2["upstream_ke_name"].astype(str).str.lower().str.contains(lowered)
            | table2["downstream_ke_name"].astype(str).str.lower().str.contains(lowered)
        ]
        if not hits.empty:
            best = hits.sort_values("n_papers_total", ascending=False).iloc[0]
            return str(best["ker_key"])
    best = table2.sort_values("n_papers_total", ascending=False).iloc[0]
    return str(best["ker_key"])


def _synthesis_input(
    table1: pd.DataFrame, table2: pd.DataFrame, ker_key: Optional[str]
) -> str:
    """
    The prompt block `synthesise_ker` would send for one edge.

    Built through the real `build_synthesis_input` so any order sensitivity in
    the prompt itself — block numbering, DOI sequence — shows up here.
    """
    if not ker_key or table2.empty:
        return ""
    match = table2[table2["ker_key"] == ker_key]
    if match.empty:
        return ""
    record_ids = [
        int(r)
        for r in str(match.iloc[0].get("record_ids") or "").split(",")
        if r.strip().isdigit()
    ]
    if not record_ids:
        return ""
    rows = table1[table1["record_id"].isin(record_ids)]
    return evidence_synthesis.build_synthesis_input(rows)


def run_stages(
    table1: pd.DataFrame, target_needle: Optional[str]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run every deterministic stage over one ordering of the rows."""
    t2_raw = table2_synthesis.compute_table2_raw(table1)
    t2_norm = table2_synthesis.compute_table2(table1, normalized=True)
    key = _target_edge_key(t2_norm, target_needle)

    return (
        {
            "table2_raw": _table2_signature(t2_raw),
            "table2_normalized": _table2_signature(t2_norm),
            "edge_counts": _edge_counts(t2_norm),
            "synthesis_input": _synthesis_input(table1, t2_norm, key),
        },
        t2_norm,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def field_drift(
    first: list[dict], other: list[dict]
) -> dict[str, tuple[int, str, str]]:
    """
    Which Table 2 fields changed, and on how many edges.

    Reported per field rather than per edge because the fix is per field: a
    field that reorders needs a sort, a field that changes content is dropping
    different material under a `limit=`. Those are different bugs and the
    summary should not blur them.
    """
    by_key = {str(r.get("ker_key")): r for r in other}
    drift: dict[str, tuple[int, str, str]] = {}
    for rec in first:
        mate = by_key.get(str(rec.get("ker_key")))
        if mate is None:
            continue
        for field, value in rec.items():
            if field in ("ker_key", "record_ids"):
                continue
            if str(value) == str(mate.get(field)):
                continue
            count, a, b = drift.get(field, (0, str(value), str(mate.get(field))))
            drift[field] = (count + 1, a, b)
    return drift


def _classify_drift(a: str, b: str) -> str:
    """Whether a changed field reordered its content or replaced it."""
    parts_a = sorted(p.strip() for p in a.split(";") if p.strip())
    parts_b = sorted(p.strip() for p in b.split(";") if p.strip())
    if parts_a and parts_a == parts_b:
        return "REORDERED (same content, different sequence)"
    if parts_a and parts_b and set(parts_a) & set(parts_b):
        return "TRUNCATED (a `limit=` kept a different subset)"
    return "REPLACED (different value chosen from the group)"


def _show_text_diff(name: str, first: str, other: str, perm: int) -> None:
    diff = list(
        difflib.unified_diff(
            first.splitlines(),
            other.splitlines(),
            fromfile="permutation 0",
            tofile=f"permutation {perm}",
            lineterm="",
            n=1,
        )
    )
    print(f"\n  {name} differs at permutation {perm}:")
    for line in diff[:40]:
        print(f"    {line}")
    if len(diff) > 40:
        print(f"    … {len(diff) - 40} more diff lines")


def _show_struct_diff(name: str, first: Any, other: Any, perm: int) -> None:
    a = json.dumps(first, indent=2, sort_keys=True, default=str).splitlines()
    b = json.dumps(other, indent=2, sort_keys=True, default=str).splitlines()
    diff = list(
        difflib.unified_diff(
            a, b, fromfile="permutation 0", tofile=f"permutation {perm}",
            lineterm="", n=1,
        )
    )
    print(f"\n  {name} differs at permutation {perm}:")
    for line in diff[:40]:
        print(f"    {line}")
    if len(diff) > 40:
        print(f"    … {len(diff) - 40} more diff lines")


def report_row_vs_paper(t2: pd.DataFrame) -> int:
    """
    Edges where the reported supporting count is not a count of papers.

    Not a determinism failure — it is stable within a run. It is reported here
    because it is the mechanism by which a stable pipeline still produces a
    different number next time: the count tracks how many rows the extractor
    happened to split a paper into, which is exactly what varies.
    """
    counts = _edge_counts(t2)
    split = [c for c in counts if c["n_supporting_rows"] != c["n_supporting_papers"]]
    print("\n" + "=" * 78)
    print("ROWS vs PAPERS")
    print("=" * 78)

    mismatched = [c for c in counts if c["n_supporting_papers"] != c["n_supporting_dois"]]
    if mismatched:
        print(f"  WARNING: {len(mismatched)} edge(s) where n_papers_supporting does")
        print("  not match the distinct supporting DOIs. The counter is wrong.")
        for c in mismatched[:5]:
            print(
                f"    reports {c['n_supporting_papers']}, DOIs say "
                f"{c['n_supporting_dois']}   {c['upstream']} -> {c['downstream']}"
            )

    if not split:
        print("  Every edge has one row per paper, so the two counts coincide here.")
        print("  The distinction still matters: it is what stops a paper split into")
        print("  several claims from reading as several papers.")
        return 0

    print(f"  {len(split)} edge(s) where papers contributed more than one claim:\n")
    for c in split:
        print(
            f"    {c['n_supporting_rows']} claim(s) from {c['n_supporting_papers']} "
            f"paper(s)   {c['upstream']} -> {c['downstream']}"
        )
    print(
        "\n  Before the counting fix these edges reported the claim count as a"
        "\n  paper count, so their support moved whenever the extractor split a"
        "\n  paper differently. They are now reported separately."
    )
    return len(split)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="aop_rag.db", type=Path)
    parser.add_argument("--permutations", type=int, default=25)
    parser.add_argument(
        "--edge",
        default=None,
        help="substring of the KE name whose synthesis prompt to compare; "
             "defaults to the edge with the most contributing rows",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No such database: {args.db}")
        return 2

    table1 = load_table1(args.db)
    if table1.empty:
        print("Table 1 is empty — nothing to check.")
        return 2

    n_canon = int(table1["upstream_ke_canonical_id"].notna().sum())
    print("=" * 78)
    print("DETERMINISM CHECK — post-extraction stages")
    print("=" * 78)
    print(f"  database          {args.db}")
    print(f"  Table 1 rows      {len(table1)}")
    print(f"  distinct DOIs     {table1['source_doi'].nunique()}")
    print(f"  rows normalized   {n_canon} / {len(table1)}")
    if n_canon == 0:
        print("  NOTE: no canonical ids are set, so the normalized view falls")
        print("        back to grouping on lowercased raw labels.")
    print(f"  permutations      {args.permutations}")

    rng = random.Random(args.seed)
    baseline: Optional[dict[str, Any]] = None
    baseline_t2: Optional[pd.DataFrame] = None
    failures: dict[str, int] = {}
    drift_totals: dict[str, dict[str, tuple[int, str, str]]] = {}

    for i in range(args.permutations):
        if i == 0:
            shuffled = table1.copy()
        else:
            order = list(range(len(table1)))
            rng.shuffle(order)
            shuffled = table1.iloc[order].reset_index(drop=True)

        result, t2 = run_stages(shuffled, args.edge)

        if baseline is None:
            baseline, baseline_t2 = result, t2
            continue

        for stage, value in result.items():
            if _digest(value) == _digest(baseline[stage]):
                continue
            failures[stage] = failures.get(stage, 0) + 1
            if stage in ("table2_raw", "table2_normalized"):
                for field, (n, a, b) in field_drift(baseline[stage], value).items():
                    prev = drift_totals.setdefault(stage, {})
                    seen_n, seen_a, seen_b = prev.get(field, (0, a, b))
                    prev[field] = (max(seen_n, n), seen_a, seen_b)
            if failures[stage] == 1 and stage == "synthesis_input":
                _show_text_diff(stage, baseline[stage], value, i)

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    stages = ["table2_raw", "table2_normalized", "edge_counts", "synthesis_input"]
    for stage in stages:
        n = failures.get(stage, 0)
        mark = "PASS" if n == 0 else "FAIL"
        detail = "order-independent" if n == 0 else f"differed in {n} permutation(s)"
        print(f"  [{mark}] {stage:22s} {detail}")

    for stage, fields in drift_totals.items():
        if not fields:
            continue
        print(f"\n  fields that moved in {stage}:")
        for field, (n, a, b) in sorted(fields.items(), key=lambda kv: -kv[1][0]):
            print(f"    {n:3d} edge(s)  {field:32s} {_classify_drift(a, b)}")

    n_inflated = report_row_vs_paper(baseline_t2) if baseline_t2 is not None else 0

    print("\n" + "=" * 78)
    if failures:
        print("Order sensitivity found. Some run-to-run variation is NOT the model.")
    else:
        print("Every post-extraction stage is a pure function of the row set.")
        print("Run-to-run variation therefore originates at or before extraction:")
        print("  which papers pass the gate, and which events each paper's chain names.")
    if n_inflated:
        print(f"Separately: {n_inflated} edge(s) report rows where they say papers.")
    print("=" * 78)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
