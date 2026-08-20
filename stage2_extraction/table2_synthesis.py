from __future__ import annotations

"""
Table 2 — cross-paper KER synthesis.

Table 2 is NOT stored persistently: it is recomputed on demand from the current
Table 1 contents, so it is always consistent with the underlying extractions and
there is no sync problem between the two.

Two views are produced, and keeping them separate is the point:

    compute_table2_raw(table1_df)
        Groups strictly on the literal KE strings each paper used. This is the
        direct-extraction view — what the papers actually said, before any
        interpretation. Use it to audit the extractor.

    compute_table2(table1_df, ...)
        Groups on canonical KE identity, so equivalent labels collapse into a
        single edge carrying the union of the evidence. This is the synthesis
        view — what the literature collectively supports.

Duplicate KER consolidation happens in the normalized view: every paper that
reports the same canonical upstream→downstream pair contributes to ONE edge,
with supporting and contradicting counts aggregated and the full provenance
rolled up so the edge can be inspected paper by paper.

Uncertainty thresholds follow the AOP-Wiki Developer's Handbook:
    Low      : 0 contradicting papers
    Moderate : 1 to 25 % of papers contradict
    High     : > 25 % of papers contradict
"""

from typing import Any, Optional, Sequence

from collections import Counter, defaultdict

import pandas as pd

from schemas import KE_LEVEL_ORDER, level_index

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _evidence_level(n_papers: int) -> str:
    """Applicability-domain coverage band, from the number of papers."""
    if n_papers <= 3:
        return "Low"
    return "Moderate" if n_papers <= 8 else "High"


def _uncertainty_level(n_total: int, n_contra: int) -> str:
    if n_total == 0 or n_contra == 0:
        return "Low"
    pct = n_contra / n_total
    if pct <= 0.25:
        return "Moderate"
    return "High"


#: Strings that mean "missing" once a value has been through `astype(str)`.
#: pandas turns a NaN float into the literal text "nan", which is not null any
#: more — `pd.isna("nan")` is False — so a missing id sails through every
#: null check and only fails later, at `int("nan")`, in whatever code trusted
#: the column. Catching it at the point of stringification is the fix; catching
#: it at each use site is whack-a-mole.
_NULL_STRINGS = {"nan", "nat", "none", "null", "<na>", ""}


def _is_missing(value: Any) -> bool:
    """True for real nulls and for the strings pandas leaves behind."""
    try:
        if value is None or pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _NULL_STRINGS


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int, tolerating floats, numeric strings and the above."""
    if _is_missing(value):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation helpers
#
# Every one of these collapses many Table 1 rows into one Table 2 value, and
# every one of them used to answer "whichever row I saw first". Row order is
# `record_id` order, which is upload order, so the same evidence uploaded in a
# different sequence produced a different Table 2 — differently ordered joined
# strings, and, wherever a `limit=` applied, silently *different content*,
# because the truncation kept whichever items happened to arrive first.
#
# That is a reproducibility failure with no model in it. These now order by
# consensus — how many rows said it — and break every tie deterministically, so
# the result is a function of the row SET rather than of the row SEQUENCE.
# `tools/stability_check.py` is the regression test.
# ---------------------------------------------------------------------------

def _rank_items(items: list[str]) -> list[str]:
    """
    Order distinct strings by support, then deterministically.

    Descending count first, so that when a `limit=` truncates the list it keeps
    what most papers actually said rather than what arrived first. Ties fall to
    the longer string — between two renderings of one fact the fuller one
    carries more information — and then to lexicographic order, which is
    arbitrary but total, so no case is left where two runs can disagree.
    """
    counts: dict[str, int] = {}
    spellings: dict[str, set[str]] = {}
    for item in items:
        key = item.lower()
        counts[key] = counts.get(key, 0) + 1
        spellings.setdefault(key, set()).add(item)

    # Which spelling survives must not depend on which row was read first
    # either, so it is chosen the same way: most characters, then alphabetical.
    canonical = {
        key: sorted(variants, key=lambda s: (-len(s), s))[0]
        for key, variants in spellings.items()
    }
    ordered = sorted(
        counts, key=lambda k: (-counts[k], -len(canonical[k]), canonical[k])
    )
    return [canonical[k] for k in ordered]


def _join_unique(series: pd.Series, sep: str = "; ", limit: Optional[int] = None) -> Optional[str]:
    """
    Combine non-null values from a Series, deduplicated and consensus-ordered.

    Most-reported first, so that where `limit` drops a tail it drops the least
    corroborated material — defensible, and unlike the previous behaviour, the
    same on every run.
    """
    items: list[str] = []
    for val in series:
        if _is_missing(val):
            continue
        for item in str(val).split(";"):
            item = item.strip()
            if item:
                items.append(item)
    if not items:
        return None
    parts = _rank_items(items)
    if limit is not None:
        parts = parts[:limit]
    return sep.join(parts) if parts else None


def _consensus_value(series: pd.Series) -> Optional[str]:
    """
    The value most rows agree on, chosen without reference to row order.

    Replaces a `_first_non_null` that returned whichever row sorted earliest by
    `record_id`. For an id column every row usually agrees and the answer is
    unchanged; for a prose column such as `ker_description` this is the
    difference between "one paper's sentence, picked by upload order" and "the
    sentence the most papers gave, with a stated tie-break".
    """
    values = [str(v).strip() for v in series if not _is_missing(v) and str(v).strip()]
    if not values:
        return None
    return _rank_items(values)[0]


#: Retained under its old name because the intent at several call sites really
#: is "any non-null value will do" — but it must still not depend on arrival
#: order, so it resolves the same way as everything else here.
_first_non_null = _consensus_value


def _mode_or_first(series: pd.Series, fallback: str) -> str:
    """Most common non-null value, breaking ties toward the upstream level."""
    values = [str(v).strip() for v in series if pd.notna(v) and str(v).strip()]
    if not values:
        return fallback
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.values())
    tied = [v for v, n in counts.items() if n == best]
    if len(tied) == 1:
        return tied[0]
    known = [v for v in tied if v in KE_LEVEL_ORDER]
    # `tied[0]` was insertion order — the last place row sequence could still
    # decide a biological level. Sorting makes the fallback total.
    return min(known, key=level_index) if known else min(tied)


#: Numeric weights used to turn an evidence profile into a single confidence
#: score. Essentiality evidence (knockout, antagonist) is the strongest single
#: signal in AOP weight-of-evidence practice, so it is weighted accordingly.
_CONFIDENCE_WEIGHTS = {
    "papers": 0.30,
    "essentiality": 0.20,
    "quantitative": 0.15,
    "provenance": 0.15,
    "extraction_confidence": 0.10,
    "aopwiki": 0.10,
}

_CONFIDENCE_LEVEL_MAP = {"High": 1.0, "Medium": 0.6, "Low": 0.25}


def _confidence_score(
    n_support: int,
    n_contra: int,
    has_essentiality: bool,
    has_quantitative: bool,
    verified_ratio: float,
    mean_extraction_confidence: float,
    in_aopwiki: bool,
    sign_conflict: bool = False,
    unsigned_ratio: float = 0.0,
    evidence_type: str = "not_stated",
) -> float:
    """
    Blend the evidence profile of a consolidated KER into a 0-1 score.

    This is a transparent heuristic, not a statistic. It exists so edges can be
    ranked and filtered consistently; the underlying counts remain visible in
    the evidence panel so a curator never has to trust the number blindly.

    Sign disagreement is treated as the serious defect it is rather than as
    extra support. Four papers reporting an edge positive and two reporting it
    negative previously scored as six supporting papers; the counts said
    "well evidenced" while the biology said "two different findings filed
    together". A conflicted edge is now halved, and an edge nobody signed is
    discounted in proportion — an unsigned link cannot be relied on for
    direction, which is most of what a relationship asserts.
    """
    total = n_support + n_contra
    # Support saturates: the fifth confirming paper adds less than the second.
    paper_term = min(1.0, n_support / 5.0)
    if total:
        paper_term *= 1.0 - 0.6 * (n_contra / total)

    score = (
        _CONFIDENCE_WEIGHTS["papers"] * paper_term
        + _CONFIDENCE_WEIGHTS["essentiality"] * (1.0 if has_essentiality else 0.0)
        + _CONFIDENCE_WEIGHTS["quantitative"] * (1.0 if has_quantitative else 0.0)
        + _CONFIDENCE_WEIGHTS["provenance"] * max(0.0, min(1.0, verified_ratio))
        + _CONFIDENCE_WEIGHTS["extraction_confidence"] * max(0.0, min(1.0, mean_extraction_confidence))
        + _CONFIDENCE_WEIGHTS["aopwiki"] * (1.0 if in_aopwiki else 0.0)
    )

    if sign_conflict:
        score *= 0.5
    score *= 1.0 - 0.25 * max(0.0, min(1.0, unsigned_ratio))

    # What kind of experiment established the direction matters more than how
    # many papers repeated it. Five papers observing two things decline
    # together is five correlations, and the count alone was scoring that
    # above one knockout.
    score *= {
        "rescue": 1.15,
        "perturbation": 1.0,
        "common_stressor": 0.85,
        "correlation": 0.7,
        "reverse_only": 0.4,
        "not_stated": 0.6,
    }.get(evidence_type, 0.6)

    return round(max(0.0, min(1.0, score)), 3)


def confidence_band(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.36:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Grouping keys
# ---------------------------------------------------------------------------

def _raw_ker_key(row: pd.Series) -> str:
    """Join key for the RAW view: literal strings only, no interpretation."""
    u = str(row.get("upstream_ke_name", "")).strip().lower()
    d = str(row.get("downstream_ke_name", "")).strip().lower()
    return f"raw::{u}::{d}"


def _paper_identity(group: pd.DataFrame) -> list[str]:
    """
    One stable identifier per contributing paper, for counting distinct papers.

    DOI where there is one. Two papers that both lack a DOI must not collapse
    into a single "paper", and two rows of the SAME DOI-less paper should not
    count as two — so the fallback is the filename, and only a row with neither
    falls back to its own record id, which counts it as its own paper. That is
    the conservative direction: it can overcount an unidentifiable row, never
    silently merge two real papers.
    """
    identities: list[str] = []
    for _, row in group.iterrows():
        doi = row.get("source_doi")
        if not _is_missing(doi):
            identities.append(f"doi:{str(doi).strip().lower()}")
            continue
        filename = row.get("source_filename")
        if not _is_missing(filename):
            identities.append(f"file:{str(filename).strip().lower()}")
            continue
        identities.append(f"record:{row.get('record_id')}")
    return identities


def _canonical_ker_key(row: pd.Series) -> str:
    """
    Join key for the NORMALIZED view.

    Prefers canonical KE ids, falls back to AOP-Wiki KE ids, and finally to
    lowercased names, so the function still behaves sensibly before
    normalization has been run.
    """
    up_canon = _as_int(row.get("upstream_ke_canonical_id"))
    down_canon = _as_int(row.get("downstream_ke_canonical_id"))
    if up_canon is not None and down_canon is not None:
        return f"canon::{up_canon}::{down_canon}"

    uid, did = row.get("upstream_ke_id"), row.get("downstream_ke_id")
    if pd.notna(uid) and pd.notna(did):
        return f"wiki::{int(uid)}::{int(did)}"

    u = str(row.get("upstream_ke_canonical") or row.get("upstream_ke_name", "")).strip().lower()
    d = str(row.get("downstream_ke_canonical") or row.get("downstream_ke_name", "")).strip().lower()
    return f"name::{u}::{d}"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def _aggregate_group(
    ker_key: str,
    group: pd.DataFrame,
    *,
    normalized: bool,
    spans_by_record: Optional[dict[int, list[dict]]] = None,
) -> dict:
    # --- Rows and papers are different quantities ---------------------------
    # `n_papers_*` used to be `len(group)` — a count of Table 1 ROWS. One paper
    # contributing two rows to the same edge counted as two supporting papers,
    # and how many rows a paper yields for a given pair is exactly what varies
    # between extraction runs. So the reported support moved without any new
    # evidence, and the confidence score and evidence-level bands moved with
    # it. Both quantities are now computed and both are reported: papers for
    # anything that claims to be about papers, rows kept alongside so the
    # extractor can still be audited.
    n_rows = len(group)
    n_rows_contra = int(group["contradicts_ker"].sum())
    n_rows_support = n_rows - n_rows_contra

    paper_ids = _paper_identity(group)
    contradicts_flags = [bool(v) for v in group["contradicts_ker"].tolist()]
    all_papers = set(paper_ids)
    contradicting_papers = {
        paper for paper, flag in zip(paper_ids, contradicts_flags) if flag
    }
    # A paper that contradicts anywhere is filed as contradicting, so the two
    # sets are disjoint and sum to the total. Counting it on both sides would
    # let supporting + contradicting exceed the number of papers, which makes
    # the contradicting fraction in `_confidence_score` incoherent.
    n_total = len(all_papers)
    n_contra = len(contradicting_papers)
    n_support = n_total - n_contra

    if normalized:
        upstream_name = (
            _first_non_null(group["upstream_ke_canonical"])
            if "upstream_ke_canonical" in group.columns
            else None
        ) or _first_non_null(group["upstream_ke_name"])
        downstream_name = (
            _first_non_null(group["downstream_ke_canonical"])
            if "downstream_ke_canonical" in group.columns
            else None
        ) or _first_non_null(group["downstream_ke_name"])
        upstream_level = _mode_or_first(
            group.get("upstream_ke_canonical_level", group["upstream_ke_level"]), "Molecular"
        )
        downstream_level = _mode_or_first(
            group.get("downstream_ke_canonical_level", group["downstream_ke_level"]), "Molecular"
        )
    else:
        upstream_name = _first_non_null(group["upstream_ke_name"])
        downstream_name = _first_non_null(group["downstream_ke_name"])
        upstream_level = _mode_or_first(group["upstream_ke_level"], "Molecular")
        downstream_level = _mode_or_first(group["downstream_ke_level"], "Molecular")

    ker_name = _first_non_null(group["ker_name"])
    if not ker_name and upstream_name and downstream_name:
        ker_name = f"{upstream_name} leads to {downstream_name}"

    # Adjacency: an edge is adjacent if ANY paper called it adjacent AND the
    # two levels are neighbouring or equal. A single "Non-adjacent" vote from
    # one paper should not hide a link that every other paper treats as direct.
    adjacency_votes = [
        str(v).strip() for v in group.get("ker_adjacency", []) if pd.notna(v)
    ]
    level_gap = abs(level_index(downstream_level) - level_index(upstream_level))
    voted_adjacent = adjacency_votes.count("Adjacent") >= max(1, len(adjacency_votes) // 2)
    adjacency = "Adjacent" if (voted_adjacent and level_gap <= 1) else "Non-adjacent"

    essentiality = _join_unique(group["essentiality_evidence"], limit=5)
    quantitative = _first_non_null(group["quantitative_relationships"])

    # --- Sign ---------------------------------------------------------------
    # An edge that some papers report as positive and others as negative is
    # not a well-supported edge with a lot of papers behind it; it is two
    # findings that have been filed as one. Before v8 the sign was not stored
    # at all, so those rows aggregated into a single confident-looking KER.
    signs = [
        str(v).strip().lower()
        for v in group.get("direction", [])
        if pd.notna(v) and str(v).strip()
    ]
    n_positive = signs.count("positive")
    n_negative = signs.count("negative")
    n_unsigned = len(group) - n_positive - n_negative

    if n_positive and n_negative:
        direction = "conflicting"
    elif n_positive:
        direction = "positive"
    elif n_negative:
        direction = "negative"
    else:
        direction = "unclear"

    # --- How the causation was established ---------------------------------
    # The strongest evidence any paper offers, not an average: one knockout
    # settles a direction that ten correlations cannot.
    _EVIDENCE_RANK = {
        "rescue": 5, "perturbation": 4, "common_stressor": 3,
        "correlation": 2, "reverse_only": 1, "not_stated": 0,
    }
    evidence_votes = [
        str(v).strip().lower()
        for v in group.get("evidence_type", [])
        if pd.notna(v) and str(v).strip()
    ]
    best_evidence = max(
        evidence_votes, key=lambda v: _EVIDENCE_RANK.get(v, 0), default="not_stated"
    )
    n_reverse_only = evidence_votes.count("reverse_only")

    cell_types = _join_unique(
        pd.concat(
            [
                group.get("upstream_cell_type", pd.Series(dtype="object")),
                group.get("downstream_cell_type", pd.Series(dtype="object")),
            ]
        ),
        limit=6,
    )

    n_spans = int(group.get("n_evidence_spans", pd.Series([0] * n_total)).fillna(0).sum())
    n_verified = int(group.get("n_verified_spans", pd.Series([0] * n_total)).fillna(0).sum())
    verified_ratio = (n_verified / n_spans) if n_spans else 0.0

    conf_values = [
        _CONFIDENCE_LEVEL_MAP.get(str(v).strip(), 0.25)
        for v in group.get("extraction_confidence", [])
        if pd.notna(v)
    ]
    mean_extraction_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.25

    aop_ids = _join_unique(group["aop_id"])
    in_aopwiki = bool(group["ker_id"].notna().any()) if "ker_id" in group.columns else False

    score = _confidence_score(
        n_support=n_support,
        n_contra=n_contra,
        has_essentiality=bool(essentiality),
        has_quantitative=bool(quantitative),
        verified_ratio=verified_ratio,
        mean_extraction_confidence=mean_extraction_confidence,
        in_aopwiki=in_aopwiki,
        sign_conflict=bool(n_positive and n_negative),
        # A row-derived numerator needs a row-derived denominator: the sign
        # counts above are per row, so this stays on rows even though the
        # paper counts above moved to papers.
        unsigned_ratio=(n_unsigned / n_rows) if n_rows else 0.0,
        evidence_type=best_evidence,
    )

    # Sorted, because this string is compared against a stored synthesis to
    # decide whether new evidence has arrived. In group order the same rows in
    # a different sequence read as a different evidence base.
    record_ids = (
        sorted(int(r) for r in group["record_id"].tolist())
        if "record_id" in group.columns
        else []
    )

    row: dict[str, Any] = {
        # Identity
        "ker_key": ker_key,
        "ker_id": _first_non_null(group["ker_id"].astype(str)),
        "ker_name": ker_name,
        "upstream_ke_name": upstream_name,
        "upstream_ke_level": upstream_level,
        "upstream_ke_id": _first_non_null(group["upstream_ke_id"].astype(str)),
        "downstream_ke_name": downstream_name,
        "downstream_ke_level": downstream_level,
        "downstream_ke_id": _first_non_null(group["downstream_ke_id"].astype(str)),
        "ker_adjacency": adjacency,
        "level_gap": level_gap,
        "aop_id": aop_ids,
        "aop_status": "existing" if (group["aop_status"] == "existing").any() else "novel",

        # Consolidation — papers and rows, kept apart on purpose
        "n_papers_total": n_total,
        "n_papers_supporting": n_support,
        "n_papers_contradicting": n_contra,
        "n_source_rows": n_rows,
        "n_rows_supporting": n_rows_support,
        "n_rows_contradicting": n_rows_contra,
        "uncertainty_level": _uncertainty_level(n_total, n_contra),
        "confidence_score": score,
        "confidence_band": confidence_band(score),

        # Sign and context
        # A link only one paper calls a marker is still a marker: nobody
        # writes "we measured X by staining for Y" as a causal claim, so the
        # marker vote is the informed one and majority rule would bury it.
        "relation_kind": (
            "marker"
            if "marker" in set(
                str(v).strip().lower()
                for v in group.get("relation_kind", [])
                if pd.notna(v)
            )
            else _mode_or_first(
                group.get("relation_kind", pd.Series(dtype="object")), "causal"
            )
        ),
        "direction": direction,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_unsigned": n_unsigned,
        "sign_conflict": bool(n_positive and n_negative),
        "evidence_type": best_evidence,
        "is_causal_evidence": best_evidence in ("rescue", "perturbation"),
        "is_stressor_evidence": best_evidence == "common_stressor",
        "n_reverse_only": n_reverse_only,
        "measured_as": _join_unique(
            group.get("measured_as", pd.Series(dtype="object")), limit=4
        ),
        "null_findings": _join_unique(
            group.get("null_findings", pd.Series(dtype="object")), limit=4
        ),
        "study_contexts": _join_unique(
            group.get("study_context", pd.Series(dtype="object")), limit=4
        ),
        "cell_types": cell_types,
        "upstream_changes": _join_unique(
            group.get("upstream_change", pd.Series(dtype="object")), limit=6
        ),
        "downstream_changes": _join_unique(
            group.get("downstream_change", pd.Series(dtype="object")), limit=6
        ),

        # Evidence narrative
        "ker_description": _first_non_null(group["ker_description"]),
        "biological_plausibility": _join_unique(group["biological_plausibility"], limit=4),
        "empirical_evidence_summary": _join_unique(group["empirical_evidence_summary"], limit=4),
        "essentiality_evidence": essentiality,

        # Applicability (union across all papers)
        "all_taxa": _join_unique(group["taxonomic_applicability"]),
        "sex_applicability": _join_unique(group["sex_applicability"]),
        "life_stage_applicability": _join_unique(group["life_stage_applicability"]),
        "study_designs": _join_unique(group["study_design"]),
        "chemical_stressors": _join_unique(group["chemical_stressor"], limit=8),
        "exposure_routes": _join_unique(group["exposure_route"], limit=5),

        # Evidence level thresholds (0-3 Low, 4-8 Moderate, 9+ High).
        # Banded on papers, not rows: the claim these make is about how much
        # independent literature covers the applicability domain, and one paper
        # split into four rows is not four studies.
        "taxonomic_evidence_level": _evidence_level(n_total),
        "sex_evidence_level": _evidence_level(n_total),
        "life_stage_evidence_level": _evidence_level(n_total),

        # Quantitative
        "quantitative_relationships": quantitative,
        "response_response_relationship": _first_non_null(group["response_response_relationship"]),
        "time_scale": _first_non_null(group["time_scale"]),
        "modulating_factors": _join_unique(group["modulating_factors"], limit=6),
        "feedforward_feedback_loops": _join_unique(group["feedforward_feedback_loops"], limit=4),

        # Provenance
        "all_source_dois": _join_unique(group["source_doi"]),
        "supporting_dois": _join_unique(group.loc[~group["contradicts_ker"], "source_doi"]),
        "contradicting_dois": _join_unique(group.loc[group["contradicts_ker"], "source_doi"]),
        "all_cited_dois": _join_unique(group["cited_evidence_dois"], limit=20),
        "record_ids": ",".join(str(r) for r in record_ids),
        "n_evidence_spans": n_spans,
        "n_verified_spans": n_verified,
        "provenance_ratio": round(verified_ratio, 3),
        "last_updated": group["extraction_date"].max(),

        # Human-review fields (empty until a reviewer fills them in)
        "uncertainty_description": None,
        "biological_plausibility_synthesis": None,
        "review_status": "Draft",
    }

    if "upstream_ke_canonical_id" in group.columns:
        row["upstream_ke_canonical_id"] = _first_non_null(
            group["upstream_ke_canonical_id"].astype(str)
        )
        row["downstream_ke_canonical_id"] = _first_non_null(
            group["downstream_ke_canonical_id"].astype(str)
        )

    # Which raw labels were folded into this edge — the audit trail for the
    # normalization step.
    if normalized:
        row["merged_upstream_labels"] = _join_unique(group["upstream_ke_name"])
        row["merged_downstream_labels"] = _join_unique(group["downstream_ke_name"])
        row["merged_ker_names"] = _join_unique(group["ker_name"], limit=10)
        row["n_distinct_raw_kers"] = int(
            group[["upstream_ke_name", "downstream_ke_name"]]
            .astype(str)
            .agg("::".join, axis=1)
            .nunique()
        )

    return row


def compute_table2(
    table1_df: pd.DataFrame,
    *,
    normalized: bool = True,
    spans_by_record: Optional[dict[int, list[dict]]] = None,
) -> pd.DataFrame:
    """
    Aggregate Table 1 rows into Table 2 KER-level summary rows.

    Parameters
    ----------
    table1_df
        Output of `table1_store.load_table1_as_dataframe()`, optionally passed
        through `ke_normalizer.apply_canonical_names()` first.
    normalized
        True  -> consolidate duplicate KERs on canonical KE identity.
        False -> group on the literal extracted strings (raw view).

    Returns one row per unique KER; empty DataFrame if the input is empty.
    """
    if table1_df is None or table1_df.empty:
        return pd.DataFrame()

    df = table1_df.copy()
    df["contradicts_ker"] = df["contradicts_ker"].astype(bool)

    key_fn = _canonical_ker_key if normalized else _raw_ker_key
    df["ker_key"] = df.apply(key_fn, axis=1)

    rows = [
        _aggregate_group(
            ker_key, group, normalized=normalized, spans_by_record=spans_by_record
        )
        for ker_key, group in df.groupby("ker_key", sort=False)
    ]

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Sort into pathway order: upstream lane first, then downstream lane, then
    # strongest evidence. This makes the table read top-to-bottom the same way
    # the map reads left-to-right.
    result["_up_idx"] = result["upstream_ke_level"].map(level_index)
    result["_down_idx"] = result["downstream_ke_level"].map(level_index)
    result = result.sort_values(
        ["_up_idx", "_down_idx", "confidence_score"], ascending=[True, True, False]
    ).drop(columns=["_up_idx", "_down_idx"])

    return result.reset_index(drop=True)


def compute_table2_raw(table1_df: pd.DataFrame) -> pd.DataFrame:
    """Direct-extraction view: no normalization, no cross-paper merging of labels."""
    return compute_table2(table1_df, normalized=False)


# ---------------------------------------------------------------------------
# Edge-level evidence assembly
# ---------------------------------------------------------------------------

def edge_evidence(
    ker_key: str,
    table2_df: pd.DataFrame,
    table1_df: pd.DataFrame,
    spans_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """
    Assemble everything needed to render the evidence panel for one KER.

    Returns a dict with:
        edge          the Table 2 row as a dict
        papers        per-paper rows contributing to this edge
        supporting    papers that support it
        contradicting papers that argue against it
        spans         evidence spans grouped by paper, with citations
        applicability taxa / sex / life-stage rollup
        confidence    score, band and the components that produced it
    """
    empty = {
        "edge": None, "papers": [], "supporting": [], "contradicting": [],
        "spans": [], "applicability": {}, "confidence": {},
    }
    if table2_df is None or table2_df.empty:
        return empty

    match = table2_df[table2_df["ker_key"] == ker_key]
    if match.empty:
        return empty
    edge = match.iloc[0].to_dict()

    record_ids = [
        int(r) for r in str(edge.get("record_ids") or "").split(",") if r.strip().isdigit()
    ]
    papers: list[dict] = []
    if table1_df is not None and not table1_df.empty and record_ids:
        subset = table1_df[table1_df["record_id"].isin(record_ids)]
        papers = subset.to_dict("records")

    supporting = [p for p in papers if not bool(p.get("contradicts_ker"))]
    contradicting = [p for p in papers if bool(p.get("contradicts_ker"))]

    spans: list[dict] = []
    if spans_df is not None and not spans_df.empty and record_ids:
        subset = spans_df[spans_df["record_id"].isin(record_ids)]
        spans = subset.to_dict("records")
        # Verified quotations first — a curator should see the solid ground
        # before the model's paraphrases.
        spans.sort(key=lambda s: (not s.get("verified", False), -float(s.get("match_ratio") or 0)))

    applicability = {
        "taxa": edge.get("all_taxa"),
        "sex": edge.get("sex_applicability"),
        "life_stage": edge.get("life_stage_applicability"),
        "study_designs": edge.get("study_designs"),
        "stressors": edge.get("chemical_stressors"),
        "exposure_routes": edge.get("exposure_routes"),
        "taxonomic_evidence_level": edge.get("taxonomic_evidence_level"),
        "sex_evidence_level": edge.get("sex_evidence_level"),
        "life_stage_evidence_level": edge.get("life_stage_evidence_level"),
    }

    confidence = {
        "score": edge.get("confidence_score"),
        "band": edge.get("confidence_band"),
        "uncertainty_level": edge.get("uncertainty_level"),
        "n_supporting": edge.get("n_papers_supporting"),
        "n_contradicting": edge.get("n_papers_contradicting"),
        "has_essentiality": bool(edge.get("essentiality_evidence")),
        "has_quantitative": bool(edge.get("quantitative_relationships")),
        "provenance_ratio": edge.get("provenance_ratio"),
        "in_aopwiki": edge.get("aop_status") == "existing",
    }

    return {
        "edge": edge,
        "papers": papers,
        "supporting": supporting,
        "contradicting": contradicting,
        "spans": spans,
        "applicability": applicability,
        "confidence": confidence,
    }


def synthesis_summary(table2_df: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the synthesis panel."""
    if table2_df is None or table2_df.empty:
        return {
            "n_kers": 0, "n_novel": 0, "n_adjacent": 0, "n_high_confidence": 0,
            "n_contradicted": 0, "n_papers": 0, "mean_confidence": 0.0,
            "provenance_coverage": 0.0,
        }

    n_spans = int(table2_df["n_evidence_spans"].fillna(0).sum())
    n_verified = int(table2_df["n_verified_spans"].fillna(0).sum())

    return {
        "n_kers": len(table2_df),
        "n_novel": int((table2_df["aop_status"] == "novel").sum()),
        "n_adjacent": int((table2_df["ker_adjacency"] == "Adjacent").sum()),
        "n_high_confidence": int((table2_df["confidence_band"] == "High").sum()),
        "n_contradicted": int((table2_df["n_papers_contradicting"] > 0).sum()),
        "n_papers": int(table2_df["n_papers_total"].sum()),
        "mean_confidence": round(float(table2_df["confidence_score"].mean()), 3),
        "provenance_coverage": round(n_verified / n_spans, 3) if n_spans else 0.0,
    }


def consolidation_report(raw_df: pd.DataFrame, normalized_df: pd.DataFrame) -> dict[str, Any]:
    """Quantify what duplicate-KER consolidation actually did."""
    n_raw = len(raw_df) if raw_df is not None else 0
    n_norm = len(normalized_df) if normalized_df is not None else 0
    merged = max(0, n_raw - n_norm)
    return {
        "n_raw_kers": n_raw,
        "n_consolidated_kers": n_norm,
        "n_edges_merged": merged,
        "reduction_pct": round(100.0 * merged / n_raw, 1) if n_raw else 0.0,
    }


# ---------------------------------------------------------------------------
# Navigating a large Table 2
#
# A corpus of any size produces more KERs than anyone reads. The failure this
# section addresses is specific: a user arrives knowing which biological event
# they care about — "oligodendrocyte differentiation" — and the table is
# organised by relationship, so they must scan every row to find the handful
# that touch it. Filtering by keyword returns dozens of rows and still leaves
# them reading all of them.
#
# The fix is to let them enter through the Key Event instead, and to put the
# corroborated relationships in front of the single-paper ones.
# ---------------------------------------------------------------------------

def relevance_sort(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order KERs by how much they deserve attention.

    Corroboration first, because a relationship two independent papers report
    is worth more of a curator's time than one that appears once; then
    confidence, then the share of quotations that could actually be located in
    the source. In a corpus where most KERs come from a single paper, this is
    what separates signal from bulk.
    """
    if table2_df is None or table2_df.empty:
        return table2_df

    df = table2_df.copy()
    for column, default in (
        ("n_papers_supporting", 0),
        ("confidence_score", 0.0),
        ("provenance_ratio", 0.0),
    ):
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default)

    return df.sort_values(
        ["n_papers_supporting", "confidence_score", "provenance_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def key_event_index(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Key Event, with how it participates in the network.

    Columns: ke_name, level, n_upstream (edges where it is the downstream
    partner), n_downstream, n_total, n_papers. Sorted by connectedness, so the
    events that actually anchor the pathway appear first — an event appearing
    in one edge is a leaf, one appearing in twelve is a hub worth opening.
    """
    if table2_df is None or table2_df.empty:
        return pd.DataFrame()

    records: dict[str, dict[str, Any]] = {}

    def touch(name: Any, level: Any, role: str, papers: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        entry = records.setdefault(
            name,
            {"ke_name": name, "level": level, "n_upstream": 0, "n_downstream": 0,
             "n_papers": 0},
        )
        # "n_downstream" counts edges leaving this event, i.e. rows where it is
        # the upstream partner. Naming follows the reader's viewpoint, not the
        # column it was read from.
        entry["n_downstream" if role == "upstream" else "n_upstream"] += 1
        try:
            entry["n_papers"] += int(papers or 0)
        except (TypeError, ValueError):
            pass
        if not entry.get("level") and level:
            entry["level"] = level

    for _, row in table2_df.iterrows():
        touch(row.get("upstream_ke_name"), row.get("upstream_ke_level"),
              "upstream", row.get("n_papers_total"))
        touch(row.get("downstream_ke_name"), row.get("downstream_ke_level"),
              "downstream", row.get("n_papers_total"))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))
    df["n_total"] = df["n_upstream"] + df["n_downstream"]
    return df.sort_values(
        ["n_total", "n_papers"], ascending=[False, False]
    ).reset_index(drop=True)


def neighbourhood(
    ke_name: str,
    table2_df: pd.DataFrame,
    *,
    include_contradicting: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Every KER one hop from `ke_name`, split by direction.

    Returns {"upstream": rows where `ke_name` is the downstream partner —
    i.e. what leads TO it, "downstream": rows where it is the upstream
    partner — what it leads to}. Both are relevance-sorted.
    """
    empty = {"upstream": pd.DataFrame(), "downstream": pd.DataFrame()}
    if table2_df is None or table2_df.empty or not ke_name:
        return empty

    df = table2_df
    if not include_contradicting and "n_papers_contradicting" in df.columns:
        df = df[df["n_papers_contradicting"].fillna(0) == 0]

    leads_to_it = df[df.get("downstream_ke_name") == ke_name]
    it_leads_to = df[df.get("upstream_ke_name") == ke_name]

    return {
        "upstream": relevance_sort(leads_to_it),
        "downstream": relevance_sort(it_leads_to),
    }


#: Levels at which an event can reasonably be an adverse outcome. An AO is a
#: consequence for the organism or the population, not a cellular observation.
_AO_LEVELS = ("Tissue", "Organ", "Individual", "Population")

#: Levels at which an event can reasonably be a molecular initiating event.
_MIE_LEVELS = ("MIE", "Molecular")


def infer_roles(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Propose an MIE / KE / AO role for every event, with the reason.

    Roles were assigned by hand, one node at a time, and everything else was
    "inferred from the graph" by position alone — which makes whatever happens
    to be terminal an adverse outcome. In the Nav1.2 corpus the terminal node
    was myelin basic protein expression: a stain used to measure maturation,
    proposed as the adverse outcome of the pathway.

    Three things decide the proposal, in order:

    1. A node reached only by `marker` or `definitional` links is not an
       event in the pathway at all. It is how another event was measured.
    2. A terminal node is an adverse outcome only if it sits at tissue level
       or above. A terminal cellular event means the corpus stops short of an
       outcome, which is a finding about the corpus and should be said rather
       than papered over.
    3. A source node is an MIE only if it is molecular. A source at cellular
       level means the initiating event is missing from the corpus.

    Returns columns: ke_name, level, role, confidence, reason, n_in, n_out.
    Nothing is written; this is a proposal for a curator to accept.
    """
    if table2_df is None or table2_df.empty:
        return pd.DataFrame()

    causal = table2_df
    if "relation_kind" in table2_df.columns:
        causal = table2_df[
            table2_df["relation_kind"].fillna("causal").astype(str) != "marker"
        ]

    levels: dict[str, str] = {}
    in_causal: Counter = Counter()
    out_causal: Counter = Counter()
    in_any: Counter = Counter()
    marker_of: dict[str, list[str]] = defaultdict(list)

    for _, row in table2_df.iterrows():
        up = str(row.get("upstream_ke_name") or "").strip()
        down = str(row.get("downstream_ke_name") or "").strip()
        if not up or not down:
            continue
        levels.setdefault(up, str(row.get("upstream_ke_level") or "Molecular"))
        levels.setdefault(down, str(row.get("downstream_ke_level") or "Molecular"))
        in_any[down] += 1
        if str(row.get("relation_kind") or "causal") == "marker":
            marker_of[down].append(up)

    for _, row in causal.iterrows():
        up = str(row.get("upstream_ke_name") or "").strip()
        down = str(row.get("downstream_ke_name") or "").strip()
        if not up or not down:
            continue
        out_causal[up] += 1
        in_causal[down] += 1

    rows = []
    for name, level in levels.items():
        n_in, n_out = in_causal[name], out_causal[name]

        if marker_of.get(name) and n_out == 0:
            measured = marker_of[name][0]
            role, confidence = "marker", "high"
            reason = (
                f"Reached only as a measurement of “{measured}”. A readout, "
                f"not an event in the pathway."
            )
        elif n_in == 0 and n_out > 0 and level in _MIE_LEVELS:
            role, confidence = "MIE", "medium"
            reason = f"Nothing upstream of it in the corpus, and it is {level}-level."
        elif n_in == 0 and n_out > 0:
            role, confidence = "KE", "low"
            reason = (
                f"Starts the chain but sits at {level} level, so the "
                f"initiating event is probably missing from the corpus."
            )
        elif n_out == 0 and n_in > 0 and level in _AO_LEVELS:
            # A candidate, not a verdict. Whether harm occurred is a judgement
            # about biology, not about how many arrows leave a box.
            role, confidence = "KE", "medium"
            reason = (
                f"Terminal at {level} level — a possible adverse outcome, if "
                f"you judge it a harm. Declare it to make it one."
            )
        elif n_out == 0 and n_in > 0:
            role, confidence = "KE", "high"
            reason = (
                f"Terminal in the corpus but only {level}-level — most likely "
                f"the last thing measured rather than an outcome."
            )
        else:
            role, confidence = "KE", "high"
            reason = "Has causes and consequences in the corpus."

        rows.append(
            {
                "ke_name": name,
                "level": level,
                "role": role,
                "confidence": confidence,
                "reason": reason,
                "n_in": n_in,
                "n_out": n_out,
            }
        )

    order = {"MIE": 0, "KE": 1, "AO": 2, "marker": 3}
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["role", "ke_name"], key=lambda c: c.map(order) if c.name == "role" else c
    ).reset_index(drop=True)


def aop_completeness(table2_df: pd.DataFrame) -> dict[str, Any]:
    """
    Whether what has been assembled is an AOP or a fragment of one.

    An AOP needs a molecular initiating event and an adverse outcome with a
    causal path between them. A corpus can be perfectly well extracted and
    still not contain one — which is a legitimate result, and much more useful
    said out loud than discovered by a reviewer.
    """
    roles = infer_roles(table2_df)
    if roles.empty:
        return {"is_aop": False, "reason": "No events yet.", "roles": roles}

    mies = roles[roles["role"] == "MIE"]["ke_name"].tolist()
    aos = roles[roles["role"] == "AO"]["ke_name"].tolist()
    markers = roles[roles["role"] == "marker"]["ke_name"].tolist()

    # Deliberately not phrased as a deficiency. Mechanistic papers describe
    # mechanism; a corpus of them that never reaches a harm has not failed at
    # anything, and telling a curator their work is incomplete because no node
    # looks like an adverse outcome invites them to promote one that isn't.
    missing = []
    if not mies:
        missing.append("no molecular initiating event yet")
    if not aos:
        missing.append("no adverse outcome declared")

    return {
        "is_aop": not missing,
        "reason": (
            "Has an initiating event and a declared adverse outcome."
            if not missing
            else "A pathway fragment: " + "; ".join(missing) + ". That is a "
                 "normal result for mechanistic papers — an adverse outcome is "
                 "a claim about harm, so it is declared rather than inferred."
        ),
        "mies": mies,
        "aos": aos,
        "markers": markers,
        "roles": roles,
    }


def causation_report(table2_df: pd.DataFrame) -> dict[str, Any]:
    """
    How much of the map is causation and how much is co-observation.

    The single question a reviewer asks of an AOP, and the one the tool could
    not answer: an arrow drawn from a knockout and an arrow drawn from two
    measurements declining together looked identical.

    `backwards` is the important list. Those are links where the only
    experiment manipulated the DOWNSTREAM event — the paper showed B causes A
    and the extraction recorded A → B. They are almost always pointing the
    wrong way and should be reversed or removed, not merely doubted.
    """
    if table2_df is None or table2_df.empty or "evidence_type" not in table2_df.columns:
        return {"counts": {}, "backwards": pd.DataFrame(), "share_causal": 0.0}

    counts = table2_df["evidence_type"].fillna("not_stated").value_counts().to_dict()
    causal = int(table2_df.get("is_causal_evidence", pd.Series(dtype=bool)).sum())

    columns = [
        c for c in ("ker_name", "upstream_ke_name", "downstream_ke_name",
                    "evidence_type", "measured_as", "n_papers_total",
                    "confidence_band", "all_source_dois")
        if c in table2_df.columns
    ]
    backwards = table2_df[table2_df["evidence_type"] == "reverse_only"][columns]
    weak = table2_df[
        table2_df["evidence_type"].isin(["correlation", "not_stated"])
    ][columns]

    return {
        "counts": counts,
        "share_causal": round(causal / len(table2_df), 3) if len(table2_df) else 0.0,
        "backwards": backwards.reset_index(drop=True),
        "correlational": weak.reset_index(drop=True),
    }


def sign_conflicts(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Edges where the papers disagree about which way the relationship runs.

    Distinct from `direction_conflicts`, which finds A→B alongside B→A. This
    finds one edge, A→B, that some papers report as the events moving together
    and others as moving oppositely. That is the shape of two different
    experiments — a knockdown and an overexpression, or the same molecule in
    two cell types — filed under one relationship, and it is invisible in any
    count of supporting papers.

    Returns one row per conflicted edge with the paper counts on each side and
    the cell types involved, which is usually where the explanation is.
    """
    if table2_df is None or table2_df.empty or "sign_conflict" not in table2_df.columns:
        return pd.DataFrame()

    conflicted = table2_df[table2_df["sign_conflict"].fillna(False).astype(bool)]
    if conflicted.empty:
        return pd.DataFrame()

    out = conflicted[
        [
            c
            for c in (
                "ker_name", "upstream_ke_name", "downstream_ke_name",
                "n_positive", "n_negative", "n_unsigned", "cell_types",
                "upstream_changes", "downstream_changes", "confidence_band",
                "all_source_dois",
            )
            if c in conflicted.columns
        ]
    ].copy()
    return out.sort_values(
        ["n_positive", "n_negative"], ascending=False
    ).reset_index(drop=True)


def unsigned_edges(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Edges no paper gave a direction to.

    An unsigned edge still draws on the map, and a reader will supply a sign
    from the arrowhead whether or not the evidence had one. Listing them is
    the cheapest way to stop that.
    """
    if table2_df is None or table2_df.empty or "direction" not in table2_df.columns:
        return pd.DataFrame()

    unsigned = table2_df[table2_df["direction"].astype(str) == "unclear"]
    if unsigned.empty:
        return pd.DataFrame()

    return unsigned[
        [
            c
            for c in (
                "ker_name", "upstream_ke_name", "downstream_ke_name",
                "n_papers_total", "confidence_band", "all_source_dois",
            )
            if c in unsigned.columns
        ]
    ].reset_index(drop=True)


def direction_conflicts(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs of Key Events asserted to act on each other in both directions.

    A → B and B → A can mean two things, and the difference matters: a genuine
    feedback loop that the AOP should represent, or two papers disagreeing
    about which event drives which. Nothing in the synthesis distinguishes
    them, and a curator cannot fix what is not surfaced — so both are listed
    and the judgement is left to a person.

    Returns one row per conflicting pair with both KER names, both confidence
    bands and both supporting-paper counts.
    """
    if table2_df is None or table2_df.empty:
        return pd.DataFrame()

    # Pair on canonical ids where normalization supplied them, since two
    # phrasings of the same event would otherwise never be recognised as the
    # two ends of one conflict. Falls back to the label.
    def endpoints(row: pd.Series) -> Optional[tuple[Any, Any]]:
        up_id = _as_int(row.get("upstream_ke_canonical_id"))
        down_id = _as_int(row.get("downstream_ke_canonical_id"))
        if up_id is not None and down_id is not None:
            return (f"id:{up_id}", f"id:{down_id}")
        up, down = row.get("upstream_ke_name"), row.get("downstream_ke_name")
        if isinstance(up, str) and isinstance(down, str):
            return (up, down)
        return None

    by_pair: dict[tuple[Any, Any], pd.Series] = {}
    labels: dict[Any, str] = {}
    for _, row in table2_df.iterrows():
        pair = endpoints(row)
        if pair is None or pair[0] == pair[1]:
            continue
        by_pair[pair] = row
        labels.setdefault(pair[0], str(row.get("upstream_ke_name") or pair[0]))
        labels.setdefault(pair[1], str(row.get("downstream_ke_name") or pair[1]))

    seen: set[frozenset] = set()
    conflicts: list[dict[str, Any]] = []
    for (up, down), row in by_pair.items():
        reverse = by_pair.get((down, up))
        if reverse is None:
            continue
        marker = frozenset((up, down))
        if marker in seen:
            continue
        seen.add(marker)
        conflicts.append(
            {
                "event_a": labels.get(up, str(up)),
                "event_b": labels.get(down, str(down)),
                "forward_ker": row.get("ker_name"),
                "forward_confidence": row.get("confidence_band"),
                "forward_papers": row.get("n_papers_supporting"),
                "reverse_ker": reverse.get("ker_name"),
                "reverse_confidence": reverse.get("confidence_band"),
                "reverse_papers": reverse.get("n_papers_supporting"),
                "forward_key": row.get("ker_key"),
                "reverse_key": reverse.get("ker_key"),
            }
        )

    if not conflicts:
        return pd.DataFrame()
    return pd.DataFrame(conflicts).sort_values(
        ["forward_papers", "reverse_papers"], ascending=[False, False]
    ).reset_index(drop=True)


__all__ = [
    "compute_table2",
    "compute_table2_raw",
    "edge_evidence",
    "synthesis_summary",
    "consolidation_report",
    "confidence_band",
    "relevance_sort",
    "key_event_index",
    "neighbourhood",
    "direction_conflicts",
    "sign_conflicts",
    "causation_report",
    "unsigned_edges",
    "infer_roles",
    "aop_completeness",
]
