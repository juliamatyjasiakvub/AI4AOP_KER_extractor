"""
Table 2 and the synthesis prompt must be functions of the row SET, not the row
ORDER.

Row order is `record_id` order, which is upload order. Re-uploading the same
corpus in a different sequence, or re-extracting after one paper failed and was
retried, permutes it. Before these tests, that permutation changed the joined
strings, silently changed *which* content survived a `limit=`, and reordered
the blocks in the synthesis prompt — so the narrative and the OECD ratings
moved with no change in the evidence.

Also covers the counting rule: `n_papers_*` counts papers, `n_rows_*` counts
Table 1 rows, and the two are not the same number whenever the extractor splits
one paper into several claims.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from stage2_extraction import evidence_synthesis, table2_synthesis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(**overrides):
    base = {
        "record_id": 1,
        "source_doi": "10.1/a",
        "source_filename": "a.pdf",
        "source_title": "A",
        "extraction_date": "2026-01-01",
        "aop_id": None,
        "aop_status": "novel",
        "upstream_ke_id": None,
        "downstream_ke_id": None,
        "ker_id": None,
        "upstream_ke_name": "increased ROS",
        "upstream_ke_level": "Molecular",
        "downstream_ke_name": "increased apoptosis",
        "downstream_ke_level": "Cellular",
        "ker_name": "ROS leads to apoptosis",
        "ker_description": "Description.",
        "ker_adjacency": "Adjacent",
        "cited_evidence_dois": None,
        "biological_plausibility": None,
        "empirical_evidence_summary": None,
        "essentiality_evidence": None,
        "contradicts_ker": False,
        "taxonomic_applicability": None,
        "sex_applicability": None,
        "life_stage_applicability": None,
        "modulating_factors": None,
        "quantitative_relationships": None,
        "response_response_relationship": None,
        "time_scale": None,
        "feedforward_feedback_loops": None,
        "study_design": None,
        "exposure_route": None,
        "chemical_stressor": None,
        "extraction_confidence": "High",
        "upstream_ke_canonical_id": None,
        "downstream_ke_canonical_id": None,
        "n_evidence_spans": 1,
        "n_verified_spans": 1,
        "direction": "positive",
        "upstream_change": "increased",
        "downstream_change": "increased",
        "upstream_cell_type": None,
        "downstream_cell_type": None,
        "relation_kind": "causal",
        "evidence_type": "perturbation",
        "measured_as": None,
        "null_findings": None,
        "study_context": None,
    }
    base.update(overrides)
    return base


def _corpus() -> pd.DataFrame:
    """Six papers on one edge, each with different applicability metadata."""
    taxa = ["Mus musculus", "Rattus norvegicus", "Danio rerio",
            "Homo sapiens", "Papio anubis", "Gallus gallus"]
    stressors = [f"stressor-{i}" for i in range(6)]
    return pd.DataFrame(
        [
            _row(
                record_id=i + 1,
                source_doi=f"10.1/paper{i}",
                source_filename=f"paper{i}.pdf",
                taxonomic_applicability=taxa[i],
                chemical_stressor=stressors[i],
                essentiality_evidence=f"knockout evidence {i}",
                ker_description=f"Mechanism as described by paper {i}.",
            )
            for i in range(6)
        ]
    )


def _shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    order = list(range(len(frame)))
    random.Random(seed).shuffle(order)
    return frame.iloc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalized", [True, False])
def test_table2_is_identical_under_any_row_order(normalized):
    corpus = _corpus()
    baseline = table2_synthesis.compute_table2(corpus, normalized=normalized)

    for seed in range(15):
        other = table2_synthesis.compute_table2(
            _shuffled(corpus, seed), normalized=normalized
        )
        assert len(other) == len(baseline)
        for column in baseline.columns:
            if column == "last_updated":
                continue
            assert list(other[column].astype(str)) == list(
                baseline[column].astype(str)
            ), f"{column} depends on row order"


def test_a_truncated_field_keeps_the_same_content_every_time():
    """
    The failure this is really about.

    `chemical_stressors` joins with `limit=8`. With more distinct stressors
    than the limit, the old implementation kept whichever arrived first — so
    two runs over identical evidence dropped *different* stressors, and neither
    said so.
    """
    corpus = pd.DataFrame(
        [
            _row(
                record_id=i + 1,
                source_doi=f"10.1/paper{i}",
                source_filename=f"paper{i}.pdf",
                chemical_stressor=f"stressor-{i:02d}",
            )
            for i in range(20)
        ]
    )
    baseline = table2_synthesis.compute_table2(corpus).iloc[0]["chemical_stressors"]
    assert len(baseline.split(";")) == 8, "the limit still applies"

    for seed in range(15):
        other = table2_synthesis.compute_table2(_shuffled(corpus, seed)).iloc[0]
        assert other["chemical_stressors"] == baseline


def test_join_unique_orders_by_how_many_rows_said_it():
    """Truncation should drop the least corroborated item, not the last-read."""
    series = pd.Series(["rare"] + ["common"] * 5 + ["mid", "mid"])
    assert table2_synthesis._join_unique(series, limit=2) == "common; mid"


def test_record_ids_are_sorted():
    corpus = _corpus()
    for seed in range(10):
        row = table2_synthesis.compute_table2(_shuffled(corpus, seed)).iloc[0]
        ids = [int(x) for x in row["record_ids"].split(",")]
        assert ids == sorted(ids)


def test_synthesis_prompt_is_identical_under_any_row_order():
    corpus = _corpus()
    baseline = evidence_synthesis.build_synthesis_input(corpus)
    for seed in range(15):
        assert evidence_synthesis.build_synthesis_input(_shuffled(corpus, seed)) == (
            baseline
        ), "the prompt sent to the model depends on upload order"


def test_synthesis_prompt_blocks_are_ordered_by_doi():
    corpus = _corpus()
    prompt = evidence_synthesis.build_synthesis_input(_shuffled(corpus, 3))
    dois = [
        line.split("DOI ")[1].split()[0]
        for line in prompt.splitlines()
        if line.startswith("[")
    ]
    assert dois == sorted(dois)


# ---------------------------------------------------------------------------
# Papers are not rows
# ---------------------------------------------------------------------------

def test_two_claims_from_one_paper_count_as_one_paper():
    corpus = pd.DataFrame(
        [
            _row(record_id=1, source_doi="10.1/x", source_filename="x.pdf"),
            _row(record_id=2, source_doi="10.1/x", source_filename="x.pdf",
                 ker_description="A second claim from the same paper."),
            _row(record_id=3, source_doi="10.1/y", source_filename="y.pdf"),
        ]
    )
    row = table2_synthesis.compute_table2(corpus).iloc[0]
    assert row["n_papers_total"] == 2, "one DOI twice is one paper"
    assert row["n_papers_supporting"] == 2
    assert row["n_source_rows"] == 3, "the claim count is still reported"
    assert row["n_rows_supporting"] == 3


def test_a_paper_that_contradicts_anywhere_is_filed_as_contradicting():
    """
    Otherwise supporting + contradicting can exceed the number of papers, and
    the contradicting fraction in the confidence score stops being a fraction.
    """
    corpus = pd.DataFrame(
        [
            _row(record_id=1, source_doi="10.1/x", source_filename="x.pdf"),
            _row(record_id=2, source_doi="10.1/x", source_filename="x.pdf",
                 contradicts_ker=True),
            _row(record_id=3, source_doi="10.1/y", source_filename="y.pdf"),
        ]
    )
    row = table2_synthesis.compute_table2(corpus).iloc[0]
    assert row["n_papers_total"] == 2
    assert row["n_papers_contradicting"] == 1
    assert row["n_papers_supporting"] == 1
    assert (
        row["n_papers_supporting"] + row["n_papers_contradicting"]
        == row["n_papers_total"]
    )


def test_papers_without_a_doi_are_not_merged_together():
    """Two unidentified papers are two papers, not one."""
    corpus = pd.DataFrame(
        [
            _row(record_id=1, source_doi=None, source_filename="first.pdf"),
            _row(record_id=2, source_doi=None, source_filename="second.pdf"),
        ]
    )
    row = table2_synthesis.compute_table2(corpus).iloc[0]
    assert row["n_papers_total"] == 2

    assert evidence_synthesis.n_contributing_papers(corpus) == 2


def test_evidence_level_bands_on_papers_not_claims():
    """
    Four claims from one paper must not read as 'Moderate' coverage of the
    applicability domain. The band is a statement about independent literature.
    """
    corpus = pd.DataFrame(
        [
            _row(record_id=i + 1, source_doi="10.1/only", source_filename="only.pdf")
            for i in range(4)
        ]
    )
    row = table2_synthesis.compute_table2(corpus).iloc[0]
    assert row["n_source_rows"] == 4
    assert row["n_papers_total"] == 1
    assert row["taxonomic_evidence_level"] == "Low"


def test_synthesis_reports_papers_and_claims_separately():
    corpus = pd.DataFrame(
        [
            _row(record_id=1, source_doi="10.1/x", source_filename="x.pdf"),
            _row(record_id=2, source_doi="10.1/x", source_filename="x.pdf"),
            _row(record_id=3, source_doi="10.1/y", source_filename="y.pdf"),
        ]
    )
    assert evidence_synthesis.n_contributing_papers(corpus) == 2
    assert len(corpus) == 3
