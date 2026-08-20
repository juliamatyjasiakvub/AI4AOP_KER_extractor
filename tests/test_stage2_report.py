"""
The Stage 2 record must describe the database, not flatter it.

This report is the artefact a reviewer reads instead of the database, so its
failure mode is not crashing — it is quietly reporting a cleaner project than
exists. These tests pin the places that would do that: an empty section must
still appear and say it is empty, a curator's "NA" must not read as a rationale,
claims must not be presented as papers, and the checks that find real problems
(two Key Events sharing a name, wordings merged on no recorded authority) must
actually fire.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stage2_extraction import stage2_report, table1_store


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    monkeypatch.setattr(table1_store, "DB_PATH", tmp_path / "test.db")
    table1_store.init_db()
    return tmp_path


# ---------------------------------------------------------------------------
# Empty is a finding, not a reason to omit
# ---------------------------------------------------------------------------

def test_every_section_appears_on_an_empty_database(empty_db):
    md = stage2_report.report_markdown(stage2_report.build_stage2_report())
    for heading in (
        "1 · Corpus and runs",
        "2 · What was extracted",
        "3 · From raw wording to Key Events",
        "4 · Curator decisions",
        "5 · Approval",
        "6 · Relationships",
        "7 · Evidence syntheses",
        "8 · Outstanding",
    ):
        assert heading in md, f"missing section: {heading}"


def test_an_unreviewed_project_says_so_rather_than_going_quiet(empty_db):
    """
    No recorded decisions means the canonical Key Events are the proposer's
    grouping with no second opinion. Omitting the section would read as "no
    problems here".
    """
    md = stage2_report.report_markdown(stage2_report.build_stage2_report())
    assert "No merge, split, mapping or rejection was recorded" in md
    assert "unreviewed" in md


def test_report_survives_a_database_with_nothing_in_it(empty_db):
    report = stage2_report.build_stage2_report()
    assert stage2_report.report_csv(report).startswith("section,subject,detail")


# ---------------------------------------------------------------------------
# Not flattering the record
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["NA", "n/a", "  ", "None", "nan", "-"])
def test_a_placeholder_rationale_is_not_shown_as_a_rationale(blank):
    """
    A curator who types "NA" has given no reason. Rendering it verbatim makes
    an unexplained decision look explained.
    """
    assert stage2_report._text(blank) == "—"


def test_real_text_is_preserved():
    assert stage2_report._text("merged: same GO term") == "merged: same GO term"


def test_a_pipe_in_a_rationale_cannot_break_the_table():
    assert "|" not in stage2_report._text("a | b").replace("\\|", "")


# ---------------------------------------------------------------------------
# The checks that find real problems
# ---------------------------------------------------------------------------

def _report_with(**overrides) -> stage2_report.Stage2Report:
    report = stage2_report.Stage2Report(generated_at="now")
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def test_two_key_events_sharing_a_name_are_reported():
    """
    They draw as two nodes with one label, which a reader will read as one
    event. Nothing else in the tool looks for this.
    """
    report = _report_with(
        canonical=pd.DataFrame(
            [
                {"canonical_name": "conduction failure", "level": "Cellular"},
                {"canonical_name": "Conduction Failure", "level": "Cellular"},
                {"canonical_name": "increased ROS", "level": "Molecular"},
            ]
        )
    )
    md = stage2_report.report_markdown(report)
    assert "used by more than one Key Event" in md
    assert "conduction failure" in md


def test_distinct_names_are_not_reported_as_duplicates():
    report = _report_with(
        canonical=pd.DataFrame(
            [{"canonical_name": "increased ROS"}, {"canonical_name": "axon loss"}]
        )
    )
    assert "used by more than one Key Event" not in stage2_report.report_markdown(report)


def test_wordings_merged_on_no_recorded_authority_are_flagged():
    report = _report_with(
        canonical=pd.DataFrame([{"canonical_name": "increased ROS"}]),
        crosswalk=pd.DataFrame(
            [
                {"raw_label": "ROS up", "merge_basis": None},
                {"raw_label": "elevated ROS", "merge_basis": "ontology"},
            ]
        ),
    )
    md = stage2_report.report_markdown(report)
    assert "1 raw wording(s) have no recorded authorising rule" in md


def test_unverified_quotations_are_reported():
    report = _report_with(
        spans=pd.DataFrame([{"verified": True}, {"verified": False}, {"verified": False}])
    )
    assert "2 quotation(s) could not be located" in stage2_report.report_markdown(report)


def test_a_clean_project_reports_nothing_outstanding():
    report = _report_with(
        spans=pd.DataFrame([{"verified": True}]),
        canonical=pd.DataFrame([{"canonical_name": "increased ROS"}]),
        crosswalk=pd.DataFrame([{"raw_label": "ROS up", "merge_basis": "ontology"}]),
    )
    assert "Nothing outstanding" in stage2_report.report_markdown(report)


# ---------------------------------------------------------------------------
# Claims are not papers
# ---------------------------------------------------------------------------

def test_claims_and_papers_are_reported_as_different_numbers(empty_db):
    """
    Three claims from two papers must not read as three papers — the whole
    reason `n_contributing_papers` exists.
    """
    rows = pd.DataFrame(
        [
            {"record_id": 1, "source_doi": "10.1/a", "source_filename": "a.pdf",
             "contradicts_ker": False},
            {"record_id": 2, "source_doi": "10.1/a", "source_filename": "a.pdf",
             "contradicts_ker": False},
            {"record_id": 3, "source_doi": "10.1/b", "source_filename": "b.pdf",
             "contradicts_ker": False},
        ]
    )
    md = stage2_report.report_markdown(_report_with(table1=rows))
    assert "| Claims (Table 1 rows) | 3 |" in md
    assert "| Distinct papers behind them | 2 |" in md


def test_a_low_verification_rate_is_called_out(empty_db):
    rows = pd.DataFrame(
        [{"record_id": 1, "source_doi": "10.1/a", "source_filename": "a.pdf",
          "contradicts_ker": False}]
    )
    spans = pd.DataFrame([{"verified": False}] * 8 + [{"verified": True}] * 2)
    md = stage2_report.report_markdown(_report_with(table1=rows, spans=spans))
    assert "Only 20% of quotations could be located" in md


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_csv_carries_decisions_approvals_and_assessments():
    report = _report_with(
        decisions=pd.DataFrame([{
            "member_ids": "[1, 2]", "action_label": "Merge as equivalent",
            "relationship": "equivalent", "curator": "Julia",
            "curator_rationale": "same GO term", "created_at": "2026-01-01",
        }]),
        approvals=pd.DataFrame([{
            "target_type": "ke", "target_key": "7", "from_state": "curated",
            "to_state": "approved", "curator": "Julia", "note": None,
            "created_at": "2026-01-02",
        }]),
        syntheses=pd.DataFrame([{
            "ker_name": "ROS leads to apoptosis", "developer_assessment": "Moderate",
            "overall_confidence": "High", "developer_curator": "Julia",
            "developer_rationale": "one model system only", "generated_at": "2026-01-03",
        }]),
    )
    csv = stage2_report.report_csv(report)
    assert "curation decision" in csv and "same GO term" in csv
    assert "approval" in csv and "curated -> approved" in csv
    assert "developer assessment" in csv and "one model system only" in csv
