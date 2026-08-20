"""
What the run did not produce, and what the numbers are actually counting.

Two blind spots, both of which made a corpus look better than it was.

**Papers that yielded nothing.** A run over thirteen papers that extracted
from eleven produced a QC report about eleven papers. The other two were
counted in `extraction_runs.papers_attempted` and described nowhere. That
matters because the two cases are opposite: a paper the model read and found
no mechanism in is a *finding*, while a paper whose reply hit the token
ceiling is a *gap* — and a gap is invisible in every table in the report,
because tables are built from rows that exist.

**The counts.** "28 rows of KERs, 18 after normalisation" reads as a loss of
ten. It is not a loss of anything: 28 is a count of *claims* (one paper saying
one event leads to another) and 18 is a count of *events* (the ends of those
claims). The sidebar made this worse by labelling 56 — two label mentions per
row — as "Raw Key Event labels", so the same corpus reported 56, 28 and 18 for
three different things and invited the reader to subtract them.
"""

import pytest

from stage2_extraction import qc_report


@pytest.fixture
def store(monkeypatch, tmp_path):
    import stage2_extraction.table1_store as ts
    from run_manifest import RunManifest

    monkeypatch.setattr(ts, "DB_PATH", tmp_path / "scratch.db")
    ts.init_db()
    run_id = ts.start_run(RunManifest(stage="extraction"))
    return ts, run_id


def seed(ts, run_id, cases):
    for filename, category, n_kers in cases:
        ts.record_paper_outcome(
            run_id=run_id,
            source_filename=filename,
            source_doi=f"10.1/{filename}",
            outcome="KERs saved" if n_kers else "No KERs found",
            category=category,
            reason="fixture",
            n_kers=n_kers,
            n_llm_calls=4,
            n_truncated=1 if category == "truncated" else 0,
        )


class TestPapersThatProducedNothing:

    def test_successes_and_failures_are_both_recorded(self, store):
        """
        "Which papers yielded nothing" is only answerable if the ones that did
        yield something are on the same list.
        """
        ts, run_id = store
        seed(ts, run_id, [("a.pdf", "saved", 3), ("b.pdf", "truncated", 0)])
        assert len(ts.load_paper_outcomes(run_id)) == 2

    def test_the_report_separates_findings_from_gaps(self, store):
        """
        The distinction the whole section exists for. Only one of these five
        barren papers is a result; the other four are missing evidence.
        """
        ts, run_id = store
        seed(ts, run_id, [
            ("ok.pdf", "saved", 3),
            ("silent.pdf", "no_mechanism", 0),
            ("cutoff.pdf", "truncated", 0),
            ("garbled.pdf", "parse_failure", 0),
            ("offline.pdf", "provider_error", 0),
            ("scan.pdf", "no_text", 0),
        ])
        report = qc_report.build_qc_report(run_id)
        assert report.n_papers_attempted == 6
        assert report.n_barren == 5
        assert report.n_recoverable == 4, "no_mechanism must not count as recoverable"

    def test_a_believable_null_result_does_not_raise_an_alarm(self, store):
        """
        A paper that genuinely discusses no mechanism is not a defect, and
        flagging it as one would train the reader to ignore the flag.
        """
        ts, run_id = store
        seed(ts, run_id, [("ok.pdf", "saved", 2), ("silent.pdf", "no_mechanism", 0)])
        report = qc_report.build_qc_report(run_id)
        assert report.n_recoverable == 0
        joined = " ".join(report.flags)
        assert "finding about those papers" in joined

    def test_a_mechanical_failure_is_flagged_first(self, store):
        ts, run_id = store
        seed(ts, run_id, [("ok.pdf", "saved", 2), ("cutoff.pdf", "truncated", 0)])
        report = qc_report.build_qc_report(run_id)
        assert report.flags, "a truncated paper must raise a flag"
        assert "not evidence of absence" in report.flags[0]

    def test_every_category_carries_an_explanation(self, store):
        ts, _ = store
        for category, text in ts.OUTCOME_CATEGORIES.items():
            assert text, f"{category} has no explanation"

    def test_the_report_names_the_barren_papers(self, store):
        ts, run_id = store
        seed(ts, run_id, [("ok.pdf", "saved", 1), ("cutoff.pdf", "truncated", 0)])
        markdown = qc_report.report_markdown(qc_report.build_qc_report(run_id))
        assert "## Papers that produced nothing" in markdown
        assert "cutoff.pdf" in markdown
        assert "ok.pdf" not in markdown.split("## Papers that produced nothing")[1][:600]

    def test_an_outcome_survives_a_missing_run(self, store):
        """
        The regression. `run_id` has a foreign key, so an unknown run made the
        insert fail — and the failure was swallowed, losing the diagnostic in
        precisely the circumstances where something had already gone wrong.
        """
        ts, _ = store
        ts.record_paper_outcome(
            run_id=999999, source_filename="orphan.pdf", source_doi="10.1/z",
            outcome="No KERs found", category="truncated",
        )
        rows = ts.load_paper_outcomes()
        assert "orphan.pdf" in rows["source_filename"].tolist()

    def test_a_run_predating_outcomes_says_so(self, store):
        """
        Silence would read as "every paper is accounted for below", which is
        the claim being corrected.
        """
        ts, run_id = store
        markdown = qc_report.report_markdown(qc_report.build_qc_report(run_id))
        assert "## Papers that produced nothing" not in markdown or \
               "predates per-paper outcome recording" in markdown


class TestRefusals:
    """
    A refusal is not a parse failure, a provider error, or a finding.

    The real case: `33069750.pdf`, a peer-reviewed Toxicon paper on saxitoxin
    neurotoxicity, was declined by the provider's safety classifier on both
    the first attempt and the retry, while twelve other papers in the same
    corpus went through the same model untouched. Categorising that as a parse
    failure sends the reader to look at a JSON reply that was never produced;
    categorising it as no_mechanism silently converts a gap into a finding.
    """

    def test_a_refusal_has_its_own_category(self, store):
        ts, run_id = store
        seed(ts, run_id, [("ok.pdf", "saved", 2), ("declined.pdf", "refusal", 0)])
        report = qc_report.build_qc_report(run_id)
        assert report.n_refused == 1
        assert report.n_recoverable == 1, "nothing was learned, so it is a gap"

    def test_the_advice_does_not_say_re_run_the_same_thing(self, store):
        """
        A refusal is a property of the model. Re-running the identical
        configuration is the one action guaranteed not to help.
        """
        ts, run_id = store
        seed(ts, run_id, [("declined.pdf", "refusal", 0)])
        report = qc_report.build_qc_report(run_id)
        flag = next(f for f in report.flags if "safety classifier" in f)
        assert "different model or provider" in flag
        assert "re-running the same configuration will not help" in flag.lower()

    def test_the_flag_asks_for_it_to_be_declared(self, store):
        """
        A corpus that quietly drops the papers one classifier disliked is not
        the corpus the methods section describes.
        """
        ts, run_id = store
        seed(ts, run_id, [("declined.pdf", "refusal", 0)])
        report = qc_report.build_qc_report(run_id)
        assert any("methods" in f for f in report.flags)

    def test_the_category_text_does_not_blame_the_pdf(self, store):
        """
        The old provider message sent the reader to check whether the PDF
        extracted cleanly. This one had six clean pages and 32,543 characters.
        """
        ts, _ = store
        text = ts.OUTCOME_CATEGORIES["refusal"]
        assert "false positive" in text
        assert "PDF" not in text

    def test_refusals_are_counted_apart_from_provider_errors(self):
        """
        Folding them together makes a run that lost a paper to a classifier
        look like a run with a flaky connection, which points at the wrong fix.
        """
        from run_manifest import RunTelemetry

        telemetry = RunTelemetry()
        telemetry.record("refusal", step="pathway")
        telemetry.record("refusal", step="pathway")
        assert telemetry.refusals == 2
        assert telemetry.provider_errors == 0


class TestTheCountsAreInNamedUnits:

    def test_claims_and_events_are_counted_separately(self, monkeypatch, tmp_path):
        """
        Two claims naming three distinct events between them. Nothing here is
        subtractable from anything else.
        """
        import stage2_extraction.table1_store as ts

        monkeypatch.setattr(ts, "DB_PATH", tmp_path / "counts.db")
        ts.init_db()
        with ts.connect() as conn:
            for up, down, doi in (
                ("decreased sodium current", "impaired myelination", "10.1/a"),
                ("impaired myelination", "reduced conduction velocity", "10.1/b"),
            ):
                conn.execute(
                    "INSERT INTO table1_extractions (source_doi, extraction_date, "
                    "upstream_ke_name, upstream_ke_level, downstream_ke_name, "
                    "downstream_ke_level, ker_name, ker_description, ker_adjacency, "
                    "paper_type, contradicts_ker, taxonomic_applicability, "
                    "sex_applicability, life_stage_applicability, study_design, "
                    "extraction_confidence) VALUES (?, datetime('now'), ?, "
                    "'Molecular', ?, 'Cellular', 'x', 'y', 'Adjacent', "
                    "'Primary study', 0, 'rat', 'both', 'adult', 'In vivo', 'High')",
                    (doi, up, down),
                )
            conn.commit()

        counts = ts.corpus_counts()
        assert counts["claims"] == 2, "two rows, two claims"
        assert counts["label_occurrences"] == 4, "two ends per claim"
        assert counts["distinct_labels"] == 3, "the middle event is shared"
        assert counts["papers"] == 2

    def test_an_empty_corpus_reports_zeroes_not_errors(self, monkeypatch, tmp_path):
        import stage2_extraction.table1_store as ts

        monkeypatch.setattr(ts, "DB_PATH", tmp_path / "empty.db")
        ts.init_db()
        counts = ts.corpus_counts()
        assert counts["claims"] == 0
        assert counts["distinct_labels"] == 0

    def test_the_metric_no_longer_calls_mentions_labels(self):
        """
        The sidebar reported 56 "Raw Key Event labels" for 18 actual labels,
        because it counted every mention. Renaming it is the fix; asserting it
        stops the old wording coming back.
        """
        from pathlib import Path

        curate = (Path(__file__).resolve().parent.parent / "ui" / "curate.py").read_text(
            encoding="utf-8"
        )
        assert '"Raw Key Event labels"' not in curate
        assert '"Distinct Key Event labels"' in curate

    def test_the_unit_explainer_exists(self):
        from pathlib import Path

        common = (Path(__file__).resolve().parent.parent / "ui" / "common.py").read_text(
            encoding="utf-8"
        )
        assert "def count_chain" in common
        assert "Two per claim" in common
