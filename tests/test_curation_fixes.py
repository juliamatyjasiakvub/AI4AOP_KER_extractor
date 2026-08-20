"""
End-to-end check of the four fixes, run against a copy of the real database.

Uses only the standard library so it runs anywhere the app runs.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LIVE_DB = REPO / "aop_rag.db"
# Copied out of the repository: the live file must never be written to, and
# SQLite needs a filesystem it can lock.
TEST_DB = Path(tempfile.gettempdir()) / "aop_rag_verify.db"


class ManifestCounters(unittest.TestCase):
    """A pathway run must count the papers that contributed."""

    def test_pathway_extracted_counts_only_contributing_papers(self):
        from run_manifest import RunTelemetry

        t = RunTelemetry()
        for n_steps in (3, 0, 1, 0):
            t.record("paper_attempted")
            t.record("pathway_extracted", n_steps=n_steps)

        self.assertEqual(t.papers_attempted, 4)
        self.assertEqual(t.papers_with_kers, 2, "empty chains must not count")
        self.assertEqual(t.kers_extracted, 4)

    def test_unknown_events_are_still_ignored(self):
        from run_manifest import RunTelemetry

        t = RunTelemetry()
        t.record("something_new", n=1)          # must not raise
        self.assertEqual(t.papers_with_kers, 0)


class AssemblyGuard(unittest.TestCase):
    """One unassemblable link must cost that link and nothing else."""

    def test_bad_link_is_warned_about_not_raised(self):
        from stage2_extraction import ker_extractor as ke

        steps = [
            ke.PathwayStep(
                from_event="A", to_event="B", from_level="Molecular",
                to_level="Cellular", description="a to b",
            ),
            ke.PathwayStep(
                from_event="B", to_event="C", from_level="Cellular",
                to_level="Cellular", description="b to c",
            ),
        ]
        pathway = ke.PathwayResult(bears_on_question=True, steps=steps)

        real_assemble = ke._assemble
        calls = {"n": 0}

        def flaky_assemble(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ke.ExtractionValidationError("level went backwards")
            return real_assemble(*args, **kwargs)

        ok_step = ke.StepResult(
            step="study_meta", ok=True, prompt="", raw_response="{}", parsed={},
        )

        ke.extract_pathway = lambda *a, **k: pathway
        ke.extract_study_meta = lambda *a, **k: ok_step
        ke.extract_applicability = lambda *a, **k: ok_step
        ke._assemble = flaky_assemble
        try:
            doc = ke.PaperDocument(
                filename="test.pdf", doi="10.0/test", title="t", full_text="text",
            )
            rows, _, warnings = ke.extract_pathway_rows(
                doc, "A", "C", cfg=None, paper_text="text",
            )
        finally:
            ke._assemble = real_assemble

        self.assertEqual(len(rows), 1, "the good link must survive")
        self.assertTrue(any("could not be assembled" in w for w in warnings))


class NormalizeAndAssign(unittest.TestCase):
    """The crashing call site, and the new assignment table, on real rows."""

    def setUp(self):
        """A fresh copy per test: one test mutates what the other asserts on."""
        if not LIVE_DB.exists():
            raise unittest.SkipTest("no live database to copy")
        shutil.copy(LIVE_DB, TEST_DB)

        from stage2_extraction import ke_normalizer, ols4_client, table1_store

        table1_store.DB_PATH = TEST_DB
        ols4_client.set_db_path(TEST_DB)
        self.store = table1_store

        self.report = ke_normalizer.normalize_table1(
            table1_store.load_table1_as_dataframe(),
            threshold=0.86,
            ols4_enabled=False,        # no network in the test
            ols4_min_score=0.45,
        )

    def tearDown(self):
        TEST_DB.unlink(missing_ok=True)

    def test_normalize_table1_signature_matches_the_ui(self):
        """The exact call ui/curate.py makes, which used to raise TypeError."""
        report = self.report

        self.assertGreater(report.n_raw, 0)
        self.assertGreater(report.n_canonical, 0)
        self.assertLessEqual(report.n_canonical, report.n_raw)
        # The UI reads both of these names.
        self.assertEqual(report.n_raw, report.n_raw_labels)

        canonical = self.store.load_canonical_kes()
        self.assertEqual(len(canonical), report.n_canonical,
                         "the result must be persisted, not just returned")

        aliases = self.store.load_alias_map()
        self.assertTrue(aliases, "every raw label needs an alias row")

        # Table 1 must be back-filled, or the map has nothing to draw.
        with sqlite3.connect(TEST_DB) as conn:
            unfilled = conn.execute(
                "SELECT COUNT(*) FROM table1_extractions "
                "WHERE upstream_ke_canonical_id IS NULL"
            ).fetchone()[0]
        self.assertEqual(unfilled, 0)

    def test_assignment_folds_synonyms_and_excludes(self):
        from stage2_extraction import canonical_groups as cg

        aliases = self.store.load_alias_map()
        canonical = self.store.load_canonical_kes()
        names = {int(r["canonical_id"]): str(r["canonical_name"])
                 for _, r in canonical.iterrows()}

        labels = sorted(aliases)
        self.assertGreaterEqual(len(labels), 3, "need a few labels to test with")

        # Point every label at its current event, then fold the second label
        # into the first and drop the third off the map.
        assignments = [(lbl, names[aliases[lbl]]) for lbl in labels]
        survivor = assignments[0][1]
        assignments[1] = (labels[1], survivor)
        excluded_name = assignments[2][1]

        result = cg.apply_assignments(
            assignments,
            excluded=[excluded_name],
            curator="verifier",
            rationale="automated check",
        )

        self.assertEqual(result["n_labels"], len(labels))
        self.assertGreaterEqual(result["n_synonyms"], 1)
        self.assertEqual(result["n_rejected"], 1)

        after = self.store.load_canonical_kes()
        by_name = {str(r["canonical_name"]): r for _, r in after.iterrows()}

        self.assertIn(survivor, by_name)
        merged_aliases = str(by_name[survivor]["aliases"])
        self.assertIn(labels[0], merged_aliases)
        self.assertIn(labels[1], merged_aliases,
                      "the folded label must survive as a synonym")

        self.assertEqual(by_name[excluded_name]["curation_status"], "rejected")
        self.assertEqual(by_name[survivor]["curation_status"], "accepted")

        # Nothing is deleted: every raw label still resolves somewhere.
        self.assertEqual(len(self.store.load_alias_map()), len(labels))

        with sqlite3.connect(TEST_DB) as conn:
            conn.row_factory = sqlite3.Row
            decision = conn.execute(
                "SELECT * FROM merge_decision WHERE action = 'assign_labels' "
                "ORDER BY decision_id DESC LIMIT 1"
            ).fetchone()
            unfilled = conn.execute(
                "SELECT COUNT(*) FROM table1_extractions "
                "WHERE downstream_ke_canonical_id IS NULL"
            ).fetchone()[0]

        self.assertIsNotNone(decision, "the decision must be auditable")
        self.assertEqual(decision["curator"], "verifier")
        self.assertTrue(decision["before_state"] and decision["after_state"])
        self.assertEqual(unfilled, 0, "Table 1 must be re-linked after assignment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
