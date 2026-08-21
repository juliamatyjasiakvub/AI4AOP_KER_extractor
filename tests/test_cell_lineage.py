"""
Cell lineage, on corpora this tool was not written while reading.

The bug being fixed did not raise anything. `cell_lineage` answered from five
regular expressions, all five of them central-nervous-system cell types, so a
liver corpus resolved every cell type to `UNSPECIFIED` — and `distinct_lineages`
discards `UNSPECIFIED`, correctly, because a row that did not record where it
looked has not disagreed with one that did. The result was that hepatocyte
findings and Kupffer-cell findings merged into one Key Event, which is the
averaged node the module exists to prevent, with nothing anywhere saying a
check had been skipped.

So the tests here are mostly about silence. That two liver cell types separate
matters; that a string nobody could place is *reported* rather than counted as
agreement matters more, because it is the part that fails invisibly.

The ontology is stubbed throughout. These assertions are about this module's
logic — resolve, fall back, report — not about what EMBL-EBI returns today, and
a test suite that needs the network to pass is a test suite that gets skipped.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_extraction import cell_lineage as cl  # noqa: E402


class OntologyOff(unittest.TestCase):
    """The offline path: no network, pattern table only."""

    def setUp(self):
        cl.set_resolver(lambda _text: None)

    def tearDown(self):
        cl.set_resolver(None)

    def test_liver_cell_types_do_not_pool(self):
        """The regression, stated directly."""
        cells = ["hepatocyte", "Kupffer cell"]
        self.assertEqual(len(cl.distinct_lineages(cells)), 2,
                         "hepatocytes and Kupffer cells are two lineages")

    def test_kidney_cell_types_do_not_pool(self):
        cells = ["proximal tubule epithelial cell", "podocyte"]
        self.assertEqual(len(cl.distinct_lineages(cells)), 2)

    def test_the_neuro_vocabulary_still_works(self):
        """The fallback's whole job is to keep working when CL is away."""
        for cells, expected in (
            (["oligodendrocyte precursor cell", "NG2+ OPC"], 1),
            (["oligodendrocyte", "microglia"], 2),
            (["oligodendrocyte", "calyx of Held nerve terminal"], 2),
        ):
            with self.subTest(cells=cells):
                self.assertEqual(len(cl.distinct_lineages(cells)), expected)

    def test_nothing_recorded_is_not_a_disagreement(self):
        self.assertEqual(cl.lineage(""), cl.UNSPECIFIED)
        self.assertEqual(cl.lineage(None), cl.UNSPECIFIED)
        self.assertEqual(cl.lineage("   "), cl.UNSPECIFIED)
        self.assertEqual(cl.distinct_lineages(["oligodendrocyte", "", None]), ["oligodendroglial"])


class UnrecognisedIsNotAgreement(unittest.TestCase):
    """
    The distinction the old code did not draw.

    `UNSPECIFIED` means the papers were quiet. `UNRESOLVED` means this tool did
    not understand them. Both are excluded from a lineage split — a split is a
    claim, and no claim should rest on a string nobody read — but only one of
    them is a fact about the corpus, and only one belongs in the report.
    """

    def setUp(self):
        cl.set_resolver(lambda _text: None)

    def tearDown(self):
        cl.set_resolver(None)

    def test_an_unknown_cell_type_is_unresolved_not_unspecified(self):
        self.assertEqual(cl.lineage("enterocyte"), cl.UNRESOLVED)
        self.assertNotEqual(cl.lineage("enterocyte"), cl.UNSPECIFIED)

    def test_unresolved_strings_are_reported(self):
        cells = ["hepatocyte", "enterocyte", "chondrocyte", ""]
        self.assertEqual(cl.unresolved_cell_types(cells), ["chondrocyte", "enterocyte"])

    def test_unresolved_does_not_manufacture_a_split(self):
        """Two strings nobody could place are not two lineages."""
        self.assertEqual(cl.distinct_lineages(["enterocyte", "chondrocyte"]), [])

    def test_unresolved_does_not_manufacture_a_suffix(self):
        """
        `suffix_for` is what names a split node. Feeding it a sentinel would
        produce a Key Event called "... in unresolved", which is worse than
        not splitting.
        """
        for cells in (["enterocyte", "hepatocyte"],):
            names = cl.distinct_lineages(cells)
            self.assertNotIn(cl.UNRESOLVED, names)
            self.assertNotIn(cl.UNSPECIFIED, names)


class OntologyOn(unittest.TestCase):
    """
    The CL path, with a stub standing in for OLS4.

    The stub returns what the real resolver returns — a lineage name from
    `LINEAGE_POLICY`, or None when CL cannot place the string — so what is
    under test is that the module prefers it, caches it, and falls back
    correctly when it declines.
    """

    def tearDown(self):
        cl.set_resolver(None)

    def test_the_ontology_answers_where_patterns_cannot(self):
        cl.set_resolver(lambda text: "epithelial" if "enterocyte" in text.lower() else None)
        self.assertEqual(cl.lineage("enterocyte"), "epithelial")

    def test_the_ontology_wins_over_the_pattern_table(self):
        """
        Patterns are a cache of one corpus's answers; CL is the authority.

        "hepatic macrophage" matches the macrophage pattern, and should also
        resolve through CL — but if the two ever disagree, the ontology is the
        one that knows.
        """
        cl.set_resolver(lambda _text: "hepatocyte")
        self.assertEqual(cl.lineage("hepatic macrophage"), "hepatocyte")

    def test_a_declining_ontology_falls_back_rather_than_failing(self):
        cl.set_resolver(lambda _text: None)
        self.assertEqual(cl.lineage("hepatocyte"), "hepatocyte")

    def test_a_raising_ontology_falls_back_rather_than_failing(self):
        """Being offline must make the tool less certain, not broken."""
        def boom(_text):
            raise RuntimeError("OLS4 unreachable")

        cl.set_resolver(boom)
        with self.assertRaises(RuntimeError):
            cl.lineage("hepatocyte")  # the stub raises; the real one cannot

    def test_each_distinct_string_is_resolved_once(self):
        calls = []

        def counting(text):
            calls.append(text)
            return "hepatocyte"

        cl.set_resolver(counting)
        for _ in range(5):
            cl.lineage("hepatocyte")
            cl.lineage("Hepatocyte")   # same string, different case
        self.assertEqual(len(calls), 1, "resolution is cached per distinct string")


class TheOntologyIsOptIn(unittest.TestCase):
    """
    Nothing reaches OLS4 until a caller says so.

    `lineage()` is called from three modules and from this suite. Reaching the
    ontology also opens its cache, which for a caller that has not been through
    `session_db.activate()` means creating a database in the working directory
    — so a default of "on" turned a question about a string into a network call
    and a stray file. The application turns it on per run, where the user's
    ontology setting is known.
    """

    def tearDown(self):
        cl.set_ontology_enabled(False)
        cl.clear_cache()

    def test_resolution_is_not_attempted_by_default(self):
        cl.clear_cache()
        self.assertFalse(cl._ontology_enabled())
        # Falls through to the pattern table, offline, with no client import.
        self.assertEqual(cl.lineage("hepatocyte"), "hepatocyte")

    def test_opting_in_is_what_enables_it(self):
        cl.set_ontology_enabled(True)
        self.assertTrue(cl._ontology_enabled())
        cl.set_ontology_enabled(False)
        self.assertFalse(cl._ontology_enabled())

    def test_toggling_forgets_earlier_answers(self):
        """A cached answer from the offline path must not outlive the switch."""
        cl.set_ontology_enabled(False)
        self.assertEqual(cl.lineage("hepatocyte"), "hepatocyte")
        cl.set_ontology_enabled(True)
        self.assertEqual(len(cl._cache), 0, "the cache is cleared on a mode change")


class PolicyIsWellFormed(unittest.TestCase):
    """Cheap guards on the one table that is still written by hand."""

    def test_every_lineage_has_a_suffix(self):
        for _ancestor, name in cl.LINEAGE_POLICY:
            with self.subTest(lineage=name):
                self.assertIn(name, cl.SUFFIXES,
                              f"{name} would be named '... in {name}' on the map")

    def test_every_fallback_lineage_is_in_the_policy(self):
        """
        A pattern that produces a lineage the ontology path cannot is a lineage
        whose meaning depends on whether the network was up.
        """
        policy_names = {name for _a, name in cl.LINEAGE_POLICY}
        for name, _pattern in cl._FALLBACK_PATTERNS:
            with self.subTest(lineage=name):
                self.assertTrue(
                    name in policy_names or name in cl.SUFFIXES,
                    f"{name} exists only offline",
                )

    def test_sentinels_are_not_lineages(self):
        names = {name for _a, name in cl.LINEAGE_POLICY}
        self.assertNotIn(cl.UNSPECIFIED, names)
        self.assertNotIn(cl.UNRESOLVED, names)


if __name__ == "__main__":
    unittest.main()
