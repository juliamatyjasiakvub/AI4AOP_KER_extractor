"""
Abbreviations come from the papers, not from a list somebody maintained.

The failure this guards against is not a crash. It is a corpus in a field
nobody extended the built-in table for: "Increased HSC activation" and
"Increased hepatic stellate cell activation" stay two canonical Key Events,
the map shows two nodes where the evidence supports one, and nothing anywhere
says that a merge was missed. The old table made that outcome a property of
which field the last maintainer happened to work in.

Three things are asserted here, in order of how badly they fail:

    * a paper's own definition merges labels the built-in table has never
      heard of, in fields the tool has never been pointed at;
    * two papers that define the same abbreviation differently do NOT get
      silently reconciled — the expansion is withheld and reported, because
      merging androgen receptor with aldose reductase is worse than leaving
      two nodes for a curator to look at;
    * calling `normalise_label` with no table behaves exactly as before, so
      every existing caller is unaffected.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_extraction.ke_normalizer import (  # noqa: E402
    agreed_abbreviations,
    build_canonical_kes,
    normalise_label,
)
from stage2_extraction.ke_synonyms import paper_abbreviations  # noqa: E402


class PaperDefinitionsDriveMerging(unittest.TestCase):
    """Fields the built-in table never covered."""

    #: One per domain the tool has no built-in vocabulary for. Each is the
    #: pair a real corpus produces: one paper spells the event out, another
    #: uses the shorthand it defined in its own introduction.
    CASES = (
        ("liver", "HSC", "hepatic stellate cell",
         "Increased HSC activation", "Increased hepatic stellate cell activation"),
        ("kidney", "PTEC", "proximal tubule epithelial cell",
         "Injury to PTEC", "Injury to proximal tubule epithelial cell"),
        ("cardiac", "APD", "action potential duration",
         "Prolonged APD", "Prolonged action potential duration"),
        ("reproductive", "AMH", "anti mullerian hormone",
         "Reduced AMH secretion", "Reduced anti mullerian hormone secretion"),
        ("respiratory", "AEC", "alveolar epithelial cell",
         "AEC apoptosis", "alveolar epithelial cell apoptosis"),
    )

    def test_a_paper_s_own_abbreviation_merges_its_labels(self):
        for domain, abbrev, long_form, short_label, long_label in self.CASES:
            with self.subTest(domain=domain):
                table = {abbrev: long_form}
                self.assertEqual(
                    normalise_label(short_label, abbreviations=table),
                    normalise_label(long_label, abbreviations=table),
                    f"{domain}: '{short_label}' and '{long_label}' are one event",
                )

    def test_without_the_table_they_stay_apart(self):
        """The regression being fixed, stated as a test rather than a claim."""
        for domain, _abbrev, _long, short_label, long_label in self.CASES:
            with self.subTest(domain=domain):
                self.assertNotEqual(
                    normalise_label(short_label),
                    normalise_label(long_label),
                    f"{domain}: built-in table should not know this field",
                )

    def test_plurals_are_expanded_too(self):
        """Papers write OPCs as often as OPC; the definition names one."""
        table = {"HSC": "hepatic stellate cell"}
        self.assertEqual(
            normalise_label("Activation of HSCs", abbreviations=table),
            normalise_label("Activation of hepatic stellate cells", abbreviations=table),
        )

    def test_longest_abbreviation_wins(self):
        """
        Expanding the short one first would eat the long one.

        A paper defining both "AEC" and "AECII" writes the second to mean
        something narrower, and replacing "AEC" first leaves "alveolar
        epithelial cellII" — a term no paper contains.
        """
        table = {"AEC": "alveolar epithelial cell",
                 "AECII": "alveolar type II cell"}
        self.assertEqual(
            normalise_label("AECII apoptosis", abbreviations=table),
            normalise_label("alveolar type II cell apoptosis", abbreviations=table),
        )


class DisagreementIsReportedNotResolved(unittest.TestCase):
    """
    The case where guessing would be worse than doing nothing.

    "AR" is androgen receptor in an endocrine paper and aldose reductase in a
    metabolic one. Picking either expands one paper's labels into the other
    paper's biology, and after the merge the record shows a single well
    evidenced Key Event rather than two findings about different proteins.
    """

    def test_papers_that_agree_produce_an_expansion(self):
        agreed, conflicts = agreed_abbreviations({
            "10.1234/a": {"HSC": "hepatic stellate cell"},
            "10.1234/b": {"HSC": "hepatic stellate cell"},
        })
        self.assertEqual(agreed["HSC"], "hepatic stellate cell")
        self.assertEqual(conflicts, [])

    def test_papers_that_disagree_produce_no_expansion(self):
        agreed, conflicts = agreed_abbreviations({
            "10.1234/a": {"AR": "androgen receptor"},
            "10.1234/b": {"AR": "aldose reductase"},
        })
        self.assertNotIn("AR", agreed, "a contested abbreviation must not be guessed")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["abbrev"], "AR")
        self.assertEqual(conflicts[0]["n_papers"], 2)
        self.assertCountEqual(
            conflicts[0]["long_forms"], ["androgen receptor", "aldose reductase"]
        )

    def test_a_conflict_does_not_suppress_the_rest(self):
        agreed, conflicts = agreed_abbreviations({
            "10.1234/a": {"AR": "androgen receptor", "HSC": "hepatic stellate cell"},
            "10.1234/b": {"AR": "aldose reductase", "HSC": "hepatic stellate cell"},
        })
        self.assertEqual(agreed.get("HSC"), "hepatic stellate cell")
        self.assertEqual([c["abbrev"] for c in conflicts], ["AR"])

    def test_case_and_blank_definitions_are_tolerated(self):
        agreed, conflicts = agreed_abbreviations({
            "10.1234/a": {"HSC": "Hepatic Stellate Cell"},
            "10.1234/b": {"HSC": "hepatic stellate cell", "  ": "ignored"},
            "10.1234/c": {"XX": ""},
        })
        self.assertEqual(agreed["HSC"], "hepatic stellate cell")
        self.assertNotIn("XX", agreed)
        self.assertEqual(conflicts, [])


class HarvestingFromRealProse(unittest.TestCase):
    """The Schwartz-Hearst pass, on text from a field with no built-in terms."""

    TEXT = (
        "Chronic injury activates hepatic stellate cells (HSCs), the principal "
        "source of extracellular matrix in the liver. We measured alanine "
        "aminotransferase (ALT) in serum and assessed proximal tubule "
        "epithelial cell (PTEC) viability. Cells were cultured in DMEM. "
        "See Fig. 2 and Table 1 for details (n = 6)."
    )

    def test_definitions_are_read_out_of_the_paper(self):
        found = paper_abbreviations(self.TEXT)
        self.assertEqual(found.get("HSC"), "hepatic stellate cells")
        self.assertEqual(found.get("PTEC"), "proximal tubule epithelial cell")
        self.assertEqual(found.get("ALT"), "alanine aminotransferase")

    def test_structural_brackets_are_not_definitions(self):
        found = paper_abbreviations(self.TEXT)
        for noise in ("Fig", "Table", "n"):
            self.assertNotIn(noise, found)

    def test_harvested_definitions_merge_a_corpus(self):
        """End to end: prose in, one canonical Key Event out."""
        table, _ = agreed_abbreviations({"10.1234/a": paper_abbreviations(self.TEXT)})
        raw = [
            ("Increased HSC activation", "Cellular", None),
            ("Increased hepatic stellate cell activation", "Cellular", None),
        ]
        without, _, _ = build_canonical_kes(raw, {})
        with_table, _, _ = build_canonical_kes(raw, {}, abbreviations=table)
        self.assertEqual(len(without), 2, "the failure this change fixes")
        self.assertEqual(len(with_table), 1, "one event, two wordings")


class ExistingCallersAreUnaffected(unittest.TestCase):
    """
    `normalise_label` is called from six places in `semantic_merge` that pass
    no table. Those must behave exactly as they did.
    """

    def test_the_builtin_table_still_applies(self):
        self.assertEqual(
            normalise_label("Increased ROS"),
            normalise_label("Increased reactive oxygen species"),
        )

    def test_an_empty_table_changes_nothing(self):
        for label in ("Increased ROS production", "Decreased T4 levels", ""):
            with self.subTest(label=label):
                self.assertEqual(
                    normalise_label(label),
                    normalise_label(label, abbreviations={}),
                )

    def test_polarity_is_still_never_folded_away(self):
        """The guard the whole module exists for, re-checked under a table."""
        table = {"HSC": "hepatic stellate cell"}
        self.assertNotEqual(
            normalise_label("Increased HSC activation", abbreviations=table),
            normalise_label("Decreased HSC activation", abbreviations=table),
        )


if __name__ == "__main__":
    unittest.main()
