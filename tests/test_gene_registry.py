"""
Which tokens get asked of HGNC, and what is done with the answer.

Two things went wrong here, and only one of them looked like a bug.

The visible one: `_NOT_SYMBOLS` was a hand-written list of capitalised words
that are not genes, and it had been written while reading one corpus. It
blocked TTX, DMEM, FBS and PBS — an electrophysiology bench — and said nothing
about ALT, AST, ALP, BUN or LDH, which a liver paper writes on every page.

The invisible one: those liver abbreviations are *real HGNC aliases*. ALT
resolves to GPT and AST to GOT1, both correctly. So the docstring's assurance
that a false positive "costs one cached HTTP request that returns nothing" was
wrong in exactly the case that mattered: it returns a plausible gene, and the
code then expanded that gene's whole group into the screening vocabulary.

The replacement is a rule about how a token matched rather than a list of what
somebody remembered. These tests pin the rule and the shape filter around it;
nothing here touches the network, so what is asserted is the decision, not
HGNC's current contents.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2_extraction import gene_registry as gr  # noqa: E402


class SymbolsAcrossFields(unittest.TestCase):
    """The shape filter has to work outside the corpus it was written for."""

    NEURO = ("SCN8A", "NaV1.6", "MBP", "GFAP", "PLP1", "KCNQ2", "PDGFRA")
    LIVER_KIDNEY = ("CYP1A1", "ALB", "HNF4A", "NR1I2", "HAVCR1", "GGT1", "SLC22A6")
    CARDIO_REPRO_IMMUNE = ("KCNH2", "MYH7", "TNNI3", "CYP19A1", "AR", "ESR1",
                           "IL6", "TNF", "NFE2L2", "TP53")

    def test_symbols_from_every_field_are_asked_about(self):
        for group in (self.NEURO, self.LIVER_KIDNEY, self.CARDIO_REPRO_IMMUNE):
            for symbol in group:
                with self.subTest(symbol=symbol):
                    self.assertTrue(
                        gr.looks_like_gene_symbol(symbol),
                        f"{symbol} would never reach HGNC",
                    )

    def test_prose_is_not_asked_about(self):
        for word in ("channel", "protein", "expression", "cells", "control",
                     "THE", "AND", "CELL", "DATA", "MICE"):
            with self.subTest(word=word):
                self.assertFalse(gr.looks_like_gene_symbol(word))

    def test_the_stoplist_holds_english_and_not_bench_vocabulary(self):
        """
        The list is allowed to name ordinary words. It is not allowed to be a
        record of one laboratory's reagents, because the next laboratory's are
        different and nobody will notice they are missing.
        """
        for reagent in ("ttx", "dmem", "fbs", "pbs", "elisa", "pcr"):
            with self.subTest(reagent=reagent):
                self.assertNotIn(reagent, gr._NOT_SYMBOLS)

    def test_nothing_absurd_gets_through(self):
        for token in ("", "   ", "a", "x" * 40, "two words"):
            with self.subTest(token=token):
                self.assertFalse(gr.looks_like_gene_symbol(token))


class MatchStrengthGatesTheFamily(unittest.TestCase):
    """
    Resolving a token is cheap and reversible. Expanding its gene group is not.

    A three-letter bare alias is where the ambiguity lives, so it gets the
    gene's own names and not its relatives. Everything with a digit, an
    approved-symbol match, or four or more characters is unambiguous enough.
    """

    def test_an_approved_symbol_is_always_strong(self):
        for symbol in ("ALB", "AR", "TNF", "MBP", "SCN8A"):
            with self.subTest(symbol=symbol):
                self.assertTrue(gr._is_strong_match(symbol, "symbol"))

    def test_a_nomenclature_alias_with_a_digit_is_strong(self):
        for alias in ("NaV1.6", "Nav1.2", "KCNQ2", "IL6"):
            with self.subTest(alias=alias):
                self.assertTrue(gr._is_strong_match(alias, "alias_symbol"))

    def test_a_long_alias_is_strong(self):
        self.assertTrue(gr._is_strong_match("PDGFRA", "alias_symbol"))
        self.assertTrue(gr._is_strong_match("HAVCR1", "prev_symbol"))

    def test_a_short_bare_alias_does_not_expand_a_family(self):
        """ALT is an alias of GPT and also the commonest liver readout."""
        for assay in ("ALT", "AST", "ALP", "BUN", "LDH"):
            with self.subTest(assay=assay):
                self.assertFalse(
                    gr._is_strong_match(assay, "alias_symbol"),
                    f"{assay} would drag its whole gene group into the vocabulary",
                )

    def test_the_same_string_is_strong_as_an_approved_symbol(self):
        """
        The rule is about the match, not the letters.

        If HGNC says a three-letter token IS the approved symbol, it is the
        gene — the coincidence with an assay name does not survive that.
        """
        self.assertFalse(gr._is_strong_match("ALT", "alias_symbol"))
        self.assertTrue(gr._is_strong_match("ALT", "symbol"))

    def test_an_unmatched_query_is_not_strong_by_accident(self):
        self.assertFalse(gr._is_strong_match("ALT", None))


class ProvenanceIsRecorded(unittest.TestCase):
    """A surprising term in a vocabulary should say where it came from."""

    def test_the_record_carries_the_index_that_matched(self):
        record = gr.GeneVocabulary(query="ALT", symbol="GPT",
                                   matched_by="alias_symbol", family_withheld=True)
        self.assertEqual(record.matched_by, "alias_symbol")
        self.assertTrue(record.family_withheld)

    def test_defaults_are_honest_about_knowing_nothing(self):
        record = gr.GeneVocabulary(query="anything")
        self.assertIsNone(record.matched_by)
        self.assertFalse(record.family_withheld)
        self.assertEqual(record.family, [])


if __name__ == "__main__":
    unittest.main()
