"""
The failures found by auditing the Nav1.2 run against its 16 source PDFs.

Each test encodes one finding from that audit. They use the real shapes the
pipeline produces, with the two mechanisms the corpus actually contains:

    oligodendroglial   reduced Nav1.2 -> loss of spiking -> impaired maturation
    microglial         raised Na+ current -> activation -> p38 -> proNGF -> death

Both were being assembled into one pathway, because the two branches shared
the string "Voltage-gated sodium channel" at one end and "Oligodendrocyte
differentiation" at the other.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd

from stage2_extraction import ke_normalizer, table2_synthesis as t2


OL = "oligodendrocyte precursor cells"
MICROGLIA = "activated microglia"


def _row(up, down, direction="positive", up_cell=None, down_cell=None,
         doi="10.0/a", contradicts=False, **kwargs):
    """One Table 1 row with only the fields the functions under test read."""
    base = {
        "record_id": abs(hash((up, down, doi))) % 100000,
        "source_doi": doi,
        "upstream_ke_name": up,
        "downstream_ke_name": down,
        "upstream_ke_level": "Molecular",
        "downstream_ke_level": "Cellular",
        "upstream_ke_id": None,
        "downstream_ke_id": None,
        "ker_id": None,
        "ker_name": f"{up} leads to {down}",
        "ker_description": "",
        "ker_adjacency": "Adjacent",
        "contradicts_ker": contradicts,
        "aop_id": None,
        "aop_status": "novel",
        "direction": direction,
        "upstream_cell_type": up_cell,
        "downstream_cell_type": down_cell,
        "upstream_change": None,
        "downstream_change": None,
        "essentiality_evidence": None,
        "quantitative_relationships": None,
        "response_response_relationship": None,
        "time_scale": None,
        "modulating_factors": None,
        "feedforward_feedback_loops": None,
        "biological_plausibility": None,
        "empirical_evidence_summary": None,
        "taxonomic_applicability": "Mouse",
        "sex_applicability": "Not specified",
        "life_stage_applicability": "Not specified",
        "study_design": "In vivo",
        "chemical_stressor": None,
        "exposure_route": None,
        "cited_evidence_dois": None,
        "extraction_confidence": "High",
        "extraction_date": "2026-08-10",
        "n_evidence_spans": 1,
        "n_verified_spans": 1,
    }
    base.update(kwargs)
    return base


class SignIsKept(unittest.TestCase):
    """"VGSC -> loss of spiking" was recorded with the wrong sign, then lost."""

    def test_opposite_findings_do_not_aggregate_into_one_confident_edge(self):
        # 28916793: raising Nav1.2 function enables spiking.
        # 34496232: deleting SCN2A abolishes it. Same two labels, opposite sign.
        df = pd.DataFrame([
            _row("Voltage-gated sodium channel", "Spiking phenotype",
                 direction="positive", doi="10.1038/s41467-017-00688-0"),
            _row("Voltage-gated sodium channel", "Spiking phenotype",
                 direction="negative", doi="10.1016/j.celrep.2021.109653"),
        ])
        table2 = t2.compute_table2(df, normalized=False)

        self.assertEqual(len(table2), 1, "same edge, so one consolidated row")
        row = table2.iloc[0]
        self.assertTrue(row["sign_conflict"])
        self.assertEqual(row["direction"], "conflicting")
        self.assertEqual(row["n_positive"], 1)
        self.assertEqual(row["n_negative"], 1)

        conflicts = t2.sign_conflicts(table2)
        self.assertEqual(len(conflicts), 1,
                         "a conflicted edge must be reported, not just scored")

    def test_conflicted_edge_scores_below_the_agreeing_one(self):
        agreeing = pd.DataFrame([
            _row("A", "B", direction="positive", doi="10.0/1"),
            _row("A", "B", direction="positive", doi="10.0/2"),
        ])
        conflicted = pd.DataFrame([
            _row("A", "B", direction="positive", doi="10.0/1"),
            _row("A", "B", direction="negative", doi="10.0/2"),
        ])
        agree_score = t2.compute_table2(agreeing, normalized=False).iloc[0]["confidence_score"]
        conflict_score = t2.compute_table2(conflicted, normalized=False).iloc[0]["confidence_score"]

        self.assertLess(conflict_score, agree_score,
                        "two papers disagreeing is not two papers agreeing")

    def test_unsigned_edges_are_listed(self):
        df = pd.DataFrame([_row("A", "B", direction="unclear")])
        table2 = t2.compute_table2(df, normalized=False)
        self.assertEqual(table2.iloc[0]["direction"], "unclear")
        self.assertEqual(len(t2.unsigned_edges(table2)), 1)


class CellTypesAreNotPooled(unittest.TestCase):
    """One label, two cell types, two mechanisms — the fusion at its source."""

    def test_conflicting_cell_types_are_reported(self):
        raw = [
            ("Voltage-gated sodium channel", "Molecular", None, OL),
            ("Voltage-gated sodium channel", "Molecular", None, MICROGLIA),
            ("p38MAPK activation", "Molecular", None, MICROGLIA),
        ]
        conflicts = ke_normalizer.find_cell_type_conflicts(raw)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["label"], "Voltage-gated sodium channel")
        self.assertCountEqual(conflicts[0]["cell_types"], [OL, MICROGLIA])

    def test_a_single_cell_type_is_not_a_conflict(self):
        raw = [
            ("Oligodendrocyte differentiation", "Cellular", None, OL),
            ("Oligodendrocyte differentiation", "Cellular", None, OL),
        ]
        self.assertEqual(ke_normalizer.find_cell_type_conflicts(raw), [])

    def test_cell_types_reach_table2(self):
        df = pd.DataFrame([
            _row("Voltage-gated sodium channel", "microglial activation",
                 up_cell=MICROGLIA, down_cell=MICROGLIA),
        ])
        row = t2.compute_table2(df, normalized=False).iloc[0]
        self.assertIn("microglia", str(row["cell_types"]).lower())


class AnchorDoesNotFuseMechanisms(unittest.TestCase):
    """The extractor must stop calling both channels the same event."""

    def test_same_event_in_a_second_cell_type_is_kept_apart(self):
        from stage2_extraction.ker_extractor import _canonical_anchor

        anchors: dict[str, str] = {}
        first = _canonical_anchor(
            "voltage gated sodium channel", "Voltage-gated sodium channel",
            "Oligodendrocyte differentiation", OL, anchors,
        )
        second = _canonical_anchor(
            "voltage gated sodium channel", "Voltage-gated sodium channel",
            "Oligodendrocyte differentiation", MICROGLIA, anchors,
        )

        self.assertEqual(first, "Voltage-gated sodium channel",
                         "the first mention still anchors")
        self.assertNotEqual(second, first,
                            "the microglial channel is not the same event")
        self.assertIn("microglia", second.lower())

    def test_same_cell_type_still_anchors(self):
        from stage2_extraction.ker_extractor import _canonical_anchor

        anchors: dict[str, str] = {}
        for _ in range(2):
            got = _canonical_anchor(
                "Voltage-gated sodium channel.", "Voltage-gated sodium channel",
                "Oligodendrocyte differentiation", OL, anchors,
            )
            self.assertEqual(got, "Voltage-gated sodium channel")

    def test_missing_cell_type_falls_back_to_the_old_behaviour(self):
        from stage2_extraction.ker_extractor import _canonical_anchor

        got = _canonical_anchor(
            "oligodendrocyte differentiation", "Voltage-gated sodium channel",
            "Oligodendrocyte differentiation", None, {},
        )
        self.assertEqual(got, "Oligodendrocyte differentiation")


class ChainsAreEvidenced(unittest.TestCase):
    """The 5-hop pathway no paper ever reported."""

    def setUp(self):
        try:
            import networkx  # noqa: F401
        except ImportError:
            self.skipTest("networkx not installed")

        from stage2_extraction import aop_visualizer as viz

        self.viz = viz
        # The real corpus: an oligodendroglial branch and a microglial branch,
        # sharing both endpoints and nothing else.
        table2 = pd.DataFrame([
            {"upstream_ke_name": "Voltage-gated sodium channel",
             "downstream_ke_name": "Action potential firing in pre-OLs",
             "upstream_ke_level": "Molecular", "downstream_ke_level": "Cellular",
             "all_source_dois": "10.1038/s41467-017-00688-0",
             "n_papers_supporting": 1, "n_papers_contradicting": 0},
            {"upstream_ke_name": "Action potential firing in pre-OLs",
             "downstream_ke_name": "Oligodendrocyte differentiation",
             "upstream_ke_level": "Cellular", "downstream_ke_level": "Cellular",
             "all_source_dois": "10.1038/s41467-017-00688-0",
             "n_papers_supporting": 1, "n_papers_contradicting": 0},
            {"upstream_ke_name": "Voltage-gated sodium channel",
             "downstream_ke_name": "microglial activation",
             "upstream_ke_level": "Molecular", "downstream_ke_level": "Cellular",
             "all_source_dois": "10.1002/(issn)1098-1136",
             "n_papers_supporting": 1, "n_papers_contradicting": 0},
            {"upstream_ke_name": "microglial activation",
             "downstream_ke_name": "proNGF production",
             "upstream_ke_level": "Cellular", "downstream_ke_level": "Molecular",
             "all_source_dois": "10.1002/(issn)1098-1136",
             "n_papers_supporting": 1, "n_papers_contradicting": 0},
            {"upstream_ke_name": "proNGF production",
             "downstream_ke_name": "Oligodendrocyte differentiation",
             "upstream_ke_level": "Molecular", "downstream_ke_level": "Cellular",
             "all_source_dois": "10.1002/(issn)1098-1136",
             "n_papers_supporting": 1, "n_papers_contradicting": 0},
        ])
        self.graph = viz.build_pathway_graph(table2)

    def test_cross_branch_paths_are_not_returned_as_chains(self):
        chains = self.viz.get_pathway_chains(self.graph)
        for chain in chains:
            joined = " → ".join(chain)
            self.assertFalse(
                "microglial activation" in joined and "pre-OLs" in joined,
                f"a chain crossed between the two mechanisms: {joined}",
            )

    def test_every_returned_chain_is_spanned_by_a_paper(self):
        for chain in self.viz.get_pathway_chains(self.graph):
            provenance = self.viz.chain_provenance(self.graph, chain)
            self.assertTrue(provenance["evidenced"])
            self.assertTrue(
                provenance["spanning_papers"],
                f"no single paper covers {' → '.join(chain)}",
            )

    def test_inferred_chains_are_still_available_but_labelled(self):
        described = self.viz.pathway_chains_with_provenance(self.graph)
        self.assertTrue(described, "exploration must still be possible")

        inferred = [c for c in described if not c["evidenced"]]
        for chain in inferred:
            self.assertTrue(
                chain["inferred_junctions"],
                "an unevidenced chain must name the junction it invented",
            )
        self.assertTrue(
            all(described[i]["evidenced"] >= described[i + 1]["evidenced"]
                for i in range(len(described) - 1)),
            "evidenced chains must be listed before inferred ones",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class CellLineageSplit(unittest.TestCase):
    """One molecule in two cell types is two Key Events, not one."""

    def test_lineage_classification_is_coarse_not_literal(self):
        from stage2_extraction import cell_lineage as cl

        for text in ("NG2+ oligodendrocyte progenitor cells",
                     "pre-myelinating oligodendrocytes (pre-OLs)",
                     "oligodendrocyte lineage cells (brainstem, MNTB)"):
            self.assertEqual(cl.lineage(text), "oligodendroglial", text)

        for text in ("calyx of Held nerve terminal", "myelinated axons in MNTB",
                     "descending motor axons"):
            self.assertEqual(cl.lineage(text), "neuronal / axonal", text)

        self.assertEqual(cl.lineage("activated microglia"), "microglial")
        self.assertEqual(cl.lineage(None), cl.UNSPECIFIED)
        self.assertEqual(cl.lineage(""), cl.UNSPECIFIED)

    def test_nine_spellings_of_one_lineage_are_not_a_conflict(self):
        from stage2_extraction import cell_lineage as cl

        self.assertEqual(
            cl.distinct_lineages([
                "oligodendrocytes", "mature oligodendrocytes",
                "immature oligodendrocytes", "oligodendrocyte precursor cells",
            ]),
            ["oligodendroglial"],
        )

    def test_unstated_cell_type_is_not_a_second_lineage(self):
        from stage2_extraction import cell_lineage as cl

        self.assertEqual(
            cl.distinct_lineages(["oligodendrocytes", None, ""]),
            ["oligodendroglial"],
        )

    def test_conflict_is_reported_across_lineages_only(self):
        from stage2_extraction import ke_normalizer

        raw = [
            ("VGSC activity", "Molecular", None, "oligodendrocytes"),
            ("VGSC activity", "Molecular", None, "mature oligodendrocytes"),
            ("VGSC activity", "Molecular", None, "calyx of Held nerve terminal"),
            ("Differentiation", "Cellular", None, "oligodendrocyte precursor cells"),
            ("Differentiation", "Cellular", None, "immature oligodendrocytes"),
        ]
        conflicts = ke_normalizer.find_cell_type_conflicts(raw)

        self.assertEqual(len(conflicts), 1, "only the two-lineage label conflicts")
        self.assertEqual(conflicts[0]["label"], "VGSC activity")
        self.assertCountEqual(
            conflicts[0]["lineages"], ["oligodendroglial", "neuronal / axonal"]
        )
