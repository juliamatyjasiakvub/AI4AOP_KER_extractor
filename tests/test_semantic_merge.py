from __future__ import annotations

"""
Guards on the merge classifier.

The first two classes are the cases named in the redesign brief. They are
regression tests in the strict sense: each is a pair that a string metric
scores high and that must never be offered as equivalent, because a merge is
destructive and silent — the aliases collapse and the resulting Key Event
reads as though the papers agreed.
"""

import pytest

from stage2_extraction.semantic_merge import (
    KERecord,
    ObjectType,
    Relationship,
    classify,
    classify_all,
    is_key_event,
    object_type,
    participants,
    rank_candidates,
    read_state,
)


def ke(label: str, level: str | None = "Cellular", **kw) -> KERecord:
    return KERecord(key=label, label=label, level=level, **kw)


# ---------------------------------------------------------------------------
# The named opposite pairs
# ---------------------------------------------------------------------------

class TestOppositePairsAreNeverEquivalent:

    def test_restored_vs_disrupted_nodal_protein_organization(self):
        a = ke("restored nodal protein organization")
        b = ke("disrupted nodal protein organization")
        result = classify(a, b)

        assert result.relationship is Relationship.CONTRADICTORY
        assert not result.mergeable
        assert result.similarity > 0.5, (
            "the pair must remain lexically similar, or the test is not "
            "exercising the case it claims to"
        )

    def test_no_change_vs_increased_opc_proliferation(self):
        a = ke("no change in OPC proliferation")
        b = ke("increased OPC proliferation")
        result = classify(a, b)

        assert result.relationship is Relationship.CONTRADICTORY
        assert not result.mergeable

    @pytest.mark.parametrize("left,right", [
        ("increased apoptosis", "decreased apoptosis"),
        ("elevated intracellular calcium", "reduced intracellular calcium"),
        ("upregulation of myelin basic protein", "downregulation of myelin basic protein"),
        ("impaired axonal conduction", "improved axonal conduction"),
        ("loss of oligodendrocytes", "increased oligodendrocyte number"),
        ("demyelination", "remyelination"),
    ])
    def test_direction_opposites_are_contradictory(self, left, right):
        result = classify(ke(left), ke(right))
        assert result.relationship is Relationship.CONTRADICTORY
        assert not result.mergeable

    def test_both_directions_of_the_argument_agree(self):
        """Classification must not depend on which record is passed first."""
        a = ke("restored nodal protein organization")
        b = ke("disrupted nodal protein organization")
        assert classify(a, b).relationship is classify(b, a).relationship


# ---------------------------------------------------------------------------
# "No change" is a study observation
# ---------------------------------------------------------------------------

class TestNoChangeIsNotAKeyEvent:

    @pytest.mark.parametrize("label", [
        "no change in OPC proliferation",
        "no significant change in myelin thickness",
        "unchanged mitochondrial membrane potential",
        "no significant difference in axon count",
        "OPC number unaffected",
        "conduction velocity not significantly altered",
    ])
    def test_rejected_as_key_event(self, label):
        ok, reason = is_key_event(label)
        assert ok is False
        assert reason

    @pytest.mark.parametrize("label", [
        "increased OPC proliferation",
        "mitochondrial dysfunction",
        "demyelination",
    ])
    def test_real_events_are_accepted(self, label):
        ok, _ = is_key_event(label)
        assert ok is True

    def test_typed_as_observation(self):
        assert object_type("no change in OPC proliferation") is ObjectType.OBSERVATION
        assert object_type("increased OPC proliferation") is ObjectType.EVENT

    def test_no_change_never_merges_with_another_no_change(self):
        """
        Two observations phrased alike are still observations. They may be
        equivalent statements, but neither is a Key Event, so the workspace
        must not quietly merge them into one node.
        """
        a = ke("no change in OPC proliferation")
        b = ke("no significant change in OPC proliferation")
        assert not classify(a, b).mergeable


# ---------------------------------------------------------------------------
# Subtype vs. class
# ---------------------------------------------------------------------------

class TestSubtypeIsNotItsClass:

    def test_nav12_is_narrower_than_the_channel_class(self):
        a = ke("NaV1.2", level="Molecular")
        b = ke("voltage-gated sodium channel", level="Molecular")
        result = classify(a, b)

        assert result.relationship is Relationship.NARROWER_THAN
        assert not result.mergeable
        assert "broader" in result.explanation.lower()

    def test_sibling_subtypes_are_distinct(self):
        result = classify(ke("NaV1.2", level="Molecular"),
                          ke("NaV1.6", level="Molecular"))
        assert result.relationship is Relationship.RELATED_DISTINCT
        assert not result.mergeable

    def test_qualified_event_is_narrower(self):
        result = classify(ke("mitochondrial ROS accumulation"),
                          ke("ROS accumulation"))
        assert result.relationship is Relationship.NARROWER_THAN
        assert not result.mergeable

    def test_ontology_ancestry_beats_string_similarity(self):
        a = KERecord(key="a", label="alpha", level="Molecular",
                     ontology_curie="GO:0000001", ontology_source="go")
        b = KERecord(key="b", label="alpha", level="Molecular",
                     ontology_curie="GO:0000002", ontology_source="go")

        def ancestors_of(curie, ontology):
            return {"GO:0000002"} if curie == "GO:0000001" else set()

        result = classify(a, b, ancestors_of=ancestors_of)
        assert result.relationship is Relationship.NARROWER_THAN
        assert not result.mergeable


# ---------------------------------------------------------------------------
# What may be merged
# ---------------------------------------------------------------------------

class TestEquivalence:

    def test_same_ontology_term_is_equivalent(self):
        a = KERecord(key="a", label="oxidative stress", level="Cellular",
                     ontology_curie="GO:0006979", ontology_source="go")
        b = KERecord(key="b", label="oxidative stress response", level="Cellular",
                     ontology_curie="GO:0006979", ontology_source="go")
        result = classify(a, b)
        assert result.relationship is Relationship.EQUIVALENT
        assert result.mergeable

    def test_reordered_wording_is_equivalent(self):
        result = classify(ke("apoptosis of hepatocytes"), ke("hepatocyte apoptosis"))
        assert result.relationship is Relationship.EQUIVALENT
        assert result.mergeable

    def test_hyphenation_variant_is_equivalent(self):
        result = classify(ke("down-regulation of myelin basic protein"),
                          ke("downregulation of myelin basic protein"))
        assert result.relationship is Relationship.EQUIVALENT

    def test_unrelated_events_are_not_equivalent(self):
        result = classify(ke("mitochondrial dysfunction"), ke("axonal degeneration"))
        assert not result.mergeable

    def test_different_levels_are_not_equivalent(self):
        result = classify(ke("apoptosis", level="Cellular"),
                          ke("apoptosis", level="Tissue"))
        assert result.relationship is Relationship.RELATED_DISTINCT
        assert not result.mergeable

    def test_high_similarity_alone_does_not_merge(self):
        """
        Two labels that are near-identical strings but state no direction and
        share no ontology term land on `uncertain`, not `equivalent`. Silence
        is not agreement.
        """
        result = classify(ke("nodal protein organization"),
                          ke("paranodal protein organization"))
        assert not result.mergeable


# ---------------------------------------------------------------------------
# Entities vs. events
# ---------------------------------------------------------------------------

class TestEntityEventSeparation:

    def test_entity_and_event_never_merge(self):
        result = classify(ke("myelin basic protein", level="Molecular"),
                          ke("decreased myelin basic protein", level="Molecular"))
        assert not result.mergeable
        assert result.relationship in {
            Relationship.RELATED_DISTINCT,
            Relationship.NARROWER_THAN,
            Relationship.BROADER_THAN,
        }

    def test_participants_ignore_direction(self):
        assert participants("increased mitochondrial ROS") == \
               participants("decreased mitochondrial ROS")

    def test_state_reading(self):
        assert read_state("restored nodal organization").sign == 1
        assert read_state("disrupted nodal organization").sign == -1
        assert read_state("no change in proliferation").no_change is True


# ---------------------------------------------------------------------------
# Nothing merges itself
# ---------------------------------------------------------------------------

class TestNoAutomaticMerging:

    def test_classify_all_returns_judgements_not_actions(self):
        records = [
            ke("increased apoptosis"),
            ke("decreased apoptosis"),
            ke("elevated apoptosis"),
        ]
        results = classify_all(records)
        assert results, "the fixture must produce candidates"
        # Every result is a verdict object; none of them has changed anything.
        assert all(hasattr(r, "relationship") for r in results)
        assert all(r.explanation for r in results)

    def test_ranker_cannot_make_a_pair_mergeable(self):
        """
        A scorer expressing total confidence still cannot license a merge.
        `mergeable` is derived from the checks, and a ranking hint has no
        route to them.
        """
        contradiction = classify(ke("increased apoptosis"), ke("decreased apoptosis"))
        assert not contradiction.mergeable

        ranked = rank_candidates([contradiction], scorer=lambda c: 1.0)
        assert ranked[0].rank_score == 1.0
        assert not ranked[0].mergeable
        assert ranked[0].relationship is Relationship.CONTRADICTORY

    def test_every_classification_carries_an_explanation(self):
        pairs = [
            (ke("increased apoptosis"), ke("decreased apoptosis")),
            (ke("NaV1.2", level="Molecular"), ke("voltage-gated sodium channel", level="Molecular")),
            (ke("apoptosis of hepatocytes"), ke("hepatocyte apoptosis")),
        ]
        for a, b in pairs:
            result = classify(a, b)
            assert result.explanation.strip()
            assert len(result.checks) == 7
