from __future__ import annotations

"""
A "±" on the map is a question, and a question needs an answer it can reach.

The mark itself was right: a Key Event whose claims record an increase in two
papers and a decrease in two others is genuinely unresolved, and averaging it
into one arrow would state something no paper supports. What was missing was
everything after the mark — which paper said what, and any way to act on it.
A flag that cannot be cleared stops being read.

These tests hold the resolution path: the conflict is legible per claim, a
curator's ruling changes the arrow, and — the one that is easy to get wrong —
ruling "the disagreement is real" does *not* tidy the mark away. That ruling
is an answer, not a dismissal, and the figure has to keep saying so.
"""

import pytest

from schemas import KERExtraction


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Four claims about one Key Event: two up, two down."""
    import stage2_extraction.table1_store as ts

    monkeypatch.setattr(ts, "DB_PATH", tmp_path / "scratch.db")
    ts.init_db()

    def claim(doi, downstream_change):
        return KERExtraction(
            upstream_ke_name="Sodium-channel activity",
            upstream_ke_level="Molecular",
            downstream_ke_name="Oligodendrocyte differentiation",
            downstream_ke_level="Cellular",
            ker_name="Sodium-channel activity affects differentiation",
            ker_description="",
            ker_adjacency="Adjacent",
            paper_type="Primary study",
            cited_evidence_dois=None,
            biological_plausibility=None,
            empirical_evidence_summary=None,
            essentiality_evidence=None,
            contradicts_ker=False,
            taxonomic_applicability="rat",
            sex_applicability="both",
            life_stage_applicability="juvenile",
            modulating_factors=None,
            quantitative_relationships=None,
            response_response_relationship=None,
            time_scale=None,
            feedforward_feedback_loops=None,
            study_design="in vivo",
            exposure_route=None,
            chemical_stressor=None,
            extraction_confidence="High",
            downstream_change=downstream_change,
        )

    for doi, change in (
        ("10.1/a", "increased"),
        ("10.1/b", "increased"),
        ("10.1/c", "decreased"),
        ("10.1/d", "decreased"),
    ):
        ts.insert_table1_row(claim(doi, change), doi, {})

    from stage2_extraction import ke_normalizer

    ke_normalizer.normalize_table1(
        ts.load_table1_as_dataframe(), threshold=0.86, ols4_enabled=False
    )
    return ts


def _differentiation_id(store):
    canonical = store.load_canonical_kes()
    match = canonical[
        canonical["canonical_name"].astype(str).str.contains(
            "differentiation", case=False
        )
    ]
    return int(match.iloc[0]["canonical_id"])


class TestTheConflictIsLegible:

    def test_the_mark_appears_only_when_the_claims_disagree(self):
        from collections import Counter

        from stage2_extraction import direction_conflicts as dc

        assert dc.dominant_change(Counter({"increased": 2, "decreased": 2})) == "±"
        assert dc.dominant_change(Counter({"increased": 3})) == "↑"
        assert dc.dominant_change(Counter({"reduced": 3})) == "↓"
        assert dc.dominant_change(Counter()) == ""

    def test_every_conflicting_claim_is_named_with_its_paper(self, store):
        """
        The half that was missing. "±" says the corpus disagrees; acting on it
        needs to know which paper said which, and that had nowhere to appear.
        """
        from stage2_extraction import direction_conflicts as dc

        claims = dc.claims_for(
            store.load_table1_as_dataframe(), _differentiation_id(store)
        )

        assert len(claims) == 4
        assert set(claims["doi"]) == {"10.1/a", "10.1/b", "10.1/c", "10.1/d"}
        assert sorted(claims["sign"]) == ["↑", "↑", "↓", "↓"]
        assert set(claims["record_id"]) == set(
            store.load_table1_as_dataframe()["record_id"]
        )

    @pytest.mark.parametrize(
        "word,expected",
        [
            ("increased", "↑"), ("elevated", "↑"), ("gain of function", "↑"),
            ("decreased", "↓"), ("reduced", "↓"), ("loss of clustering", "↓"),
            ("abolished", "↓"), ("impaired", "↓"),
            ("altered", ""), ("", ""),
        ],
    )
    def test_a_recorded_change_is_read_the_same_way_everywhere(self, word, expected):
        from stage2_extraction import direction_conflicts as dc

        assert dc.sign_of(word) == expected


class TestRulingOnIt:

    def test_a_ruling_changes_the_arrow(self, store):
        from stage2_extraction import direction_conflicts as dc

        canonical_id = _differentiation_id(store)
        store.set_ke_direction(
            canonical_id, "decreased",
            curator="julia",
            rationale="The two increases are from the injury model.",
        )

        rulings = store.load_ke_directions()
        assert dc.resolved_change(canonical_id, "±", rulings) == "↓"

    def test_ruling_the_conflict_real_keeps_the_mark(self, store):
        """
        The failure worth guarding: treating every ruling as a resolution
        would let "I have looked at this and the literature genuinely splits"
        erase the very mark that says so. It is an answer, not a dismissal.
        """
        from stage2_extraction import direction_conflicts as dc

        canonical_id = _differentiation_id(store)
        store.set_ke_direction(
            canonical_id, "conflicted", curator="julia",
            rationale="Both directions are well supported in different models.",
        )

        rulings = store.load_ke_directions()
        assert dc.resolved_change(canonical_id, "±", rulings) == "±"
        assert rulings[canonical_id]["acknowledged"] == 1

    def test_a_ruling_records_who_and_why(self, store):
        canonical_id = _differentiation_id(store)
        store.set_ke_direction(
            canonical_id, "increased", curator="julia", rationale="Because.",
        )
        ruling = store.load_ke_directions()[canonical_id]

        assert ruling["curator"] == "julia"
        assert ruling["rationale"] == "Because."
        assert ruling["updated_at"]

    def test_a_ruling_can_be_withdrawn(self, store):
        from stage2_extraction import direction_conflicts as dc

        canonical_id = _differentiation_id(store)
        store.set_ke_direction(canonical_id, "increased", curator="julia")
        store.clear_ke_direction(canonical_id)

        rulings = store.load_ke_directions()
        assert dc.resolved_change(canonical_id, "±", rulings) == "±"

    def test_an_unknown_direction_is_refused(self, store):
        with pytest.raises(ValueError):
            store.set_ke_direction(_differentiation_id(store), "sideways")

    def test_correcting_the_misread_claim_resolves_it_at_source(self, store):
        """
        The outcome the panel is really for. A ruling states a judgement over
        a corpus that disagrees; correcting the row that was misread means the
        corpus no longer disagrees, and no judgement is needed.
        """
        from collections import Counter

        from stage2_extraction import direction_conflicts as dc

        table1 = store.load_table1_as_dataframe()
        wrong = table1[table1["source_doi"] == "10.1/a"].iloc[0]
        store.update_table1_row(
            int(wrong["record_id"]),
            {"downstream_change": "decreased"},
            curator="julia",
            rationale="Figure 3 is the rescue condition, not the treated one.",
        )
        second = store.load_table1_as_dataframe()
        second_wrong = second[second["source_doi"] == "10.1/b"].iloc[0]
        store.update_table1_row(
            int(second_wrong["record_id"]),
            {"downstream_change": "decreased"},
            curator="julia",
            rationale="Same misreading.",
        )

        claims = dc.claims_for(
            store.load_table1_as_dataframe(), _differentiation_id(store)
        )
        counted = Counter(c.lower() for c in claims["change"])
        assert dc.dominant_change(counted) == "↓"
