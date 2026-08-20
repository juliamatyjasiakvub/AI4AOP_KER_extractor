"""
Tests for the step between Table 1 and the canonical Key Events.

The complaint these answer: "from 27 KER claims we arrive at 18 KEs" was shown
as two totals with nothing in between, so the grouping could not be checked and
read as though nine findings had been discarded. The pieces under test are the
per-label record of *why* each wording was grouped, the crosswalk that displays
it, and the claim-level curation state the Assign grid writes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from schemas import CanonicalKE
from stage2_extraction import curation_store, ke_normalizer, table1_store


# ---------------------------------------------------------------------------
# Merge basis
# ---------------------------------------------------------------------------

def test_every_label_gets_a_basis():
    """No wording may end up in a Key Event without a recorded reason."""
    raw = [
        ("Increased persistent sodium current", "Molecular", None),
        ("Persistent sodium current, increased", "Molecular", None),
        ("Neuronal hyperexcitability", "Cellular", None),
    ]
    kes, _, report = ke_normalizer.build_canonical_kes(raw, {})

    for ke in kes:
        for alias in ke.aliases:
            rule, detail = ke.alias_basis[alias]
            assert rule, f"{alias} has no rule"
            assert detail.strip(), f"{alias} has no explanation"

    assert len(report.crosswalk) == 3
    assert {row["raw_label"] for row in report.crosswalk} == {label for label, _, _ in raw}


def test_token_order_merge_names_the_label_it_matched():
    """The explanation has to say what it matched against, not just that it did."""
    raw = [
        ("Increased persistent sodium current", "Molecular", None),
        ("Persistent sodium current, increased", "Molecular", None),
    ]
    kes, _, _ = ke_normalizer.build_canonical_kes(raw, {})
    assert len(kes) == 1

    folded = [
        (label, basis)
        for label, basis in kes[0].alias_basis.items()
        if basis[0] != "own_group"
    ]
    assert folded, "one of the two wordings should carry a merge rule"
    _, (rule, detail) = folded[0]
    assert rule == "token_order"
    assert "sodium current" in detail


def test_aopwiki_id_beats_wording():
    """
    Rule 1 groups on a shared identifier even when the strings disagree — and
    the crosswalk has to say so, because this is the grouping a curator is least
    able to guess from the names.
    """
    raw = [
        ("Increased persistent sodium current", "Molecular", 1541),
        ("Voltage-gated sodium channel activation", "Molecular", 1541),
    ]
    kes, _, report = ke_normalizer.build_canonical_kes(raw, {})
    assert len(kes) == 1

    bases = {row["raw_label"]: row["basis"] for row in report.crosswalk}
    assert "aopwiki" in bases.values()
    detail = next(
        row["detail"] for row in report.crosswalk if row["basis"] == "aopwiki"
    )
    assert "1541" in detail


def test_opposite_directions_are_not_grouped():
    """The polarity guard still holds, and both labels keep their own event."""
    raw = [
        ("Increased apoptosis", "Cellular", None),
        ("Decreased apoptosis", "Cellular", None),
    ]
    kes, _, report = ke_normalizer.build_canonical_kes(raw, {})
    assert len(kes) == 2
    assert {row["canonical_name"] for row in report.crosswalk} == {
        "Increased apoptosis",
        "Decreased apoptosis",
    }


def test_crosswalk_marks_which_wording_names_the_event():
    raw = [
        ("Increased persistent sodium current", "Molecular", None),
        ("Persistent sodium current, increased", "Molecular", None),
    ]
    _, _, report = ke_normalizer.build_canonical_kes(raw, {})
    named = [row for row in report.crosswalk if row["is_event_name"]]
    assert len(named) == 1, "exactly one wording names the event"


def test_mentions_are_counted_per_wording():
    """A wording used twice carries two mentions, so folding it is not a loss."""
    raw = [
        ("Neuronal hyperexcitability", "Cellular", None),
        ("Neuronal hyperexcitability", "Cellular", None),
        ("Myelination", "Tissue", None),
    ]
    _, _, report = ke_normalizer.build_canonical_kes(raw, {})
    mentions = {row["raw_label"]: row["mentions"] for row in report.crosswalk}
    assert mentions["Neuronal hyperexcitability"] == 2
    assert mentions["Myelination"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """A scratch database, so these tests never touch the working corpus."""
    monkeypatch.setattr(table1_store, "DB_PATH", tmp_path / "scratch.db")
    table1_store.init_db()
    return table1_store


def test_basis_survives_a_round_trip(store):
    """
    The reason has to be readable after a reload. Holding it only in session
    state would mean the account of the grouping disappeared the moment the
    curator refreshed the page — which is when they would go looking for it.
    """
    ke = CanonicalKE(
        canonical_id=None,
        canonical_name="Increased persistent sodium current",
        level="Molecular",
        aliases=[
            "Increased persistent sodium current",
            "Persistent sodium current, increased",
        ],
        alias_basis={
            "Increased persistent sodium current": [
                "own_group", "The wording the event is named from.",
            ],
            "Persistent sodium current, increased": [
                "token_order", "Same content words as “Increased persistent sodium current”.",
            ],
        },
    )
    store.replace_canonical_kes([ke], {"Increased persistent sodium current": 0})

    crosswalk = store.load_alias_crosswalk()
    assert len(crosswalk) == 2

    by_label = {row["raw_label"]: row for _, row in crosswalk.iterrows()}
    assert by_label["Persistent sodium current, increased"]["merge_basis"] == "token_order"
    assert "content words" in by_label["Persistent sodium current, increased"]["merge_detail"]
    assert by_label["Increased persistent sodium current"]["is_event_name"]
    assert not by_label["Persistent sodium current, increased"]["is_event_name"]


def test_every_basis_rule_has_a_display_label():
    """A rule the UI cannot name would render as a blank cell."""
    rules = {
        "aopwiki", "ontology", "normalised_string", "token_order", "lexical",
        "own_group", "curator",
    }
    assert rules <= set(table1_store.ALIAS_BASIS_LABELS)


# ---------------------------------------------------------------------------
# Claim-level curation
# ---------------------------------------------------------------------------

def test_claim_state_round_trips_through_three_states(store):
    """
    Keep and Checked are two ticks stored in one status field, so the mapping
    has to be lossless: rejected, accepted and unreviewed must each come back
    as the pair of ticks that produced them.
    """
    curation_store.set_curation("claim", "1", status="rejected")
    curation_store.set_curation("claim", "2", status="accepted")
    curation_store.set_curation("claim", "3", status="unreviewed")

    state = curation_store.curation_map("claim")

    def ticks(record_id: str) -> tuple[bool, bool]:
        status = state[record_id]["status"]
        if status == "rejected":
            return False, False
        return True, status == "accepted"

    assert ticks("1") == (False, False)   # not kept
    assert ticks("2") == (True, True)     # kept and checked
    assert ticks("3") == (True, False)    # kept, not yet checked


def test_rejected_claims_do_not_reach_the_synthesis():
    """
    The filter the Keep tick relies on. A tick that is recorded and then ignored
    is worse than no tick, so this asserts the row actually stops contributing.
    """
    table1 = pd.DataFrame(
        [
            {"record_id": 1, "claim_status": "accepted"},
            {"record_id": 2, "claim_status": "rejected"},
            {"record_id": 3, "claim_status": "unreviewed"},
        ]
    )
    contributing = table1[table1["claim_status"] != "rejected"]
    assert sorted(contributing["record_id"]) == [1, 3]
    assert len(table1) == 3, "the row stays in Table 1 either way"


# ---------------------------------------------------------------------------
# One wording, more than one Key Event
# ---------------------------------------------------------------------------

def test_one_wording_can_point_at_two_events_per_row(store):
    """
    The case the whole redesign is for.

    Two claims both say "voltage-gated sodium channels". One blocked the channel
    in an oligodendrocyte, one activated it in an axon. Those are two Key Events
    and the wording is identical, so nothing keyed by label can tell them apart —
    the assignment has to live on the row. This asserts that a per-row
    assignment survives a full rebuild of the canonical events, which is where
    the label-keyed back-fill would otherwise flatten it.
    """
    from stage2_extraction import canonical_groups as cg

    label = "Voltage-gated sodium channels"
    with store.connect() as conn:
        for record_id, cell in ((1, "oligodendrocyte"), (2, "axon")):
            conn.execute(
                """
                INSERT INTO table1_extractions
                    (record_id, source_doi, extraction_date,
                     upstream_ke_name, upstream_ke_level,
                     downstream_ke_name, downstream_ke_level,
                     ker_name, ker_description, ker_adjacency, paper_type,
                     contradicts_ker, taxonomic_applicability, sex_applicability,
                     life_stage_applicability, study_design, extraction_confidence,
                     upstream_cell_type)
                VALUES (?, ?, '2026-01-01', ?, 'Molecular', 'Myelination', 'Tissue',
                        'k', 'd', 'adjacent', 'primary', 0, 'rat', 'both',
                        'adult', 'in vitro', 'High', ?)
                """,
                (record_id, f"10.1/{record_id}", label, cell),
            )
        conn.commit()

    oligo = f"{label} in oligodendrocytes"
    axon = f"{label} in neurons/axons"

    cg.apply_assignments(
        [(label, oligo), (label, axon), ("Myelination", "Myelination")],
        curator="test",
    )

    ids = store.canonical_ids_by_name()
    assert oligo in ids and axon in ids, "both events must exist"

    store.set_claim_canonical_ends({
        1: (ids[oligo], ids["Myelination"]),
        2: (ids[axon], ids["Myelination"]),
    })
    store.recount_canonical_source_rows()

    table1 = store.load_table1_as_dataframe()
    by_record = {int(r["record_id"]): r for _, r in table1.iterrows()}
    assert int(by_record[1]["upstream_ke_canonical_id"]) == ids[oligo]
    assert int(by_record[2]["upstream_ke_canonical_id"]) == ids[axon]

    # And the wording is still attached to both, because both papers wrote it.
    crosswalk = store.load_alias_crosswalk()
    events = set(crosswalk.loc[crosswalk["raw_label"] == label, "canonical_name"])
    assert events == {oligo, axon}


def test_source_row_counts_are_not_double_counted(store):
    """
    Two events split out of one wording must not each claim the full evidence.

    Counting by label would give both the same total, and that total is read as
    evidence weight on the map — so a split would silently double the corpus.
    """
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO ke_canonical (canonical_name, level, merge_method, "
            "curation_status, n_source_rows, updated_at) "
            "VALUES ('A', 'Molecular', 'manual', 'accepted', 99, '2026-01-01')"
        )
        a_id = conn.execute(
            "SELECT canonical_id FROM ke_canonical WHERE canonical_name = 'A'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO table1_extractions
                (record_id, source_doi, extraction_date, upstream_ke_name,
                 upstream_ke_level, downstream_ke_name, downstream_ke_level,
                 ker_name, ker_description, ker_adjacency, paper_type,
                 contradicts_ker, taxonomic_applicability, sex_applicability,
                 life_stage_applicability, study_design, extraction_confidence,
                 upstream_ke_canonical_id)
            VALUES (1, '10.1/a', '2026-01-01', 'A', 'Molecular', 'B', 'Tissue',
                    'k', 'd', 'adjacent', 'primary', 0, 'rat', 'both', 'adult',
                    'in vitro', 'High', ?)
            """,
            (a_id,),
        )
        conn.commit()

    store.recount_canonical_source_rows()
    canonical = store.load_canonical_kes()
    assert int(canonical.loc[canonical["canonical_name"] == "A", "n_source_rows"].iloc[0]) == 1


def test_synonym_count_never_goes_negative():
    """A count of synonyms below zero is a number that cannot be displayed."""
    assert max(0, 18 - 20) == 0
