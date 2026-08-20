from __future__ import annotations

"""
Coarsening is a decision, not a loophole.

`merge_as_equivalent` refuses anything the classifier did not call equivalent,
and that refusal has to stay: folding a subtype into its class *as the same
event* is how evidence about NaV1.2 silently becomes evidence about sodium
channels in general. But a curator working at the class level is not making
that mistake — they are saying the subtype distinction is not one this AOP
turns on, which the tool previously had no way to express.

`collapse_into_broader` allows it and records it as what it is. The tests that
matter are therefore not "does it merge" but: does the equivalence guard still
hold, is the coarsening distinguishable from an equivalence afterwards, and is
it reversible. The last one is not hypothetical — the first implementation
folded the records correctly and left them permanent, because `undo` had an
allow-list of actions and the new one was not on it.
"""

import pytest

from stage2_extraction.semantic_merge import Classification, Relationship


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A scratch database with three records: one class, two subtypes."""
    import stage2_extraction.table1_store as ts

    monkeypatch.setattr(ts, "DB_PATH", tmp_path / "scratch.db")
    ts.init_db()

    from stage2_extraction import canonical_groups as cg

    def add(name: str, rows: int) -> int:
        with ts.connect() as conn:
            kid = conn.execute(
                "INSERT INTO ke_canonical (canonical_name, level, n_source_rows, "
                "curation_status, merge_method, updated_at) "
                "VALUES (?, 'Molecular', ?, 'unreviewed', 'auto', datetime('now'))",
                (name, rows),
            ).lastrowid
            conn.execute(
                "INSERT INTO ke_alias (raw_label, canonical_id) VALUES (?, ?)",
                (name, kid),
            )
            conn.commit()
        return kid

    # The class is the *worst* evidenced of the three on purpose: survivor
    # selection by evidence volume would keep a subtype, which is exactly the
    # wrong answer and the reason the survivor is asked for rather than inferred.
    ids = {
        "broad": add("Voltage-gated sodium channel activity", 3),
        "nav12": add("NaV1.2 activity", 12),
        "nav16": add("NaV1.6 activity", 2),
    }
    return ts, cg, ids


def verdict(relationship: Relationship, a: int, b: int) -> Classification:
    return Classification(
        source=str(a), target=str(b), relationship=relationship,
        explanation="Test fixture.",
    )


class TestTheEquivalenceGuardStillHolds:

    def test_merge_as_equivalent_refuses_broader_than(self, store):
        """The guard this feature must not weaken."""
        ts, cg, ids = store
        with pytest.raises(cg.MergeRefused):
            cg.merge_as_equivalent(
                [ids["broad"], ids["nav12"]],
                classification=verdict(Relationship.BROADER_THAN, ids["nav12"], ids["broad"]),
            )

    def test_collapse_refuses_contradictory_pairs(self, store):
        """
        The one classification that is never a question of grain. Two records
        the classifier calls incompatible describe opposite findings; pooling
        them does not coarsen anything, it discards one of them.
        """
        ts, cg, ids = store
        with pytest.raises(cg.MergeRefused):
            cg.collapse_into_broader(
                [ids["broad"], ids["nav12"]],
                survivor_id=ids["broad"],
                classification=verdict(Relationship.CONTRADICTORY, ids["nav12"], ids["broad"]),
            )

    def test_the_survivor_must_be_one_of_the_members(self, store):
        ts, cg, ids = store
        with pytest.raises(ValueError):
            cg.collapse_into_broader([ids["broad"], ids["nav12"]], survivor_id=99999)


class TestCollapsing:

    def test_the_curators_survivor_wins_over_evidence_volume(self, store):
        """
        `preview_merge` picks a survivor by row count, and the best-evidenced
        record here is a subtype. Which record is the broader one is the whole
        content of the decision, so it is asked for and must be obeyed.
        """
        ts, cg, ids = store
        auto = cg.preview_merge(list(ids.values()))
        assert auto.survivor_id == ids["nav12"], "fixture no longer exercises the conflict"

        cg.collapse_into_broader(
            list(ids.values()), survivor_id=ids["broad"],
            classification=verdict(Relationship.BROADER_THAN, ids["nav12"], ids["broad"]),
            curator="tester",
        )
        with ts.connect() as conn:
            names = [r[0] for r in conn.execute("SELECT canonical_name FROM ke_canonical")]
        assert names == ["Voltage-gated sodium channel activity"]

    def test_the_subtype_names_survive_as_aliases(self, store):
        """What makes a coarsening legible afterwards instead of merely done."""
        ts, cg, ids = store
        cg.collapse_into_broader(
            list(ids.values()), survivor_id=ids["broad"], curator="tester",
        )
        with ts.connect() as conn:
            aliases = sorted(
                r[0] for r in conn.execute(
                    "SELECT raw_label FROM ke_alias WHERE canonical_id = ?",
                    (ids["broad"],),
                )
            )
        assert aliases == [
            "NaV1.2 activity",
            "NaV1.6 activity",
            "Voltage-gated sodium channel activity",
        ]

    def test_it_is_logged_as_coarsening_not_equivalence(self, store):
        """
        Both operations leave identical database state, so the log is the only
        place the difference survives.
        """
        ts, cg, ids = store
        result = cg.collapse_into_broader(
            [ids["broad"], ids["nav12"]], survivor_id=ids["broad"], curator="tester",
        )
        with ts.connect() as conn:
            action, method = conn.execute(
                "SELECT action, method FROM merge_decision WHERE decision_id = ?",
                (result["decision_id"],),
            ).fetchone()
        assert action == "collapse_broader"
        assert method == "coarsening"

    def test_it_appears_in_the_merge_history(self, store):
        """
        Filtering the log on 'merge_equivalent' alone would hide it here — and
        since undo is driven from this view, hiding it makes it permanent.
        """
        ts, cg, ids = store
        cg.collapse_into_broader(
            [ids["broad"], ids["nav12"]], survivor_id=ids["broad"], curator="tester",
        )
        groups = cg.canonical_groups()
        assert len(groups) == 1
        assert groups.iloc[0]["action"] == "collapse_broader"
        assert groups.iloc[0]["action_label"] == "Collapse into the broader Key Event"


class TestReversibility:

    def test_undo_restores_every_record(self, store):
        """The regression: `undo` had an allow-list and this action was not on it."""
        ts, cg, ids = store
        result = cg.collapse_into_broader(
            list(ids.values()), survivor_id=ids["broad"], curator="tester",
        )
        cg.undo(result["decision_id"], curator="tester")
        with ts.connect() as conn:
            names = sorted(
                r[0] for r in conn.execute("SELECT canonical_name FROM ke_canonical")
            )
        assert names == [
            "NaV1.2 activity",
            "NaV1.6 activity",
            "Voltage-gated sodium channel activity",
        ]

    def test_undone_collapses_leave_the_live_log(self, store):
        ts, cg, ids = store
        result = cg.collapse_into_broader(
            [ids["broad"], ids["nav12"]], survivor_id=ids["broad"], curator="tester",
        )
        cg.undo(result["decision_id"], curator="tester")
        assert cg.canonical_groups().empty
        assert len(cg.canonical_groups(include_reverted=True)) == 1


class TestTheActionIsRegistered:

    def test_it_is_a_known_action(self):
        """`record_decision` rejects any action not in ACTIONS."""
        from stage2_extraction import canonical_groups as cg

        assert "collapse_broader" in cg.ACTIONS
        assert cg.ACTION_LABELS["collapse_broader"]

    def test_it_is_exported(self):
        from stage2_extraction import canonical_groups as cg

        assert "collapse_into_broader" in cg.__all__

    def test_a_collapsed_pair_stops_being_re_suggested(self, store):
        """
        A tool that keeps asking a question you already answered trains people
        to click through it.
        """
        ts, cg, ids = store
        cg.collapse_into_broader(
            [ids["broad"], ids["nav12"]], survivor_id=ids["broad"], curator="tester",
        )
        decided = cg.decided_pairs()
        assert frozenset({ids["broad"], ids["nav12"]}) in decided
