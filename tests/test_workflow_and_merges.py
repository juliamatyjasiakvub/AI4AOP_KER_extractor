from __future__ import annotations

"""
The gate, the staleness rules, and the reversibility of a merge.

These run against a real SQLite file rather than mocks: the behaviour under
test is mostly what the database does when several tables change together, and
a mock would only assert that the code calls the functions it calls.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from stage2_extraction import table1_store


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def fresh_db(tmp: Path):
    """Point every store module at a new empty database and seed it."""
    table1_store.DB_PATH = tmp / "test.db"
    if table1_store.DB_PATH.exists():
        table1_store.DB_PATH.unlink()
    table1_store.init_db()

    from stage2_extraction import ols4_client
    ols4_client.set_db_path(table1_store.DB_PATH)

    with table1_store.connect() as conn:
        conn.executemany(
            "INSERT INTO ke_canonical (canonical_id, canonical_name, level, "
            "ontology_curie, n_source_rows, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "increased ROS", "Cellular", "GO:0000001", 3, "2026-01-01"),
                (2, "elevated ROS", "Cellular", "GO:0000001", 1, "2026-01-01"),
                (3, "axonal degeneration", "Tissue", None, 2, "2026-01-01"),
                (4, "decreased ROS", "Cellular", None, 1, "2026-01-01"),
            ],
        )
        conn.executemany(
            "INSERT INTO ke_alias (canonical_id, raw_label, n_uses) VALUES (?, ?, ?)",
            [
                (1, "increased ROS", 3),
                (1, "ROS increase", 1),
                (2, "elevated ROS", 1),
                (3, "axonal degeneration", 2),
                (4, "decreased ROS", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO table1_extractions "
            "(record_id, source_doi, extraction_date, upstream_ke_name, "
            " upstream_ke_level, downstream_ke_name, downstream_ke_level, "
            " ker_name, ker_description, ker_adjacency, paper_type, "
            " contradicts_ker, taxonomic_applicability, sex_applicability, "
            " life_stage_applicability, study_design, extraction_confidence, "
            " upstream_ke_canonical_id, downstream_ke_canonical_id) "
            "VALUES (?, ?, '2026-01-01', ?, 'Cellular', ?, 'Tissue', ?, '', "
            "'Adjacent', 'Primary study', 0, ?, 'Mixed', 'Adult', 'In vivo', "
            "'High', ?, ?)",
            [
                (10, "10.1/a", "increased ROS", "axonal degeneration",
                 "ROS drives degeneration", "Rat", 1, 3),
                (11, "10.1/b", "elevated ROS", "axonal degeneration",
                 "ROS drives degeneration", "Mouse", 2, 3),
            ],
        )
        conn.commit()


class Ctx:
    """Test context that rebuilds the database for each use."""

    def __enter__(self):
        # `ignore_cleanup_errors` because on Windows something still holds a
        # handle to test.db when the block exits, and Windows will not unlink
        # an open file where POSIX will. These tests therefore passed on Linux
        # and failed on Windows during teardown — after every assertion in the
        # body had already succeeded, which made seven green tests report as
        # failures. The subject here is merge and workflow logic; a temp
        # directory outliving the run is the operating system's problem, and a
        # suite that is red on the maintainer's own machine stops being read.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        path = Path(self._tmp.name)
        self._previous = table1_store.DB_PATH
        fresh_db(path)
        return self

    def __exit__(self, *exc):
        table1_store.DB_PATH = self._previous
        self._tmp.cleanup()
        return False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestApprovalGate:

    def test_synthesis_blocked_before_approval(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            result = wf.gate(ke_ids=[1, 3])
            assert result.allowed is False
            assert len(result.blocking) == 2
            assert "not yet approved" in result.reason

    def test_synthesis_allowed_once_everything_is_approved(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            for ke_id in (1, 3):
                wf.set_state("ke", ke_id, wf.State.CURATED)
                wf.approve("ke", ke_id, curator="jm")
            assert wf.gate(ke_ids=[1, 3]).allowed is True

    def test_cannot_skip_curation(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            with pytest.raises(wf.TransitionError):
                wf.set_state("ke", 1, wf.State.APPROVED)

    def test_require_approved_raises(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            with pytest.raises(wf.NotApproved):
                wf.require_approved(ke_ids=[1])

    def test_retracting_is_always_allowed(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            wf.set_state("ke", 1, wf.State.CURATED)
            wf.approve("ke", 1, curator="jm")
            wf.retract("ke", 1, curator="jm")
            assert wf.get_state("ke", 1) is wf.State.CURATED


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestStaleness:

    def _approve_and_synthesise(self, wf):
        for ke_id in (1, 3):
            wf.set_state("ke", ke_id, wf.State.CURATED)
            wf.approve("ke", ke_id, curator="jm")
        with table1_store.connect() as conn:
            conn.execute(
                "INSERT INTO ker_synthesis (ker_key, ker_name, n_papers, "
                "overall_confidence, generated_at) VALUES ('1->3', "
                "'increased ROS -> axonal degeneration', 2, 'Moderate', '2026-01-01')",
                )
            conn.commit()
        wf.set_state("ker", "1->3", wf.State.CURATED)
        wf.approve("ker", "1->3", curator="jm")
        wf.set_state("ker", "1->3", wf.State.SYNTHESIZED)

    def test_changing_an_approved_ke_marks_dependents_stale(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            self._approve_and_synthesise(wf)

            table1_store.rename_canonical_ke(1, "increased reactive oxygen species")
            wf.invalidate_for_ke(1, reason="renamed")

            stale = wf.stale_syntheses()
            assert not stale.empty
            assert "1->3" in set(stale["ker_key"])

    def test_previous_version_is_preserved(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            self._approve_and_synthesise(wf)
            table1_store.rename_canonical_ke(1, "increased reactive oxygen species")
            wf.invalidate_for_ke(1, reason="renamed")

            history = wf.synthesis_history("1->3")
            assert len(history) == 1
            payload = json.loads(history.iloc[0]["payload"])
            assert payload["overall_confidence"] == "Moderate"

    def test_drifted_approval_fails_the_gate(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            wf.set_state("ke", 1, wf.State.CURATED)
            wf.approve("ke", 1, curator="jm")
            assert wf.gate(ke_ids=[1]).allowed is True

            table1_store.rename_canonical_ke(1, "something else entirely")

            status = wf.get_status("ke", 1)
            assert status.drifted is True
            assert status.is_approved is False
            assert wf.gate(ke_ids=[1]).allowed is False

    def test_graph_snapshot_is_invalidated(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            self._approve_and_synthesise(wf)
            with table1_store.connect() as conn:
                conn.execute(
                    "INSERT INTO aop_snapshot (name, payload, created_at) "
                    "VALUES ('v1', '{}', '2026-01-01')"
                )
                conn.commit()

            wf.invalidate_for_ke(1, reason="renamed")

            with table1_store.connect() as conn:
                stale = conn.execute(
                    "SELECT stale FROM aop_snapshot WHERE name = 'v1'"
                ).fetchone()[0]
            assert stale == 1

    def test_approval_is_logged(self):
        with Ctx():
            from stage2_extraction import workflow_state as wf
            wf.set_state("ke", 1, wf.State.CURATED)
            wf.approve("ke", 1, curator="jm", note="looks right")
            log = wf.approval_log("ke")
            assert len(log) == 2
            assert set(log["to_state"]) == {"curated", "approved"}
            assert "jm" in set(log["curator"].dropna())


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

class TestMerge:

    def test_preview_reports_consequences_without_writing(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            preview = cg.preview_merge([1, 2])

            assert preview.survivor_id == 1          # better supported
            assert preview.absorbed_ids == [2]
            assert "elevated ROS" in preview.aliases_moving
            assert preview.evidence_reassigned == 1
            assert preview.kers_consolidated          # both point at KE 3

            with table1_store.connect() as conn:
                still_there = conn.execute(
                    "SELECT COUNT(*) FROM ke_canonical WHERE canonical_id = 2"
                ).fetchone()[0]
            assert still_there == 1, "preview must not write"

    def test_preview_flags_direction_conflict(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            preview = cg.preview_merge([1, 4])   # increased vs decreased ROS
            assert preview.direction_conflicts
            assert preview.blocking

    def test_merge_refused_when_direction_conflicts(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            with pytest.raises(cg.MergeRefused):
                cg.merge_as_equivalent([1, 4], curator="jm")

    def test_merge_refused_for_non_equivalent_classification(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            from stage2_extraction.semantic_merge import KERecord, classify

            verdict = classify(
                KERecord(key="1", label="NaV1.2", level="Molecular"),
                KERecord(key="2", label="voltage-gated sodium channel", level="Molecular"),
            )
            with pytest.raises(cg.MergeRefused):
                cg.merge_as_equivalent([1, 2], classification=verdict, curator="jm")

    def test_merge_moves_aliases_and_evidence(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            result = cg.merge_as_equivalent([1, 2], curator="jm",
                                            rationale="same event, two wordings")
            assert result["survivor_id"] == 1

            with table1_store.connect() as conn:
                gone = conn.execute(
                    "SELECT COUNT(*) FROM ke_canonical WHERE canonical_id = 2"
                ).fetchone()[0]
                aliases = {
                    r[0] for r in conn.execute(
                        "SELECT raw_label FROM ke_alias WHERE canonical_id = 1"
                    )
                }
                links = conn.execute(
                    "SELECT COUNT(*) FROM table1_extractions "
                    "WHERE upstream_ke_canonical_id = 1"
                ).fetchone()[0]

            assert gone == 0
            assert "elevated ROS" in aliases
            assert links == 2

    def test_merge_appears_in_canonical_groups_with_provenance(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            cg.merge_as_equivalent([1, 2], curator="jm", rationale="same event")
            groups = cg.canonical_groups()

            assert len(groups) == 1
            row = groups.iloc[0]
            assert row["canonical_ke"] == "increased ROS"
            assert "elevated ROS" in row["original_aliases"]
            assert row["curator"] == "jm"
            assert row["rationale"] == "same event"
            assert row["n_claims"] == 2
            assert row["date"]

    def test_merge_is_reversible(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            result = cg.merge_as_equivalent([1, 2], curator="jm")
            cg.undo(result["decision_id"], curator="jm")

            with table1_store.connect() as conn:
                restored = conn.execute(
                    "SELECT canonical_name FROM ke_canonical WHERE canonical_id = 2"
                ).fetchone()
                alias_owner = conn.execute(
                    "SELECT canonical_id FROM ke_alias WHERE raw_label = 'elevated ROS'"
                ).fetchone()[0]
                link_owner = conn.execute(
                    "SELECT upstream_ke_canonical_id FROM table1_extractions "
                    "WHERE record_id = 11"
                ).fetchone()[0]

            assert restored is not None
            assert restored[0] == "elevated ROS"
            assert alias_owner == 2
            assert link_owner == 2

    def test_undo_twice_is_refused(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            result = cg.merge_as_equivalent([1, 2], curator="jm")
            cg.undo(result["decision_id"])
            with pytest.raises(ValueError):
                cg.undo(result["decision_id"])

    def test_split_pulls_one_alias_out(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            new_id = cg.split_alias(1, "ROS increase", curator="jm")

            with table1_store.connect() as conn:
                owner = conn.execute(
                    "SELECT canonical_id FROM ke_alias WHERE raw_label = 'ROS increase'"
                ).fetchone()[0]
                name = conn.execute(
                    "SELECT canonical_name FROM ke_canonical WHERE canonical_id = ?",
                    (new_id,),
                ).fetchone()[0]
            assert owner == new_id
            assert name == "ROS increase"


# ---------------------------------------------------------------------------
# Broader mapping is not a merge
# ---------------------------------------------------------------------------

class TestBroaderMapping:

    def test_mapping_leaves_the_key_event_alone(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            cg.map_to_broader(
                1, curie="GO:9999999", label="reactive oxygen species metabolism",
                source="go", curator="jm",
            )
            with table1_store.connect() as conn:
                name = conn.execute(
                    "SELECT canonical_name FROM ke_canonical WHERE canonical_id = 1"
                ).fetchone()[0]
                own_term = conn.execute(
                    "SELECT ontology_curie FROM ke_canonical WHERE canonical_id = 1"
                ).fetchone()[0]

            assert name == "increased ROS", "mapping must not rename anything"
            assert own_term == "GO:0000001", "the KE's own term must be untouched"

            mappings = cg.ontology_mappings(1)
            assert len(mappings) == 1
            assert mappings.iloc[0]["curie"] == "GO:9999999"
            assert mappings.iloc[0]["relation"] == "broader"

    def test_mapping_does_not_pool_evidence(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            cg.map_to_broader(1, curie="GO:9999999", label="parent", curator="jm")
            cg.map_to_broader(3, curie="GO:9999999", label="parent", curator="jm")

            with table1_store.connect() as conn:
                distinct = conn.execute(
                    "SELECT COUNT(*) FROM ke_canonical WHERE canonical_id IN (1, 3)"
                ).fetchone()[0]
            assert distinct == 2, (
                "two Key Events sharing a broader concept must stay two Key Events"
            )


# ---------------------------------------------------------------------------
# Decisions are not re-asked
# ---------------------------------------------------------------------------

class TestDecisionMemory:

    def test_keep_separate_is_remembered(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            cg.keep_separate([1, 2], curator="jm", rationale="different compartments")
            assert frozenset({1, 2}) in cg.decided_pairs()

    def test_decision_log_records_the_action(self):
        with Ctx():
            from stage2_extraction import canonical_groups as cg
            cg.mark_unresolved([1, 2], curator="jm")
            log = cg.decision_log()
            assert log.iloc[0]["action"] == "mark_unresolved"
            assert log.iloc[0]["action_label"] == "Mark as unresolved"
