from __future__ import annotations

"""
Merges, the record of them, and the way back.

Two things live here that the old code treated as one:

    merge_as_equivalent   these records are the same Key Event, fold them
                          into one and move the aliases across
    map_to_broader        this Key Event is a *kind of* that ontology term;
                          keep it exactly as it is and attach the parent

Collapsing those two into a single "OLS4 match" button is how evidence about
NaV1.2 ends up filed under voltage-gated sodium channels. A finding about one
channel subtype then reads as a finding about the whole class, and after the
merge there is nothing left in the record to show that it happened.

Everything a curator does is written to `merge_decision` with the classifier's
verdict, the explanation shown at the time, the curator's own rationale, and
JSON snapshots of the state before and after. `undo` replays the snapshot;
`split` pulls a single alias back out into its own Key Event. Provenance that
cannot be reversed is a log, not provenance.
"""

import datetime
import json
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from stage2_extraction import workflow_state
from stage2_extraction.semantic_merge import (
    Classification,
    KERecord,
    Relationship,
)
from stage2_extraction.table1_store import connect


#: What a curator can decide about a selected group of records.
ACTIONS = (
    "merge_equivalent",
    "collapse_broader",
    "keep_separate",
    "map_broader",
    "record_relation",
    "reject_not_ke",
    "mark_unresolved",
    "assign_labels",
)

ACTION_LABELS = {
    "merge_equivalent": "Merge as equivalent",
    "collapse_broader": "Collapse into the broader Key Event",
    "keep_separate": "Keep separate",
    "map_broader": "Map to a broader ontology concept",
    "record_relation": "Record a biological relationship",
    "reject_not_ke": "Reject as not being a Key Event",
    "mark_unresolved": "Mark as unresolved",
    "assign_labels": "Assign raw labels to Key Events",
}

#: Biological relationships a curator can record between two distinct KEs.
RELATION_TYPES = (
    "upstream_of",
    "downstream_of",
    "part_of",
    "has_part",
    "marker_for",
    "co_occurs_with",
    "modulates",
)


class MergeRefused(RuntimeError):
    """A merge the classifier does not permit."""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

@dataclass
class MergePreview:
    """
    Exactly what a merge would do, computed before anything is written.

    Shown in full rather than summarised. A curator asked to confirm "merge 4
    records" is being asked to approve consequences they cannot see; the same
    curator told "9 aliases move, 3 KERs consolidate, 1 becomes a self-loop"
    can decline for a reason.
    """

    survivor_id: int
    survivor_name: str
    absorbed_ids: list[int] = field(default_factory=list)
    aliases_moving: list[str] = field(default_factory=list)
    kers_consolidated: list[str] = field(default_factory=list)
    evidence_reassigned: int = 0
    self_loops: list[str] = field(default_factory=list)
    direction_conflicts: list[str] = field(default_factory=list)
    level_conflicts: list[str] = field(default_factory=list)
    applicability_conflicts: list[str] = field(default_factory=list)
    approved_records_touched: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[str]:
        """Problems that make the merge wrong, not merely notable."""
        return self.self_loops + self.direction_conflicts

    @property
    def warnings(self) -> list[str]:
        return self.level_conflicts + self.applicability_conflicts

    @property
    def is_clean(self) -> bool:
        return not self.blocking and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "survivor_id": self.survivor_id,
            "survivor_name": self.survivor_name,
            "absorbed_ids": self.absorbed_ids,
            "aliases_moving": self.aliases_moving,
            "kers_consolidated": self.kers_consolidated,
            "evidence_reassigned": self.evidence_reassigned,
            "self_loops": self.self_loops,
            "direction_conflicts": self.direction_conflicts,
            "level_conflicts": self.level_conflicts,
            "applicability_conflicts": self.applicability_conflicts,
            "approved_records_touched": self.approved_records_touched,
        }


def preview_merge(member_ids: Sequence[int], survivor_id: Optional[int] = None) -> MergePreview:
    """
    Work out the consequences of merging `member_ids` without writing anything.

    The survivor defaults to the record backed by the most rows, so the
    best-supported name is the one that stays by default.
    """
    ids = [int(i) for i in member_ids]
    if len(ids) < 2:
        raise ValueError("A merge needs at least two records.")

    with connect() as conn:
        rows = {
            int(r["canonical_id"]): dict(r)
            for r in conn.execute(
                f"SELECT * FROM ke_canonical WHERE canonical_id IN "
                f"({','.join('?' * len(ids))})",
                ids,
            )
        }
        missing = [i for i in ids if i not in rows]
        if missing:
            raise ValueError(f"No such canonical Key Event: {missing}")

        if survivor_id is None:
            survivor_id = max(ids, key=lambda i: (rows[i]["n_source_rows"] or 0, -i))
        survivor_id = int(survivor_id)
        absorbed = [i for i in ids if i != survivor_id]

        preview = MergePreview(
            survivor_id=survivor_id,
            survivor_name=str(rows[survivor_id]["canonical_name"]),
            absorbed_ids=absorbed,
        )

        # Aliases that will move.
        if absorbed:
            preview.aliases_moving = [
                str(r[0])
                for r in conn.execute(
                    f"SELECT raw_label FROM ke_alias WHERE canonical_id IN "
                    f"({','.join('?' * len(absorbed))}) ORDER BY raw_label",
                    absorbed,
                )
            ]

        # Evidence rows pointing at an absorbed record.
        if absorbed:
            placeholders = ",".join("?" * len(absorbed))
            preview.evidence_reassigned = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM table1_extractions WHERE "
                    f"upstream_ke_canonical_id IN ({placeholders}) OR "
                    f"downstream_ke_canonical_id IN ({placeholders})",
                    absorbed + absorbed,
                ).fetchone()[0]
            )

        _preview_kers(conn, ids, survivor_id, preview)
        _preview_conflicts(conn, ids, rows, survivor_id, preview)

        for i in ids:
            status = workflow_state.get_status("ke", str(i))
            if status.is_approved:
                preview.approved_records_touched.append(
                    f"{rows[i]['canonical_name']} (approved "
                    f"{status.approved_at or 'previously'})"
                )

    return preview


def _preview_kers(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    survivor_id: int,
    preview: MergePreview,
) -> None:
    """Which relationships collapse together, and which become self-loops."""
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT upstream_ke_canonical_id AS up, downstream_ke_canonical_id AS down, "
        f"ker_name, COUNT(*) AS n FROM table1_extractions "
        f"WHERE upstream_ke_canonical_id IN ({placeholders}) "
        f"   OR downstream_ke_canonical_id IN ({placeholders}) "
        f"GROUP BY up, down, ker_name",
        list(ids) + list(ids),
    ).fetchall()

    member = set(ids)
    consolidated: dict[tuple[Any, Any], list[str]] = {}

    for row in rows:
        up = survivor_id if row["up"] in member else row["up"]
        down = survivor_id if row["down"] in member else row["down"]

        if up is not None and up == down:
            preview.self_loops.append(
                f"“{row['ker_name']}” would point {preview.survivor_name} at "
                f"itself"
            )
            continue

        consolidated.setdefault((up, down), []).append(str(row["ker_name"]))

    for (up, down), names in consolidated.items():
        if len(names) > 1:
            preview.kers_consolidated.append(
                f"{len(names)} relationships collapse into one: "
                + "; ".join(sorted(set(names))[:3])
                + ("…" if len(set(names)) > 3 else "")
            )


def _preview_conflicts(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    rows: dict[int, dict],
    survivor_id: int,
    preview: MergePreview,
) -> None:
    """Direction, level and applicability disagreements among the members."""
    from stage2_extraction.semantic_merge import read_state

    names = {i: str(rows[i]["canonical_name"]) for i in ids}

    signs = {i: read_state(names[i]).sign for i in ids}
    positives = [names[i] for i in ids if signs[i] > 0]
    negatives = [names[i] for i in ids if signs[i] < 0]
    if positives and negatives:
        preview.direction_conflicts.append(
            f"Opposed directions in the same group: "
            f"{', '.join(positives)} against {', '.join(negatives)}"
        )

    levels = {str(rows[i]["level"]) for i in ids if rows[i]["level"]}
    if len(levels) > 1:
        preview.level_conflicts.append(
            f"Members sit at different biological levels ({', '.join(sorted(levels))}); "
            f"the merged record would take “{rows[survivor_id]['level']}”"
        )

    curies = {
        str(rows[i]["ontology_curie"]) for i in ids if rows[i]["ontology_curie"]
    }
    if len(curies) > 1:
        preview.level_conflicts.append(
            f"Members are annotated to different ontology terms "
            f"({', '.join(sorted(curies))}); only the survivor's is kept"
        )

    placeholders = ",".join("?" * len(ids))
    for column, what in (
        ("taxonomic_applicability", "taxon"),
        ("sex_applicability", "sex"),
        ("life_stage_applicability", "life stage"),
    ):
        values = {
            str(r[0]).strip()
            for r in conn.execute(
                f"SELECT DISTINCT {column} FROM table1_extractions "
                f"WHERE upstream_ke_canonical_id IN ({placeholders}) "
                f"   OR downstream_ke_canonical_id IN ({placeholders})",
                list(ids) + list(ids),
            )
            if r[0] and str(r[0]).strip().lower() not in {"none", "not specified", ""}
        }
        if len(values) > 1:
            preview.applicability_conflicts.append(
                f"Evidence spans several {what} values "
                f"({', '.join(sorted(values)[:4])}"
                + ("…" if len(values) > 4 else "")
                + ") — check they describe one applicability domain"
            )


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def record_decision(
    *,
    action: str,
    member_ids: Sequence[int],
    classification: Optional[Classification] = None,
    survivor_id: Optional[int] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
    method: str = "manual",
    before_state: Optional[dict] = None,
    after_state: Optional[dict] = None,
    group_uid: Optional[str] = None,
) -> int:
    """Write one curator decision to the audit trail. Returns its id."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}")

    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO merge_decision "
            "(group_uid, action, relationship, member_ids, survivor_id, "
            " similarity, explanation, curator_rationale, curator, method, "
            " before_state, after_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                group_uid or uuid.uuid4().hex,
                action,
                classification.relationship.value if classification else None,
                json.dumps([int(i) for i in member_ids]),
                int(survivor_id) if survivor_id is not None else None,
                classification.similarity if classification else None,
                classification.explanation if classification else None,
                rationale,
                curator,
                method,
                json.dumps(before_state, default=str) if before_state else None,
                json.dumps(after_state, default=str) if after_state else None,
                _now(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _snapshot(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[str, Any]:
    """Everything needed to put these records back exactly as they were."""
    placeholders = ",".join("?" * len(ids))
    canonical = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM ke_canonical WHERE canonical_id IN ({placeholders})",
            list(ids),
        )
    ]
    aliases = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM ke_alias WHERE canonical_id IN ({placeholders})",
            list(ids),
        )
    ]
    extractions = [
        {"record_id": r["record_id"], "up": r["upstream_ke_canonical_id"],
         "down": r["downstream_ke_canonical_id"]}
        for r in conn.execute(
            f"SELECT record_id, upstream_ke_canonical_id, downstream_ke_canonical_id "
            f"FROM table1_extractions WHERE upstream_ke_canonical_id IN ({placeholders}) "
            f"OR downstream_ke_canonical_id IN ({placeholders})",
            list(ids) + list(ids),
        )
    ]
    mappings = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM ontology_mapping WHERE canonical_id IN ({placeholders})",
            list(ids),
        )
    ]
    return {
        "ke_canonical": canonical,
        "ke_alias": aliases,
        "table1_links": extractions,
        "ontology_mapping": mappings,
        "taken_at": _now(),
    }


def merge_as_equivalent(
    member_ids: Sequence[int],
    *,
    classification: Optional[Classification] = None,
    survivor_id: Optional[int] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
    allow_conflicts: bool = False,
) -> dict[str, Any]:
    """
    Fold several canonical Key Events into one.

    Refuses when `classification` says anything other than `equivalent`. The
    guard is here rather than only in the UI because this is the function that
    destroys information, and a destructive operation should not rely on its
    caller having checked.

    Passing no classification is allowed — a curator may merge two records the
    classifier never paired up — but the decision is then recorded with
    method "manual_override" so the audit trail shows nothing vouched for it.
    """
    ids = [int(i) for i in member_ids]
    if len(ids) < 2:
        raise ValueError("A merge needs at least two records.")

    if classification is not None and classification.relationship is not Relationship.EQUIVALENT:
        raise MergeRefused(
            f"These records are classified “{classification.relationship.label}”, "
            f"not Equivalent, so they cannot be merged. "
            f"{classification.explanation}"
        )

    preview = preview_merge(ids, survivor_id)
    if preview.blocking and not allow_conflicts:
        raise MergeRefused(
            "The merge would create: " + "; ".join(preview.blocking)
        )

    survivor = preview.survivor_id
    absorbed = preview.absorbed_ids
    group_uid = uuid.uuid4().hex

    with connect() as conn:
        before = _snapshot(conn, ids)

        for source_id in absorbed:
            _fold(conn, source_id, survivor)

        conn.execute(
            "UPDATE ke_canonical SET merge_method = 'curator', updated_at = ? "
            "WHERE canonical_id = ?",
            (_now(), survivor),
        )
        after = _snapshot(conn, [survivor])
        conn.commit()

    decision_id = record_decision(
        action="merge_equivalent",
        member_ids=ids,
        classification=classification,
        survivor_id=survivor,
        curator=curator,
        rationale=rationale,
        method="assisted" if classification is not None else "manual_override",
        before_state=before,
        after_state=after,
        group_uid=group_uid,
    )

    affected = workflow_state.invalidate_for_ke(
        survivor, reason=f"merged {len(absorbed)} record(s) into it"
    )

    return {
        "decision_id": decision_id,
        "survivor_id": survivor,
        "absorbed_ids": absorbed,
        "aliases_moved": len(preview.aliases_moving),
        "syntheses_invalidated": affected,
        "preview": preview,
    }


def collapse_into_broader(
    member_ids: Sequence[int],
    *,
    survivor_id: int,
    classification: Optional[Classification] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
    allow_conflicts: bool = False,
) -> dict[str, Any]:
    """
    Pool a specific Key Event into the broader one, knowingly.

    `merge_as_equivalent` refuses anything the classifier did not call
    equivalent, and that refusal is right for its own purpose: two records
    being folded together as *the same event* when one is a subtype of the
    other is how evidence about NaV1.2 silently becomes evidence about sodium
    channels in general.

    But a curator working at the class level is not making that mistake, they
    are making a decision — this AOP is about sodium-channel function, the
    subtype distinction is not one it turns on, and three records for three
    isoforms are three fragments of one Key Event. Refusing that outright left
    only `map_to_broader`, which keeps both records and so does not do what was
    asked.

    So the operation is allowed and recorded as what it is. The mechanics are
    identical to an equivalence merge — same fold, same reversible snapshot —
    but the decision is logged with action `collapse_broader` rather than
    `merge_equivalent`, so the audit trail distinguishes "these were the same
    event" from "these were pooled at a coarser grain". The absorbed records'
    names survive as aliases either way, which is what makes the coarsening
    legible afterwards rather than merely irreversible.

    `survivor_id` is required and is not inferred: which record is the broader
    one is the whole content of the decision, and `preview_merge` picks a
    survivor on evidence volume, which has nothing to do with breadth.
    """
    ids = [int(i) for i in member_ids]
    if len(ids) < 2:
        raise ValueError("A collapse needs at least two records.")
    if int(survivor_id) not in ids:
        raise ValueError("The surviving record must be one of the members.")

    if classification is not None and classification.relationship in (
        Relationship.CONTRADICTORY,
    ):
        # The one classification that is never a grain question. Two records
        # the classifier calls incompatible describe opposite findings, and
        # pooling those is not coarsening, it is losing one of them.
        raise MergeRefused(
            "These records are classified “Contradictory or incompatible”. "
            "That is not a difference of grain — pooling them would discard "
            f"one of two opposite findings. {classification.explanation}"
        )

    preview = preview_merge(ids, int(survivor_id))
    if preview.blocking and not allow_conflicts:
        raise MergeRefused("The collapse would create: " + "; ".join(preview.blocking))

    survivor = preview.survivor_id
    absorbed = preview.absorbed_ids
    group_uid = uuid.uuid4().hex

    with connect() as conn:
        before = _snapshot(conn, ids)
        for source_id in absorbed:
            _fold(conn, source_id, survivor)
        conn.execute(
            "UPDATE ke_canonical SET merge_method = 'curator_coarsened', "
            "updated_at = ? WHERE canonical_id = ?",
            (_now(), survivor),
        )
        after = _snapshot(conn, [survivor])
        conn.commit()

    decision_id = record_decision(
        action="collapse_broader",
        member_ids=ids,
        classification=classification,
        survivor_id=survivor,
        curator=curator,
        rationale=rationale or (
            "Pooled into the broader Key Event; the subtype distinction is not "
            "one this AOP turns on."
        ),
        method="coarsening",
        before_state=before,
        after_state=after,
        group_uid=group_uid,
    )

    affected = workflow_state.invalidate_for_ke(
        survivor, reason=f"collapsed {len(absorbed)} narrower record(s) into it"
    )

    return {
        "decision_id": decision_id,
        "survivor_id": survivor,
        "absorbed_ids": absorbed,
        "aliases_moved": len(preview.aliases_moving),
        "syntheses_invalidated": affected,
        "preview": preview,
    }


def _fold(conn: sqlite3.Connection, source_id: int, target_id: int) -> None:
    """Move one canonical record's aliases, links and counts onto another."""
    conn.execute(
        "UPDATE OR REPLACE ke_alias SET canonical_id = ? WHERE canonical_id = ?",
        (target_id, source_id),
    )
    conn.execute(
        "UPDATE table1_extractions SET upstream_ke_canonical_id = ? "
        "WHERE upstream_ke_canonical_id = ?",
        (target_id, source_id),
    )
    conn.execute(
        "UPDATE table1_extractions SET downstream_ke_canonical_id = ? "
        "WHERE downstream_ke_canonical_id = ?",
        (target_id, source_id),
    )
    conn.execute(
        "UPDATE OR IGNORE ontology_mapping SET canonical_id = ? WHERE canonical_id = ?",
        (target_id, source_id),
    )
    conn.execute(
        "UPDATE ke_canonical SET n_source_rows = n_source_rows + "
        "(SELECT COALESCE(n_source_rows, 0) FROM ke_canonical WHERE canonical_id = ?) "
        "WHERE canonical_id = ?",
        (source_id, target_id),
    )
    conn.execute("DELETE FROM ke_canonical WHERE canonical_id = ?", (source_id,))


def map_to_broader(
    canonical_id: int,
    *,
    curie: str,
    label: Optional[str] = None,
    iri: Optional[str] = None,
    source: Optional[str] = None,
    score: float = 0.0,
    relation: str = "broader",
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """
    Attach an ontology parent while leaving the Key Event exactly as it is.

    The point of the separate function: NaV1.2 stays NaV1.2 and gains a link
    to the voltage-gated sodium channel class. Evidence about the subtype is
    still filed against the subtype, and a reader who wants the class view can
    roll up through the mapping instead of finding the two already fused.
    """
    with connect() as conn:
        before = _snapshot(conn, [canonical_id])
        conn.execute(
            "INSERT OR REPLACE INTO ontology_mapping "
            "(canonical_id, relation, curie, iri, label, source, score, "
            " curator, rationale, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(canonical_id), relation, curie.strip().upper(), iri, label,
             source, float(score), curator, rationale, _now()),
        )
        after = _snapshot(conn, [canonical_id])
        conn.commit()

    return record_decision(
        action="map_broader",
        member_ids=[canonical_id],
        survivor_id=canonical_id,
        curator=curator,
        rationale=rationale or f"Mapped to broader concept {curie}",
        before_state=before,
        after_state=after,
    )


def record_relation(
    source_id: int,
    target_id: int,
    relation: str,
    *,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """Note that two distinct Key Events are biologically related."""
    if relation not in RELATION_TYPES:
        raise ValueError(f"Unknown relation {relation!r}")
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ke_relation "
            "(source_id, target_id, relation, curator, rationale, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(source_id), int(target_id), relation, curator, rationale, _now()),
        )
        conn.commit()
    return record_decision(
        action="record_relation",
        member_ids=[source_id, target_id],
        curator=curator,
        rationale=rationale or f"{source_id} {relation} {target_id}",
    )


def keep_separate(
    member_ids: Sequence[int],
    *,
    classification: Optional[Classification] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """
    Record that a suggested duplicate is not one.

    Worth storing rather than simply dismissing: without it the same pair is
    re-suggested on every run and the curator re-decides it every time.
    """
    return record_decision(
        action="keep_separate",
        member_ids=member_ids,
        classification=classification,
        curator=curator,
        rationale=rationale,
    )


def apply_assignments(
    assignments: Sequence[tuple[str, str]],
    *,
    excluded: Iterable[str] = (),
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> dict[str, Any]:
    """
    Rebuild the canonical Key Events from an explicit label-to-event mapping.

    `assignments` is (raw_label, target_event_name) for every raw label in
    Table 1. Labels sharing a target become one Key Event with the others as
    its synonyms; a label given its own name stays its own event. Names in
    `excluded` are kept — nothing is deleted, and their evidence stays
    readable — but are marked rejected so they do not reach the map.

    This is the bulk counterpart to the pairwise merge actions. Pairwise
    classification is the right tool for two labels a curator is unsure
    about; it is the wrong tool for the first pass over a fresh run, where
    the curator already knows which labels name the same event and only
    wants to say so once.

    Returns a summary dict.
    """
    from schemas import CanonicalKE
    from stage2_extraction import ke_normalizer, table1_store

    pairs = [
        (str(label).strip(), str(target).strip())
        for label, target in assignments
        if str(label).strip() and str(target).strip()
    ]
    if not pairs:
        raise ValueError("No assignments were given.")

    excluded_names = {str(name).strip() for name in excluded if str(name).strip()}

    table1 = table1_store.load_table1_as_dataframe()
    previous = table1_store.load_canonical_kes()

    # Level and AOP-Wiki id come from what the papers said about each label,
    # not from the curator's grouping — the grouping decides identity, the
    # extraction decides biology.
    raw = ke_normalizer.collect_raw_kes(table1)
    levels_by_label: dict[str, Counter] = defaultdict(Counter)
    wiki_by_label: dict[str, Counter] = defaultdict(Counter)
    rows_by_label: Counter = Counter()
    for label, level, wiki_id in raw:
        levels_by_label[label][level] += 1
        rows_by_label[label] += 1
        if wiki_id is not None:
            wiki_by_label[label][wiki_id] += 1

    # Ontology annotation is expensive and already paid for; carry it across
    # by name so a rename does not silently drop the term.
    ontology_by_name: dict[str, dict[str, Any]] = {}
    if not previous.empty:
        for _, row in previous.iterrows():
            ontology_by_name[str(row["canonical_name"])] = {
                "ontology_curie": row.get("ontology_curie"),
                "ontology_iri": row.get("ontology_iri"),
                "ontology_label": row.get("ontology_label"),
                "ontology_source": row.get("ontology_source"),
                "ontology_score": float(row.get("ontology_score") or 0.0),
            }

    before_state = {
        "ke_canonical": (
            previous[["canonical_id", "canonical_name", "level", "n_source_rows"]]
            .to_dict("records") if not previous.empty else []
        ),
    }

    grouped: dict[str, list[str]] = {}
    for label, target in pairs:
        grouped.setdefault(target, [])
        if label not in grouped[target]:
            grouped[target].append(label)

    canonical_kes: list[CanonicalKE] = []
    label_to_index: dict[str, int] = {}

    for index, (target, labels) in enumerate(sorted(grouped.items())):
        levels: Counter = Counter()
        wiki: Counter = Counter()
        n_rows = 0
        for label in labels:
            levels.update(levels_by_label.get(label, {}))
            wiki.update(wiki_by_label.get(label, {}))
            n_rows += rows_by_label.get(label, 0)
            label_to_index[label] = index

        # A curator decision is a basis like any other, and the crosswalk has
        # to keep saying something after a manual pass. Leaving these blank
        # would make the assignments the curator is most sure of look like the
        # ones with no reason recorded.
        alias_basis = {
            label: [
                "curator",
                (
                    f"Assigned to “{target}” by hand"
                    + (f" — {rationale}" if rationale else "")
                    + "."
                    if str(label).strip().casefold() != str(target).strip().casefold()
                    else f"Kept as its own Key Event by hand."
                ),
            ]
            for label in labels
        }

        ontology = ontology_by_name.get(target, {})
        canonical_kes.append(
            CanonicalKE(
                canonical_id=None,
                canonical_name=target,
                level=levels.most_common(1)[0][0] if levels else "Molecular",
                aliases=list(labels),
                alias_basis=alias_basis,
                aopwiki_ke_id=wiki.most_common(1)[0][0] if wiki else None,
                merge_method="manual",
                curation_status="rejected" if target in excluded_names else "accepted",
                n_source_rows=n_rows,
                **ontology,
            )
        )

    assigned_ids = table1_store.replace_canonical_kes(canonical_kes, label_to_index)

    # `replace_canonical_kes` preserves the previous status by name so that a
    # re-run of clustering does not wipe curation. Here the status IS the
    # decision being made, so it is written afterwards, deliberately.
    accepted: list[int] = []
    rejected: list[int] = []
    for ke, canonical_id in zip(canonical_kes, assigned_ids):
        table1_store.set_canonical_ke_status(canonical_id, ke.curation_status)
        (rejected if ke.curation_status == "rejected" else accepted).append(canonical_id)

    workflow_state.bulk_set(
        [("ke", str(i)) for i in accepted],
        workflow_state.State.CURATED,
        curator=curator,
        note="assigned in the curation table",
    )

    after_state = {
        "ke_canonical": [
            {
                "canonical_id": canonical_id,
                "canonical_name": ke.canonical_name,
                "level": ke.level,
                "aliases": ke.aliases,
                "curation_status": ke.curation_status,
            }
            for ke, canonical_id in zip(canonical_kes, assigned_ids)
        ],
    }

    decision_id = record_decision(
        action="assign_labels",
        member_ids=assigned_ids,
        curator=curator,
        rationale=rationale,
        before_state=before_state,
        after_state=after_state,
    )

    return {
        "decision_id": decision_id,
        "n_events": len(canonical_kes),
        "n_labels": len(label_to_index),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        # A wording sent to two events — one channel measured in two cell types
        # — counts once in `label_to_index` and twice in `canonical_kes`, so the
        # difference can go negative. It is a count of synonyms, and there is no
        # such thing as minus two synonyms.
        "n_synonyms": max(0, len(label_to_index) - len(canonical_kes)),
    }


def split_by_cell_lineage(
    canonical_id: int,
    *,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> dict[str, Any]:
    """
    Split one Key Event into one per cell lineage its evidence came from.

    The unit of the split is the Table 1 ROW, not the label, and that is the
    whole difficulty. Every paper wrote "voltage-gated sodium channel
    activity"; what differs is the cell each one measured it in, which lives
    on the row. So the existing alias-level split cannot do this — the label
    is identical in every case — and the rows have to be repointed one at a
    time at whichever new Key Event matches their own cell type.

    Rows whose cell type was never stated stay with the original Key Event.
    They are not evidence for either lineage, and guessing one for them would
    manufacture the certainty this whole exercise is meant to avoid.

    Returns a summary. Reversible only by re-normalising.
    """
    from stage2_extraction import cell_lineage

    with connect() as conn:
        row = conn.execute(
            "SELECT canonical_name, level FROM ke_canonical WHERE canonical_id = ?",
            (int(canonical_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"No Key Event {canonical_id}")
        base_name, level = str(row["canonical_name"]), str(row["level"])

        affected = conn.execute(
            "SELECT record_id, upstream_ke_canonical_id, downstream_ke_canonical_id, "
            "       upstream_cell_type, downstream_cell_type "
            "FROM table1_extractions "
            "WHERE upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?",
            (int(canonical_id), int(canonical_id)),
        ).fetchall()

        # Which lineage does each row's endpoint belong to?
        plan: list[tuple[int, str, str]] = []   # (record_id, side, lineage)
        for record in affected:
            for side in ("upstream", "downstream"):
                if record[f"{side}_ke_canonical_id"] != canonical_id:
                    continue
                name = cell_lineage.lineage(record[f"{side}_cell_type"])
                if name not in (cell_lineage.UNSPECIFIED, cell_lineage.UNRESOLVED):
                    plan.append((int(record["record_id"]), side, name))

        lineages = sorted({name for _, _, name in plan})
        if len(lineages) < 2:
            raise ValueError(
                f"“{base_name}” has evidence from {len(lineages)} identified "
                f"cell lineage(s); there is nothing to split."
            )

        before_state = {
            "ke_canonical": [
                {"canonical_id": int(canonical_id), "canonical_name": base_name,
                 "level": level}
            ],
            "n_rows": len(affected),
        }

        now = _now()
        new_ids: dict[str, int] = {}
        for name in lineages:
            new_name = f"{base_name} {cell_lineage.suffix_for(name)}"
            existing = conn.execute(
                "SELECT canonical_id FROM ke_canonical WHERE canonical_name = ?",
                (new_name,),
            ).fetchone()
            if existing:
                new_ids[name] = int(existing["canonical_id"])
                continue
            cursor = conn.execute(
                "INSERT INTO ke_canonical "
                "(canonical_name, level, merge_method, curation_status, "
                " n_source_rows, updated_at) VALUES (?, ?, 'manual', 'accepted', 0, ?)",
                (new_name, level, now),
            )
            new_ids[name] = int(cursor.lastrowid)

        moved = 0
        for record_id, side, name in plan:
            conn.execute(
                f"UPDATE table1_extractions SET {side}_ke_canonical_id = ? "
                f"WHERE record_id = ?",
                (new_ids[name], record_id),
            )
            moved += 1

        # Aliases follow the rows: each new Key Event keeps the original
        # wording, because that is still what the papers wrote.
        for new_id in set(new_ids.values()):
            conn.execute(
                "INSERT OR IGNORE INTO ke_alias (canonical_id, raw_label) VALUES (?, ?)",
                (new_id, base_name),
            )
            conn.execute(
                "UPDATE ke_canonical SET n_source_rows = ("
                "  SELECT COUNT(*) FROM table1_extractions "
                "  WHERE upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?"
                ") WHERE canonical_id = ?",
                (new_id, new_id, new_id),
            )

        remaining = int(
            conn.execute(
                "SELECT COUNT(*) FROM table1_extractions "
                "WHERE upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?",
                (int(canonical_id), int(canonical_id)),
            ).fetchone()[0]
        )
        if remaining == 0:
            conn.execute(
                "DELETE FROM ke_canonical WHERE canonical_id = ?", (int(canonical_id),)
            )
            conn.execute(
                "DELETE FROM workflow_state WHERE target_type = 'ke' AND target_key = ?",
                (str(canonical_id),),
            )
        else:
            conn.execute(
                "UPDATE ke_canonical SET n_source_rows = ? WHERE canonical_id = ?",
                (remaining, int(canonical_id)),
            )

        # Splitting a node invalidates every relationship that touched it.
        conn.execute(
            "DELETE FROM workflow_state WHERE target_type = 'ker' "
            "AND (target_key LIKE ? OR target_key LIKE ?)",
            (f"{canonical_id}->%", f"%->{canonical_id}"),
        )
        conn.commit()

    for new_id in set(new_ids.values()):
        workflow_state.set_state(
            "ke", str(new_id), workflow_state.State.CURATED,
            curator=curator, note="split out by cell lineage", force=True,
        )

    decision_id = record_decision(
        action="assign_labels",
        member_ids=sorted(set(new_ids.values())),
        curator=curator,
        rationale=rationale or f"Split “{base_name}” by cell lineage.",
        before_state=before_state,
        after_state={
            "ke_canonical": [
                {"canonical_id": i, "canonical_name":
                 f"{base_name} {cell_lineage.suffix_for(n)}"}
                for n, i in sorted(new_ids.items())
            ],
        },
    )

    return {
        "decision_id": decision_id,
        "base_name": base_name,
        "lineages": lineages,
        "new_ids": new_ids,
        "rows_moved": moved,
        "rows_left_unassigned": remaining,
    }


def reject_not_ke(
    canonical_id: int,
    *,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """Mark a record as not a Key Event — a study observation, or an entity."""
    with connect() as conn:
        conn.execute(
            "UPDATE ke_canonical SET curation_status = 'rejected', updated_at = ? "
            "WHERE canonical_id = ?",
            (_now(), int(canonical_id)),
        )
        conn.commit()
    workflow_state.invalidate_for_ke(canonical_id, reason="rejected as not a Key Event")
    return record_decision(
        action="reject_not_ke",
        member_ids=[canonical_id],
        survivor_id=canonical_id,
        curator=curator,
        rationale=rationale,
    )


def mark_unresolved(
    member_ids: Sequence[int],
    *,
    classification: Optional[Classification] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """Park a group without deciding. Keeps it visible instead of silently open."""
    return record_decision(
        action="mark_unresolved",
        member_ids=member_ids,
        classification=classification,
        curator=curator,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# The Canonical Groups view
# ---------------------------------------------------------------------------

def canonical_groups(include_reverted: bool = False) -> pd.DataFrame:
    """
    Every completed merge, with the provenance the redesign asks for.

    Columns: canonical KE, original aliases, ontology mapping, source
    publications, number of claims, merge method, curator, rationale, date,
    and the decision id needed to undo it.
    """
    with connect() as conn:
        sql = (
            "SELECT decision_id, group_uid, member_ids, survivor_id, relationship, "
            "       explanation, curator_rationale, curator, method, action, "
            "       created_at, reverted, reverted_at, before_state, after_state "
            # A coarsening folds records exactly as an equivalence merge does,
            # so it belongs in the same log and needs the same undo. Filtering
            # on 'merge_equivalent' alone would have made it invisible here
            # and, in practice, permanent.
            "FROM merge_decision WHERE action IN ('merge_equivalent', 'collapse_broader')"
        )
        if not include_reverted:
            sql += " AND reverted = 0"
        sql += " ORDER BY decision_id DESC"
        decisions = [dict(r) for r in conn.execute(sql)]

        rows = []
        for d in decisions:
            survivor_id = d["survivor_id"]
            survivor = conn.execute(
                "SELECT canonical_name, level, ontology_curie, ontology_label, "
                "       n_source_rows FROM ke_canonical WHERE canonical_id = ?",
                (survivor_id,),
            ).fetchone()

            before = json.loads(d["before_state"] or "{}")
            original_names = sorted(
                {str(r.get("canonical_name")) for r in before.get("ke_canonical", [])}
            )
            original_aliases = sorted(
                {str(r.get("raw_label")) for r in before.get("ke_alias", [])}
            )

            papers = [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT source_doi FROM table1_extractions "
                    "WHERE upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?",
                    (survivor_id, survivor_id),
                )
                if r[0]
            ]
            n_claims = int(
                conn.execute(
                    "SELECT COUNT(*) FROM table1_extractions "
                    "WHERE upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?",
                    (survivor_id, survivor_id),
                ).fetchone()[0]
            )
            parents = [
                f"{r[0]} ({r[1]})"
                for r in conn.execute(
                    "SELECT curie, relation FROM ontology_mapping WHERE canonical_id = ?",
                    (survivor_id,),
                )
            ]

            status = workflow_state.get_status("ke", str(survivor_id))

            rows.append(
                {
                    "decision_id": d["decision_id"],
                    "action": d["action"],
                    "action_label": ACTION_LABELS.get(d["action"], d["action"]),
                    "canonical_ke": survivor["canonical_name"] if survivor else "(deleted)",
                    "level": survivor["level"] if survivor else "",
                    "original_names": "; ".join(original_names),
                    "original_aliases": "; ".join(original_aliases),
                    "ontology_term": (
                        f"{survivor['ontology_curie']} — {survivor['ontology_label']}"
                        if survivor and survivor["ontology_curie"] else ""
                    ),
                    "broader_concepts": "; ".join(parents),
                    "source_publications": "; ".join(sorted(papers)),
                    "n_publications": len(papers),
                    "n_claims": n_claims,
                    "merge_method": d["method"],
                    "classification": d["relationship"] or "",
                    "explanation": d["explanation"] or "",
                    "curator": d["curator"] or "",
                    "rationale": d["curator_rationale"] or "",
                    "date": d["created_at"],
                    "workflow_state": status.effective_state.label,
                    "reverted": bool(d["reverted"]),
                    "reverted_at": d["reverted_at"] or "",
                }
            )

    return pd.DataFrame(rows)


def group_detail(decision_id: int) -> dict[str, Any]:
    """Before-and-after state for one merge, for the expander in the UI."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM merge_decision WHERE decision_id = ?", (decision_id,)
        ).fetchone()
    if row is None:
        return {}
    return {
        "decision": dict(row),
        "before": json.loads(row["before_state"] or "{}"),
        "after": json.loads(row["after_state"] or "{}"),
        "member_ids": json.loads(row["member_ids"] or "[]"),
    }


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------

def undo(decision_id: int, *, curator: Optional[str] = None) -> dict[str, Any]:
    """
    Put a merge back exactly as it was.

    Replays the `before_state` snapshot: the absorbed canonical rows are
    recreated with their original ids, their aliases are moved back, and every
    Table 1 link is restored to the record it pointed at. Ids are reused
    deliberately so that anything referring to them — a saved layout, an
    export, a note in someone's manuscript — still resolves.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM merge_decision WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No decision {decision_id}")
        if row["reverted"]:
            raise ValueError("That decision has already been undone.")
        # A coarsening folds records exactly as an equivalence merge does, so
        # the same snapshot replay restores it. Leaving it out of this set was
        # the difference between a reversible decision and a permanent one.
        if row["action"] not in {"merge_equivalent", "collapse_broader", "map_broader"}:
            raise ValueError(f"{row['action']} is not reversible in this way.")

        before = json.loads(row["before_state"] or "{}")
        if not before:
            raise ValueError("No before-state was recorded; this cannot be undone.")

        restored_ids = []
        for record in before.get("ke_canonical", []):
            columns = ", ".join(record.keys())
            placeholders = ", ".join("?" * len(record))
            conn.execute(
                f"INSERT OR REPLACE INTO ke_canonical ({columns}) "
                f"VALUES ({placeholders})",
                list(record.values()),
            )
            restored_ids.append(int(record["canonical_id"]))

        for alias in before.get("ke_alias", []):
            conn.execute(
                "INSERT OR REPLACE INTO ke_alias "
                "(alias_id, canonical_id, raw_label, n_uses) VALUES (?, ?, ?, ?)",
                (alias.get("alias_id"), alias["canonical_id"],
                 alias["raw_label"], alias.get("n_uses", 1)),
            )

        for link in before.get("table1_links", []):
            conn.execute(
                "UPDATE table1_extractions SET upstream_ke_canonical_id = ?, "
                "downstream_ke_canonical_id = ? WHERE record_id = ?",
                (link["up"], link["down"], link["record_id"]),
            )

        if restored_ids:
            conn.execute(
                "DELETE FROM ontology_mapping WHERE canonical_id IN "
                f"({','.join('?' * len(restored_ids))})",
                restored_ids,
            )
        for mapping in before.get("ontology_mapping", []):
            conn.execute(
                "INSERT OR REPLACE INTO ontology_mapping "
                "(mapping_id, canonical_id, relation, curie, iri, label, source, "
                " score, curator, rationale, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mapping.get("mapping_id"), mapping["canonical_id"],
                 mapping["relation"], mapping["curie"], mapping.get("iri"),
                 mapping.get("label"), mapping.get("source"),
                 mapping.get("score", 0.0), mapping.get("curator"),
                 mapping.get("rationale"), mapping.get("created_at", _now())),
            )

        conn.execute(
            "UPDATE merge_decision SET reverted = 1, reverted_at = ?, reverted_by = ? "
            "WHERE decision_id = ?",
            (_now(), curator, decision_id),
        )
        conn.commit()

    for canonical_id in restored_ids:
        workflow_state.invalidate_for_ke(
            canonical_id, reason=f"merge {decision_id} undone"
        )

    return {"decision_id": decision_id, "restored_ids": restored_ids}


def split_alias(
    canonical_id: int,
    raw_label: str,
    *,
    level: Optional[str] = None,
    curator: Optional[str] = None,
    rationale: Optional[str] = None,
) -> int:
    """
    Pull one alias out of a canonical group into a Key Event of its own.

    The finer-grained counterpart to `undo`: used when a merge was mostly
    right and one label does not belong. Returns the new canonical id.
    """
    label = (raw_label or "").strip()
    if not label:
        raise ValueError("No alias given.")

    with connect() as conn:
        parent = conn.execute(
            "SELECT * FROM ke_canonical WHERE canonical_id = ?", (int(canonical_id),)
        ).fetchone()
        if parent is None:
            raise ValueError(f"No canonical Key Event {canonical_id}")

        before = _snapshot(conn, [int(canonical_id)])

        cursor = conn.execute(
            "INSERT INTO ke_canonical "
            "(canonical_name, level, merge_method, curation_status, "
            " n_source_rows, updated_at) VALUES (?, ?, 'curator', 'unreviewed', 0, ?)",
            (label, level or parent["level"], _now()),
        )
        new_id = int(cursor.lastrowid)

        conn.execute(
            "UPDATE ke_alias SET canonical_id = ? WHERE canonical_id = ? AND raw_label = ?",
            (new_id, int(canonical_id), label),
        )
        conn.execute(
            "UPDATE table1_extractions SET upstream_ke_canonical_id = ? "
            "WHERE upstream_ke_canonical_id = ? AND upstream_ke_name = ?",
            (new_id, int(canonical_id), label),
        )
        conn.execute(
            "UPDATE table1_extractions SET downstream_ke_canonical_id = ? "
            "WHERE downstream_ke_canonical_id = ? AND downstream_ke_name = ?",
            (new_id, int(canonical_id), label),
        )
        moved = int(
            conn.execute(
                "SELECT COUNT(*) FROM table1_extractions WHERE "
                "upstream_ke_canonical_id = ? OR downstream_ke_canonical_id = ?",
                (new_id, new_id),
            ).fetchone()[0]
        )
        conn.execute(
            "UPDATE ke_canonical SET n_source_rows = ? WHERE canonical_id = ?",
            (moved, new_id),
        )
        conn.execute(
            "UPDATE ke_canonical SET n_source_rows = MAX(0, n_source_rows - ?) "
            "WHERE canonical_id = ?",
            (moved, int(canonical_id)),
        )
        after = _snapshot(conn, [int(canonical_id), new_id])
        conn.commit()

    record_decision(
        action="keep_separate",
        member_ids=[canonical_id, new_id],
        survivor_id=new_id,
        curator=curator,
        rationale=rationale or f"Split “{label}” out of canonical {canonical_id}",
        before_state=before,
        after_state=after,
    )
    workflow_state.invalidate_for_ke(canonical_id, reason=f"“{label}” split out")
    return new_id


# ---------------------------------------------------------------------------
# Decisions already taken
# ---------------------------------------------------------------------------

def decided_pairs() -> set[frozenset[int]]:
    """
    Pairs a curator has already ruled on.

    Used to stop the workspace re-suggesting a pair that was deliberately kept
    separate. A tool that keeps asking the same question trains people to click
    through it.
    """
    out: set[frozenset[int]] = set()
    with connect() as conn:
        for row in conn.execute(
            "SELECT member_ids FROM merge_decision "
            "WHERE reverted = 0 AND action IN ('keep_separate', 'merge_equivalent', "
            "'collapse_broader', 'record_relation', 'map_broader')"
        ):
            try:
                ids = [int(i) for i in json.loads(row[0] or "[]")]
            except (ValueError, TypeError):
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    out.add(frozenset({ids[i], ids[j]}))
    return out


def decision_log(limit: int = 200) -> pd.DataFrame:
    """The full curation trail, newest first."""
    with connect() as conn:
        frame = pd.read_sql_query(
            "SELECT decision_id, created_at, action, relationship, member_ids, "
            "       survivor_id, curator, curator_rationale, explanation, "
            "       method, reverted "
            "FROM merge_decision ORDER BY decision_id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    if not frame.empty:
        frame["action_label"] = frame["action"].map(
            lambda a: ACTION_LABELS.get(a, a)
        )
    return frame


def ontology_mappings(canonical_id: Optional[int] = None) -> pd.DataFrame:
    """Broader-concept mappings, optionally for one Key Event."""
    sql = (
        "SELECT m.mapping_id, m.canonical_id, k.canonical_name, m.relation, "
        "       m.curie, m.label, m.source, m.score, m.curator, m.rationale, "
        "       m.created_at "
        "FROM ontology_mapping m "
        "LEFT JOIN ke_canonical k ON k.canonical_id = m.canonical_id"
    )
    params: tuple = ()
    if canonical_id is not None:
        sql += " WHERE m.canonical_id = ?"
        params = (int(canonical_id),)
    sql += " ORDER BY k.canonical_name, m.relation"
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def remove_mapping(mapping_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM ontology_mapping WHERE mapping_id = ?", (int(mapping_id),))
        conn.commit()


def ke_relations() -> pd.DataFrame:
    """Recorded biological relationships between distinct Key Events."""
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT r.relation_id, r.relation, r.rationale, r.curator, r.created_at, "
            "       s.canonical_name AS source_name, t.canonical_name AS target_name "
            "FROM ke_relation r "
            "LEFT JOIN ke_canonical s ON s.canonical_id = r.source_id "
            "LEFT JOIN ke_canonical t ON t.canonical_id = r.target_id "
            "ORDER BY r.relation_id DESC",
            conn,
        )


__all__ = [
    "ACTIONS",
    "ACTION_LABELS",
    "RELATION_TYPES",
    "MergeRefused",
    "MergePreview",
    "preview_merge",
    "merge_as_equivalent",
    "collapse_into_broader",
    "map_to_broader",
    "record_relation",
    "keep_separate",
    "reject_not_ke",
    "mark_unresolved",
    "record_decision",
    "canonical_groups",
    "group_detail",
    "undo",
    "split_alias",
    "decided_pairs",
    "decision_log",
    "ontology_mappings",
    "remove_mapping",
    "ke_relations",
]
