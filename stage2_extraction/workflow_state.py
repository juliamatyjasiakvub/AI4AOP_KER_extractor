from __future__ import annotations

"""
The workflow gate: nothing is synthesised before it is approved.

States
------
    raw                     as extracted, untouched
    normalization_proposed  the tool has suggested a canonical form
    curated                 a human has decided what the record is
    approved                a human has signed off that it is fit to build on
    synthesized             evidence has been synthesised from it

The order is not decorative. Synthesis writes prose that reads as settled —
"the weight of evidence is moderate" — over whatever Key Events happen to be
in the table at the time. Run it before curation and it launders a duplicate,
a contradiction or a mis-merge into an authoritative-sounding paragraph, and
the reader has no way to tell. `gate` is the function that refuses.

Staleness
---------
Approval is approval *of something specific*. `content_hash` fingerprints what
was approved, so a Key Event that is renamed, re-levelled, re-merged or
re-annotated after sign-off no longer matches its own approval. Everything
built on it is then marked stale, the graph snapshot is invalidated, and both
have to be regenerated and re-approved. The previous version is archived
rather than deleted: it records what was believed and on what basis, which is
worth keeping even once it is wrong.
"""

import datetime
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from stage2_extraction.table1_store import connect


class State(str, Enum):
    RAW = "raw"
    NORMALIZATION_PROPOSED = "normalization_proposed"
    CURATED = "curated"
    APPROVED = "approved"
    SYNTHESIZED = "synthesized"

    @property
    def label(self) -> str:
        return {
            "raw": "Raw",
            "normalization_proposed": "Normalization proposed",
            "curated": "Curated",
            "approved": "Approved",
            "synthesized": "Synthesized",
        }[self.value]

    @property
    def rank(self) -> int:
        return _ORDER.index(self)


_ORDER: tuple[State, ...] = (
    State.RAW,
    State.NORMALIZATION_PROPOSED,
    State.CURATED,
    State.APPROVED,
    State.SYNTHESIZED,
)

#: Transitions that are allowed. Forward moves go one step at a time so no
#: record can jump from raw to approved without someone having curated it.
#: Backward moves to any earlier state are always allowed — retracting an
#: approval must never be harder than granting one.
_FORWARD: dict[State, set[State]] = {
    State.RAW: {State.NORMALIZATION_PROPOSED, State.CURATED},
    State.NORMALIZATION_PROPOSED: {State.CURATED},
    State.CURATED: {State.APPROVED},
    State.APPROVED: {State.SYNTHESIZED},
    State.SYNTHESIZED: set(),
}

TARGET_TYPES = ("ke", "ker")


class TransitionError(RuntimeError):
    """A state change that the workflow does not permit."""


class NotApproved(RuntimeError):
    """Raised when synthesis is attempted over unapproved records."""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

#: The fields whose change invalidates an approval of a Key Event.
#:
#: Chosen as "everything a downstream reader would have relied on". The name
#: and level appear on the map; the ontology term decides what the KE is
#: claimed to be; the alias list decides which raw records were folded in, and
#: so which evidence the synthesis will find. `updated_at` is deliberately
#: absent — a touch that changes nothing should not invalidate anything.
_KE_HASH_FIELDS = (
    "canonical_name",
    "level",
    "ontology_curie",
    "aopwiki_ke_id",
)


def content_hash(payload: Any) -> str:
    """Stable fingerprint of whatever is being approved."""
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def ke_hash(conn: sqlite3.Connection, canonical_id: int) -> Optional[str]:
    """Fingerprint one canonical Key Event, aliases included."""
    row = conn.execute(
        f"SELECT {', '.join(_KE_HASH_FIELDS)} FROM ke_canonical WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchone()
    if row is None:
        return None

    aliases = sorted(
        str(r[0])
        for r in conn.execute(
            "SELECT raw_label FROM ke_alias WHERE canonical_id = ?", (canonical_id,)
        )
    )
    parents = sorted(
        f"{r[0]}:{r[1]}"
        for r in conn.execute(
            "SELECT relation, curie FROM ontology_mapping WHERE canonical_id = ?",
            (canonical_id,),
        )
    )
    return content_hash(
        {
            "fields": [row[f] for f in _KE_HASH_FIELDS],
            "aliases": aliases,
            "parents": parents,
        }
    )


def ker_hash(conn: sqlite3.Connection, ker_key: str, upstream_id: Optional[int],
             downstream_id: Optional[int]) -> str:
    """
    Fingerprint a KER by its own key and the state of both endpoints.

    A KER is a claim about two Key Events, so re-approving it has to be
    triggered by a change to either of them. Folding the endpoint hashes in
    means a rename on one side propagates without anything needing to walk the
    graph looking for dependants.
    """
    ends = [
        ke_hash(conn, upstream_id) if upstream_id is not None else None,
        ke_hash(conn, downstream_id) if downstream_id is not None else None,
    ]
    return content_hash({"ker": ker_key, "ends": ends})


# ---------------------------------------------------------------------------
# Reading and writing state
# ---------------------------------------------------------------------------

@dataclass
class Status:
    """Where one record sits, and whether its approval still holds."""

    target_type: str
    target_key: str
    state: State
    content_hash: Optional[str] = None
    current_hash: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    note: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def drifted(self) -> bool:
        """Approved, but the thing approved has since changed."""
        if self.state.rank < State.APPROVED.rank:
            return False
        if not self.content_hash or not self.current_hash:
            return False
        return self.content_hash != self.current_hash

    @property
    def effective_state(self) -> State:
        """The state after accounting for drift."""
        return State.CURATED if self.drifted else self.state

    @property
    def is_approved(self) -> bool:
        return self.effective_state.rank >= State.APPROVED.rank


def get_state(target_type: str, target_key: str) -> State:
    """The stored state, defaulting to `raw` for anything never seen."""
    with connect() as conn:
        row = conn.execute(
            "SELECT state FROM workflow_state WHERE target_type = ? AND target_key = ?",
            (target_type, str(target_key)),
        ).fetchone()
    return State(row[0]) if row else State.RAW


def get_status(target_type: str, target_key: str) -> Status:
    """Full status including a freshly computed fingerprint."""
    with connect() as conn:
        return _status(conn, target_type, str(target_key))


def _status(conn: sqlite3.Connection, target_type: str, target_key: str) -> Status:
    row = conn.execute(
        "SELECT state, content_hash, approved_by, approved_at, note, updated_at "
        "FROM workflow_state WHERE target_type = ? AND target_key = ?",
        (target_type, target_key),
    ).fetchone()

    current = None
    if target_type == "ke":
        try:
            current = ke_hash(conn, int(target_key))
        except (TypeError, ValueError):
            current = None

    if row is None:
        return Status(target_type, target_key, State.RAW, current_hash=current)

    return Status(
        target_type=target_type,
        target_key=target_key,
        state=State(row[0]),
        content_hash=row[1],
        current_hash=current,
        approved_by=row[2],
        approved_at=row[3],
        note=row[4],
        updated_at=row[5],
    )


def set_state(
    target_type: str,
    target_key: str,
    state: State | str,
    *,
    curator: Optional[str] = None,
    note: Optional[str] = None,
    force: bool = False,
) -> Status:
    """
    Move one record to `state`, logging the transition.

    Raises `TransitionError` on a forward jump that skips a step. `force`
    exists for migrations and tests, not for the UI: a button that skips
    curation is the same as having no gate.
    """
    state = State(state)
    key = str(target_key)
    if target_type not in TARGET_TYPES:
        raise ValueError(f"Unknown target type {target_type!r}")

    with connect() as conn:
        before = _status(conn, target_type, key)

        if not force and state is not before.state:
            forward = state.rank > before.state.rank
            if forward and state not in _FORWARD[before.state]:
                raise TransitionError(
                    f"Cannot go from {before.state.label} straight to "
                    f"{state.label}. The intermediate step exists so that "
                    f"someone has to look at the record."
                )

        stamp = _now()
        fingerprint = before.current_hash if state.rank >= State.APPROVED.rank else None
        approved_by = curator if state.rank >= State.APPROVED.rank else None
        approved_at = stamp if state.rank >= State.APPROVED.rank else None

        # Re-approving something already approved keeps the original approver
        # and date unless a new curator is named.
        if state is State.SYNTHESIZED and before.state is State.APPROVED:
            fingerprint = before.content_hash or before.current_hash
            approved_by = before.approved_by or curator
            approved_at = before.approved_at or stamp

        conn.execute(
            "INSERT INTO workflow_state "
            "(target_type, target_key, state, content_hash, approved_by, "
            " approved_at, note, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(target_type, target_key) DO UPDATE SET "
            "  state = excluded.state, "
            "  content_hash = excluded.content_hash, "
            "  approved_by = excluded.approved_by, "
            "  approved_at = excluded.approved_at, "
            "  note = excluded.note, "
            "  updated_at = excluded.updated_at",
            (target_type, key, state.value, fingerprint, approved_by,
             approved_at, note, stamp),
        )
        conn.execute(
            "INSERT INTO approval_log "
            "(target_type, target_key, from_state, to_state, curator, note, "
            " content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (target_type, key, before.state.value, state.value, curator, note,
             fingerprint, stamp),
        )

        # Dropping below approval invalidates whatever was built on it.
        if before.state.rank >= State.APPROVED.rank > state.rank:
            _invalidate_dependants(conn, target_type, key,
                                   reason=f"{before.state.label} → {state.label}")

        conn.commit()
        return _status(conn, target_type, key)


def approve(target_type: str, target_key: str, curator: Optional[str] = None,
            note: Optional[str] = None) -> Status:
    return set_state(target_type, target_key, State.APPROVED,
                     curator=curator, note=note)


def retract(target_type: str, target_key: str, curator: Optional[str] = None,
            note: Optional[str] = None) -> Status:
    """Pull an approval back, invalidating everything downstream."""
    return set_state(target_type, target_key, State.CURATED,
                     curator=curator, note=note)


def bulk_set(
    targets: Iterable[tuple[str, str]],
    state: State | str,
    *,
    curator: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, int]:
    """Apply one state to many records, reporting what happened."""
    done = skipped = 0
    for target_type, key in targets:
        try:
            set_state(target_type, key, state, curator=curator, note=note)
            done += 1
        except TransitionError:
            skipped += 1
    return {"changed": done, "skipped": skipped}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Whether synthesis may run, and if not, precisely what is missing."""

    allowed: bool
    blocking: list[Status] = field(default_factory=list)
    reason: str = ""

    @property
    def summary(self) -> str:
        if self.allowed:
            return "All required Key Events and relationships are approved."
        names = ", ".join(f"{s.target_type}:{s.target_key}" for s in self.blocking[:6])
        more = "" if len(self.blocking) <= 6 else f" (+{len(self.blocking) - 6} more)"
        return f"{self.reason} Waiting on {names}{more}."


def gate(
    ke_ids: Sequence[int] = (),
    ker_keys: Sequence[str] = (),
) -> GateResult:
    """
    May evidence be synthesised over these records?

    Answers no unless every named Key Event and KER is approved *and* its
    approval still matches its content. A record whose fingerprint has drifted
    counts as unapproved even though the column still says "approved" — that
    is the whole reason the fingerprint exists.
    """
    blocking: list[Status] = []
    with connect() as conn:
        for ke_id in ke_ids:
            status = _status(conn, "ke", str(ke_id))
            if not status.is_approved:
                blocking.append(status)
        for key in ker_keys:
            status = _status(conn, "ker", str(key))
            if not status.is_approved:
                blocking.append(status)

    if not blocking:
        return GateResult(allowed=True)

    drifted = [s for s in blocking if s.drifted]
    if drifted and len(drifted) == len(blocking):
        reason = ("Approved records have changed since sign-off and need "
                  "re-approval.")
    else:
        reason = "Some records are not yet approved."

    return GateResult(allowed=False, blocking=blocking, reason=reason)


def require_approved(ke_ids: Sequence[int] = (), ker_keys: Sequence[str] = ()) -> None:
    """Gate as an assertion, for call sites that should never proceed."""
    result = gate(ke_ids, ker_keys)
    if not result.allowed:
        raise NotApproved(result.summary)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def _invalidate_dependants(
    conn: sqlite3.Connection, target_type: str, target_key: str, *, reason: str
) -> int:
    """
    Mark everything built on a record as stale.

    Archives before flagging, so the superseded synthesis survives. Nothing is
    deleted here — the curator decides whether to regenerate, and until they do
    the UI shows the old text with a stale banner rather than an empty panel.
    """
    affected = 0

    if target_type == "ke":
        keys = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT ker_key FROM ker_synthesis "
                "WHERE ker_key LIKE ? OR ker_key LIKE ?",
                (f"{target_key}->%", f"%->{target_key}"),
            )
        ]
        # KER keys are canonical name pairs rather than ids in older rows, so
        # also match on the canonical name.
        row = conn.execute(
            "SELECT canonical_name FROM ke_canonical WHERE canonical_id = ?",
            (target_key,),
        ).fetchone()
        if row:
            name = str(row[0])
            keys += [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT ker_key FROM ker_synthesis "
                    "WHERE ker_key LIKE ? OR ker_key LIKE ?",
                    (f"{name}%", f"%{name}"),
                )
            ]
    else:
        keys = [target_key]

    for key in dict.fromkeys(keys):
        affected += _mark_synthesis_stale(conn, key, reason=reason)

    conn.execute(
        "UPDATE aop_snapshot SET stale = 1, stale_reason = ? WHERE stale = 0",
        (f"{target_type} {target_key}: {reason}",),
    )
    return affected


def _mark_synthesis_stale(conn: sqlite3.Connection, ker_key: str, *, reason: str) -> int:
    row = conn.execute(
        "SELECT * FROM ker_synthesis WHERE ker_key = ?", (ker_key,)
    ).fetchone()
    if row is None:
        return 0
    if row["stale"]:
        return 0

    conn.execute(
        "INSERT INTO synthesis_history (ker_key, payload, reason, archived_at) "
        "VALUES (?, ?, ?, ?)",
        (ker_key, json.dumps(dict(row), default=str), reason, _now()),
    )
    conn.execute(
        "UPDATE ker_synthesis SET stale = 1, stale_reason = ? WHERE ker_key = ?",
        (reason, ker_key),
    )
    # A synthesised KER whose inputs moved is back to being merely approved.
    conn.execute(
        "UPDATE workflow_state SET state = 'approved', updated_at = ? "
        "WHERE target_type = 'ker' AND target_key = ? AND state = 'synthesized'",
        (_now(), ker_key),
    )
    return 1


def invalidate_for_ke(canonical_id: int, reason: str = "Key Event changed") -> int:
    """
    Public entry point: a curated Key Event has been edited.

    Called by the curation workspace after any write that changes what a Key
    Event is. Returns the number of syntheses marked stale.
    """
    with connect() as conn:
        affected = _invalidate_dependants(conn, "ke", str(canonical_id), reason=reason)
        status = _status(conn, "ke", str(canonical_id))
        if status.state.rank >= State.APPROVED.rank and status.drifted:
            conn.execute(
                "UPDATE workflow_state SET state = 'curated', updated_at = ? "
                "WHERE target_type = 'ke' AND target_key = ?",
                (_now(), str(canonical_id)),
            )
            conn.execute(
                "INSERT INTO approval_log (target_type, target_key, from_state, "
                "to_state, curator, note, content_hash, created_at) "
                "VALUES ('ke', ?, 'approved', 'curated', NULL, ?, NULL, ?)",
                (str(canonical_id), f"Auto-retracted: {reason}", _now()),
            )
        conn.commit()
    return affected


def invalidate_for_ker(ker_key: str, reason: str = "relationship changed") -> int:
    """
    Public entry point: the evidence under a relationship has changed.

    The sibling of `invalidate_for_ke`, and missing until now because until now
    nothing could change a Table 1 row after extraction wrote it. Adding,
    correcting or deleting a claim changes what the relationship's evidence
    page was built from, and an approved KER whose underlying claims have moved
    is exactly the "approved, but not this" state the workflow exists to catch.

    Returns the number of syntheses marked stale.
    """
    with connect() as conn:
        affected = _invalidate_dependants(conn, "ker", str(ker_key), reason=reason)
        status = _status(conn, "ker", str(ker_key))
        if status.state.rank >= State.APPROVED.rank and status.drifted:
            conn.execute(
                "UPDATE workflow_state SET state = 'curated', updated_at = ? "
                "WHERE target_type = 'ker' AND target_key = ?",
                (_now(), str(ker_key)),
            )
            conn.execute(
                "INSERT INTO approval_log (target_type, target_key, from_state, "
                "to_state, curator, note, content_hash, created_at) "
                "VALUES ('ker', ?, 'approved', 'curated', NULL, ?, NULL, ?)",
                (str(ker_key), f"Auto-retracted: {reason}", _now()),
            )
        conn.commit()
    return affected


def stale_syntheses() -> pd.DataFrame:
    """Every synthesis currently flagged stale."""
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT ker_key, ker_name, stale_reason, generated_at "
            "FROM ker_synthesis WHERE stale = 1 ORDER BY ker_name",
            conn,
        )


def synthesis_history(ker_key: str) -> pd.DataFrame:
    """Superseded versions of one KER's synthesis, newest first."""
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT history_id, reason, archived_at, payload FROM synthesis_history "
            "WHERE ker_key = ? ORDER BY history_id DESC",
            conn,
            params=(ker_key,),
        )


# ---------------------------------------------------------------------------
# Overviews
# ---------------------------------------------------------------------------

def state_frame(target_type: str = "ke") -> pd.DataFrame:
    """
    Every record of one type with its state and drift flag.

    Drift is recomputed rather than read, because the whole point is that the
    stored state can be wrong.
    """
    with connect() as conn:
        if target_type == "ke":
            base = pd.read_sql_query(
                "SELECT canonical_id AS target_key, canonical_name AS name, level "
                "FROM ke_canonical ORDER BY canonical_name",
                conn,
            )
        else:
            base = pd.read_sql_query(
                "SELECT DISTINCT target_key, target_key AS name, '' AS level "
                "FROM workflow_state WHERE target_type = 'ker'",
                conn,
            )
        if base.empty:
            return pd.DataFrame(
                columns=["target_key", "name", "level", "state", "state_label",
                         "drifted", "approved_by", "approved_at"]
            )

        rows = []
        for _, r in base.iterrows():
            status = _status(conn, target_type, str(r["target_key"]))
            rows.append(
                {
                    "target_key": str(r["target_key"]),
                    "name": r["name"],
                    "level": r["level"],
                    "state": status.effective_state.value,
                    "state_label": status.effective_state.label,
                    "drifted": status.drifted,
                    "approved_by": status.approved_by,
                    "approved_at": status.approved_at,
                }
            )
    return pd.DataFrame(rows)


def counts(target_type: str = "ke") -> dict[str, int]:
    """How many records sit in each state."""
    frame = state_frame(target_type)
    out = {s.value: 0 for s in State}
    if frame.empty:
        return out
    for value, n in frame["state"].value_counts().items():
        out[str(value)] = int(n)
    return out


def approval_log(target_type: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
    """The audit trail, newest first."""
    sql = ("SELECT log_id, target_type, target_key, from_state, to_state, "
           "curator, note, created_at FROM approval_log")
    params: tuple = ()
    if target_type:
        sql += " WHERE target_type = ?"
        params = (target_type,)
    sql += " ORDER BY log_id DESC LIMIT ?"
    params = params + (limit,)
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


__all__ = [
    "State",
    "Status",
    "GateResult",
    "TransitionError",
    "NotApproved",
    "TARGET_TYPES",
    "get_state",
    "get_status",
    "set_state",
    "approve",
    "retract",
    "bulk_set",
    "gate",
    "require_approved",
    "invalidate_for_ke",
    "stale_syntheses",
    "synthesis_history",
    "state_frame",
    "counts",
    "approval_log",
    "content_hash",
    "ke_hash",
    "ker_hash",
]
