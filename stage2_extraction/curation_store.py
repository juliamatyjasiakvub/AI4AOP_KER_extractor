from __future__ import annotations

"""
Expert curation state and persistent layout.

Two closely related concerns live here:

**Curation.** An automatically assembled AOP is a draft, not a result. A curator
must be able to accept an edge, reject one, rename it, or merge two edges that
say the same thing — and those decisions must survive re-extraction. Curation
records are therefore keyed by stable content keys (canonical KE names for
nodes, canonical KE-pair keys for edges) rather than by database ids, which are
renumbered every time normalization re-runs.

**Layout.** Force-directed layouts are not reproducible: the same graph re-flows
differently on every load, so a map a curator spent an hour arranging is
destroyed by a page refresh. Node coordinates, lanes, groups and pin flags are
saved here and re-applied on load; only genuinely new nodes get an automatic
position.
"""

import datetime
import json
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from schemas import CurationRecord, LayoutPosition
from stage2_extraction.table1_store import connect

DEFAULT_LAYOUT_NAME = "default"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

def set_curation(
    target_type: str,
    target_key: str,
    *,
    status: Optional[str] = None,
    display_name: Optional[str] = None,
    note: Optional[str] = None,
    merged_into: Optional[str] = None,
    curator: Optional[str] = None,
) -> None:
    """
    Upsert a curation record.

    Only the fields you pass are changed; passing None leaves the stored value
    alone. Pass an empty string to clear a text field.
    """
    with connect() as conn:
        existing = conn.execute(
            "SELECT * FROM ker_curation WHERE target_type = ? AND target_key = ?",
            (target_type, target_key),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO ker_curation
                    (target_type, target_key, status, display_name, note,
                     merged_into, curator, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_type,
                    target_key,
                    status or "unreviewed",
                    _blank_to_none(display_name),
                    _blank_to_none(note),
                    _blank_to_none(merged_into),
                    curator,
                    _now(),
                ),
            )
        else:
            conn.execute(
                """
                UPDATE ker_curation
                SET status = ?, display_name = ?, note = ?, merged_into = ?,
                    curator = ?, updated_at = ?
                WHERE target_type = ? AND target_key = ?
                """,
                (
                    status if status is not None else existing["status"],
                    _resolve(display_name, existing["display_name"]),
                    _resolve(note, existing["note"]),
                    _resolve(merged_into, existing["merged_into"]),
                    curator if curator is not None else existing["curator"],
                    _now(),
                    target_type,
                    target_key,
                ),
            )
        conn.commit()


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve(new: Optional[str], old: Any) -> Optional[str]:
    """None means 'leave as is'; empty string means 'clear'."""
    if new is None:
        return old
    new = new.strip()
    return new or None


def get_curation(target_type: str, target_key: str) -> Optional[CurationRecord]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM ker_curation WHERE target_type = ? AND target_key = ?",
            (target_type, target_key),
        ).fetchone()
    if row is None:
        return None
    return CurationRecord(
        target_type=row["target_type"],
        target_key=row["target_key"],
        status=row["status"],
        display_name=row["display_name"],
        note=row["note"],
        merged_into=row["merged_into"],
        curator=row["curator"],
        updated_at=row["updated_at"],
    )


def load_curation(target_type: Optional[str] = None) -> pd.DataFrame:
    with connect() as conn:
        if target_type is None:
            return pd.read_sql_query(
                "SELECT * FROM ker_curation ORDER BY target_type, target_key", conn
            )
        return pd.read_sql_query(
            "SELECT * FROM ker_curation WHERE target_type = ? ORDER BY target_key",
            conn,
            params=(target_type,),
        )


def curation_map(target_type: str) -> dict[str, dict[str, Any]]:
    """target_key -> {status, display_name, note, merged_into}."""
    df = load_curation(target_type)
    if df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row["target_key"])] = {
            "status": row["status"],
            "display_name": row["display_name"],
            "note": row["note"],
            "merged_into": row["merged_into"],
        }
    return out


def accept(target_type: str, target_key: str, curator: Optional[str] = None) -> None:
    set_curation(target_type, target_key, status="accepted", curator=curator)


def reject(
    target_type: str,
    target_key: str,
    note: Optional[str] = None,
    curator: Optional[str] = None,
) -> None:
    set_curation(target_type, target_key, status="rejected", note=note, curator=curator)


def reset_curation(target_type: str, target_key: str) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM ker_curation WHERE target_type = ? AND target_key = ?",
            (target_type, target_key),
        )
        conn.commit()


def clear_all_curation() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM ker_curation")
        conn.commit()


def curation_summary() -> dict[str, dict[str, int]]:
    """Counts of accepted / rejected / unreviewed, split by target type."""
    df = load_curation()
    summary = {
        "ke": {"accepted": 0, "rejected": 0, "unreviewed": 0, "renamed": 0, "merged": 0},
        "ker": {"accepted": 0, "rejected": 0, "unreviewed": 0, "renamed": 0, "merged": 0},
    }
    if df.empty:
        return summary
    for _, row in df.iterrows():
        bucket = summary.setdefault(
            str(row["target_type"]),
            {"accepted": 0, "rejected": 0, "unreviewed": 0, "renamed": 0, "merged": 0},
        )
        status = str(row["status"] or "unreviewed")
        bucket[status] = bucket.get(status, 0) + 1
        if row["display_name"]:
            bucket["renamed"] += 1
        if row["merged_into"]:
            bucket["merged"] += 1
    return summary


def apply_ker_curation(
    table2_df: pd.DataFrame,
    *,
    hide_rejected: bool = True,
    key_column: str = "ker_key",
) -> pd.DataFrame:
    """
    Overlay curation decisions onto a Table 2 frame.

    Adds `curation_status`, `curator_note` and applies curator renames to
    `ker_name`. Edges merged into another edge are dropped, as are rejected
    edges when `hide_rejected` is True.
    """
    if table2_df is None or table2_df.empty:
        return table2_df

    df = table2_df.copy()
    decisions = curation_map("ker")

    statuses, notes, names = [], [], []
    drop_rows: list[int] = []

    for position, (_, row) in enumerate(df.iterrows()):
        key = str(row.get(key_column, ""))
        decision = decisions.get(key)
        if decision is None:
            statuses.append("unreviewed")
            notes.append(None)
            names.append(row.get("ker_name"))
            continue

        statuses.append(decision["status"])
        notes.append(decision["note"])
        names.append(decision["display_name"] or row.get("ker_name"))

        if decision["merged_into"]:
            drop_rows.append(position)
        elif hide_rejected and decision["status"] == "rejected":
            drop_rows.append(position)

    df["curation_status"] = statuses
    df["curator_note"] = notes
    df["ker_name"] = names

    if drop_rows:
        df = df.drop(df.index[drop_rows])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------

def save_positions(
    positions: Iterable[LayoutPosition] | Iterable[dict],
    layout_name: str = DEFAULT_LAYOUT_NAME,
    *,
    mark_pinned: bool = False,
) -> int:
    """
    Persist node coordinates.

    Accepts either LayoutPosition objects or plain dicts with at least
    `node_key`, `x` and `y`. Returns the number of nodes written.

    `mark_pinned=True` records every saved node as pinned, which is what you
    want when the curator explicitly clicks "save this arrangement": those
    nodes should never be moved by an automatic re-layout again.
    """
    rows: list[tuple] = []
    now = _now()

    for item in positions:
        if isinstance(item, LayoutPosition):
            node_key, x, y = item.node_key, item.x, item.y
            lane, group, pinned = item.lane, item.group, item.pinned
            name = item.layout_name or layout_name
        else:
            node_key = str(item.get("node_key") or item.get("id") or "").strip()
            if not node_key:
                continue
            x, y = item.get("x"), item.get("y")
            if x is None or y is None:
                continue
            lane = item.get("lane") or item.get("level")
            group = item.get("group")
            pinned = bool(item.get("pinned", False))
            name = item.get("layout_name") or layout_name

        rows.append(
            (
                name,
                node_key,
                float(x),
                float(y),
                lane,
                group,
                int(bool(pinned or mark_pinned)),
                now,
            )
        )

    if not rows:
        return 0

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO layout_state (layout_name, node_key, x, y, lane, "group", pinned, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(layout_name, node_key) DO UPDATE SET
                x = excluded.x,
                y = excluded.y,
                lane = COALESCE(excluded.lane, layout_state.lane),
                "group" = COALESCE(excluded."group", layout_state."group"),
                pinned = excluded.pinned,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def save_positions_json(
    payload: str,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    *,
    mark_pinned: bool = True,
) -> int:
    """
    Save coordinates from the JSON blob the graph component emits.

    Accepts either a bare list of `{id, x, y}` objects or an object with a
    `positions` key, which is what the vis.js export produces.
    """
    if not payload or not payload.strip():
        return 0
    try:
        data = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Layout payload is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("positions") or data.get("nodes") or []
    if not isinstance(data, list):
        raise ValueError("Layout payload must be a list of node positions.")

    return save_positions(data, layout_name, mark_pinned=mark_pinned)


def load_positions(layout_name: str = DEFAULT_LAYOUT_NAME) -> dict[str, dict[str, Any]]:
    """node_key -> {x, y, lane, group, pinned}."""
    with connect() as conn:
        rows = conn.execute(
            'SELECT node_key, x, y, lane, "group", pinned FROM layout_state WHERE layout_name = ?',
            (layout_name,),
        ).fetchall()
    return {
        row["node_key"]: {
            "x": row["x"],
            "y": row["y"],
            "lane": row["lane"],
            "group": row["group"],
            "pinned": bool(row["pinned"]),
        }
        for row in rows
    }


def load_positions_frame(layout_name: str = DEFAULT_LAYOUT_NAME) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM layout_state WHERE layout_name = ? ORDER BY node_key",
            conn,
            params=(layout_name,),
        )


def set_pinned(node_key: str, pinned: bool, layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE layout_state SET pinned = ?, updated_at = ? "
            "WHERE layout_name = ? AND node_key = ?",
            (int(pinned), _now(), layout_name, node_key),
        )
        conn.commit()


def set_group(node_key: str, group: Optional[str], layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
    with connect() as conn:
        conn.execute(
            'UPDATE layout_state SET "group" = ?, updated_at = ? '
            "WHERE layout_name = ? AND node_key = ?",
            (_blank_to_none(group), _now(), layout_name, node_key),
        )
        conn.commit()


def rename_node_key(old_key: str, new_key: str, layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
    """
    Follow a node's saved position when its canonical name changes.

    Without this, renaming a KE would silently orphan its coordinates and the
    node would jump to a fresh automatic position.
    """
    with connect() as conn:
        conn.execute(
            "DELETE FROM layout_state WHERE layout_name = ? AND node_key = ?",
            (layout_name, new_key),
        )
        conn.execute(
            "UPDATE layout_state SET node_key = ?, updated_at = ? "
            "WHERE layout_name = ? AND node_key = ?",
            (new_key, _now(), layout_name, old_key),
        )
        conn.commit()


def clear_layout(layout_name: str = DEFAULT_LAYOUT_NAME) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM layout_state WHERE layout_name = ?", (layout_name,))
        conn.commit()


def list_layouts() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT layout_name,
                   COUNT(*)          AS n_nodes,
                   SUM(pinned)       AS n_pinned,
                   MAX(updated_at)   AS updated_at
            FROM layout_state
            GROUP BY layout_name
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def copy_layout(source: str, destination: str) -> int:
    """Duplicate a saved layout under a new name (e.g. to branch a curation)."""
    with connect() as conn:
        conn.execute("DELETE FROM layout_state WHERE layout_name = ?", (destination,))
        cur = conn.execute(
            """
            INSERT INTO layout_state (layout_name, node_key, x, y, lane, "group", pinned, updated_at)
            SELECT ?, node_key, x, y, lane, "group", pinned, ?
            FROM layout_state WHERE layout_name = ?
            """,
            (destination, _now(), source),
        )
        conn.commit()
        return cur.rowcount


def export_layout(layout_name: str = DEFAULT_LAYOUT_NAME) -> str:
    """Serialise a layout to JSON so a curated map can be shared or archived."""
    positions = load_positions(layout_name)
    return json.dumps(
        {
            "layout_name": layout_name,
            "exported_at": _now(),
            "positions": [
                {"node_key": key, **value} for key, value in sorted(positions.items())
            ],
        },
        indent=2,
    )


def import_layout(payload: str, layout_name: Optional[str] = None) -> int:
    """Restore a layout previously produced by `export_layout`."""
    data = json.loads(payload)
    name = layout_name or data.get("layout_name") or DEFAULT_LAYOUT_NAME
    return save_positions(data.get("positions", []), name)


__all__ = [
    "DEFAULT_LAYOUT_NAME",
    "set_curation",
    "get_curation",
    "load_curation",
    "curation_map",
    "accept",
    "reject",
    "reset_curation",
    "clear_all_curation",
    "curation_summary",
    "apply_ker_curation",
    "save_positions",
    "save_positions_json",
    "load_positions",
    "load_positions_frame",
    "set_pinned",
    "set_group",
    "rename_node_key",
    "clear_layout",
    "list_layouts",
    "copy_layout",
    "export_layout",
    "import_layout",
]
