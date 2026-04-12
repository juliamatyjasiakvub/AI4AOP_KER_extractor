from __future__ import annotations

"""
Lightweight client for the AOP-Wiki REST API.

Looks up KE IDs, KER IDs, and AOP IDs by name / ID pair.
All functions return None (not raise) on failed lookups so the pipeline
can continue without matches — a missing ID just means the entity is novel.
"""

import time
from typing import Optional

import requests

BASE = "https://aopwiki.org"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "AOP-RAG/2.0"})

_RATE_LIMIT_DELAY = 0.5  # seconds between requests — be polite to the API


def _get(path: str, params: Optional[dict] = None) -> Optional[list | dict]:
    """GET helper. Returns parsed JSON or None on any error."""
    try:
        r = SESSION.get(f"{BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        time.sleep(_RATE_LIMIT_DELAY)
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# KE lookup
# ---------------------------------------------------------------------------

def lookup_ke_id(ke_name: str) -> Optional[int]:
    """
    Search AOP-Wiki for a Key Event by name.

    Returns the numeric KE ID of the best match, or None if not found.
    Matching is case-insensitive exact match first, then substring.
    """
    if not ke_name or not ke_name.strip():
        return None

    data = _get("/key_events.json", params={"search": ke_name.strip()})
    if not data or not isinstance(data, list):
        return None

    name_lower = ke_name.strip().lower()

    # 1. Exact match
    for item in data:
        title = (item.get("title") or "").strip().lower()
        if title == name_lower:
            return item.get("id")

    # 2. Substring match — return first hit
    for item in data:
        title = (item.get("title") or "").strip().lower()
        if name_lower in title or title in name_lower:
            return item.get("id")

    return None


# ---------------------------------------------------------------------------
# KER lookup
# ---------------------------------------------------------------------------

def lookup_ker_id(upstream_ke_id: int, downstream_ke_id: int) -> Optional[int]:
    """
    Look up a KER by its upstream and downstream KE IDs.

    Returns the KER ID or None.
    """
    data = _get(
        "/relationships.json",
        params={"upstream_id": upstream_ke_id, "downstream_id": downstream_ke_id},
    )
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    return data[0].get("id")


# ---------------------------------------------------------------------------
# AOP lookup
# ---------------------------------------------------------------------------

def lookup_aop_ids_for_ker(ker_id: int) -> Optional[str]:
    """
    Return a semicolon-separated string of AOP IDs that contain this KER,
    or None if the KER is not part of any AOP.
    """
    data = _get(f"/relationships/{ker_id}.json")
    if not data or not isinstance(data, dict):
        return None

    aops = data.get("aops") or []
    if not aops:
        return None

    ids = [str(a.get("id")) for a in aops if a.get("id") is not None]
    return ";".join(ids) if ids else None


# ---------------------------------------------------------------------------
# Full enrichment — call once per KERExtraction
# ---------------------------------------------------------------------------

def enrich_ker(
    upstream_ke_name: str,
    downstream_ke_name: str,
) -> dict:
    """
    Given upstream and downstream KE names, look up all IDs from AOP-Wiki.

    Returns a dict with keys:
        upstream_ke_id, downstream_ke_id, ker_id, aop_id, aop_status
    All values may be None if the entity is not found (novel).
    """
    upstream_ke_id = lookup_ke_id(upstream_ke_name)
    downstream_ke_id = lookup_ke_id(downstream_ke_name)

    ker_id = None
    aop_id = None

    if upstream_ke_id is not None and downstream_ke_id is not None:
        ker_id = lookup_ker_id(upstream_ke_id, downstream_ke_id)
        if ker_id is not None:
            aop_id = lookup_aop_ids_for_ker(ker_id)

    aop_status = "existing" if aop_id else "novel"

    return {
        "upstream_ke_id": upstream_ke_id,
        "downstream_ke_id": downstream_ke_id,
        "ker_id": ker_id,
        "aop_id": aop_id,
        "aop_status": aop_status,
    }
