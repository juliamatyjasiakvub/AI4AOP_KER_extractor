from __future__ import annotations

"""
Backwards-compatible shim around the offline XML-dump client.

The previous version of this module called `https://aopwiki.org/*.json`
endpoints that do not actually exist on the site (every request returned 404),
so all KE/KER/AOP IDs ended up None and every KER was silently flagged "novel".

The real lookup now lives in `aopwiki_xml.py`, which loads a bundled XML dump.
This module is kept so existing imports (`from .aopwiki_client import enrich_ker`)
continue to work without changes elsewhere.
"""

from typing import Optional

from stage2_extraction.aopwiki_xml import enrich_ker as _enrich_ker
from stage2_extraction.aopwiki_xml import get_index


def enrich_ker(upstream_ke_name: str, downstream_ke_name: str) -> dict:
    """See `aopwiki_xml.enrich_ker` for the contract."""
    return _enrich_ker(upstream_ke_name, downstream_ke_name)


def lookup_ke_id(ke_name: str) -> Optional[int]:
    """Legacy helper retained for compatibility — uses the offline index."""
    idx = get_index()
    if idx is None or not ke_name:
        return None
    name_l = ke_name.strip().lower()
    if not name_l:
        return None
    hit = idx.ke_by_title.get(name_l)
    if hit:
        return hit.get("wiki_id")
    for title, entry in idx.ke_by_title.items():
        if name_l in title or title in name_l:
            return entry.get("wiki_id")
    return None


__all__ = ["enrich_ker", "lookup_ke_id"]
