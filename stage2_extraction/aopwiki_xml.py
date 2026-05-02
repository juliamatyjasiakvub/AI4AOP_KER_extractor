from __future__ import annotations

"""
Offline AOP-Wiki client backed by the official quarterly XML dump.

Why local instead of HTTP?
- AOP-Wiki does NOT expose a public JSON REST API. The previous `*.json`
  endpoints used by the old `aopwiki_client` always returned 404, so every
  KER was silently flagged "novel".
- The maintainers ship a gzipped XML dump containing every KE, KER, and AOP
  along with their numeric `aop-wiki-id` values. Loading it once at startup
  is fast (~10 MB compressed, ~50 MB uncompressed) and removes all per-KER
  network calls.

Files in `stage2_extraction/aopwiki_data/`:
    aop-wiki-xml-YYYY-MM-DD.gz   ← the bundled dump

Public API:
    get_index()                       -> AOPWikiIndex (cached, lazy)
    enrich_ker(upstream, downstream)  -> dict (same shape as the old client)
    get_local_version()               -> "YYYY-MM-DD" or None
    get_latest_remote_version()       -> "YYYY-MM-DD" or None (scrapes the site)
    download_dump(version)            -> path of newly downloaded gz
    update_to_latest()                -> (was_updated, new_version)
"""

import gzip
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "aopwiki_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_PAGE = "https://aopwiki.org/downloads"
DUMP_URL_TMPL  = "https://aopwiki.org/downloads/aop-wiki-xml-{date}.gz"

# AOP-XML namespace (every element is in this namespace in the dump)
NS  = "{http://www.aopkb.org/aop-xml}"
TAG_KE     = NS + "key-event"
TAG_KER    = NS + "key-event-relationship"
TAG_AOP    = NS + "aop"
TAG_TITLE  = NS + "title"
TAG_USID   = NS + "upstream-id"
TAG_DSID   = NS + "downstream-id"
TAG_KE_RELS = NS + "key-event-relationships"
TAG_REL    = NS + "relationship"

TAG_KE_REF      = NS + "key-event-reference"
TAG_KER_REF     = NS + "key-event-relationship-reference"
TAG_AOP_REF     = NS + "aop-reference"

VERSION_RE = re.compile(r"aop-wiki-xml-(\d{4}-\d{2}-\d{2})\.gz$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _list_local_dumps() -> list[Path]:
    return sorted(DATA_DIR.glob("aop-wiki-xml-*.gz"))


def get_local_version() -> Optional[str]:
    """Return the date string of the newest bundled dump, or None."""
    versions: list[str] = []
    for p in _list_local_dumps():
        m = VERSION_RE.search(p.name)
        if m:
            versions.append(m.group(1))
    return max(versions) if versions else None


def _local_dump_path(version: str) -> Path:
    return DATA_DIR / f"aop-wiki-xml-{version}.gz"


def get_latest_remote_version(timeout: int = 10) -> Optional[str]:
    """
    Scrape the AOP-Wiki downloads page and return the newest dump's date string
    (YYYY-MM-DD), or None on any error.
    """
    try:
        r = requests.get(DOWNLOADS_PAGE, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return None
    versions = re.findall(r"aop-wiki-xml-(\d{4}-\d{2}-\d{2})\.gz", r.text)
    return max(versions) if versions else None


def download_dump(version: str, timeout: int = 120) -> Path:
    """
    Download a specific dump version into DATA_DIR and return its path.
    Raises RuntimeError on failure.
    """
    url = DUMP_URL_TMPL.format(date=version)
    target = _local_dump_path(version)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as exc:
        if target.exists():
            target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return target


def update_to_latest() -> tuple[bool, Optional[str]]:
    """
    Compare local vs remote dump version. If remote is newer, download it and
    return (True, new_version). Otherwise return (False, current_version).
    """
    local  = get_local_version()
    remote = get_latest_remote_version()
    if remote is None:
        return False, local
    if local == remote:
        return False, local
    if local is not None and local > remote:
        return False, local
    download_dump(remote)
    # Drop the cached index so the next call reloads with the new dump.
    _INDEX_CACHE["index"] = None
    return True, remote


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------

@dataclass
class AOPWikiIndex:
    version: str
    # KE: title (lowercased) -> {"uuid": str, "wiki_id": int|None}
    ke_by_title: dict[str, dict] = field(default_factory=dict)
    # KE: uuid -> wiki_id
    ke_uuid_to_wiki_id: dict[str, int] = field(default_factory=dict)
    # KER: (upstream_uuid, downstream_uuid) -> ker_uuid
    ker_by_pair: dict[tuple[str, str], str] = field(default_factory=dict)
    # KER: uuid -> wiki_id
    ker_uuid_to_wiki_id: dict[str, int] = field(default_factory=dict)
    # KER uuid -> list of AOP wiki_ids it appears in
    ker_to_aop_wiki_ids: dict[str, list[int]] = field(default_factory=dict)


_INDEX_CACHE: dict[str, Optional[AOPWikiIndex]] = {"index": None}


def get_index(force_reload: bool = False) -> Optional[AOPWikiIndex]:
    """Lazily load and cache the index from the newest local dump."""
    if not force_reload and _INDEX_CACHE.get("index") is not None:
        return _INDEX_CACHE["index"]

    version = get_local_version()
    if version is None:
        _INDEX_CACHE["index"] = None
        return None

    gz_path = _local_dump_path(version)
    if not gz_path.exists():
        _INDEX_CACHE["index"] = None
        return None

    # gzip.open works as a file-like object, which iterparse accepts via name=...
    # but we need a real path or stream. Easiest is to open and pass the stream.
    with gzip.open(gz_path, "rb") as f:
        # ET.iterparse accepts a file object too
        # We re-implement _build_index_from_xml inline to use the file stream.
        idx = _build_index_from_stream(f, version)

    _INDEX_CACHE["index"] = idx
    return idx


def _build_index_from_stream(stream, version: str) -> AOPWikiIndex:
    """Same logic as _build_index_from_xml but consumes a binary stream."""
    idx = AOPWikiIndex(version=version)
    aop_uuid_to_wiki_id: dict[str, int] = {}
    aop_uuid_to_ker_uuids: dict[str, list[str]] = {}

    for _, elem in ET.iterparse(stream, events=("end",)):
        tag = elem.tag

        if tag == TAG_KE:
            uuid = elem.get("id") or ""
            title_el = elem.find(NS + "title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            if uuid and title:
                idx.ke_by_title.setdefault(title.lower(), {"uuid": uuid, "wiki_id": None})
            elem.clear()

        elif tag == TAG_KER:
            uuid = elem.get("id") or ""
            title_el = elem.find(NS + "title")
            if title_el is not None:
                u_el = title_el.find(NS + "upstream-id")
                d_el = title_el.find(NS + "downstream-id")
                u = (u_el.text or "").strip() if u_el is not None else ""
                d = (d_el.text or "").strip() if d_el is not None else ""
                if uuid and u and d:
                    idx.ker_by_pair[(u, d)] = uuid
            elem.clear()

        elif tag == TAG_AOP:
            aop_uuid = elem.get("id") or ""
            rels_el = elem.find(NS + "key-event-relationships")
            if aop_uuid and rels_el is not None:
                ker_uuids = [
                    r.get("id") for r in rels_el.findall(NS + "relationship")
                    if r.get("id")
                ]
                aop_uuid_to_ker_uuids[aop_uuid] = ker_uuids
            elem.clear()

        elif tag == TAG_KE_REF:
            uuid = elem.get("id") or ""
            wid  = elem.get("aop-wiki-id")
            if uuid and wid and wid.isdigit():
                idx.ke_uuid_to_wiki_id[uuid] = int(wid)

        elif tag == TAG_KER_REF:
            uuid = elem.get("id") or ""
            wid  = elem.get("aop-wiki-id")
            if uuid and wid and wid.isdigit():
                idx.ker_uuid_to_wiki_id[uuid] = int(wid)

        elif tag == TAG_AOP_REF:
            uuid = elem.get("id") or ""
            wid  = elem.get("aop-wiki-id")
            if uuid and wid and wid.isdigit():
                aop_uuid_to_wiki_id[uuid] = int(wid)

    for entry in idx.ke_by_title.values():
        wid = idx.ke_uuid_to_wiki_id.get(entry["uuid"])
        if wid is not None:
            entry["wiki_id"] = wid

    for aop_uuid, ker_uuids in aop_uuid_to_ker_uuids.items():
        aop_wiki_id = aop_uuid_to_wiki_id.get(aop_uuid)
        if aop_wiki_id is None:
            continue
        for ker_uuid in ker_uuids:
            idx.ker_to_aop_wiki_ids.setdefault(ker_uuid, []).append(aop_wiki_id)

    return idx


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _lookup_ke(idx: AOPWikiIndex, ke_name: str) -> Optional[dict]:
    """Find a KE by name. Returns {'uuid':..., 'wiki_id':...} or None."""
    if not ke_name:
        return None
    name_l = ke_name.strip().lower()
    if not name_l:
        return None

    # 1. Exact match
    hit = idx.ke_by_title.get(name_l)
    if hit:
        return hit

    # 2. Substring match — short titles win to avoid runaway matches
    candidates = [
        (title, entry) for title, entry in idx.ke_by_title.items()
        if name_l in title or title in name_l
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: abs(len(t[0]) - len(name_l)))
    return candidates[0][1]


def enrich_ker(upstream_ke_name: str, downstream_ke_name: str) -> dict:
    """
    Resolve numeric AOP-Wiki IDs for a KER.

    Returns a dict with keys:
        upstream_ke_id, downstream_ke_id, ker_id, aop_id, aop_status
    All values may be None (-> aop_status='novel') if no match is found.
    Output shape matches the legacy aopwiki_client.enrich_ker.
    """
    idx = get_index()
    if idx is None:
        # No dump available — caller should warn the user.
        return {
            "upstream_ke_id":   None,
            "downstream_ke_id": None,
            "ker_id":           None,
            "aop_id":           None,
            "aop_status":       "novel",
        }

    u = _lookup_ke(idx, upstream_ke_name)
    d = _lookup_ke(idx, downstream_ke_name)

    upstream_wiki_id   = u["wiki_id"] if u else None
    downstream_wiki_id = d["wiki_id"] if d else None

    ker_wiki_id: Optional[int] = None
    aop_id_str:  Optional[str] = None

    if u and d:
        ker_uuid = idx.ker_by_pair.get((u["uuid"], d["uuid"]))
        if ker_uuid:
            ker_wiki_id = idx.ker_uuid_to_wiki_id.get(ker_uuid)
            aop_ids = idx.ker_to_aop_wiki_ids.get(ker_uuid) or []
            if aop_ids:
                aop_id_str = ";".join(str(x) for x in sorted(set(aop_ids)))

    return {
        "upstream_ke_id":   upstream_wiki_id,
        "downstream_ke_id": downstream_wiki_id,
        "ker_id":           ker_wiki_id,
        "aop_id":           aop_id_str,
        "aop_status":       "existing" if aop_id_str else "novel",
    }
