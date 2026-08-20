from __future__ import annotations

"""
One database per browser session, so no user can read or delete another's work.

The problem
-----------
Every store in this codebase resolved its file from a module-level constant —
`table1_store.DB_PATH`, and a `_DB_PATH` apiece in `ols4_client`, `ke_synonyms`
and `gene_registry`, all defaulting to `aop_rag.db` in the working directory.
Module globals are per *process*, and Streamlit serves every browser session
from one process. So on a shared host there is exactly one database and it
belongs to whoever is looking at it:

* one curator's Table 1 rows appear in another's, attributed to a run they
  never made;
* **Clear all extraction data** and **Reset everything** empty the corpus for
  everybody, with a confirmation dialog that says "your" data;
* two people extracting at once interleave rows into the same `run_id`, so the
  manifest describes conditions that were never true of half the rows it
  covers.

None of that is visible from inside a session. It looks like a working app.

The fix
-------
`activate()` at the top of every script run points all four stores at a file
belonging to this session and nothing else. The path comes from
`AOP_RAG_DB` when that is set — the single-user, keep-my-corpus case — and
otherwise from a private temporary directory keyed by Streamlit's session id.

Thread-local, not global
------------------------
Streamlit runs each session's script in its own thread and they interleave
freely, so setting a module global at the top of a script run is not isolation:
between the assignment and the query, another session's thread can reassign it
and the query lands in the wrong file. Every store therefore keeps a
`threading.local` override, set here, and falls back to its module constant
when none is set — which is what keeps the code importable outside Streamlit
and keeps the existing tests, which monkeypatch the constant, working
unchanged.

What this does not do
---------------------
It does not authenticate anybody. A session id is a bearer token in a cookie:
it stops sessions from colliding, it does not stop someone who obtains one.
Anything genuinely confidential needs a login in front of the app.
"""

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "PERSISTENT_ENV_VAR",
    "activate",
    "current_path",
    "is_persistent",
    "session_id",
    "sessions_root",
    "sweep_stale",
    "discard_session",
]

#: Set this to a file path to opt out of per-session isolation and keep one
#: persistent database — the single-user desktop case. Deployments simply do
#: not set it, so the safe behaviour is the one you get by doing nothing.
PERSISTENT_ENV_VAR = "AOP_RAG_DB"

#: Session databases older than this with no live session are swept. Long
#: enough that a curator reading a paper over lunch does not lose their run;
#: short enough that a shared host does not accumulate them forever.
STALE_AFTER_SECONDS = 12 * 3600

_SESSIONS_DIRNAME = "ai4aop-sessions"


def sessions_root() -> Path:
    """The directory holding per-session databases."""
    root = Path(tempfile.gettempdir()) / _SESSIONS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def persistent_path() -> Optional[Path]:
    """The configured single-database path, or None when isolation applies."""
    raw = (os.getenv(PERSISTENT_ENV_VAR) or "").strip()
    return Path(raw).expanduser() if raw else None


def is_persistent() -> bool:
    return persistent_path() is not None


def session_id() -> str:
    """
    This browser session's id, or a per-process stand-in outside Streamlit.

    Read through Streamlit's internal script-run context, which is where the id
    lives and has moved between releases — hence the defensive import. Falling
    back to one shared id would silently restore the bug this module exists to
    fix, so the fallback is a *unique* id instead: worst case each script run
    gets its own database, which is wasteful and safe rather than tidy and
    wrong.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "session_id", None):
            return str(ctx.session_id)
    except Exception:  # noqa: BLE001 - a private API that may move again
        pass

    import uuid

    return f"anon-{uuid.uuid4().hex}"


def current_path() -> Path:
    """The database file this session should be using."""
    configured = persistent_path()
    if configured is not None:
        return configured
    return sessions_root() / f"{_safe(session_id())}.db"


def _safe(name: str) -> str:
    """A session id reduced to something safe to put in a filename."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]


# ---------------------------------------------------------------------------
# Pointing the stores at it
# ---------------------------------------------------------------------------

def activate() -> dict[str, Any]:
    """
    Point every store at this session's database. Call once per script run.

    Cheap and idempotent — it assigns four thread-local values — so calling it
    at the top of every rerun is correct and is what keeps a long-lived
    Streamlit thread from drifting onto another session's file.
    """
    path = current_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    from stage2_extraction import (
        gene_registry,
        ke_synonyms,
        ols4_client,
        table1_store,
    )

    # The satellite caches live in the same file as the rows. They hold public
    # reference data — OLS4 terms, HGNC symbols — so sharing them would be
    # harmless in itself, but the *lookups* are the curator's search terms and
    # a shared cache file is one more thing to reason about. One file per
    # session is the claim that is actually checkable: nothing this session
    # writes is readable by another.
    for module in (table1_store, ols4_client, ke_synonyms, gene_registry):
        module.set_db_path(path)

    return {
        "path": path,
        "persistent": is_persistent(),
        "session_id": session_id(),
    }


def discard_session(path: Optional[Path] = None) -> bool:
    """
    Delete this session's database. Refuses to touch a persistent one.

    The refusal matters: the same button in the same place must not mean
    "clear my scratch copy" for a deployed user and "delete the corpus I have
    been building for a month" for a local one.
    """
    if is_persistent():
        return False
    target = Path(path) if path is not None else current_path()
    try:
        target.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def sweep_stale(
    *,
    older_than: int = STALE_AFTER_SECONDS,
    active_ids: Optional[set[str]] = None,
) -> int:
    """
    Delete session databases whose session is gone.

    Streamlit exposes no supported "session ended" hook, so cleanup is a sweep
    rather than a callback. A file is removed when it is older than
    `older_than` AND its session is not among `active_ids` — belt and braces,
    because the active-session list comes from an internal API that may be
    unavailable, and deleting a live session's database would look exactly like
    data loss.
    """
    if is_persistent():
        return 0

    if active_ids is None:
        active_ids = _active_session_ids()

    cutoff = time.time() - max(0, older_than)
    removed = 0
    for candidate in sessions_root().glob("*.db"):
        if candidate.stem in active_ids:
            continue
        try:
            if candidate.stat().st_mtime > cutoff:
                continue
            candidate.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _active_session_ids() -> set[str]:
    """
    Session ids Streamlit still considers live, as filename stems.

    Returns an empty set when the internal API is unavailable — which makes the
    sweep fall back to age alone, and age alone is already conservative.
    """
    try:
        from streamlit.runtime import get_instance

        runtime = get_instance()
        sessions = runtime._session_mgr.list_active_sessions()
        return {
            _safe(str(getattr(s, "session_id", getattr(s, "id", ""))))
            for s in sessions
        }
    except Exception:  # noqa: BLE001 - private API, absent outside a server
        return set()


@atexit.register
def _cleanup_on_exit() -> None:
    """Remove the whole session directory when the process stops."""
    if is_persistent():
        return
    try:
        shutil.rmtree(sessions_root(), ignore_errors=True)
    except Exception:  # noqa: BLE001 - never let cleanup break shutdown
        pass
