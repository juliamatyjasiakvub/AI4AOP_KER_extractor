"""
No session may read or delete another session's corpus.

Every store resolved its file from a module-level constant defaulting to
`aop_rag.db`. Module globals are per process and Streamlit serves every browser
session from one process, so on a shared host there was exactly one database
and it belonged to whoever was looking at it: one curator's rows appeared in
another's, "Clear all extraction data" emptied the corpus for everybody, and
two people extracting at once wrote into the same run.

The isolation is thread-local rather than global, because Streamlit runs each
session's script in its own thread and those threads interleave — a global
assigned at the top of a script run can be reassigned by another session before
the query that depends on it. These tests run genuine concurrent threads, since
that is the only way the failure they guard against actually shows up.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

import run_manifest
import session_db
from stage2_extraction import ker_extractor, table1_store
from stage2_extraction.llm_providers import LLMConfig


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_two_sessions_get_two_files(monkeypatch, tmp_path):
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)

    monkeypatch.setattr(session_db, "session_id", lambda: "session-aaa")
    first = session_db.current_path()
    monkeypatch.setattr(session_db, "session_id", lambda: "session-bbb")
    second = session_db.current_path()

    assert first != second
    assert first.parent == second.parent == tmp_path


def test_the_env_var_opts_back_into_one_shared_file(monkeypatch, tmp_path):
    """The single-user case: one lasting corpus, chosen deliberately."""
    target = tmp_path / "aop_rag.db"
    monkeypatch.setenv(session_db.PERSISTENT_ENV_VAR, str(target))

    monkeypatch.setattr(session_db, "session_id", lambda: "session-aaa")
    first = session_db.current_path()
    monkeypatch.setattr(session_db, "session_id", lambda: "session-bbb")
    second = session_db.current_path()

    assert first == second == target
    assert session_db.is_persistent()


def test_a_missing_session_id_does_not_collapse_into_one_database(monkeypatch, tmp_path):
    """
    The id is read through a private Streamlit API. If that moves again the
    fallback must stay unique per caller — a shared fallback id would restore
    the exact bug this module exists to prevent, silently and with no error.
    """
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)

    assert session_db.session_id() != session_db.session_id()
    assert session_db.current_path() != session_db.current_path()


def test_a_session_id_cannot_escape_the_sessions_directory(monkeypatch, tmp_path):
    """A session id reaches a filename, so it has to be neutralised first."""
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)
    monkeypatch.setattr(session_db, "session_id", lambda: "../../etc/passwd")

    path = session_db.current_path()
    assert path.parent == tmp_path
    assert ".." not in path.name


# ---------------------------------------------------------------------------
# Concurrency — the reason this is thread-local
# ---------------------------------------------------------------------------

def test_concurrent_threads_write_to_their_own_database(tmp_path):
    """
    Two sessions extracting at once. With a module global the later assignment
    wins for both threads and one corpus lands in the other's file; with a
    thread-local each thread keeps its own.
    """
    results: dict[str, list] = {}
    started = threading.Barrier(2)

    def session(name: str) -> None:
        db = tmp_path / f"{name}.db"
        table1_store.set_db_path(db)
        table1_store.init_db()
        started.wait()          # force the two threads to interleave
        with table1_store.connect() as conn:
            conn.execute(
                "INSERT INTO ke_canonical (canonical_name, level, merge_method, "
                "curation_status, n_source_rows, updated_at) "
                "VALUES (?, 'Molecular', 'manual', 'accepted', 0, '2026-01-01')",
                (f"event-{name}",),
            )
            conn.commit()
        with sqlite3.connect(db) as check:
            results[name] = [
                r[0] for r in check.execute(
                    "SELECT canonical_name FROM ke_canonical"
                )
            ]
        table1_store.set_db_path(None)

    threads = [threading.Thread(target=session, args=(n,)) for n in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["alice"] == ["event-alice"]
    assert results["bob"] == ["event-bob"], (
        "bob's database must not contain alice's Key Event"
    )


def test_the_module_constant_is_still_the_fallback(monkeypatch, tmp_path):
    """
    Existing tests monkeypatch `DB_PATH` directly and must keep working, and
    the stores must stay usable outside Streamlit where no override is set.
    """
    table1_store.set_db_path(None)
    monkeypatch.setattr(table1_store, "DB_PATH", tmp_path / "fallback.db")
    assert table1_store.current_db_path() == tmp_path / "fallback.db"

    table1_store.set_db_path(tmp_path / "override.db")
    try:
        assert table1_store.current_db_path() == tmp_path / "override.db"
    finally:
        table1_store.set_db_path(None)

    assert table1_store.current_db_path() == tmp_path / "fallback.db"


def test_the_satellite_caches_follow_the_session(tmp_path):
    """
    The ontology, synonym and gene caches live in the same file as the rows, so
    leaving them on the module default would have kept one shared file open
    alongside the isolated one.
    """
    from stage2_extraction import gene_registry, ke_synonyms, ols4_client

    target = tmp_path / "session.db"
    for module in (ols4_client, ke_synonyms, gene_registry):
        module.set_db_path(target)
        assert module._db() == target
        module.set_db_path(None)
        assert module._db() == module._DB_PATH


# ---------------------------------------------------------------------------
# Credentials and telemetry
# ---------------------------------------------------------------------------

def test_a_fallback_api_key_does_not_cross_between_sessions():
    """
    `LLMConfig` carries an `api_key`. Held in a module global, one session's
    fallback credentials would be picked up — and billed — by every other
    session in the process the moment their primary model declined.
    """
    seen: dict[str, object] = {}
    ready = threading.Barrier(2)

    def session(name: str, cfg) -> None:
        ker_extractor.set_refusal_fallback(cfg)
        ready.wait()
        seen[name] = ker_extractor.refusal_fallback()

    alice = LLMConfig(provider="anthropic", model="claude-haiku-4-5", api_key="alice-key")
    bob = LLMConfig(provider="openai", model="gpt-4o", api_key="bob-key")

    threads = [
        threading.Thread(target=session, args=("alice", alice)),
        threading.Thread(target=session, args=("bob", bob)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["alice"].api_key == "alice-key"
    assert seen["bob"].api_key == "bob-key"


def test_two_runs_do_not_share_one_set_of_counters():
    """
    The manifest exists to say what conditions produced a given set of rows. A
    process-wide recorder made each concurrent run's manifest describe the
    other's calls as its own — worse than having no record, because it reads
    as one.
    """
    totals: dict[str, int] = {}
    ready = threading.Barrier(2)

    def session(name: str, n_calls: int) -> None:
        telemetry = run_manifest.start_run()
        ready.wait()
        for _ in range(n_calls):
            run_manifest.record("llm_call")
        totals[name] = telemetry.llm_calls
        run_manifest.end_run()

    threads = [
        threading.Thread(target=session, args=("alice", 3)),
        threading.Thread(target=session, args=("bob", 7)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert totals == {"alice": 3, "bob": 7}


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def test_a_live_session_is_never_swept(monkeypatch, tmp_path):
    """Deleting a database out from under an open tab looks exactly like data loss."""
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)

    live = tmp_path / "live.db"
    stale = tmp_path / "stale.db"
    for path in (live, stale):
        path.write_bytes(b"")

    removed = session_db.sweep_stale(older_than=0, active_ids={"live"})

    assert live.exists(), "a session Streamlit still considers active is untouchable"
    assert not stale.exists()
    assert removed == 1


def test_a_recent_database_is_never_swept(monkeypatch, tmp_path):
    """Age alone is the fallback when the active-session list is unavailable."""
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)

    recent = tmp_path / "recent.db"
    recent.write_bytes(b"")

    assert session_db.sweep_stale(older_than=3600, active_ids=set()) == 0
    assert recent.exists()


def test_the_persistent_database_is_never_swept_or_discarded(monkeypatch, tmp_path):
    """
    The same button must not mean "clear my scratch copy" for a deployed user
    and "delete the corpus I built over a month" for a local one.
    """
    target = tmp_path / "aop_rag.db"
    target.write_bytes(b"important")
    monkeypatch.setenv(session_db.PERSISTENT_ENV_VAR, str(target))

    assert session_db.sweep_stale(older_than=0) == 0
    assert session_db.discard_session() is False
    assert target.read_bytes() == b"important"


def test_discarding_removes_only_this_session(monkeypatch, tmp_path):
    monkeypatch.delenv(session_db.PERSISTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_db, "sessions_root", lambda: tmp_path)
    monkeypatch.setattr(session_db, "session_id", lambda: "mine")

    mine = tmp_path / "mine.db"
    theirs = tmp_path / "theirs.db"
    for path in (mine, theirs):
        path.write_bytes(b"")

    assert session_db.discard_session() is True
    assert not mine.exists()
    assert theirs.exists()
