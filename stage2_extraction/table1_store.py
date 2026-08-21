from __future__ import annotations

"""
SQLite-backed store for Table 1 and everything that hangs off it.

A single `aop_rag.db` file is created in the working directory holding:

    schema_meta         schema version marker, so upgrades are detected
    extraction_runs     one row per run: model, settings, robustness counters
    table1_extractions  one row per KER per paper (raw, as extracted)
    evidence_spans      verbatim quotations with page / section / chunk
    ke_canonical        merged Key Events with ontology annotation
    ke_alias            every raw label that maps onto a canonical KE
    physio_map_link     tissue/organ KEs linked to physiological map entities
    ker_curation        expert accept / reject / rename / merge decisions
    layout_state        persisted node coordinates, lanes, groups, pins
    ols4_cache          ontology lookup cache (managed by ols4_client)

    merge_decision      one row per curator decision on a candidate group,
                        with the semantic classification, the explanation
                        shown at the time, and before/after snapshots
    ontology_mapping    broader/related ontology parents, kept deliberately
                        apart from equivalence merges
    workflow_state      raw -> normalization_proposed -> curated -> approved
                        -> synthesized, per Key Event and per KER
    approval_log        every state transition, so approval is auditable
    synthesis_history   superseded syntheses, preserved when one goes stale
    ke_role             MIE / KE / AO assignment and its approval
    aop_snapshot        the approved graph, frozen at approval time
    layout_offset       vertical-only user nudges; horizontal position is
                        always recomputed from causal order

The store is deliberately the only module that writes SQL, so the schema can
evolve in one place.
"""

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from run_manifest import RunManifest, RunTelemetry
from schemas import KE_LEVEL_ORDER, CanonicalKE, EvidenceSpan, KERExtraction
from stage2_extraction import citations
from stage2_extraction.pdf_reader import strip_control_chars

DB_PATH = Path("aop_rag.db")


class SchemaTooNewError(RuntimeError):
    """The database was written by a newer version of this tool than is running."""

#: Bump this whenever the schema changes in a way that old rows cannot satisfy.
#: v3 added `extraction_runs` and the `run_id` on extractions and spans, so a
#: v2 row cannot say which model or prompts produced it. v4 added the output
#: budget multiplier, which changes how much of each reply survives and so
#: belongs in the manifest alongside the model. v5 added the extraction mode
#: and its target Key Events, without which an open run and a targeted run are
#: indistinguishable — and they mean different things by an absent result.
#: v6 added `ker_synthesis`, the consolidated weight-of-evidence assessment.
#: v7 added the curation-workflow tables — merge decisions with their semantic
#: classification, ontology mappings held apart from equivalence, workflow
#: state and its approval log, synthesis history, KE roles and approved graph
#: snapshots. Unlike earlier bumps this one is purely additive, so a v6
#: database is migrated in place rather than rebuilt: the extracted rows are
#: expensive to reproduce and nothing about them changed.
#: v8 records the sign of every relationship and the cell type it was observed
#: in. Both were being extracted already and thrown away — the direction into
#: a prose field, the cell type nowhere at all — which let a loss-of-function
#: result and a gain-of-function result in different cell types collapse onto
#: one edge. Purely additive, so a v7 database keeps its rows; the new columns
#: are simply empty until those rows are re-extracted.
#: v10 records who put each row there. Until now every Table 1 row was, by
#: construction, something a model said about a paper, so "extracted" needed no
#: marking. A curator can now enter a claim the extraction missed, and a
#: curator-entered row that looks identical to a model-extracted one would
#: quietly undo the one promise this tool makes. Purely additive.
#:
#: v11 records, on every alias, why that raw label was put in its Key Event.
#: The grouping was always computed and never written down, so the only
#: account of it the curator got was a pair of totals — "18 canonical Key
#: Events proposed from 31 raw labels" — which says nothing about which
#: wording went where or on what authority. Purely additive.
SCHEMA_VERSION = 12

#: Versions that `_migrate` can upgrade in place, keyed by the version being
#: upgraded *from*. Anything older still triggers a rebuild.
#: v11 joins the list because v12 only ADDS `ker_synthesis.n_rows`, with a
#: default. Nothing a v11 database holds changes meaning, so dropping every
#: table — which is what a version bump does to anything not listed here —
#: would destroy a curated corpus to add one nullable column.
_IN_PLACE_UPGRADES = {6, 7, 8, 9, 10, 11}

#: Sign and context on the relationship itself, added in v8.
_TABLE1_V8_COLUMNS = (
    # positive | negative | none | unclear — how the two events moved together
    # in THIS paper's experiment.
    ("direction", "TEXT"),
    # What the paper actually did to each end: "increased"/"decreased"/etc.
    # "Reduced Nav1.2 -> loss of spiking" and "increased Na+ current ->
    # activation" are the same edge without these.
    ("upstream_change", "TEXT"),
    ("downstream_change", "TEXT"),
    # The cell type each event was observed in. Oligodendroglial Nav1.2 and
    # microglial Na+ current are not one Key Event, and nothing in v7 could
    # say so.
    ("upstream_cell_type", "TEXT"),
    ("downstream_cell_type", "TEXT"),
    # causal | marker | definitional. "Oligodendrocyte differentiation ->
    # myelin basic protein expression" is not a causal step: MBP is how
    # differentiation is measured. Drawn as an edge it adds a terminal node
    # that looks like an adverse outcome and is really a staining result.
    ("relation_kind", "TEXT"),
)

#: v9: how each link's direction was actually established.
#:
#: Without these an AOP cannot distinguish a knockout from a correlation, and
#: every downstream judgement — adjacency, confidence, whether an arrow should
#: be drawn at all — was being made on paper counts instead.
_TABLE1_V9_COLUMNS = (
    ("evidence_type", "TEXT"),      # rescue|perturbation|correlation|reverse_only
    ("measured_as", "TEXT"),        # the assay behind the claim
    ("null_findings", "TEXT"),      # what was measured and did NOT change
    ("study_context", "TEXT"),
    ("upstream_target", "TEXT"),   # SCN2A/Nav1.2, not "sodium channel"
    ("downstream_target", "TEXT"),      # development, injury model, knockout, culture
)

#: v10: provenance of the row itself, as opposed to provenance of its claim.
#:
#: `origin` is the one that matters. Everything downstream — the QC
#: verification rate, the confidence distribution, the arrow on the map — was
#: written assuming every row came from a model reading a paper, and each of
#: those readings is wrong for a row a curator typed. A default of 'llm' is
#: correct for every row that existed before this column did.
_TABLE1_V10_COLUMNS = (
    ("origin", "TEXT NOT NULL DEFAULT 'llm'"),
    ("entered_by", "TEXT"),
    ("entry_rationale", "TEXT"),
    ("entered_at", "TEXT"),
)

#: v10 on the canonical table: whether this Key Event was derived from raw
#: labels or asserted by a curator. Normalization rebuilds the derived ones
#: from scratch every time it runs and must not take the asserted ones with
#: them.
_KE_CANONICAL_V10_COLUMNS = (
    ("origin", "TEXT NOT NULL DEFAULT 'derived'"),
    ("created_by", "TEXT"),
    ("create_rationale", "TEXT"),
)

#: Why each raw label sits in the Key Event it sits in, added in v11.
#:
#: `merge_basis` is one of the rule names in `ALIAS_BASIS_LABELS` below and
#: `merge_detail` is the sentence that names the evidence — the AOP-Wiki id,
#: the ontology CURIE and its score, the label it was matched against and the
#: similarity. An alias written before v11 has both NULL, which displays as
#: "recorded before this was tracked" rather than as a blank the curator has
#: to interpret.
_KE_ALIAS_V11_COLUMNS = (
    ("merge_basis", "TEXT"),
    ("merge_detail", "TEXT"),
)

#: How each grouping rule is described in the UI. The keys are what the
#: normalizer writes; the values are what a curator reads. Kept next to the
#: column definition so the two cannot drift apart.
ALIAS_BASIS_LABELS = {
    "aopwiki": "Same AOP-Wiki Key Event id",
    "ontology": "Same ontology term",
    "normalised_string": "Identical after normalisation",
    "token_order": "Same content words, different order",
    "lexical": "Lexical similarity above threshold",
    "own_group": "Its own event — nothing merged into it",
    "curator": "Curator assigned it",
}

#: Row origins. 'llm' is a model reading a paper; 'curator' is a claim typed by
#: a person; 'curator_edited' is a model row a person has since corrected;
#: 'imported' arrived in bulk from a spreadsheet. The last three share one
#: property that the code cares about: no model produced them, so no model
#: quality measure applies to them.
LLM_ORIGIN = "llm"
CURATOR_ORIGINS = ("curator", "curator_edited", "imported")


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_SCHEMA_META_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

#: One row per extraction run. Columns mirror `RunManifest.as_row()` followed by
#: `RunTelemetry.as_row()` — the conditions of the run, then what went wrong
#: during it. Without this table two rows in Table 1 cannot be compared, since
#: nothing records whether they came from the same model or the same prompts.
CREATE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stage               TEXT    NOT NULL DEFAULT 'extraction',
    mode                TEXT    NOT NULL DEFAULT 'open',
    target_upstream     TEXT,
    target_downstream   TEXT,
    started_at          TEXT    NOT NULL,
    finished_at         TEXT,
    status              TEXT    NOT NULL DEFAULT 'running',

    provider            TEXT,
    model               TEXT,
    model_reported      TEXT,
    endpoint_host       TEXT,
    temperature         REAL,
    top_p               REAL,
    max_output_tokens   INTEGER,
    num_ctx             INTEGER,
    seed                INTEGER,

    prompt_fingerprint  TEXT,
    budget_scale        REAL,
    chunking_enabled    INTEGER,
    chunk_char_budget   INTEGER,
    chunk_min_score     REAL,
    chunk_scorer        TEXT,
    llm_triage          INTEGER,
    ols4_enabled        INTEGER,
    ols4_min_score      REAL,
    max_kers            INTEGER,

    aopwiki_version     TEXT,
    code_version        TEXT,
    schema_version      INTEGER,
    python_version      TEXT,
    platform            TEXT,

    llm_calls           INTEGER NOT NULL DEFAULT 0,
    step_failures       INTEGER NOT NULL DEFAULT 0,
    provider_errors     INTEGER NOT NULL DEFAULT 0,
    provider_retries    INTEGER NOT NULL DEFAULT 0,
    json_repairs        INTEGER NOT NULL DEFAULT 0,
    json_failures       INTEGER NOT NULL DEFAULT 0,
    truncated_steps     INTEGER NOT NULL DEFAULT 0,
    empty_replies       INTEGER NOT NULL DEFAULT 0,
    papers_attempted    INTEGER NOT NULL DEFAULT 0,
    papers_with_kers    INTEGER NOT NULL DEFAULT 0,
    kers_extracted      INTEGER NOT NULL DEFAULT 0,
    chunks_total        INTEGER NOT NULL DEFAULT 0,
    chunks_selected     INTEGER NOT NULL DEFAULT 0,
    chars_total         INTEGER NOT NULL DEFAULT 0,
    chars_sent          INTEGER NOT NULL DEFAULT 0,
    dropped_params      TEXT,
    renamed_params      TEXT,
    notes               TEXT
)
"""

CREATE_TABLE1_SQL = """
CREATE TABLE IF NOT EXISTS table1_extractions (
    record_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doi              TEXT    NOT NULL,
    source_filename         TEXT,
    source_title            TEXT,
    extraction_date         TEXT    NOT NULL,
    run_id                  INTEGER,
    aop_id                  TEXT,
    aop_status              TEXT,
    upstream_ke_id          INTEGER,
    downstream_ke_id        INTEGER,
    ker_id                  INTEGER,
    upstream_ke_name        TEXT    NOT NULL,
    upstream_ke_level       TEXT    NOT NULL,
    downstream_ke_name      TEXT    NOT NULL,
    downstream_ke_level     TEXT    NOT NULL,
    ker_name                TEXT    NOT NULL,
    ker_description         TEXT    NOT NULL,
    ker_adjacency           TEXT    NOT NULL,
    paper_type              TEXT    NOT NULL,
    cited_evidence_dois     TEXT,
    biological_plausibility TEXT,
    empirical_evidence_summary TEXT,
    essentiality_evidence   TEXT,
    contradicts_ker         INTEGER NOT NULL,   -- 0/1 boolean
    taxonomic_applicability TEXT    NOT NULL,
    sex_applicability       TEXT    NOT NULL,
    life_stage_applicability TEXT   NOT NULL,
    modulating_factors      TEXT,
    quantitative_relationships TEXT,
    response_response_relationship TEXT,
    time_scale              TEXT,
    feedforward_feedback_loops TEXT,
    study_design            TEXT    NOT NULL,
    exposure_route          TEXT,
    chemical_stressor       TEXT,
    extraction_confidence   TEXT    NOT NULL,
    upstream_ke_canonical_id   INTEGER,
    downstream_ke_canonical_id INTEGER,
    n_evidence_spans        INTEGER NOT NULL DEFAULT 0,
    n_verified_spans        INTEGER NOT NULL DEFAULT 0,
    direction               TEXT,   -- positive | negative | none | unclear
    upstream_change         TEXT,   -- what the paper did to the upstream event
    downstream_change       TEXT,   -- what it observed at the downstream one
    upstream_cell_type      TEXT,
    downstream_cell_type    TEXT,
    relation_kind           TEXT,   -- causal | marker | definitional
    evidence_type           TEXT,   -- rescue|perturbation|correlation|reverse_only
    measured_as             TEXT,
    null_findings           TEXT,
    study_context           TEXT,
    upstream_target         TEXT,
    downstream_target       TEXT,
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE SET NULL
)
"""

CREATE_EVIDENCE_SQL = """
CREATE TABLE IF NOT EXISTS evidence_spans (
    span_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id       INTEGER NOT NULL,
    run_id          INTEGER,
    quote           TEXT    NOT NULL,
    field           TEXT    NOT NULL,
    section         TEXT,
    section_kind    TEXT,
    page_start      INTEGER,
    page_end        INTEGER,
    chunk_id        TEXT,
    char_start      INTEGER,
    char_end        INTEGER,
    verified        INTEGER NOT NULL DEFAULT 0,
    match_ratio     REAL    NOT NULL DEFAULT 0.0,
    source_doi      TEXT,
    source_filename TEXT,
    FOREIGN KEY (record_id) REFERENCES table1_extractions(record_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE SET NULL
)
"""

CREATE_KE_CANONICAL_SQL = """
CREATE TABLE IF NOT EXISTS ke_canonical (
    canonical_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name   TEXT    NOT NULL,
    level            TEXT    NOT NULL,
    ontology_curie   TEXT,
    ontology_iri     TEXT,
    ontology_label   TEXT,
    ontology_source  TEXT,
    ontology_score   REAL    NOT NULL DEFAULT 0.0,
    aopwiki_ke_id    INTEGER,
    merge_method     TEXT    NOT NULL DEFAULT 'auto',
    curation_status  TEXT    NOT NULL DEFAULT 'unreviewed',
    n_source_rows    INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT
)
"""

CREATE_KE_ALIAS_SQL = """
CREATE TABLE IF NOT EXISTS ke_alias (
    alias_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id INTEGER NOT NULL,
    raw_label    TEXT    NOT NULL,
    n_uses       INTEGER NOT NULL DEFAULT 1,
    UNIQUE (canonical_id, raw_label),
    FOREIGN KEY (canonical_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE
)
"""

CREATE_PHYSIO_LINK_SQL = """
CREATE TABLE IF NOT EXISTS physio_map_link (
    link_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id INTEGER NOT NULL,
    provider     TEXT    NOT NULL,
    entity_label TEXT    NOT NULL,
    entity_id    TEXT    NOT NULL,
    url          TEXT    NOT NULL,
    confidence   REAL    NOT NULL DEFAULT 0.0,
    UNIQUE (canonical_id, provider, entity_id),
    FOREIGN KEY (canonical_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE
)
"""

CREATE_CURATION_SQL = """
CREATE TABLE IF NOT EXISTS ker_curation (
    target_type  TEXT NOT NULL,           -- 'ke' | 'ker'
    target_key   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'unreviewed',
    display_name TEXT,
    note         TEXT,
    merged_into  TEXT,
    curator      TEXT,
    updated_at   TEXT,
    PRIMARY KEY (target_type, target_key)
)
"""

CREATE_LAYOUT_SQL = """
CREATE TABLE IF NOT EXISTS layout_state (
    layout_name TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    lane        TEXT,
    "group"     TEXT,
    pinned      INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT,
    PRIMARY KEY (layout_name, node_key)
)
"""

#: Consolidated weight-of-evidence assessments, one per KER.
#:
#: Persisted rather than recomputed because it costs a model call, and keyed
#: with the record ids it was built from so it can be told apart from a
#: synthesis written before new papers arrived — a stale assessment that still
#: reads authoritatively is the failure mode worth guarding against.
CREATE_SYNTHESIS_SQL = """
CREATE TABLE IF NOT EXISTS ker_synthesis (
    ker_key                        TEXT PRIMARY KEY,
    ker_name                       TEXT NOT NULL,
    n_papers                       INTEGER NOT NULL DEFAULT 0,
    n_rows                         INTEGER NOT NULL DEFAULT 0,
    record_ids                     TEXT,
    mechanistic_basis              TEXT,
    biological_plausibility        TEXT,
    biological_plausibility_rating TEXT,
    empirical_evidence             TEXT,
    empirical_evidence_rating      TEXT,
    essentiality                   TEXT,
    essentiality_rating            TEXT,
    quantitative_understanding     TEXT,
    applicability_domain           TEXT,
    uncertainties                  TEXT,
    overall_confidence             TEXT,
    generated_at                   TEXT,
    model                          TEXT,
    run_id                         INTEGER
)
"""

CREATE_OLS4_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS ols4_cache (
    cache_key   TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    ontologies  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    fetched_at  REAL NOT NULL
)
"""

# ---------------------------------------------------------------------------
# v7 — curation workflow
# ---------------------------------------------------------------------------

#: One row per decision a curator makes about a candidate group.
#:
#: `action` is what the curator did; `relationship` is what the classifier
#: said the records were to each other. Both are stored because they can
#: disagree — a curator may keep two records separate that the classifier
#: called equivalent, and that disagreement is the interesting part of the
#: audit trail. `before_state` and `after_state` hold JSON snapshots so a
#: merge can be undone exactly, rather than approximately.
CREATE_MERGE_DECISION_SQL = """
CREATE TABLE IF NOT EXISTS merge_decision (
    decision_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    group_uid         TEXT    NOT NULL,
    action            TEXT    NOT NULL,
    relationship      TEXT,
    member_ids        TEXT    NOT NULL,
    survivor_id       INTEGER,
    similarity        REAL,
    explanation       TEXT,
    curator_rationale TEXT,
    curator           TEXT,
    method            TEXT    NOT NULL DEFAULT 'manual',
    before_state      TEXT,
    after_state       TEXT,
    reverted          INTEGER NOT NULL DEFAULT 0,
    reverted_at       TEXT,
    reverted_by       TEXT,
    created_at        TEXT    NOT NULL
)
"""

#: Ontology parents attached to a Key Event *without* collapsing it.
#:
#: Deliberately not the `ontology_curie` column on `ke_canonical`: that column
#: means "this KE is that term", whereas a row here means "this KE is a kind of
#: that term". Pooling evidence about NaV1.2 with evidence about voltage-gated
#: sodium channels in general is exactly the error the separation prevents.
CREATE_ONTOLOGY_MAPPING_SQL = """
CREATE TABLE IF NOT EXISTS ontology_mapping (
    mapping_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id INTEGER NOT NULL,
    relation     TEXT    NOT NULL DEFAULT 'broader',
    curie        TEXT    NOT NULL,
    iri          TEXT,
    label        TEXT,
    source       TEXT,
    score        REAL    NOT NULL DEFAULT 0.0,
    curator      TEXT,
    rationale    TEXT,
    created_at   TEXT    NOT NULL,
    UNIQUE (canonical_id, relation, curie),
    FOREIGN KEY (canonical_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE
)
"""

#: Biological relationships between two Key Events that are *not* the same
#: event and are not an ontology parent — "upstream of", "part of", "marker
#: for". Recording one is a way of saying "I looked at this pair and they are
#: related like so", which stops the pair being re-suggested as a duplicate.
CREATE_KE_RELATION_SQL = """
CREATE TABLE IF NOT EXISTS ke_relation (
    relation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL,
    target_id    INTEGER NOT NULL,
    relation     TEXT    NOT NULL,
    curator      TEXT,
    rationale    TEXT,
    created_at   TEXT    NOT NULL,
    UNIQUE (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE
)
"""

#: A curator's ruling on which way a Key Event moves.
#:
#: The map derives a node's arrow by counting the changes its claims recorded.
#: When some say increased and some say decreased it draws "±", which is an
#: honest statement of the corpus and a dead end for the reader: there was no
#: way to act on it and no way to make it go away. Three things can actually
#: resolve one — a misextracted claim, two events wearing one name, or a real
#: split in the literature that the developer judges — and only the third
#: needs storing, because the other two are edits to the rows themselves.
#:
#: `direction` is what the AOP asserts. `acknowledged` records the other
#: answer: the conflict is real, it stays visible, and the curator has seen it.
CREATE_KE_DIRECTION_SQL = """
CREATE TABLE IF NOT EXISTS ke_direction (
    canonical_id INTEGER PRIMARY KEY,
    direction    TEXT    NOT NULL,   -- increased | decreased | conflicted
    acknowledged INTEGER NOT NULL DEFAULT 0,
    rationale    TEXT,
    curator      TEXT,
    updated_at   TEXT    NOT NULL,
    FOREIGN KEY (canonical_id) REFERENCES ke_canonical(canonical_id)
        ON DELETE CASCADE
)
"""

#: Every version of a Table 1 row that is no longer the current one.
#:
#: The README promises that decisions are auditable and that nothing is
#: silently overwritten. Editing an extracted row is exactly the operation that
#: could break that promise: the row on screen would say one thing, the model
#: said another, and there would be no way to tell which. Archiving the prior
#: version first is what keeps "the curator changed this" distinguishable from
#: "the model said this".
#:
#: Deliberately no foreign key to `table1_extractions` — the whole point is
#: that this survives the row's deletion.
CREATE_RECORD_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS record_history (
    history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL,
    payload     TEXT    NOT NULL,   -- the row as it was, JSON
    spans       TEXT,               -- its evidence spans, JSON
    action      TEXT    NOT NULL,   -- edited | deleted
    reason      TEXT,
    curator     TEXT,
    archived_at TEXT    NOT NULL
)
"""

#: Where each Key Event and each KER sits in the workflow.
#:
#: `content_hash` is the fingerprint of what was approved. Comparing it against
#: the current content is how a silently-changed approval is detected: the
#: state still says "approved" but the thing approved is no longer what is on
#: screen, and everything derived from it has to be regenerated.
CREATE_WORKFLOW_STATE_SQL = """
CREATE TABLE IF NOT EXISTS workflow_state (
    target_type  TEXT NOT NULL,
    target_key   TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'raw',
    content_hash TEXT,
    approved_by  TEXT,
    approved_at  TEXT,
    note         TEXT,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (target_type, target_key)
)
"""

CREATE_APPROVAL_LOG_SQL = """
CREATE TABLE IF NOT EXISTS approval_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type  TEXT NOT NULL,
    target_key   TEXT NOT NULL,
    from_state   TEXT,
    to_state     TEXT NOT NULL,
    curator      TEXT,
    note         TEXT,
    content_hash TEXT,
    created_at   TEXT NOT NULL
)
"""

#: Superseded syntheses, kept verbatim.
#:
#: When an approved KE or KER changes, the synthesis built on it is wrong but
#: not worthless — it records what was believed and on what basis. Archiving
#: rather than deleting means the previous version is still readable next to
#: its replacement.
CREATE_SYNTHESIS_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS synthesis_history (
    history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ker_key     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    reason      TEXT,
    archived_at TEXT NOT NULL
)
"""

CREATE_KE_ROLE_SQL = """
CREATE TABLE IF NOT EXISTS ke_role (
    canonical_id INTEGER PRIMARY KEY,
    role         TEXT    NOT NULL DEFAULT 'KE',
    curator      TEXT,
    rationale    TEXT,
    approved     INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT,
    FOREIGN KEY (canonical_id) REFERENCES ke_canonical(canonical_id) ON DELETE CASCADE
)
"""

#: The text the extractor actually read, kept after the run.
#:
#: Chunks lived only in memory: a paper was parsed, scored, sent to the model
#: and forgotten, leaving quotations that point at a `chunk_id` no longer
#: attached to anything. That makes the obvious next question unanswerable —
#: "this chain stops at a marker; does the paper say what happens downstream?"
#: — without re-uploading the PDFs. Storing the text turns that into a
#: query. It is the single largest table in the database and worth it: the
#: PDFs are the one input the tool cannot regenerate for itself.
CREATE_PAPER_CHUNK_SQL = """
CREATE TABLE IF NOT EXISTS paper_chunk (
    chunk_row_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER,
    source_doi      TEXT,
    source_filename TEXT,
    chunk_id        TEXT    NOT NULL,
    text            TEXT    NOT NULL,
    section         TEXT,
    section_kind    TEXT,
    page_start      INTEGER,
    page_end        INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    relevance_score REAL    NOT NULL DEFAULT 0.0,
    selected        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    UNIQUE (source_filename, chunk_id, run_id)
)
"""

#: The abbreviations each paper defines for itself.
#:
#: Papers introduce their own shorthand on first use — "oligodendrocyte
#: precursor cell (OPC)", "hepatic stellate cell (HSC)" — and which shorthand a
#: lab picks is its convention, not something a curator can predict or a
#: maintainer can enumerate in advance. `ke_synonyms.paper_abbreviations()`
#: already reads those definitions out of the text; this is where the answer is
#: kept so that normalisation, which runs over the whole corpus long after the
#: PDFs are gone, can still expand a label the way its own paper meant it.
#:
#: Keyed by DOI rather than by run: the same paper means the same thing by
#: "HSC" in every run it appears in, and re-extracting it should refresh the
#: definitions rather than accumulate copies of them.
CREATE_PAPER_ABBREV_SQL = """
CREATE TABLE IF NOT EXISTS paper_abbrev (
    abbrev_row_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doi      TEXT    NOT NULL,
    source_filename TEXT,
    run_id          INTEGER,
    abbrev          TEXT    NOT NULL,
    long_form       TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    UNIQUE (source_doi, abbrev)
)
"""

#: The graph as it stood when it was approved.
#:
#: The map is a claim about the evidence, so it is frozen rather than rendered
#: live: a graph that quietly redraws itself as rows arrive cannot be cited.
CREATE_AOP_SNAPSHOT_SQL = """
CREATE TABLE IF NOT EXISTS aop_snapshot (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    content_hash TEXT,
    stale        INTEGER NOT NULL DEFAULT 0,
    stale_reason TEXT,
    created_by   TEXT,
    created_at   TEXT    NOT NULL
)
"""

#: Vertical nudges only.
#:
#: `layout_state` stored x and y, which let a saved layout drag a node into a
#: causal column it does not belong to — a late Key Event parked to the left of
#: the MIE reads as a claim about ordering that nobody made. Horizontal
#: position is now always recomputed from causal depth and only the vertical
#: offset is the curator's to keep.
CREATE_LAYOUT_OFFSET_SQL = """
CREATE TABLE IF NOT EXISTS layout_offset (
    layout_name TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    y           REAL NOT NULL,
    updated_at  TEXT,
    PRIMARY KEY (layout_name, node_key)
)
"""

#: What happened to every paper handed to a run, including the ones that
#: produced nothing.
#:
#: A run reporting "11 papers yielded KERs" out of 13 says nothing about the
#: other two, and those are the two worth knowing about: a paper that yielded
#: nothing because it genuinely discusses no mechanism is a result, and a
#: paper that yielded nothing because its reply hit the token ceiling is a
#: silent false negative that looks identical. The per-paper outcome existed
#: only as a table drawn during the run and thrown away on the next rerun, so
#: the distinction was unrecoverable ten seconds after it was made.
CREATE_PAPER_OUTCOME_SQL = """
CREATE TABLE IF NOT EXISTS paper_outcome (
    outcome_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER,
    source_filename TEXT,
    source_doi      TEXT,
    outcome         TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT 'unknown',
    reason          TEXT,
    n_kers          INTEGER NOT NULL DEFAULT 0,
    n_llm_calls     INTEGER NOT NULL DEFAULT 0,
    n_truncated     INTEGER NOT NULL DEFAULT 0,
    chars_sent      INTEGER,
    chars_total     INTEGER,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE
)
"""

#: Why a paper produced nothing. The categories are the ones that call for
#: different action: re-run it with a bigger budget, believe the result, or
#: fix the configuration.
OUTCOME_CATEGORIES = {
    "saved": "KERs extracted and saved.",
    "no_mechanism": (
        "The model read the paper and reported no mechanistic link. Believe "
        "this one — it is a finding about the paper."
    ),
    "truncated": (
        "The reply hit the output-token ceiling and was cut off. The paper may "
        "well contain KERs that were never written out. Re-run it with a "
        "higher budget multiplier."
    ),
    "parse_failure": (
        "The model replied but the JSON could not be parsed even after repair. "
        "Usually a model too small for the task."
    ),
    "refusal": (
        "The provider's safety classifier declined to answer, twice. On "
        "peer-reviewed toxicology this is a false positive — the same model "
        "read the rest of the corpus — and it is a property of that model, "
        "not of the paper. Nothing was learned about this paper. Run it "
        "through a different model or provider; that is the honest workaround "
        "and the only one this tool offers."
    ),
    "provider_error": (
        "The model could not be reached — network, authentication, or a wrong "
        "model name. Nothing about the paper was learned."
    ),
    "no_text": (
        "No text could be read from the PDF. Usually a scanned image with no "
        "text layer; it needs OCR before it can be extracted."
    ),
    "chunking_dropped": (
        "Chunk scoring selected only part of the paper and the model saw no "
        "relevant passage. Lower the relevance threshold or raise the "
        "character budget — the evidence may have been discarded before the "
        "model ever saw it."
    ),
    "error": "The paper failed with an unexpected error.",
    "unknown": "No reason was recorded.",
}


_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_t1_doi ON table1_extractions(source_doi)",
    "CREATE INDEX IF NOT EXISTS idx_t1_up_canon ON table1_extractions(upstream_ke_canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_t1_down_canon ON table1_extractions(downstream_ke_canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_span_record ON evidence_spans(record_id)",
    "CREATE INDEX IF NOT EXISTS idx_alias_canonical ON ke_alias(canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_alias_label ON ke_alias(raw_label)",
    "CREATE INDEX IF NOT EXISTS idx_t1_run ON table1_extractions(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_span_run ON evidence_spans(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_merge_group ON merge_decision(group_uid)",
    "CREATE INDEX IF NOT EXISTS idx_merge_live ON merge_decision(reverted, action)",
    "CREATE INDEX IF NOT EXISTS idx_ontmap_canon ON ontology_mapping(canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_kerel_source ON ke_relation(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_wf_state ON workflow_state(state)",
    "CREATE INDEX IF NOT EXISTS idx_applog_target ON approval_log(target_type, target_key)",
    "CREATE INDEX IF NOT EXISTS idx_synthhist_key ON synthesis_history(ker_key)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_run ON paper_outcome(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_t1_origin ON table1_extractions(origin)",
    "CREATE INDEX IF NOT EXISTS idx_rechist_record ON record_history(record_id)",
)

#: Dropped in this order on a version mismatch. Children before parents, so a
#: foreign key never points at a table that has already gone.
_ALL_TABLES = (
    "paper_outcome",
    "evidence_spans",
    "table1_extractions",
    "extraction_runs",
    "ke_alias",
    "physio_map_link",
    "ontology_mapping",
    "ke_relation",
    "ke_role",
    "ke_canonical",
    "ker_curation",
    "layout_state",
    "layout_offset",
    "ker_synthesis",
    "synthesis_history",
    "merge_decision",
    "workflow_state",
    "approval_log",
    "aop_snapshot",
    "paper_chunk",
    "paper_abbrev",
    "record_history",
    "ke_direction",
)

#: Columns added to `ker_synthesis` in v7, applied with ALTER TABLE so a v6
#: database keeps the syntheses it already paid a model call for.
_SYNTHESIS_V7_COLUMNS = (
    ("input_hash", "TEXT"),
    ("stale", "INTEGER NOT NULL DEFAULT 0"),
    ("stale_reason", "TEXT"),
    ("developer_assessment", "TEXT"),
    ("developer_rationale", "TEXT"),
    ("developer_curator", "TEXT"),
    ("approved_by", "TEXT"),
    ("approved_at", "TEXT"),
    ("version", "INTEGER NOT NULL DEFAULT 1"),
)

#: `n_papers` used to hold a row count. It now holds a count of distinct
#: papers, and the row count moves here beside it, so a synthesis whose rows
#: changed while its papers held still can be told from one where new
#: literature arrived. Added with ALTER TABLE: an existing database keeps its
#: syntheses, and their `n_rows` reads 0 rather than a wrong number.
_SYNTHESIS_V12_COLUMNS = (
    ("n_rows", "INTEGER NOT NULL DEFAULT 0"),
)

#: Why paper text was allowed to leave the machine on a hosted run.
#:
#: Added with ALTER TABLE and deliberately WITHOUT bumping SCHEMA_VERSION. A
#: version bump sends any database whose version is not in `_IN_PLACE_UPGRADES`
#: down the DROP TABLE path, and getting that list wrong destroys a curated
#: corpus — a risk with no upside here, because the change is one nullable
#: column that older code simply ignores. `_add_missing_columns` is idempotent,
#: so this is applied on every `init_db` and is a no-op once present.
_RUNS_PROVENANCE_COLUMNS = (
    ("transmission_ack", "TEXT"),
)


# ---------------------------------------------------------------------------
# Connection + schema management
# ---------------------------------------------------------------------------

#: Per-thread override of `DB_PATH`, set by `session_db.activate()`.
#:
#: Thread-local rather than global because Streamlit serves every browser
#: session from one process, running each session's script in its own thread,
#: and those threads interleave. A module global assigned at the top of a
#: script run is not isolation — between the assignment and the query another
#: session can reassign it, and the query lands in someone else's database.
#:
#: `DB_PATH` stays as the fallback so the stores remain importable outside
#: Streamlit, and so tests that monkeypatch it keep working untouched.
_LOCAL = threading.local()


def set_db_path(path: Any) -> None:
    """Point this thread at `path`. Pass None to fall back to `DB_PATH`."""
    _LOCAL.db_path = None if path is None else Path(path)


def current_db_path() -> Path:
    """The database this thread should use."""
    return getattr(_LOCAL, "db_path", None) or DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(current_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _add_missing_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[tuple[str, str]],
) -> list[str]:
    """
    Add any of `columns` the table does not already have.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, and re-running the ALTER raises
    rather than passing silently, so the existing columns are read first. Used
    to widen `ker_synthesis` in place instead of dropping it.
    """
    try:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return []
    if not existing:
        return []

    added: list[str] = []
    for name, decl in columns:
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        added.append(name)
    return added


def _create_all(conn: sqlite3.Connection) -> None:
    for ddl in (
        CREATE_SCHEMA_META_SQL,
        CREATE_RUNS_SQL,
        CREATE_TABLE1_SQL,
        CREATE_EVIDENCE_SQL,
        CREATE_KE_CANONICAL_SQL,
        CREATE_KE_ALIAS_SQL,
        CREATE_PHYSIO_LINK_SQL,
        CREATE_CURATION_SQL,
        CREATE_LAYOUT_SQL,
        CREATE_SYNTHESIS_SQL,
        CREATE_OLS4_CACHE_SQL,
        CREATE_MERGE_DECISION_SQL,
        CREATE_ONTOLOGY_MAPPING_SQL,
        CREATE_KE_RELATION_SQL,
        CREATE_WORKFLOW_STATE_SQL,
        CREATE_APPROVAL_LOG_SQL,
        CREATE_SYNTHESIS_HISTORY_SQL,
        CREATE_KE_ROLE_SQL,
        CREATE_PAPER_CHUNK_SQL,
        CREATE_PAPER_ABBREV_SQL,
        CREATE_RECORD_HISTORY_SQL,
        CREATE_KE_DIRECTION_SQL,
        CREATE_AOP_SNAPSHOT_SQL,
        CREATE_LAYOUT_OFFSET_SQL,
        CREATE_PAPER_OUTCOME_SQL,
        # Bibliographic cache. Deliberately absent from `_ALL_TABLES`: it holds
        # nothing this tool produced, only what Crossref said about a DOI, so a
        # schema reset has no reason to throw it away and make the next session
        # re-fetch every paper.
        citations.CREATE_PAPER_CITATION_SQL,
    ):
        conn.execute(ddl)
    _add_missing_columns(conn, "ker_synthesis", _SYNTHESIS_V7_COLUMNS)
    _add_missing_columns(conn, "ker_synthesis", _SYNTHESIS_V12_COLUMNS)
    _add_missing_columns(conn, "extraction_runs", _RUNS_PROVENANCE_COLUMNS)
    _add_missing_columns(conn, "table1_extractions", _TABLE1_V8_COLUMNS)
    _add_missing_columns(conn, "table1_extractions", _TABLE1_V9_COLUMNS)
    _add_missing_columns(conn, "table1_extractions", _TABLE1_V10_COLUMNS)
    _add_missing_columns(conn, "ke_canonical", _KE_CANONICAL_V10_COLUMNS)
    _add_missing_columns(conn, "ke_alias", _KE_ALIAS_V11_COLUMNS)
    for index_sql in _INDEXES:
        conn.execute(index_sql)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def get_setting(key: str, default: str = "") -> str:
    """Read one small persistent preference out of `schema_meta`."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = ?", (str(key),)
            ).fetchone()
    except sqlite3.Error:
        return default
    return str(row[0]) if row and row[0] is not None else default


def set_setting(key: str, value: str) -> None:
    """Persist one small preference. Used for things a user should type once."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            (str(key), str(value)),
        )
        conn.commit()


def get_schema_version(conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    own = conn is None
    conn = conn or _connect()
    try:
        conn.execute(CREATE_SCHEMA_META_SQL)
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError):
        return None
    finally:
        if own:
            conn.close()


def init_db(*, reset_on_version_mismatch: bool = True) -> dict[str, Any]:
    """
    Ensure the database exists and matches `SCHEMA_VERSION`.

    The v1 schema had no provenance, no canonical KEs and no curation state, so
    its rows cannot be upgraded in place. When a v1 database is found it is
    rebuilt from scratch and the caller is told, so the UI can say so plainly
    rather than failing with a confusing missing-column error.

    A version listed in `_IN_PLACE_UPGRADES` is different: it is missing tables
    and columns but nothing it already holds has changed meaning, so it is
    migrated rather than dropped. Re-extracting a paper costs a model call and
    the curator's time, and losing that to a schema bump that only added
    workflow bookkeeping would be gratuitous.

    Returns a dict describing what happened:
        {"created": bool, "reset": bool, "migrated": bool,
         "previous_version": int | None}
    """
    db_path = current_db_path()
    existed = db_path.exists()
    previous = get_schema_version() if existed else None
    was_reset = False
    was_migrated = False
    backup_path: Optional[Path] = None

    # A database NEWER than the code is not a stale database. It is this
    # database, opened by an older copy of the program — a second app window,
    # a session still holding modules imported before an upgrade, a colleague
    # on last month's checkout. Treating that as a version mismatch and
    # dropping every table destroys current data to satisfy old code, which is
    # exactly backwards. It has happened once; it does not get to happen twice.
    if existed and previous is not None and previous > SCHEMA_VERSION:
        raise SchemaTooNewError(
            f"{db_path} was written by a newer version of this tool "
            f"(database schema v{previous}, this code understands v{SCHEMA_VERSION}). "
            f"Refusing to touch it. Close any other running copy of the app and "
            f"restart this one, or update the code."
        )

    with _connect() as conn:
        stale = existed and previous != SCHEMA_VERSION
        if stale and previous in _IN_PLACE_UPGRADES:
            was_migrated = True
        elif stale and reset_on_version_mismatch:
            # Never drop irreplaceable rows without leaving a copy behind.
            # Re-extracting a whole corpus costs money and an afternoon; a
            # file copy costs nothing and is the difference between a bad
            # moment and a lost corpus.
            backup_path = _backup_database(previous)
            for table in _ALL_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            was_reset = True
        _create_all(conn)
        if was_migrated:
            _backfill_workflow_state(conn)
        conn.commit()

    # Keep the side caches pointed at the same file.
    try:
        from stage2_extraction import ols4_client

        ols4_client.set_db_path(db_path)
    except Exception:
        pass

    for module_name in ("ke_synonyms", "gene_registry"):
        try:
            import importlib

            module = importlib.import_module(f"stage2_extraction.{module_name}")
            module.set_db_path(db_path)
            module.init_cache()
        except Exception:
            pass

    return {
        "created": not existed,
        "reset": was_reset,
        "migrated": was_migrated,
        "previous_version": previous,
        "backup_path": str(backup_path) if backup_path else None,
    }


def _backup_database(previous_version: Optional[int]) -> Optional[Path]:
    """Copy the database aside before anything destructive happens to it."""
    import shutil

    source = current_db_path()
    if not source.exists():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = source.with_name(
        f"{source.stem}.backup-v{previous_version or 'unknown'}-{stamp}{source.suffix}"
    )
    try:
        shutil.copy2(source, target)
        return target
    except OSError:
        return None


def _backfill_workflow_state(conn: sqlite3.Connection) -> None:
    """
    Give every pre-existing Key Event a workflow state.

    Everything from before v7 starts at `raw`, including Key Events whose
    `curation_status` said "accepted". That looks harsh, but the old accept
    button meant "this name is fine", not "this is approved for synthesis" —
    the second decision was never asked for and cannot be inferred. Starting
    at raw makes the curator answer it once, explicitly.
    """
    now = _now()
    rows = conn.execute("SELECT canonical_id FROM ke_canonical").fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO workflow_state "
        "(target_type, target_key, state, updated_at) VALUES ('ke', ?, 'raw', ?)",
        [(str(r[0]), now) for r in rows],
    )


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def _runs_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(extraction_runs)")}


def start_run(manifest: RunManifest) -> int:
    """
    Persist a manifest at the start of a run and return its `run_id`.

    Written before any extraction happens, so a run that crashes half way
    still leaves a record saying what was attempted and under which settings.
    """
    manifest.schema_version = SCHEMA_VERSION
    with _connect() as conn:
        allowed = _runs_columns(conn)
        row = {k: v for k, v in manifest.as_row().items() if k in allowed}
        cols = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        cur = conn.execute(
            f"INSERT INTO extraction_runs ({cols}) VALUES ({placeholders})", row
        )
        conn.commit()
        run_id = int(cur.lastrowid)
    manifest.run_id = run_id
    return run_id


def start_manual_run(curator: str = "", note: str = "") -> int:
    """
    Open a run row for claims a person is about to enter by hand.

    Manual rows need a `run_id` like any other, or they fall out of every
    join that groups Table 1 by run. But the manifest fields describe a model
    call — provider, model, temperature, prompt fingerprint — and none of them
    apply. They are left null on purpose: a null provider beside `stage =
    'manual'` is a truthful description of a row nobody's model produced,
    where 'manual'/'n-a' would be a fabricated one.

    One run per curator per day, reused on subsequent entries, so a working
    session reads as one event in the run list rather than forty.
    """
    today = datetime.date.today().isoformat()
    tag = f"manual:{curator or 'unattributed'}:{today}"

    with _connect() as conn:
        existing = conn.execute(
            "SELECT run_id FROM extraction_runs WHERE stage = 'manual' "
            "AND target_upstream = ? ORDER BY run_id DESC LIMIT 1",
            (tag,),
        ).fetchone()
        if existing is not None:
            return int(existing[0])

        cur = conn.execute(
            "INSERT INTO extraction_runs "
            "(stage, mode, target_upstream, target_downstream, started_at, "
            " finished_at, status, code_version, schema_version) "
            "VALUES ('manual', 'manual', ?, ?, ?, ?, 'complete', ?, ?)",
            (tag, note or None, _now(), _now(), _code_version(), SCHEMA_VERSION),
        )
        conn.commit()
        return int(cur.lastrowid)


def _code_version() -> str:
    try:
        from run_manifest import code_version

        return str(code_version() or "")
    except Exception:
        return ""


def finish_run(
    run_id: int,
    telemetry: Optional[RunTelemetry] = None,
    *,
    status: str = "completed",
    model_reported: Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Close a run, writing back the robustness counters gathered during it.

    `fields` carries anything only knowable once the run has started — the
    chunk scorer that ended up being used, for instance, since LLM triage can
    fall back to the heuristic scorer mid-run. Unknown keys are ignored rather
    than raising, so an older database stays writable.
    """
    updates: dict[str, Any] = {
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status,
    }
    if model_reported:
        updates["model_reported"] = model_reported
    if telemetry is not None:
        updates.update(telemetry.as_row())
    updates.update({k: v for k, v in fields.items() if v is not None})

    with _connect() as conn:
        allowed = _runs_columns(conn)
        updates = {k: v for k, v in updates.items() if k in allowed}
        assignments = ", ".join(f"{k} = :{k}" for k in updates)
        conn.execute(
            f"UPDATE extraction_runs SET {assignments} WHERE run_id = :run_id",
            {**updates, "run_id": int(run_id)},
        )
        conn.commit()


def close_orphaned_runs() -> int:
    """
    Mark runs still flagged `running` as interrupted, and report how many.

    A run is opened before any work starts and closed only when the loop ends
    normally. Anything in between — a browser refresh, a rerun triggered by a
    stray click, the process being stopped — leaves the row saying "running"
    forever, and a stale "running" row is indistinguishable from work in
    progress. Since Streamlit runs one script at a time, any such row found at
    startup belongs to a run that is definitively over.
    """
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE extraction_runs SET status = 'interrupted', finished_at = ? "
            "WHERE status = 'running'",
            (datetime.datetime.now().isoformat(timespec="seconds"),),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def load_runs(stage: Optional[str] = None) -> pd.DataFrame:
    """All runs, newest first."""
    with _connect() as conn:
        if stage:
            return pd.read_sql_query(
                "SELECT * FROM extraction_runs WHERE stage = ? ORDER BY run_id DESC",
                conn,
                params=(stage,),
            )
        return pd.read_sql_query(
            "SELECT * FROM extraction_runs ORDER BY run_id DESC", conn
        )


def get_run(run_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM extraction_runs WHERE run_id = ?", (int(run_id),)
        ).fetchone()
    return dict(row) if row else None


def latest_run_id(stage: str = "extraction") -> Optional[int]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT run_id FROM extraction_runs WHERE stage = ? "
            "ORDER BY run_id DESC LIMIT 1",
            (stage,),
        ).fetchone()
    return int(row[0]) if row else None


# ---------------------------------------------------------------------------
# KER syntheses
# ---------------------------------------------------------------------------

def save_synthesis(synthesis: Any, run_id: Optional[int] = None) -> None:
    """Store (or replace) the consolidated assessment for one KER."""
    row = {
        "ker_key": synthesis.ker_key,
        "ker_name": synthesis.ker_name,
        "n_papers": int(synthesis.n_papers or 0),
        "n_rows": int(getattr(synthesis, "n_rows", 0) or 0),
        "record_ids": json.dumps(sorted(synthesis.record_ids or [])),
        "mechanistic_basis": synthesis.mechanistic_basis,
        "biological_plausibility": synthesis.biological_plausibility,
        "biological_plausibility_rating": synthesis.biological_plausibility_rating,
        "empirical_evidence": synthesis.empirical_evidence,
        "empirical_evidence_rating": synthesis.empirical_evidence_rating,
        "essentiality": synthesis.essentiality,
        "essentiality_rating": synthesis.essentiality_rating,
        "quantitative_understanding": synthesis.quantitative_understanding,
        "applicability_domain": synthesis.applicability_domain,
        "uncertainties": synthesis.uncertainties,
        "overall_confidence": synthesis.overall_confidence,
        "generated_at": synthesis.generated_at,
        "model": synthesis.model,
        "run_id": int(run_id) if run_id is not None else None,
    }
    row = {k: strip_control_chars(v) for k, v in row.items()}
    cols = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    with _connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO ker_synthesis ({cols}) VALUES ({placeholders})",
            row,
        )
        conn.commit()


def load_synthesis(ker_key: str) -> Optional[dict[str, Any]]:
    """
    Fetch a stored synthesis, with `record_ids` decoded.

    The caller compares those ids against the rows currently contributing to
    the edge; a mismatch means new evidence has arrived since it was written.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ker_synthesis WHERE ker_key = ?", (str(ker_key),)
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["record_ids"] = json.loads(data.get("record_ids") or "[]")
    except (ValueError, TypeError):
        data["record_ids"] = []
    return data


def load_all_syntheses() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM ker_synthesis ORDER BY generated_at DESC", conn
        )


def delete_synthesis(ker_key: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM ker_synthesis WHERE ker_key = ?", (str(ker_key),))
        conn.commit()


# ---------------------------------------------------------------------------
# Table 1 writes
# ---------------------------------------------------------------------------

def insert_table1_row(
    extraction: KERExtraction,
    source_doi: str,
    wiki_ids: dict,
    *,
    source_filename: Optional[str] = None,
    source_title: Optional[str] = None,
    run_id: Optional[int] = None,
    origin: str = LLM_ORIGIN,
    entered_by: Optional[str] = None,
    entry_rationale: Optional[str] = None,
) -> int:
    """
    Insert one KERExtraction plus its evidence spans.

    `origin` defaults to 'llm' because that is what every caller inside the
    extraction pipeline is. Manual entry passes 'curator' and, with it, who
    entered the row and why — see `manual_entry.save_manual_claim`, which is
    the only thing that should be calling this with a curator origin.

    Returns the new record_id.
    """
    today = datetime.date.today().isoformat()
    spans = list(getattr(extraction, "evidence_spans", None) or [])

    row = {
        "source_doi": source_doi,
        "source_filename": source_filename,
        "source_title": source_title,
        "extraction_date": today,
        "run_id": int(run_id) if run_id is not None else None,
        "aop_id": wiki_ids.get("aop_id"),
        "aop_status": wiki_ids.get("aop_status"),
        "upstream_ke_id": wiki_ids.get("upstream_ke_id"),
        "downstream_ke_id": wiki_ids.get("downstream_ke_id"),
        "ker_id": wiki_ids.get("ker_id"),
        "upstream_ke_name": extraction.upstream_ke_name,
        "upstream_ke_level": extraction.upstream_ke_level,
        "downstream_ke_name": extraction.downstream_ke_name,
        "downstream_ke_level": extraction.downstream_ke_level,
        "ker_name": extraction.ker_name,
        "ker_description": extraction.ker_description,
        "ker_adjacency": extraction.ker_adjacency,
        "paper_type": extraction.paper_type,
        "cited_evidence_dois": extraction.cited_evidence_dois,
        "biological_plausibility": extraction.biological_plausibility,
        "empirical_evidence_summary": extraction.empirical_evidence_summary,
        "essentiality_evidence": extraction.essentiality_evidence,
        "contradicts_ker": int(extraction.contradicts_ker),
        "taxonomic_applicability": extraction.taxonomic_applicability,
        "sex_applicability": extraction.sex_applicability,
        "life_stage_applicability": extraction.life_stage_applicability,
        "modulating_factors": extraction.modulating_factors,
        "quantitative_relationships": extraction.quantitative_relationships,
        "response_response_relationship": extraction.response_response_relationship,
        "time_scale": extraction.time_scale,
        "feedforward_feedback_loops": extraction.feedforward_feedback_loops,
        "study_design": extraction.study_design,
        "exposure_route": extraction.exposure_route,
        "chemical_stressor": extraction.chemical_stressor,
        "extraction_confidence": extraction.extraction_confidence,
        "upstream_ke_canonical_id": None,
        "downstream_ke_canonical_id": None,
        "n_evidence_spans": len(spans),
        "n_verified_spans": sum(1 for s in spans if s.verified),
        "direction": getattr(extraction, "direction", None) or "unclear",
        "upstream_change": getattr(extraction, "upstream_change", None),
        "downstream_change": getattr(extraction, "downstream_change", None),
        "upstream_cell_type": getattr(extraction, "upstream_cell_type", None),
        "downstream_cell_type": getattr(extraction, "downstream_cell_type", None),
        "relation_kind": getattr(extraction, "relation_kind", None) or "causal",
        "evidence_type": getattr(extraction, "evidence_type", None) or "not_stated",
        "measured_as": getattr(extraction, "measured_as", None),
        "null_findings": getattr(extraction, "null_findings", None),
        "study_context": getattr(extraction, "study_context", None),
        "upstream_target": getattr(extraction, "upstream_target", None),
        "downstream_target": getattr(extraction, "downstream_target", None),
        "origin": str(origin or LLM_ORIGIN),
        "entered_by": entered_by,
        "entry_rationale": entry_rationale,
        "entered_at": _now() if origin and origin != LLM_ORIGIN else None,
    }

    # Scrub control characters before they are stored. A NUL that arrived from
    # a badly encoded PDF is invisible in the UI and only announces itself much
    # later, when an export fails — so it does not get to enter the database.
    row = {k: strip_control_chars(v) for k, v in row.items()}

    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    sql = f"INSERT INTO table1_extractions ({cols}) VALUES ({placeholders})"

    with _connect() as conn:
        cur = conn.execute(sql, row)
        record_id = cur.lastrowid
        if spans:
            _insert_spans(
                conn, record_id, spans, source_doi, source_filename, run_id=run_id
            )
        conn.commit()
        return record_id


def _insert_spans(
    conn: sqlite3.Connection,
    record_id: int,
    spans: Sequence[EvidenceSpan],
    source_doi: Optional[str],
    source_filename: Optional[str],
    *,
    run_id: Optional[int] = None,
) -> None:
    conn.executemany(
        """
        INSERT INTO evidence_spans
            (record_id, run_id, quote, field, section, section_kind, page_start,
             page_end, chunk_id, char_start, char_end, verified, match_ratio,
             source_doi, source_filename)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record_id,
                int(run_id) if run_id is not None else None,
                strip_control_chars(span.quote),
                span.field,
                strip_control_chars(span.section),
                span.section_kind,
                span.page_start,
                span.page_end,
                span.chunk_id,
                span.char_start,
                span.char_end,
                int(bool(span.verified)),
                float(span.match_ratio or 0.0),
                span.source_doi or source_doi,
                span.source_filename or source_filename,
            )
            for span in spans
        ],
    )


# ---------------------------------------------------------------------------
# Table 1 reads
# ---------------------------------------------------------------------------

def store_chunks(
    chunks: Sequence[Any],
    *,
    source_doi: Optional[str],
    source_filename: Optional[str],
    run_id: Optional[int] = None,
) -> int:
    """
    Keep the text of one paper after its run finishes.

    Returns the number of chunks written. Re-running the same paper in the
    same run replaces rather than duplicates, so a retry does not double the
    corpus.
    """
    if not chunks:
        return 0

    now = _now()
    rows = [
        {
            "run_id": int(run_id) if run_id is not None else None,
            "source_doi": source_doi,
            "source_filename": source_filename,
            "chunk_id": str(getattr(c, "chunk_id", "") or ""),
            "text": strip_control_chars(getattr(c, "text", "") or ""),
            "section": getattr(c, "section", None),
            "section_kind": getattr(c, "section_kind", None),
            "page_start": int(getattr(c, "page_start", 0) or 0),
            "page_end": int(getattr(c, "page_end", 0) or 0),
            "char_start": int(getattr(c, "char_start", 0) or 0),
            "char_end": int(getattr(c, "char_end", 0) or 0),
            "relevance_score": float(getattr(c, "relevance_score", 0.0) or 0.0),
            "selected": int(bool(getattr(c, "selected", False))),
            "created_at": now,
        }
        for c in chunks
    ]

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO paper_chunk "
            "(run_id, source_doi, source_filename, chunk_id, text, section, "
            " section_kind, page_start, page_end, char_start, char_end, "
            " relevance_score, selected, created_at) "
            "VALUES (:run_id, :source_doi, :source_filename, :chunk_id, :text, "
            " :section, :section_kind, :page_start, :page_end, :char_start, "
            " :char_end, :relevance_score, :selected, :created_at)",
            rows,
        )
        conn.commit()
    return len(rows)


def store_paper_abbreviations(
    abbreviations: dict[str, str],
    *,
    source_doi: Optional[str],
    source_filename: Optional[str] = None,
    run_id: Optional[int] = None,
) -> int:
    """
    Keep the abbreviations one paper defined for itself.

    Re-running the same paper replaces its definitions rather than adding to
    them: a second extraction of the same DOI is a better reading of the same
    document, not a second document.
    """
    if not abbreviations or not (source_doi or "").strip():
        return 0

    doi = str(source_doi).strip()
    now = _now()
    rows = [
        {
            "source_doi": doi,
            "source_filename": source_filename,
            "run_id": int(run_id) if run_id is not None else None,
            "abbrev": strip_control_chars(str(abbrev or "").strip()),
            "long_form": strip_control_chars(str(long_form or "").strip()),
            "created_at": now,
        }
        for abbrev, long_form in abbreviations.items()
        if str(abbrev or "").strip() and str(long_form or "").strip()
    ]
    if not rows:
        return 0

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO paper_abbrev "
            "(source_doi, source_filename, run_id, abbrev, long_form, created_at) "
            "VALUES (:source_doi, :source_filename, :run_id, :abbrev, :long_form, "
            " :created_at)",
            rows,
        )
        conn.commit()
    return len(rows)


def load_paper_abbreviations(
    source_doi: Optional[str] = None,
) -> dict[str, dict[str, str]]:
    """
    Every paper's abbreviations, as {doi: {abbrev: long form}}.

    Returns the whole corpus by default, because the caller that needs this —
    normalisation — works across all papers at once and would otherwise make
    one query per DOI.
    """
    sql = "SELECT source_doi, abbrev, long_form FROM paper_abbrev"
    params: list[Any] = []
    if source_doi and str(source_doi).strip():
        sql += " WHERE source_doi = ?"
        params.append(str(source_doi).strip())

    out: dict[str, dict[str, str]] = {}
    try:
        with _connect() as conn:
            for doi, abbrev, long_form in conn.execute(sql, params):
                out.setdefault(str(doi), {})[str(abbrev)] = str(long_form)
    except sqlite3.Error:
        # An older database has no such table. Callers treat an empty map as
        # "no paper-derived expansions", which is the pre-existing behaviour.
        return {}
    return out


def load_chunks(
    source_doi: Optional[str] = None,
    source_filename: Optional[str] = None,
    *,
    contains: Optional[str] = None,
    section_kinds: Optional[Sequence[str]] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Retrieve stored paper text.

    `contains` does a plain case-insensitive substring match. That is enough
    for the question this exists to answer — "which papers say anything about
    myelination downstream of this event" — and avoids standing up a search
    index for a corpus of this size.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if source_doi:
        clauses.append("source_doi = ?")
        params.append(source_doi)
    if source_filename:
        clauses.append("source_filename = ?")
        params.append(source_filename)
    if contains:
        clauses.append("LOWER(text) LIKE ?")
        params.append(f"%{contains.lower()}%")
    if section_kinds:
        placeholders = ",".join("?" * len(section_kinds))
        clauses.append(f"section_kind IN ({placeholders})")
        params.extend(list(section_kinds))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT source_doi, source_filename, chunk_id, section, section_kind, "
        "       page_start, page_end, relevance_score, selected, text "
        f"FROM paper_chunk {where} "
        "ORDER BY source_filename, chunk_id LIMIT ?"
    )
    params.append(int(limit))

    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def chunk_coverage() -> pd.DataFrame:
    """Which papers have their text stored, and how much of it."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT source_filename, source_doi, COUNT(*) AS n_chunks, "
            "       SUM(LENGTH(text)) AS n_chars, "
            "       SUM(selected) AS n_selected "
            "FROM paper_chunk GROUP BY source_filename, source_doi "
            "ORDER BY source_filename",
            conn,
        )


def load_table1_as_dataframe() -> pd.DataFrame:
    """Load all Table 1 rows as a pandas DataFrame."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM table1_extractions ORDER BY record_id", conn
        )


def load_evidence_spans(record_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """
    Load evidence spans, optionally restricted to a set of record ids.

    Returned columns match `evidence_spans` plus a convenience `citation`
    column formatted as "DOI — Section, p. N".
    """
    with _connect() as conn:
        if record_ids is None:
            df = pd.read_sql_query(
                "SELECT * FROM evidence_spans ORDER BY record_id, span_id", conn
            )
        else:
            ids = [int(r) for r in record_ids]
            if not ids:
                return pd.DataFrame()
            placeholders = ",".join("?" * len(ids))
            df = pd.read_sql_query(
                f"SELECT * FROM evidence_spans WHERE record_id IN ({placeholders}) "
                "ORDER BY record_id, span_id",
                conn,
                params=ids,
            )

    if df.empty:
        return df

    def _cite(row) -> str:
        bits = []
        if row.get("source_doi"):
            bits.append(str(row["source_doi"]))
        if row.get("section"):
            bits.append(str(row["section"]))
        ps, pe = row.get("page_start"), row.get("page_end")
        if pd.notna(ps):
            if pd.notna(pe) and pe != ps:
                bits.append(f"pp. {int(ps)}–{int(pe)}")
            else:
                bits.append(f"p. {int(ps)}")
        return " — ".join(bits) if bits else "location unknown"

    df["citation"] = df.apply(_cite, axis=1)
    df["verified"] = df["verified"].astype(bool)
    return df


def load_record(record_id: int) -> Optional[dict[str, Any]]:
    """One Table 1 row as a plain dict, or None if it is gone."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM table1_extractions WHERE record_id = ?", (int(record_id),)
        ).fetchone()
    return dict(row) if row is not None else None


def _archive_record(
    conn: sqlite3.Connection,
    record_id: int,
    *,
    action: str,
    reason: Optional[str],
    curator: Optional[str],
) -> None:
    """Copy a row and its spans into `record_history` before changing it."""
    row = conn.execute(
        "SELECT * FROM table1_extractions WHERE record_id = ?", (int(record_id),)
    ).fetchone()
    if row is None:
        return
    spans = [
        dict(s)
        for s in conn.execute(
            "SELECT * FROM evidence_spans WHERE record_id = ?", (int(record_id),)
        ).fetchall()
    ]
    conn.execute(
        "INSERT INTO record_history "
        "(record_id, payload, spans, action, reason, curator, archived_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            int(record_id),
            json.dumps(dict(row), default=str),
            json.dumps(spans, default=str),
            action,
            reason,
            curator,
            _now(),
        ),
    )


#: Fields a curator may change on an existing row. Deliberately excludes
#: `source_doi`, `run_id` and the canonical ids: the first two say where the
#: row came from and editing them would let a claim be reattributed to a paper
#: that never made it, and the third is normalization's to assign.
EDITABLE_FIELDS = (
    "upstream_ke_name", "upstream_ke_level",
    "downstream_ke_name", "downstream_ke_level",
    "ker_name", "ker_description", "ker_adjacency",
    "direction", "relation_kind", "evidence_type",
    "upstream_change", "downstream_change",
    "upstream_cell_type", "downstream_cell_type",
    "upstream_target", "downstream_target",
    "paper_type", "study_design", "study_context",
    "biological_plausibility", "empirical_evidence_summary",
    "essentiality_evidence", "null_findings", "measured_as",
    "contradicts_ker",
    "taxonomic_applicability", "sex_applicability", "life_stage_applicability",
    "modulating_factors", "quantitative_relationships",
    "response_response_relationship", "time_scale",
    "feedforward_feedback_loops", "exposure_route", "chemical_stressor",
    "extraction_confidence",
)


def update_table1_row(
    record_id: int,
    changes: dict[str, Any],
    *,
    curator: str = "",
    rationale: str = "",
) -> dict[str, Any]:
    """
    Change fields on an existing row, archiving what was there first.

    An edited model row becomes `curator_edited` rather than staying `llm`.
    That matters more than it looks: a row that reads as a model extraction
    but has been corrected by hand is the one state in which the provenance
    drawer would be actively lying about where the claim came from.

    Returns {"changed": [field names], "origin": new origin}. Fields not in
    `EDITABLE_FIELDS` are ignored rather than raising, so a form that grows a
    field cannot quietly start writing somewhere it should not.
    """
    record_id = int(record_id)
    with _connect() as conn:
        current = conn.execute(
            "SELECT * FROM table1_extractions WHERE record_id = ?", (record_id,)
        ).fetchone()
        if current is None:
            raise ValueError(f"no Table 1 row with record_id {record_id}")
        current = dict(current)

        clean = {
            k: strip_control_chars(v)
            for k, v in changes.items()
            if k in EDITABLE_FIELDS
        }
        # Normalise the one boolean before comparing, or False and 0 read as a
        # change every time the form is saved.
        if "contradicts_ker" in clean:
            clean["contradicts_ker"] = int(bool(clean["contradicts_ker"]))

        changed = [k for k, v in clean.items() if current.get(k) != v]
        if not changed:
            return {"changed": [], "origin": current.get("origin", LLM_ORIGIN)}

        _archive_record(
            conn, record_id, action="edited", reason=rationale, curator=curator
        )

        origin = current.get("origin", LLM_ORIGIN)
        new_origin = "curator" if origin == "curator" else "curator_edited"

        assignments = ", ".join(f"{k} = :{k}" for k in changed)
        params = {k: clean[k] for k in changed}
        params.update(
            record_id=record_id,
            origin=new_origin,
            entered_by=curator or current.get("entered_by"),
            entry_rationale=rationale or current.get("entry_rationale"),
            entered_at=_now(),
        )
        conn.execute(
            f"UPDATE table1_extractions SET {assignments}, origin = :origin, "
            "entered_by = :entered_by, entry_rationale = :entry_rationale, "
            "entered_at = :entered_at WHERE record_id = :record_id",
            params,
        )
        conn.commit()

    return {"changed": changed, "origin": new_origin}


def delete_record(
    record_id: int, *, curator: str = "", reason: str = ""
) -> None:
    """
    Delete a single Table 1 row and its evidence spans.

    The row is archived to `record_history` first. Deleting a claim is a
    curation decision like any other, and an AOP that shrank between two
    exports should be able to say which claim went and who removed it.
    """
    with _connect() as conn:
        _archive_record(
            conn, int(record_id), action="deleted", reason=reason, curator=curator
        )
        conn.execute("DELETE FROM evidence_spans WHERE record_id = ?", (record_id,))
        conn.execute("DELETE FROM table1_extractions WHERE record_id = ?", (record_id,))
        conn.commit()


def load_record_history(record_id: Optional[int] = None) -> pd.DataFrame:
    """Superseded and deleted versions of Table 1 rows, newest first."""
    sql = (
        "SELECT history_id, record_id, action, reason, curator, archived_at, "
        "       payload, spans FROM record_history "
    )
    params: tuple = ()
    if record_id is not None:
        sql += "WHERE record_id = ? "
        params = (int(record_id),)
    sql += "ORDER BY history_id DESC"
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def clear_all_table1() -> None:
    """Delete all extraction rows, their spans, and derived canonical KEs."""
    with _connect() as conn:
        conn.execute("DELETE FROM evidence_spans")
        conn.execute("DELETE FROM table1_extractions")
        conn.execute("DELETE FROM ke_alias")
        conn.execute("DELETE FROM physio_map_link")
        conn.execute("DELETE FROM ke_canonical")
        conn.commit()


def clear_everything() -> dict[str, int]:
    """
    Empty every table in the database, leaving the schema in place.

    The blunt instrument behind the sidebar's reset button. `clear_all_table1`
    deliberately spares runs, curation decisions and syntheses so a re-run does
    not lose the human judgements attached to it; this one spares nothing, which
    is the point — it exists for the case where the state is wrong and the
    fastest correct move is to start from empty.

    Returns the row counts that were deleted, so the caller can say what went.
    """
    removed: dict[str, int] = {}
    with _connect() as conn:
        for table in _ALL_TABLES:
            try:
                before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                continue  # table absent on an older schema — nothing to clear
            if before:
                removed[table] = int(before)
            conn.execute(f"DELETE FROM {table}")
        # Cached ontology lookups and Key Event synonym sets live in the same
        # file but are not part of _ALL_TABLES, since a schema bump has no
        # reason to discard them. A full reset does.
        for side_cache in ("ols4_cache", "ke_synonym_cache", "hgnc_cache"):
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {side_cache}").fetchone()[0]
                if n:
                    removed[side_cache] = int(n)
                conn.execute(f"DELETE FROM {side_cache}")
            except sqlite3.Error:
                pass
        # Reset AUTOINCREMENT counters so the next run is #1 again, matching
        # the empty state the user just asked for.
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except sqlite3.Error:
            pass
        conn.commit()
        try:
            conn.execute("VACUUM")  # reclaim the file space, not just the rows
        except sqlite3.Error:
            pass  # cosmetic; the data is already gone
    return removed


def list_source_papers() -> pd.DataFrame:
    """One row per ingested paper, with KER and provenance counts."""
    with _connect() as conn:
        return pd.read_sql_query(
            """
            SELECT
                source_doi,
                MAX(source_filename)            AS source_filename,
                MAX(source_title)               AS source_title,
                COUNT(*)                        AS n_kers,
                SUM(n_evidence_spans)           AS n_spans,
                SUM(n_verified_spans)           AS n_verified_spans,
                MAX(extraction_date)            AS last_extraction
            FROM table1_extractions
            GROUP BY source_doi
            ORDER BY last_extraction DESC, source_doi
            """,
            conn,
        )


# ---------------------------------------------------------------------------
# Canonical KEs
# ---------------------------------------------------------------------------

def replace_canonical_kes(
    canonical_kes: Sequence[CanonicalKE],
    label_to_index: dict[str, int],
) -> list[int]:
    """
    Persist a freshly computed set of canonical KEs, replacing any previous set.

    Curation decisions (accept/reject/rename) are keyed by canonical NAME
    rather than by id, so they survive a re-normalisation that renumbers ids.
    Aliases are written for every raw label, and Table 1 rows are back-filled
    with their canonical ids.

    Returns the assigned canonical ids, positionally aligned with the input.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")

    with _connect() as conn:
        # Preserve curation status by name across the rebuild.
        previous_status = {
            row["canonical_name"]: row["curation_status"]
            for row in conn.execute(
                "SELECT canonical_name, curation_status FROM ke_canonical"
            ).fetchall()
        }

        # Curator-asserted Key Events are not derived from raw labels, so
        # re-deriving cannot reproduce them. Deleting the lot and rebuilding
        # would silently drop every event a curator added because no paper
        # named it — which is precisely the case they were added for. They keep
        # their ids too, so their approval state survives.
        asserted = {
            str(row["canonical_name"]).strip().casefold()
            for row in conn.execute(
                "SELECT canonical_name FROM ke_canonical WHERE origin = 'curator'"
            ).fetchall()
        }

        conn.execute(
            "DELETE FROM ke_alias WHERE canonical_id IN "
            "(SELECT canonical_id FROM ke_canonical WHERE origin != 'curator')"
        )
        conn.execute(
            "DELETE FROM physio_map_link WHERE canonical_id IN "
            "(SELECT canonical_id FROM ke_canonical WHERE origin != 'curator')"
        )
        conn.execute("DELETE FROM ke_canonical WHERE origin != 'curator'")

        # A name that now arrives from the evidence is no longer an assertion:
        # the placeholder has been overtaken by the thing it stood in for, and
        # keeping both would put two identical events on the map.
        superseded = [
            ke.canonical_name for ke in canonical_kes
            if str(ke.canonical_name).strip().casefold() in asserted
        ]
        if superseded:
            conn.executemany(
                "DELETE FROM ke_canonical WHERE origin = 'curator' "
                "AND LOWER(TRIM(canonical_name)) = LOWER(TRIM(?))",
                [(name,) for name in superseded],
            )

        assigned: list[int] = []
        for ke in canonical_kes:
            status = previous_status.get(ke.canonical_name, ke.curation_status or "unreviewed")
            cur = conn.execute(
                """
                INSERT INTO ke_canonical
                    (canonical_name, level, ontology_curie, ontology_iri, ontology_label,
                     ontology_source, ontology_score, aopwiki_ke_id, merge_method,
                     curation_status, n_source_rows, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ke.canonical_name,
                    ke.level,
                    ke.ontology_curie,
                    ke.ontology_iri,
                    ke.ontology_label,
                    ke.ontology_source,
                    float(ke.ontology_score or 0.0),
                    ke.aopwiki_ke_id,
                    ke.merge_method,
                    status,
                    int(ke.n_source_rows or 0),
                    now,
                ),
            )
            canonical_id = cur.lastrowid
            assigned.append(canonical_id)
            ke.canonical_id = canonical_id

            # The basis travels with the alias. Writing the grouping without
            # writing its reason is what made "Propose canonical Key Events"
            # unauditable: the result was on screen and the argument for it
            # was not, so a curator could neither check it nor cite it.
            basis_by_label = getattr(ke, "alias_basis", None) or {}
            for alias in ke.aliases:
                basis = basis_by_label.get(alias) or ()
                rule = str(basis[0]) if len(basis) > 0 and basis[0] else None
                detail = str(basis[1]) if len(basis) > 1 and basis[1] else None
                conn.execute(
                    "INSERT OR IGNORE INTO ke_alias "
                    "(canonical_id, raw_label, merge_basis, merge_detail) "
                    "VALUES (?, ?, ?, ?)",
                    (canonical_id, alias, rule, detail),
                )

        # Canonical ids are reassigned on every rebuild, but workflow state is
        # keyed by id. A row left behind by a deleted Key Event does not stay
        # harmlessly orphaned: the next rebuild hands that id to a different
        # event, which then inherits an approval nobody gave it. Clearing them
        # here is the only place that knows the renumbering happened.
        conn.execute(
            "DELETE FROM workflow_state WHERE target_type = 'ke' "
            "AND CAST(target_key AS INTEGER) NOT IN "
            "(SELECT canonical_id FROM ke_canonical)"
        )
        # Relationship keys are built from the same ids, so they rot too.
        conn.execute(
            "DELETE FROM workflow_state WHERE target_type = 'ker' "
            "AND (CAST(SUBSTR(target_key, 1, INSTR(target_key, '->') - 1) AS INTEGER) "
            "     NOT IN (SELECT canonical_id FROM ke_canonical) "
            " OR CAST(SUBSTR(target_key, INSTR(target_key, '->') + 2) AS INTEGER) "
            "     NOT IN (SELECT canonical_id FROM ke_canonical))"
        )

        # Back-fill Table 1 with canonical ids.
        for label, index in label_to_index.items():
            if index >= len(assigned):
                continue
            canonical_id = assigned[index]
            conn.execute(
                "UPDATE table1_extractions SET upstream_ke_canonical_id = ? "
                "WHERE upstream_ke_name = ?",
                (canonical_id, label),
            )
            conn.execute(
                "UPDATE table1_extractions SET downstream_ke_canonical_id = ? "
                "WHERE downstream_ke_name = ?",
                (canonical_id, label),
            )

        conn.commit()
        return assigned


def load_canonical_kes() -> pd.DataFrame:
    """Canonical KEs with their aliases collapsed into a semicolon list."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM ke_canonical ORDER BY canonical_id", conn
        )
        aliases = pd.read_sql_query(
            "SELECT canonical_id, raw_label FROM ke_alias ORDER BY canonical_id, raw_label",
            conn,
        )

    if df.empty:
        return df

    if aliases.empty:
        df["aliases"] = ""
        df["n_aliases"] = 0
        return df

    grouped = (
        aliases.groupby("canonical_id")["raw_label"]
        .agg(lambda s: "; ".join(s))
        .rename("aliases")
    )
    counts = aliases.groupby("canonical_id")["raw_label"].count().rename("n_aliases")
    df = df.merge(grouped, on="canonical_id", how="left").merge(
        counts, on="canonical_id", how="left"
    )
    df["aliases"] = df["aliases"].fillna("")
    df["n_aliases"] = df["n_aliases"].fillna(0).astype(int)
    return df


def canonical_ids_by_name() -> dict[str, int]:
    """Canonical Key Event name -> id, for resolving a name a curator picked."""
    with _connect() as conn:
        return {
            str(row["canonical_name"]): int(row["canonical_id"])
            for row in conn.execute(
                "SELECT canonical_id, canonical_name FROM ke_canonical"
            )
        }


def set_claim_canonical_ends(
    assignments: dict[int, tuple[Optional[int], Optional[int]]]
) -> int:
    """
    Point individual Table 1 rows at the Key Events their curator chose.

    The unit here is the row, not the label, and that is the point. Many
    papers wrote "voltage-gated sodium channels"; one blocked the channel in an
    oligodendrocyte and another activated it in an axon, and those are not the
    same Key Event however identical the wording. `replace_canonical_kes`
    back-fills by label and therefore cannot express that — it gives every row
    sharing a wording the same id — so any per-row decision has to be written
    afterwards, here. `split_by_cell_lineage` already worked this way; this is
    the same mechanism made available to the curation grid.

    `assignments` maps record_id to (upstream_canonical_id,
    downstream_canonical_id). A None leaves that end as it is. Returns the
    number of rows touched.
    """
    if not assignments:
        return 0

    touched = 0
    with _connect() as conn:
        for record_id, (upstream_id, downstream_id) in assignments.items():
            sets, params = [], []
            if upstream_id is not None:
                sets.append("upstream_ke_canonical_id = ?")
                params.append(int(upstream_id))
            if downstream_id is not None:
                sets.append("downstream_ke_canonical_id = ?")
                params.append(int(downstream_id))
            if not sets:
                continue
            params.append(int(record_id))
            cursor = conn.execute(
                f"UPDATE table1_extractions SET {', '.join(sets)} WHERE record_id = ?",
                params,
            )
            touched += cursor.rowcount
        conn.commit()
    return touched


def recount_canonical_source_rows() -> None:
    """
    Recompute `n_source_rows` from the rows actually pointing at each event.

    Needed after any per-row repointing. Counting by label instead would give
    two Key Events split out of one wording the same total — the full count for
    each — and that total is read as evidence weight on the map, so it would
    overstate both.
    """
    with _connect() as conn:
        conn.execute(
            """
            UPDATE ke_canonical SET n_source_rows = (
                SELECT COUNT(*) FROM table1_extractions
                WHERE upstream_ke_canonical_id = ke_canonical.canonical_id
                   OR downstream_ke_canonical_id = ke_canonical.canonical_id
            )
            """
        )
        conn.commit()


def load_alias_crosswalk() -> pd.DataFrame:
    """
    One row per raw label: which Key Event it went to, and why.

    This is the missing account of the step between Table 1 and the canonical
    events. Table 1 is one row per claim in the paper's own words; the
    canonical events are the nodes those claims resolve to. Between them sits
    a decision per label, and until this query existed the only trace of it
    was two counts that did not obviously relate to each other.

    Columns: raw_label, canonical_id, canonical_name, level, merge_basis,
    merge_detail, is_event_name. `is_event_name` marks the label the event is
    named after, so the rest read as its synonyms.
    """
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT a.raw_label        AS raw_label,
                   a.canonical_id     AS canonical_id,
                   c.canonical_name   AS canonical_name,
                   c.level            AS level,
                   c.merge_method     AS merge_method,
                   a.merge_basis      AS merge_basis,
                   a.merge_detail     AS merge_detail
            FROM ke_alias a
            JOIN ke_canonical c ON c.canonical_id = a.canonical_id
            ORDER BY c.canonical_name, a.raw_label
            """,
            conn,
        )

    if df.empty:
        return df

    df["is_event_name"] = (
        df["raw_label"].astype(str).str.strip().str.casefold()
        == df["canonical_name"].astype(str).str.strip().str.casefold()
    )
    return df


def load_alias_map() -> dict[str, int]:
    """Raw label -> canonical_id."""
    with _connect() as conn:
        rows = conn.execute("SELECT raw_label, canonical_id FROM ke_alias").fetchall()
    return {row["raw_label"]: row["canonical_id"] for row in rows}


def record_paper_outcome(
    *,
    run_id: Optional[int],
    source_filename: Optional[str],
    source_doi: Optional[str],
    outcome: str,
    category: str = "unknown",
    reason: str = "",
    n_kers: int = 0,
    n_llm_calls: int = 0,
    n_truncated: int = 0,
    chars_sent: Optional[int] = None,
    chars_total: Optional[int] = None,
) -> None:
    """
    Write what happened to one paper, whether or not it produced anything.

    Called for every paper, including the successes: "which papers yielded
    nothing" is only answerable if the ones that did yield something are on
    the same list.

    A failure here must not cost the extraction it describes, so errors are
    swallowed — but only after one retry with the run link dropped. The
    foreign key on `run_id` rejects a run row that does not exist, and the
    first version of this simply lost the outcome when that happened: the
    diagnostic disappeared in exactly the circumstances where something had
    already gone wrong. An outcome with no run attached is still worth far
    more than no outcome.
    """
    values = (
        source_filename, source_doi, outcome,
        category if category in OUTCOME_CATEGORIES else "unknown",
        strip_control_chars(reason or "")[:1000],
        int(n_kers), int(n_llm_calls), int(n_truncated),
        chars_sent, chars_total, _now(),
    )
    sql = (
        "INSERT INTO paper_outcome (run_id, source_filename, source_doi, "
        "outcome, category, reason, n_kers, n_llm_calls, n_truncated, "
        "chars_sent, chars_total, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for attempt_run_id in (run_id, None):
        try:
            with _connect() as conn:
                conn.execute(sql, (attempt_run_id, *values))
                conn.commit()
            return
        except sqlite3.Error:
            if attempt_run_id is None:
                return
            continue


def load_paper_outcomes(run_id: Optional[int] = None) -> pd.DataFrame:
    """Every paper's outcome, most recent run first unless one is named."""
    sql = "SELECT * FROM paper_outcome"
    params: list[Any] = []
    if run_id is not None:
        sql += " WHERE run_id = ?"
        params.append(int(run_id))
    sql += " ORDER BY outcome_id"
    try:
        with _connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def corpus_counts() -> dict[str, int]:
    """
    The chain of counts, in the units they are actually in.

    These numbers get compared to each other constantly and they are not
    commensurable: 28 extracted claims becoming 18 canonical Key Events is not
    a loss of ten anything, because a claim is a relationship between two
    events and an event is an event. Reporting them separately, with the unit
    named, is the whole point.
    """
    out = {
        "papers": 0, "claims": 0, "label_occurrences": 0, "distinct_labels": 0,
        "canonical_kes": 0, "relationships": 0, "aliases": 0,
    }
    try:
        with _connect() as conn:
            def one(sql: str) -> int:
                try:
                    return int(conn.execute(sql).fetchone()[0] or 0)
                except sqlite3.Error:
                    return 0

            out["papers"] = one(
                "SELECT COUNT(DISTINCT source_doi) FROM table1_extractions")
            out["claims"] = one("SELECT COUNT(*) FROM table1_extractions")
            # Two per row, because every claim names an upstream and a
            # downstream event. This is the number the sidebar metric used to
            # show as "Raw Key Event labels", which is why 28 rows appeared to
            # produce 56 labels.
            out["label_occurrences"] = one(
                "SELECT COUNT(*) FROM ("
                " SELECT upstream_ke_name FROM table1_extractions"
                " UNION ALL SELECT downstream_ke_name FROM table1_extractions)")
            out["distinct_labels"] = one(
                "SELECT COUNT(*) FROM ("
                " SELECT upstream_ke_name AS n FROM table1_extractions"
                " UNION SELECT downstream_ke_name FROM table1_extractions)")
            out["canonical_kes"] = one("SELECT COUNT(*) FROM ke_canonical")
            out["aliases"] = one("SELECT COUNT(*) FROM ke_alias")
            out["relationships"] = one(
                "SELECT COUNT(*) FROM ("
                " SELECT DISTINCT upstream_ke_canonical_id, "
                " downstream_ke_canonical_id FROM table1_extractions"
                " WHERE upstream_ke_canonical_id IS NOT NULL"
                " AND downstream_ke_canonical_id IS NOT NULL)")
    except sqlite3.Error:
        pass
    return out


#: What a curator can rule about a Key Event whose claims disagree on sign.
KE_DIRECTIONS = ("increased", "decreased", "conflicted")


def set_ke_direction(
    canonical_id: int,
    direction: str,
    *,
    curator: str = "",
    rationale: str = "",
) -> None:
    """
    Record which way a Key Event moves, where the claims do not agree.

    'conflicted' is a real answer and not a refusal to answer: it says the
    split in the literature is genuine, the map should keep showing it, and
    somebody has looked. Storing it is what turns "±" from an unanswered
    question into an answered one.
    """
    if direction not in KE_DIRECTIONS:
        raise ValueError(f"{direction!r} is not one of {KE_DIRECTIONS}")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO ke_direction "
            "(canonical_id, direction, acknowledged, rationale, curator, updated_at) "
            "VALUES (?, ?, 1, ?, ?, ?) "
            "ON CONFLICT(canonical_id) DO UPDATE SET direction = excluded.direction, "
            "acknowledged = 1, rationale = excluded.rationale, "
            "curator = excluded.curator, updated_at = excluded.updated_at",
            (int(canonical_id), direction, rationale or None,
             curator or None, _now()),
        )
        conn.commit()


def clear_ke_direction(canonical_id: int) -> None:
    """Withdraw a ruling, putting the event back to what the claims say."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM ke_direction WHERE canonical_id = ?", (int(canonical_id),)
        )
        conn.commit()


def load_ke_directions() -> dict[int, dict[str, Any]]:
    """Curator direction rulings, keyed by canonical id."""
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT * FROM ke_direction").fetchall()
    except sqlite3.Error:
        return {}
    return {int(r["canonical_id"]): dict(r) for r in rows}


def create_manual_canonical_ke(
    name: str,
    level: str,
    *,
    curator: str = "",
    rationale: str = "",
) -> int:
    """
    Add a Key Event nothing in the corpus named.

    The narrow case this exists for: an adverse outcome or molecular initiating
    event the developer knows belongs in the pathway, where the papers gathered
    so far only cover the middle of it. Every *other* way of adding a Key Event
    should go through a Table 1 row, because a Key Event with a relationship is
    evidence and this is only an assertion.

    Returns the canonical_id, or the existing one if the name is already taken
    — adding the same event twice is a slip, not an instruction.
    """
    name = strip_control_chars(str(name)).strip()
    if not name:
        raise ValueError("a Key Event needs a name")
    if level not in KE_LEVEL_ORDER:
        raise ValueError(f"{level!r} is not one of {KE_LEVEL_ORDER}")

    with _connect() as conn:
        existing = conn.execute(
            "SELECT canonical_id FROM ke_canonical "
            "WHERE LOWER(TRIM(canonical_name)) = LOWER(TRIM(?))",
            (name,),
        ).fetchone()
        if existing is not None:
            return int(existing[0])

        cur = conn.execute(
            "INSERT INTO ke_canonical "
            "(canonical_name, level, merge_method, curation_status, "
            " n_source_rows, updated_at, origin, created_by, create_rationale) "
            "VALUES (?, ?, 'manual', 'unreviewed', 0, ?, 'curator', ?, ?)",
            (name, level, _now(), curator or None, rationale or None),
        )
        canonical_id = int(cur.lastrowid)
        # No alias row: an alias records a wording some paper used, and no
        # paper used this one.
        conn.commit()
    return canonical_id


def rename_canonical_ke(canonical_id: int, new_name: str) -> None:
    """Curator rename of a canonical KE. Aliases are untouched."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE ke_canonical SET canonical_name = ?, merge_method = 'manual', updated_at = ? "
            "WHERE canonical_id = ?",
            (new_name.strip(), now, canonical_id),
        )
        conn.commit()


def set_canonical_ke_level(canonical_id: int, level: str) -> None:
    """Curator override of a KE's biological level (moves it between lanes)."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE ke_canonical SET level = ?, merge_method = 'manual', updated_at = ? "
            "WHERE canonical_id = ?",
            (level, now, canonical_id),
        )
        conn.commit()


def set_canonical_ke_status(canonical_id: int, status: str) -> None:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE ke_canonical SET curation_status = ?, updated_at = ? WHERE canonical_id = ?",
            (status, now, canonical_id),
        )
        conn.commit()


def merge_canonical_kes(source_id: int, target_id: int) -> None:
    """
    Fold canonical KE `source_id` into `target_id`.

    Aliases move across, Table 1 pointers are repointed, and the source record
    is removed. This is the manual counterpart to automatic normalization.
    """
    if source_id == target_id:
        return
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE OR IGNORE ke_alias SET canonical_id = ? WHERE canonical_id = ?",
            (target_id, source_id),
        )
        conn.execute("DELETE FROM ke_alias WHERE canonical_id = ?", (source_id,))
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
            "UPDATE ke_canonical SET n_source_rows = n_source_rows + "
            "(SELECT COALESCE(n_source_rows, 0) FROM ke_canonical WHERE canonical_id = ?), "
            "merge_method = 'manual', updated_at = ? WHERE canonical_id = ?",
            (source_id, now, target_id),
        )
        conn.execute("DELETE FROM ke_canonical WHERE canonical_id = ?", (source_id,))
        conn.commit()


def split_alias_to_new_ke(raw_label: str, level: str) -> int:
    """
    Pull one raw label out of its canonical group into a brand-new KE.

    The inverse of a merge, for when normalization over-merged.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO ke_canonical
                (canonical_name, level, merge_method, curation_status, n_source_rows, updated_at)
            VALUES (?, ?, 'manual', 'unreviewed', 0, ?)
            """,
            (raw_label, level, now),
        )
        new_id = cur.lastrowid
        conn.execute("DELETE FROM ke_alias WHERE raw_label = ?", (raw_label,))
        conn.execute(
            "INSERT INTO ke_alias (canonical_id, raw_label) VALUES (?, ?)",
            (new_id, raw_label),
        )
        conn.execute(
            "UPDATE table1_extractions SET upstream_ke_canonical_id = ? WHERE upstream_ke_name = ?",
            (new_id, raw_label),
        )
        conn.execute(
            "UPDATE table1_extractions SET downstream_ke_canonical_id = ? WHERE downstream_ke_name = ?",
            (new_id, raw_label),
        )
        conn.commit()
        return new_id


# ---------------------------------------------------------------------------
# Physiological map links
# ---------------------------------------------------------------------------

def replace_physio_links(links: Sequence[Any]) -> None:
    """Replace all physiological-map links with a freshly computed set."""
    with _connect() as conn:
        conn.execute("DELETE FROM physio_map_link")
        conn.executemany(
            """
            INSERT OR IGNORE INTO physio_map_link
                (canonical_id, provider, entity_label, entity_id, url, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(link.canonical_id),
                    link.provider,
                    link.entity_label,
                    link.entity_id,
                    link.url,
                    float(link.confidence or 0.0),
                )
                for link in links
            ],
        )
        conn.commit()


def load_physio_links() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            """
            SELECT p.*, k.canonical_name, k.level
            FROM physio_map_link p
            JOIN ke_canonical k ON k.canonical_id = p.canonical_id
            ORDER BY k.level, k.canonical_name
            """,
            conn,
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def table_counts() -> dict[str, int]:
    """Row counts for every table — used by the diagnostics panel."""
    counts: dict[str, int] = {}
    with _connect() as conn:
        for table in _ALL_TABLES + ("ols4_cache",):
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                counts[table] = 0
    return counts


def connect() -> sqlite3.Connection:
    """Expose a connection for sibling store modules (curation, layout)."""
    return _connect()


__all__ = [
    "DB_PATH",
    "current_db_path",
    "set_db_path",
    "SCHEMA_VERSION",
    "init_db",
    "get_schema_version",
    "connect",
    "start_run",
    "finish_run",
    "close_orphaned_runs",
    "load_runs",
    "get_run",
    "latest_run_id",
    "save_synthesis",
    "load_synthesis",
    "load_all_syntheses",
    "delete_synthesis",
    "insert_table1_row",
    "load_table1_as_dataframe",
    "load_evidence_spans",
    "delete_record",
    "clear_all_table1",
    "clear_everything",
    "list_source_papers",
    "replace_canonical_kes",
    "load_canonical_kes",
    "load_alias_map",
    "load_alias_crosswalk",
    "ALIAS_BASIS_LABELS",
    "rename_canonical_ke",
    "set_canonical_ke_level",
    "set_canonical_ke_status",
    "merge_canonical_kes",
    "split_alias_to_new_ke",
    "replace_physio_links",
    "load_physio_links",
    "table_counts",
]
