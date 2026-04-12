from __future__ import annotations

"""
SQLite-backed store for Table 1 (per-paper KER extraction rows).

A single `aop_rag.db` file is created in the working directory.
All columns map 1-to-1 with Table1Row fields in schemas.py.
"""

import datetime
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from schemas import KERExtraction, Table1Row

DB_PATH = Path("aop_rag.db")

CREATE_TABLE1_SQL = """
CREATE TABLE IF NOT EXISTS table1_extractions (
    record_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doi              TEXT    NOT NULL,
    extraction_date         TEXT    NOT NULL,
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
    extraction_confidence   TEXT    NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.execute(CREATE_TABLE1_SQL)
        conn.commit()


def insert_table1_row(
    extraction: KERExtraction,
    source_doi: str,
    wiki_ids: dict,
) -> int:
    """
    Insert one KERExtraction into table1_extractions.

    Parameters
    ----------
    extraction : KERExtraction
        LLM-extracted KER data.
    source_doi : str
        DOI of the uploaded paper.
    wiki_ids : dict
        Output of aopwiki_client.enrich_ker() containing upstream_ke_id,
        downstream_ke_id, ker_id, aop_id, aop_status.

    Returns
    -------
    int
        The new record_id (SQLite autoincrement).
    """
    today = datetime.date.today().isoformat()

    row = {
        "source_doi": source_doi,
        "extraction_date": today,
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
    }

    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    sql = f"INSERT INTO table1_extractions ({cols}) VALUES ({placeholders})"

    with _connect() as conn:
        cur = conn.execute(sql, row)
        conn.commit()
        return cur.lastrowid


def load_table1_as_dataframe() -> pd.DataFrame:
    """Load all Table 1 rows as a pandas DataFrame."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM table1_extractions ORDER BY record_id", conn
        )


def delete_record(record_id: int) -> None:
    """Delete a single Table 1 row by record_id."""
    with _connect() as conn:
        conn.execute("DELETE FROM table1_extractions WHERE record_id = ?", (record_id,))
        conn.commit()


def clear_all_table1() -> None:
    """Delete all rows — used for testing / reset."""
    with _connect() as conn:
        conn.execute("DELETE FROM table1_extractions")
        conn.commit()
