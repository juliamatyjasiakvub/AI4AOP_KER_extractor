from __future__ import annotations

import io
import os
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from schemas import PubMedRecord, ScreeningDecision
from stage1_search.pubmed_search import search_pubmed
from stage1_search.screening import screen_record
from stage1_search.export import build_export_dataframe, dataframe_to_csv_bytes
from stage2_extraction.pdf_reader import extract_text_from_pdf, truncate_to_token_budget
from stage2_extraction.ker_extractor import extract_kers_from_text, ExtractionError
from stage2_extraction.aopwiki_client import enrich_ker
from stage2_extraction.table1_store import init_db, insert_table1_row, load_table1_as_dataframe, clear_all_table1
from stage2_extraction.table2_synthesis import compute_table2

# ---------------------------------------------------------------------------
# App-wide setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AOP_RAG", layout="wide")
init_db()  # ensure SQLite tables exist

# ---------------------------------------------------------------------------
# Sidebar — shared settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    st.subheader("Stage 1 — Search & screen")
    ollama_model = st.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="Local model for title/abstract screening",
    )
    year_start = st.number_input("Year start", min_value=1900, max_value=2100, value=2010, step=1)
    year_end   = st.number_input("Year end",   min_value=1900, max_value=2100, value=2026, step=1)
    max_records = st.number_input("Max records", min_value=1, max_value=500, value=25, step=1)

    st.divider()

    st.subheader("Stage 2 — KER extraction")
    anthropic_key = st.text_input(
        "Anthropic API key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Used for full-text KER extraction. Set ANTHROPIC_API_KEY env var to avoid re-entering.",
    )

    if st.button("Clear all Table 1 data", type="secondary"):
        clear_all_table1()
        st.success("Table 1 cleared.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Stage 1 — Search & screen", "Stage 2 — KER extraction"])


# ===========================================================================
# TAB 1 — PubMed search + Ollama screening (existing functionality, unchanged)
# ===========================================================================

with tab1:
    st.header("PubMed search & title/abstract screening")
    st.caption("Search PubMed and screen results using a local Ollama model.")

    query = st.text_area(
        "PubMed query", height=80,
        placeholder="e.g. oxidative stress AND hepatotoxicity",
    )
    inclusion_criteria = st.text_area(
        "Inclusion criteria (optional)", height=100,
        placeholder="e.g. mammalian studies with mechanistic evidence",
    )
    exclusion_criteria = st.text_area(
        "Exclusion criteria (optional)", height=100,
        placeholder="e.g. reviews only, non-English, in vitro only",
    )

    run_search = st.button("Search and screen", type="primary", key="run_search")

    if "results_df" not in st.session_state:
        st.session_state.results_df = None

    if run_search:
        if not query.strip():
            st.error("Please enter a PubMed query.")
        else:
            try:
                with st.spinner("Fetching PubMed records..."):
                    records = search_pubmed(
                        query=query,
                        year_start=int(year_start),
                        year_end=int(year_end),
                        max_records=int(max_records),
                    )

                if not records:
                    st.warning("No PubMed records found.")
                else:
                    screened_pairs: List[Tuple[PubMedRecord, ScreeningDecision]] = []
                    progress = st.progress(0, text="Screening with Ollama...")
                    for idx, record in enumerate(records, start=1):
                        decision = screen_record(
                            record=record,
                            query=query,
                            inclusion_criteria=inclusion_criteria,
                            exclusion_criteria=exclusion_criteria,
                            model=ollama_model,
                        )
                        screened_pairs.append((record, decision))
                        progress.progress(idx / len(records), text=f"Screened {idx}/{len(records)}")
                    progress.empty()

                    df = build_export_dataframe(
                        screened_pairs,
                        query=query,
                        inclusion_criteria=inclusion_criteria,
                        exclusion_criteria=exclusion_criteria,
                    )
                    st.session_state.results_df = df
            except Exception as e:
                st.exception(e)

    if st.session_state.results_df is not None:
        df: pd.DataFrame = st.session_state.results_df
        st.subheader("Screening results")

        counts = df["screening_decision"].value_counts(dropna=False).to_dict()
        c1, c2, c3 = st.columns(3)
        c1.metric("Yes", counts.get("yes", 0))
        c2.metric("Maybe", counts.get("maybe", 0))
        c3.metric("No", counts.get("no", 0))

        st.dataframe(df, use_container_width=True, height=500)
        st.download_button(
            "Download CSV",
            data=dataframe_to_csv_bytes(df),
            file_name="aop_rag_screening_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Run a search to see results.")


# ===========================================================================
# TAB 2 — Full-text KER extraction
# ===========================================================================

with tab2:
    st.header("KER extraction from full-text papers")
    st.caption(
        "Upload PDFs of papers that passed Stage 1 screening. "
        "Claude extracts KERs, then AOP-Wiki IDs are looked up automatically."
    )

    # -----------------------------------------------------------------------
    # Upload + extract
    # -----------------------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more full-text papers as PDFs.",
    )

    paper_doi = st.text_input(
        "DOI of the uploaded paper",
        placeholder="10.1016/j.tox.2022.01.001",
        help=(
            "If uploading multiple PDFs in one batch, enter the DOI of the "
            "first paper. You can process papers one at a time for precise DOI tracking."
        ),
    )

    run_extraction = st.button("Extract KERs", type="primary", key="run_extraction")

    if run_extraction:
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        elif not paper_doi.strip():
            st.error("Please enter the DOI of the paper.")
        elif not anthropic_key.strip():
            st.error("Please enter your Anthropic API key in the sidebar.")
        else:
            for uploaded_file in uploaded_files:
                st.markdown(f"**Processing:** `{uploaded_file.name}`")

                # Step 1 — extract text
                with st.spinner("Extracting text from PDF..."):
                    try:
                        raw_text = extract_text_from_pdf(uploaded_file)
                        paper_text = truncate_to_token_budget(raw_text)
                        st.caption(f"Text extracted: {len(raw_text):,} chars → {len(paper_text):,} chars sent to API")
                    except RuntimeError as e:
                        st.error(str(e))
                        continue

                # Step 2 — LLM extraction
                with st.spinner("Sending to Claude for KER extraction..."):
                    try:
                        extractions, warnings = extract_kers_from_text(
                            paper_text=paper_text,
                            api_key=anthropic_key.strip(),
                        )
                    except ExtractionError as e:
                        st.error(str(e))
                        continue

                if warnings:
                    for w in warnings:
                        st.warning(w)

                if not extractions:
                    st.warning(f"No KERs extracted from `{uploaded_file.name}`.")
                    continue

                st.success(f"Extracted {len(extractions)} KER(s). Looking up AOP-Wiki IDs...")

                # Step 3 — AOP-Wiki enrichment + insert
                inserted = 0
                wiki_progress = st.progress(0)
                for i, extraction in enumerate(extractions):
                    with st.spinner(f"AOP-Wiki lookup {i+1}/{len(extractions)}: {extraction.ker_name[:60]}..."):
                        wiki_ids = enrich_ker(
                            upstream_ke_name=extraction.upstream_ke_name,
                            downstream_ke_name=extraction.downstream_ke_name,
                        )
                    insert_table1_row(
                        extraction=extraction,
                        source_doi=paper_doi.strip(),
                        wiki_ids=wiki_ids,
                    )
                    inserted += 1
                    wiki_progress.progress(inserted / len(extractions))

                wiki_progress.empty()
                st.success(f"Saved {inserted} row(s) to Table 1.")

    st.divider()

    # -----------------------------------------------------------------------
    # Table 1 viewer
    # -----------------------------------------------------------------------

    st.subheader("Table 1 — per-paper extraction rows")

    t1_df = load_table1_as_dataframe()

    if t1_df.empty:
        st.info("No rows yet. Extract KERs from a paper above.")
    else:
        # Confidence filter
        conf_filter = st.multiselect(
            "Filter by extraction confidence",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        filtered = t1_df[t1_df["extraction_confidence"].isin(conf_filter)]

        st.dataframe(filtered, use_container_width=True, height=400)
        st.caption(f"{len(filtered)} of {len(t1_df)} rows shown")

        csv_t1 = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Table 1 CSV", csv_t1, "table1_extractions.csv", "text/csv")

    st.divider()

    # -----------------------------------------------------------------------
    # Table 2 viewer — computed on demand
    # -----------------------------------------------------------------------

    st.subheader("Table 2 — KER synthesis (cross-paper)")

    if st.button("Compute Table 2 from current Table 1", key="compute_t2"):
        if t1_df.empty:
            st.warning("Table 1 is empty — add papers first.")
        else:
            t2_df = compute_table2(t1_df)
            st.session_state.table2_df = t2_df

    if "table2_df" in st.session_state and not st.session_state.table2_df.empty:
        t2: pd.DataFrame = st.session_state.table2_df

        # Uncertainty filter
        unc_filter = st.multiselect(
            "Filter by uncertainty level",
            options=["Low", "Moderate", "High"],
            default=["Low", "Moderate", "High"],
            key="unc_filter",
        )
        t2_filtered = t2[t2["uncertainty_level"].isin(unc_filter)]

        st.dataframe(t2_filtered, use_container_width=True, height=400)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique KERs", len(t2))
        c2.metric("Novel KERs", int((t2["aop_status"] == "novel").sum()))
        c3.metric("High uncertainty", int((t2["uncertainty_level"] == "High").sum()))
        c4.metric("Papers processed", int(t2["n_papers_total"].sum()))

        csv_t2 = t2_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Table 2 CSV", csv_t2, "table2_synthesis.csv", "text/csv")

        st.info(
            "Fields marked for human review: **uncertainty_description**, "
            "**biological_plausibility_synthesis**, **review_status**. "
            "Edit these in the downloaded CSV and re-import, or add a review interface in a future release."
        )
