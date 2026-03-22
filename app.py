from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import streamlit as st

from export import build_export_dataframe, dataframe_to_csv_bytes
from pubmed_search import search_pubmed
from screening import screen_record
from schemas import PubMedRecord, ScreeningDecision

st.set_page_config(page_title="AOP_RAG Release 1", layout="wide")

st.title("AOP_RAG — Release 1")
st.caption("User-defined PubMed search with Ollama-based title/abstract screening")

with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    year_start = st.number_input("Year start", min_value=1900, max_value=2100, value=2010, step=1)
    year_end = st.number_input("Year end", min_value=1900, max_value=2100, value=2026, step=1)
    max_records = st.number_input("Max records", min_value=1, max_value=500, value=25, step=1)

query = st.text_area("PubMed query", height=100, placeholder="e.g. oxidative stress AND hepatotoxicity")
inclusion_criteria = st.text_area("Optional inclusion criteria", height=120, placeholder="e.g. mammalian studies with mechanistic evidence")
exclusion_criteria = st.text_area("Optional exclusion criteria", height=120, placeholder="e.g. reviews only, non-English, in vitro only")

run = st.button("Search and screen", type="primary")

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if run:
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
                st.warning("No PubMed records found for that query.")
            else:
                screened_pairs: List[Tuple[PubMedRecord, ScreeningDecision]] = []
                progress = st.progress(0, text="Screening records with Ollama...")
                for idx, record in enumerate(records, start=1):
                    decision = screen_record(
                        record=record,
                        query=query,
                        inclusion_criteria=inclusion_criteria,
                        exclusion_criteria=exclusion_criteria,
                        model=ollama_model,
                    )
                    screened_pairs.append((record, decision))
                    progress.progress(idx / len(records), text=f"Screened {idx}/{len(records)} records")
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
    st.dataframe(df, use_container_width=True, height=600)

    decision_counts = df["screening_decision"].value_counts(dropna=False).to_dict()
    col1, col2, col3 = st.columns(3)
    col1.metric("Yes", decision_counts.get("yes", 0))
    col2.metric("Maybe", decision_counts.get("maybe", 0))
    col3.metric("No", decision_counts.get("no", 0))

    csv_bytes = dataframe_to_csv_bytes(df)
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="aop_rag_screening_results.csv",
        mime="text/csv",
    )
else:
    st.info("Run a search to generate screened PubMed results.")
