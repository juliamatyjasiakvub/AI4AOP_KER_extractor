from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from export import dataframe_to_csv_bytes, screened_records_to_dataframe
from pubmed_search import PubMedSearchError, fetch_pubmed_records
from screening import ScreeningError, screen_records

st.set_page_config(page_title="AOP_RAG Release 1", layout="wide")

CURRENT_YEAR = datetime.now().year
DEFAULT_MAX_RECORDS = 25
HARD_MAX_RECORDS = 200


def _init_state() -> None:
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "screened_records" not in st.session_state:
        st.session_state["screened_records"] = None


_init_state()

st.title("AOP_RAG — Release 1")
st.caption("User-defined PubMed search with AI-assisted title/abstract screening.")

with st.sidebar:
    st.header("Search settings")
    pubmed_query = st.text_area(
        "PubMed query",
        placeholder="Example: (oxidative stress) AND (liver toxicity)",
        height=120,
    )
    inclusion_criteria = st.text_area(
        "Optional inclusion criteria",
        placeholder="Example: Include mammalian in vivo or in vitro mechanistic studies related to hepatotoxicity.",
        height=120,
    )
    exclusion_criteria = st.text_area(
        "Optional exclusion criteria",
        placeholder="Example: Exclude reviews, non-English papers, ecotoxicology, or purely clinical outcome studies.",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        year_start = st.number_input("Year start", min_value=1900, max_value=CURRENT_YEAR, value=max(1900, CURRENT_YEAR - 10))
    with col2:
        year_end = st.number_input("Year end", min_value=1900, max_value=CURRENT_YEAR, value=CURRENT_YEAR)

    max_records = st.number_input(
        "Max records",
        min_value=1,
        max_value=HARD_MAX_RECORDS,
        value=DEFAULT_MAX_RECORDS,
        step=1,
        help=f"Keep this small for the demo. Hard cap: {HARD_MAX_RECORDS}.",
    )

    llm_model = st.text_input(
        "LLM model",
        value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Used for title/abstract screening.",
    )

    run_clicked = st.button("Search and screen", type="primary", use_container_width=True)

if run_clicked:
    if not pubmed_query.strip():
        st.error("Please enter a PubMed query.")
    elif year_start > year_end:
        st.error("Year start cannot be greater than year end.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in the environment.")
    else:
        try:
            with st.spinner("Fetching PubMed records..."):
                records = fetch_pubmed_records(
                    query=pubmed_query,
                    year_start=int(year_start),
                    year_end=int(year_end),
                    max_records=int(max_records),
                )

            if not records:
                st.warning("No PubMed records found for this query.")
            else:
                with st.spinner(f"Screening {len(records)} records with the LLM..."):
                    screened = screen_records(
                        records=records,
                        query=pubmed_query,
                        inclusion_criteria=inclusion_criteria or None,
                        exclusion_criteria=exclusion_criteria or None,
                        model=llm_model,
                    )

                df = screened_records_to_dataframe(screened)
                st.session_state["screened_records"] = screened
                st.session_state["results_df"] = df
        except PubMedSearchError as exc:
            st.error(f"PubMed retrieval failed: {exc}")
        except ScreeningError as exc:
            st.error(f"Screening failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

results_df: pd.DataFrame | None = st.session_state.get("results_df")

if results_df is not None and not results_df.empty:
    st.subheader("Screening results")

    decision_counts = results_df["screening_decision"].value_counts(dropna=False)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", int(len(results_df)))
    m2.metric("Yes", int(decision_counts.get("yes", 0)))
    m3.metric("Maybe", int(decision_counts.get("maybe", 0)))
    m4.metric("No", int(decision_counts.get("no", 0)))

    st.dataframe(results_df, use_container_width=True, height=550)

    csv_bytes = dataframe_to_csv_bytes(results_df)
    st.download_button(
        label="Download screening CSV",
        data=csv_bytes,
        file_name="aop_rag_screening_results.csv",
        mime="text/csv",
    )
else:
    st.info("Run a search to see screening results here.")
