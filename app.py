from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
import streamlit as st

from schemas import PubMedRecord, ScreeningDecision
from stage1_search.pubmed_search import search_pubmed
from stage1_search.screening import screen_record
from stage1_search.export import build_export_dataframe, dataframe_to_csv_bytes
from stage2_extraction.pdf_reader import extract_text_from_pdf, truncate_to_token_budget, extract_doi_from_pdf, _strip_references
from stage2_extraction.ker_extractor import extract_kers_from_text, ExtractionError
from stage2_extraction.llm_providers import LLMConfig
from stage2_extraction.aopwiki_client import enrich_ker
from stage2_extraction import aopwiki_xml
from stage2_extraction.table1_store import init_db, insert_table1_row, load_table1_as_dataframe, clear_all_table1
from stage2_extraction.table2_synthesis import compute_table2
from stage2_extraction.aop_visualizer import build_pathway_graph, render_interactive_graph, get_pathway_chains

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
    PROVIDER_DEFAULTS = {
        "Ollama (local)": {
            "provider": "ollama",
            "default_model": "llama3.1:8b",
            "default_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "needs_key": False,
            "model_help": "Local Ollama model tag, e.g. llama3.1:8b, qwen2.5:14b.",
        },
        "Anthropic Claude": {
            "provider": "anthropic",
            "default_model": "claude-sonnet-4-5",
            "default_url": "https://api.anthropic.com/v1/messages",
            "needs_key": True,
            "env_key": "ANTHROPIC_API_KEY",
            "model_help": "e.g. claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5.",
        },
        "OpenAI GPT": {
            "provider": "openai",
            "default_model": "gpt-4o",
            "default_url": "https://api.openai.com/v1/chat/completions",
            "needs_key": True,
            "env_key": "OPENAI_API_KEY",
            "model_help": "e.g. gpt-4o, gpt-4o-mini, gpt-4.1.",
        },
    }
    provider_label = st.selectbox(
        "LLM provider",
        list(PROVIDER_DEFAULTS.keys()),
        index=0,
        help=(
            "Local Ollama is free but limited by your hardware. Cloud providers "
            "(Claude, GPT) accept much larger inputs and usually give better "
            "extraction quality."
        ),
    )
    provider_cfg = PROVIDER_DEFAULTS[provider_label]

    extraction_model = st.text_input(
        "Model name",
        value=provider_cfg["default_model"],
        key=f"model_{provider_cfg['provider']}",
        help=provider_cfg["model_help"],
    )

    api_base_url = st.text_input(
        "API base URL",
        value=provider_cfg["default_url"],
        key=f"url_{provider_cfg['provider']}",
        help="Override only if you proxy the API or run a private endpoint.",
    )

    api_key_value = ""
    if provider_cfg["needs_key"]:
        env_key = provider_cfg.get("env_key", "")
        api_key_value = st.text_input(
            "API key",
            value=os.getenv(env_key, ""),
            type="password",
            key=f"key_{provider_cfg['provider']}",
            help=f"Used only for this session. Falls back to ${env_key} if blank.",
        )

    st.divider()

    st.subheader("AOP-Wiki dump")
    local_v = aopwiki_xml.get_local_version()
    if local_v:
        st.caption(f"Local dump: **{local_v}**")
    else:
        st.warning("No local AOP-Wiki dump found in `stage2_extraction/aopwiki_data/`.")

    if st.button("Check for updates", key="aop_check_updates"):
        with st.spinner("Querying aopwiki.org/downloads..."):
            remote_v = aopwiki_xml.get_latest_remote_version()
        if remote_v is None:
            st.error("Could not reach aopwiki.org.")
        elif local_v == remote_v:
            st.success(f"Up to date ({local_v}).")
        else:
            st.session_state["aop_remote_v"] = remote_v
            st.info(f"Newer dump available: **{remote_v}** (local: {local_v or 'none'}).")

    if st.session_state.get("aop_remote_v") and st.session_state["aop_remote_v"] != local_v:
        if st.button(f"Download {st.session_state['aop_remote_v']}", key="aop_download"):
            with st.spinner("Downloading XML dump (~10 MB)..."):
                try:
                    aopwiki_xml.download_dump(st.session_state["aop_remote_v"])
                    aopwiki_xml.get_index(force_reload=True)
                    st.success(f"Updated to {st.session_state['aop_remote_v']}.")
                    st.session_state.pop("aop_remote_v", None)
                except Exception as exc:
                    st.error(f"Download failed: {exc}")

    st.divider()
    if st.button("Clear all Table 1 data", type="secondary"):
        clear_all_table1()
        st.success("Table 1 cleared.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Stage 1 — Search & screen", "Stage 2 — KER extraction", "Pathway Visualization"])


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
        elif not ollama_model.strip():
            st.error("Please enter an Ollama model name in the sidebar (e.g. llama3.1:8b).")
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
        "A local Ollama model extracts KERs, then AOP-Wiki IDs are looked up automatically."
    )

    # -----------------------------------------------------------------------
    # Upload + extract
    # -----------------------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more full-text papers as PDFs. The DOI is detected automatically.",
    )

    # Auto-detect a DOI for each uploaded PDF and let the user override if wrong.
    doi_overrides: dict[str, str] = {}
    if uploaded_files:
        st.markdown("**Detected DOIs** (edit any field to override):")
        for f in uploaded_files:
            cache_key = f"_auto_doi::{f.name}::{f.size}"
            if cache_key not in st.session_state:
                try:
                    st.session_state[cache_key] = extract_doi_from_pdf(f) or ""
                except Exception:
                    st.session_state[cache_key] = ""
            auto_doi = st.session_state[cache_key]
            placeholder = "10.1016/j.tox.2022.01.001 (not found — please enter)"
            doi_overrides[f.name] = st.text_input(
                f.name,
                value=auto_doi,
                key=f"doi_input::{f.name}",
                placeholder=placeholder,
                help="Auto-extracted from the PDF; edit if it looks wrong.",
            )

    run_extraction = st.button("Extract KERs", type="primary", key="run_extraction")

    if run_extraction:
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        elif any(not (doi_overrides.get(f.name) or "").strip() for f in uploaded_files):
            missing = [f.name for f in uploaded_files if not (doi_overrides.get(f.name) or "").strip()]
            st.error("Missing DOI for: " + ", ".join(missing))
        elif not extraction_model.strip():
            st.error("Please enter an Ollama model name in the sidebar (e.g. llama3.1:8b).")
        else:
            for uploaded_file in uploaded_files:
                paper_doi = doi_overrides[uploaded_file.name].strip()
                st.markdown(f"**Processing:** `{uploaded_file.name}` — DOI `{paper_doi}`")

                # Step 1 — extract text
                with st.spinner("Extracting text from PDF..."):
                    try:
                        raw_text = extract_text_from_pdf(uploaded_file)
                        text_no_refs = _strip_references(raw_text)
                        paper_text = truncate_to_token_budget(text_no_refs)
                        st.caption(f"Text extracted: {len(raw_text):,} chars → {len(text_no_refs):,} chars (after removing references) → {len(paper_text):,} chars sent to API")
                        print('paper_text', paper_text, 'end of paper_text')
                    except RuntimeError as e:
                        st.error(str(e))
                        continue

                # Step 2 — LLM extraction via Ollama (multi-step pipeline)
                debug_container = st.expander("Per-step LLM debug (prompts + raw responses)", expanded=False)
                step_log: list = []

                def _on_step(step_result, _container=debug_container, _log=step_log):
                    _log.append(step_result)
                    status = "OK" if step_result.ok else "FAIL"
                    with _container:
                        st.markdown(f"**[{status}] {step_result.step}**")
                        if step_result.error:
                            st.error(step_result.error)
                        with st.expander("Prompt", expanded=False):
                            st.code(step_result.prompt, language="text")
                        with st.expander("Raw response", expanded=False):
                            st.code(step_result.raw_response or "<empty>", language="json")
                        if step_result.ok and step_result.parsed is not None:
                            with st.expander("Parsed JSON", expanded=False):
                                st.json(step_result.parsed)
                        st.divider()

                llm_cfg = LLMConfig(
                    provider=provider_cfg["provider"],
                    model=extraction_model.strip(),
                    api_key=(api_key_value or None) if provider_cfg["needs_key"] else None,
                    base_url=api_base_url.strip() or None,
                )

                with st.spinner(f"Running stepwise extraction with {extraction_model} ({provider_label}) — this may take a few minutes..."):
                    try:
                        extractions, warnings = extract_kers_from_text(
                            paper_text=paper_text,
                            cfg=llm_cfg,
                            on_step=_on_step,
                        )
                    except ExtractionError as e:
                        st.error(str(e))
                        continue

                st.caption(f"LLM calls executed: {len(step_log)} (failures: {sum(1 for s in step_log if not s.ok)})")

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


# ===========================================================================
# TAB 3 — AOP Pathway Visualization
# ===========================================================================

with tab3:
    st.header("AOP Pathway Visualization")
    st.caption(
        "Visualize the synthesized Key Event Relationships as an interactive directed graph. "
        "Nodes are Key Events (KEs), edges are KERs with evidence from all papers."
    )

    # Load Table 1 and compute Table 2
    t1_df = load_table1_as_dataframe()

    if t1_df.empty:
        st.warning("No data in Table 1. Extract KERs from papers in Stage 2 first.")
    else:
        # Compute Table 2 if not already computed
        if "table2_df" not in st.session_state or st.session_state.table2_df is None or st.session_state.table2_df.empty:
            with st.spinner("Computing Table 2 synthesis..."):
                table2_df = compute_table2(t1_df)
                st.session_state.table2_df = table2_df
        else:
            table2_df = st.session_state.table2_df

        if table2_df.empty:
            st.warning("Table 2 is empty. No KERs to visualize.")
        else:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_confidence = st.selectbox(
                    "Minimum uncertainty level",
                    options=["High", "Moderate", "Low"],
                    index=2,  # Default to Low (show all)
                    help="Show only KERs with LOW uncertainty or better",
                    key="vis_uncertainty",
                )
            with col2:
                include_novel = st.checkbox(
                    "Include novel KERs",
                    value=True,
                    help="Show KERs not yet in AOP-Wiki",
                    key="vis_novel",
                )

            # Filter table2 based on criteria
            table2_filtered = table2_df.copy()

            # Filter by uncertainty
            uncertainty_order = ["Low", "Moderate", "High"]
            min_idx = uncertainty_order.index(min_confidence)
            valid_uncertainties = uncertainty_order[:min_idx+1]
            table2_filtered = table2_filtered[table2_filtered["uncertainty_level"].isin(valid_uncertainties)]

            # Filter by novelty
            if not include_novel:
                table2_filtered = table2_filtered[table2_filtered["aop_status"] == "existing"]

            if table2_filtered.empty:
                st.info("No KERs match the current filter criteria.")
            else:
                # Build and render graph
                with st.spinner("Building pathway graph..."):
                    graph = build_pathway_graph(table2_filtered)
                    html_graph = render_interactive_graph(graph, height=800, physics=True)

                st.components.v1.html(html_graph, height=850)

                # Summary stats
                st.subheader("Graph Statistics")
                cols = st.columns(4)
                cols[0].metric("Key Events (nodes)", graph.number_of_nodes())
                cols[1].metric("KERs (edges)", graph.number_of_edges())
                cols[2].metric("Table 1 rows", len(t1_df))
                cols[3].metric("Table 2 KERs", len(table2_filtered))

                # Pathway chains
                st.subheader("Identified Pathways")
                chains = get_pathway_chains(graph, max_length=8)

                if chains:
                    st.write(f"Found **{len(chains)}** mechanistic pathways (longest chains shown first):")
                    for i, chain in enumerate(chains[:10], 1):  # Show top 10
                        chain_str = " → ".join(chain)
                        st.caption(f"**{i}.** {chain_str} ({len(chain)} events)")
                else:
                    st.info("No complete pathways detected (graph may be disconnected or cyclic).")

                # Detailed KER table
                st.subheader("KER Details")
                display_cols = [
                    "ker_name",
                    "upstream_ke_name",
                    "downstream_ke_name",
                    "n_supporting_papers",
                    "n_contradicting_papers",
                    "uncertainty_level",
                    "aop_status",
                ]
                available_cols = [c for c in display_cols if c in table2_filtered.columns]
                st.dataframe(table2_filtered[available_cols], use_container_width=True, height=400)
