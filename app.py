from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

from schemas import KE_LEVEL_ORDER, PubMedRecord, ScreeningDecision
from stage1_search.pubmed_search import NCBICredentials, search_pubmed
from stage1_search.screening import screen_record
from stage1_search.export import build_export_dataframe, dataframe_to_csv_bytes

from stage2_extraction import (
    aopwiki_xml,
    gene_registry,
    ke_normalizer,
    ke_synonyms,
    ols4_client,
)
import run_manifest
import session_db
from legal import (
    CLOUD_TRANSFER_WARNING,
    LEGAL_SUMMARY_MD,
    LOCAL_PROCESSING_NOTE,
    UPLOAD_ACKNOWLEDGEMENT,
    with_disclaimer,
)
from run_manifest import REPRODUCIBILITY_NOTE, RunManifest, RunTelemetry
from stage2_extraction.aopwiki_client import enrich_ker
from stage2_extraction.chunk_scorer import prepare_paper_text
from stage2_extraction import ker_extractor
from stage2_extraction.ker_extractor import ExtractionError, extract_kers_from_document
from stage2_extraction.llm_providers import LLMAuthError, LLMConfig
from stage2_extraction.pdf_reader import extract_document, strip_control_chars
from stage2_extraction import (
    citations,
    curation_store,
    evidence_synthesis,
    manual_entry,
    qc_report,
    table1_store,
    table2_synthesis,
)
from stage2_extraction.table1_store import (
    clear_all_table1,
    init_db,
    insert_table1_row,
    list_source_papers,
    load_canonical_kes,
    load_evidence_spans,
    load_table1_as_dataframe,
)
from stage2_extraction.table2_synthesis import (
    compute_table2,
    compute_table2_raw,
    consolidation_report,
)
from ui import aop_map as ui_aop_map
from ui import approve as ui_approve
from ui import common as ui_common
from ui import curate as ui_curate
from ui import manual_claim as ui_manual_claim
from ui import synthesis as ui_synthesis

# ---------------------------------------------------------------------------
# App-wide setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI4AOP — KER extractor", layout="wide")

# Point every store at this session's own database, before anything opens one.
#
# This must run on EVERY rerun, not once behind a session_state guard: it sets
# thread-local state, Streamlit hands a session's script to whichever thread is
# free, and a thread that last served another session would otherwise still be
# pointed at that session's file. It is four assignments, so doing it every
# time is cheap and is the only version that is actually correct.
_db_session = session_db.activate()

if "_db_status" not in st.session_state:
    st.session_state["_db_status"] = init_db()
    # Delete databases belonging to sessions that have ended. Streamlit offers
    # no supported teardown hook, so cleanup is a sweep at start-up rather than
    # a callback, and it is deliberately conservative: a file has to be both
    # old and unclaimed by a live session before it is removed.
    try:
        session_db.sweep_stale()
    except Exception:  # noqa: BLE001 - housekeeping must never block the app
        pass
    # Any run still marked 'running' when the app boots was cut short — a
    # refresh, a rerun, a stopped process. Close them now so the run history
    # does not accumulate rows that look like work still in flight.
    st.session_state["_orphaned_runs"] = table1_store.close_orphaned_runs()

_db_status = st.session_state["_db_status"]


# ---------------------------------------------------------------------------
# Full reset
# ---------------------------------------------------------------------------

#: Session-state keys that survive a reset. Everything else is discarded,
#: because the whole point of the button is that the user cannot tell which
#: leftover value is the one confusing them.
_RESET_KEEP_PREFIXES = (
    "curator_name",     # who is curating is not stale data
    # The reset widgets themselves: deleting the key of a checkbox that is
    # mid-render is how you get a confirmation box that unticks itself.
    "_confirm_full_reset",
    "_btn_full_reset",
    "_confirm_page_reset",
    "_btn_page_reset",
)


def _reset_everything() -> dict[str, int]:
    """
    Empty the database and every cache the app holds, in one action.

    Four separate stores accumulate state, and until now only the first had a
    button: the SQLite database, the OLS4 ontology lookup cache, the AOP-Wiki
    index held in module memory, and Streamlit's session state — which quietly
    remembers the DOI detected for each uploaded PDF under a
    `_auto_doi::<name>::<size>` key, so re-uploading a file never re-reads it.
    """
    removed = table1_store.clear_everything()

    # The ontology cache may be pointed at a different file than the main DB,
    # in which case clear_everything did not reach it.
    try:
        ols4_client.clear_cache()
    except Exception:
        pass

    for module in (ke_synonyms, gene_registry):
        try:
            module.clear_cache()
        except Exception:
            pass

    # Drop the in-memory AOP-Wiki index; it reloads lazily from the dump on
    # disk, which is a downloaded artefact and deliberately kept.
    try:
        aopwiki_xml.get_index(force_reload=True)
    except Exception:
        pass

    for key in list(st.session_state.keys()):
        if not str(key).startswith(_RESET_KEEP_PREFIXES):
            del st.session_state[key]

    # init_db() was consumed on first load; re-run it so the schema banner and
    # connection are valid for the rest of this rerun.
    st.session_state["_db_status"] = init_db()
    return removed


# ---------------------------------------------------------------------------
# Shared data pipeline
# ---------------------------------------------------------------------------

def load_pipeline(force: bool = False) -> dict[str, Any]:
    """
    Load Table 1, apply canonical KE names, and compute both Table 2 views.

    Cached in session state so switching tabs does not recompute the synthesis;
    pass force=True after any write.
    """
    if not force and "_pipeline" in st.session_state:
        return st.session_state["_pipeline"]

    t1 = load_table1_as_dataframe()
    canonical_df = load_canonical_kes()

    if not t1.empty and not canonical_df.empty:
        alias_map = table1_store.load_alias_map()
        id_to_row = {int(r["canonical_id"]): r for _, r in canonical_df.iterrows()}

        def canonical_for(label: Any) -> tuple[Optional[str], Optional[str], Optional[int]]:
            if pd.isna(label):
                return None, None, None
            canonical_id = alias_map.get(str(label).strip())
            if canonical_id is None or canonical_id not in id_to_row:
                return str(label).strip(), None, None
            row = id_to_row[canonical_id]
            return str(row["canonical_name"]), str(row["level"]), int(canonical_id)

        for prefix in ("upstream", "downstream"):
            names, levels, ids = [], [], []
            for value in t1[f"{prefix}_ke_name"]:
                name, level, canonical_id = canonical_for(value)
                names.append(name)
                levels.append(level)
                ids.append(canonical_id)
            t1[f"{prefix}_ke_canonical"] = names
            t1[f"{prefix}_ke_canonical_level"] = [
                lvl if lvl else t1[f"{prefix}_ke_level"].iloc[i]
                for i, lvl in enumerate(levels)
            ]
            t1[f"{prefix}_ke_canonical_id"] = ids

    # Claim-level curation. A curator who unticks **Keep** on a claim in the
    # Assign grid is saying that row should stop contributing evidence — it
    # misread the paper, or it is a duplicate of a row entered by hand. Without
    # this filter the tick was recorded and then ignored, which is worse than
    # not offering it. The row itself stays in Table 1 with its quotations: this
    # decides what the synthesis reads, not what the corpus contains.
    if not t1.empty:
        claim_state = curation_store.curation_map("claim")
        t1["claim_status"] = [
            str((claim_state.get(str(int(record_id))) or {}).get("status")
                or "unreviewed")
            for record_id in t1["record_id"]
        ]
        contributing = t1[t1["claim_status"] != "rejected"]
    else:
        contributing = t1

    normalized = (
        compute_table2(contributing, normalized=True)
        if not contributing.empty else pd.DataFrame()
    )
    raw = compute_table2_raw(contributing) if not contributing.empty else pd.DataFrame()

    if not normalized.empty:
        normalized = curation_store.apply_ker_curation(normalized, hide_rejected=False)

    pipeline = {
        "table1": t1,
        "table1_contributing": contributing,
        "canonical": canonical_df,
        "table2_normalized": normalized,
        "table2_raw": raw,
        "consolidation": consolidation_report(raw, normalized),
    }
    st.session_state["_pipeline"] = pipeline
    return pipeline


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Render a DataFrame as disclaimer-stamped CSV, safely.

    Rows extracted before control characters were scrubbed on write still
    carry them, and a single NUL anywhere in the frame makes `to_csv` raise
    "need to escape, but no escapechar set" — which surfaces as a crashed
    download button with no indication of which cell is at fault. Cleaning
    here means an old database still exports.
    """
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(strip_control_chars)
    return with_disclaimer(cleaned.to_csv(index=False)).encode("utf-8")


def invalidate_pipeline() -> None:
    st.session_state.pop("_pipeline", None)


def _fmt(value: Any, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    text = str(value).strip()
    return text if text and text.lower() != "none" else dash


# ---------------------------------------------------------------------------
# Sidebar — shared settings
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "Ollama (local)": {
        "provider": "ollama",
        "default_model": "llama3.1:8b",
        #: Used as the fallback default when this provider is also the primary,
        #: so "same provider, different model" is offered rather than the model
        #: already in use.
        "sibling_model": "qwen2.5:14b",
        "default_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "needs_key": False,
        "model_help": "Local Ollama model tag, e.g. llama3.1:8b, qwen2.5:14b.",
    },
    "Anthropic Claude": {
        "provider": "anthropic",
        "default_model": "claude-sonnet-4-5",
        "sibling_model": "claude-haiku-4-5",
        "default_url": "https://api.anthropic.com/v1/messages",
        "needs_key": True,
        "env_key": "ANTHROPIC_API_KEY",
        "model_help": "e.g. claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5.",
    },
    "OpenAI GPT": {
        "provider": "openai",
        "default_model": "gpt-4o",
        "sibling_model": "gpt-4o-mini",
        "default_url": "https://api.openai.com/v1/chat/completions",
        "needs_key": True,
        "env_key": "OPENAI_API_KEY",
        "model_help": "e.g. gpt-4o, gpt-4o-mini, gpt-4.1.",
    },
}

#: The two halves of the tool, and the top-level choice between them.
#:
#: They were six sibling tabs, which made "Search & screen" look like the first
#: of six equal steps rather than a separate stage that happens first and is
#: then done with. The five Stage 2 steps are a sequence you work through; the
#: search is not part of that sequence.
STAGE_1 = "Stage 1 — Find papers"
STAGE_2 = "Stage 2 — Build the AOP"

with st.sidebar:
    st.header("Settings")

    if _db_status.get("reset"):
        st.warning(
            "The database used an older schema without provenance or canonical "
            "Key Events, so it was rebuilt. Previous extractions were cleared."
        )
    elif _db_status.get("migrated"):
        st.info(
            f"The database was upgraded from schema "
            f"v{_db_status.get('previous_version')} to "
            f"v{table1_store.SCHEMA_VERSION}. Extractions were kept. Existing "
            f"Key Events start at **Raw** and need approving before anything "
            f"can be synthesised."
        )

    if st.session_state.get("_orphaned_runs"):
        n_orphaned = st.session_state["_orphaned_runs"]
        st.warning(
            f"{n_orphaned} previous run(s) never finished and have been marked "
            "**interrupted**. A run stops early if the page is refreshed or a "
            "control is touched while it is working, so any papers after the "
            "point it stopped were never processed."
        )

    # No curator box. Decisions are attributed to the account running the app
    # (see ui.common.curator_name), which is the same answer this would have
    # collected and does not have to be retyped after every restart.
    st.divider()

    # The stage picker, and the reason the rest of the sidebar is conditional.
    #
    # Both stages' provider settings used to sit here at once, so a curator
    # normalising Key Events was looking at a "Model name" box for the search
    # step they finished last week, and there were two boxes with that label
    # either way. Only the stage you are in can be configured, because only its
    # settings can affect anything you are about to do.
    active_stage = st.radio(
        "Stage",
        (STAGE_1, STAGE_2),
        key="_active_stage",
        help=(
            "Stage 1 finds and screens the literature. Stage 2 turns the "
            "papers you kept into an AOP, in five steps."
        ),
    )
    st.caption(
        "Stage 1 finds papers. Stage 2 builds the pathway from them."
        if active_stage == STAGE_1
        else "Five steps, in order. Each one needs the one before it signed off."
    )
    st.divider()


if active_stage == STAGE_1:
    with st.sidebar:
        st.subheader("Stage 1 — Search & screen")
        s1_provider_label = st.selectbox(
            "LLM provider (Stage 1)",
            list(PROVIDER_DEFAULTS.keys()),
            index=0,
            help=(
                "Local Ollama is free but limited by your hardware. Cloud providers "
                "accept much larger inputs and usually give better screening quality."
            ),
            key="s1_provider_select",
        )
        s1_provider_cfg = PROVIDER_DEFAULTS[s1_provider_label]

        s1_model = st.text_input(
            "Model name",
            value=s1_provider_cfg["default_model"],
            key=f"s1_model_{s1_provider_cfg['provider']}",
            help=s1_provider_cfg["model_help"],
        )
        s1_api_base_url = st.text_input(
            "API base URL",
            value=s1_provider_cfg["default_url"],
            key=f"s1_url_{s1_provider_cfg['provider']}",
            help="Override only if you proxy the API or run a private endpoint.",
        )
        s1_api_key_value = ""
        if s1_provider_cfg["needs_key"]:
            s1_env_key = s1_provider_cfg.get("env_key", "")
            s1_api_key_value = st.text_input(
                "API key",
                value=os.getenv(s1_env_key, ""),
                type="password",
                key=f"s1_key_{s1_provider_cfg['provider']}",
                help=f"Used only for this session. Falls back to ${s1_env_key} if blank.",
            ).strip()

        # --- Identifying this client to NCBI -----------------------------
        #
        # E-utilities are free and usable without registration, but NCBI's
        # usage policy asks every request to name the tool and give a contact
        # address, so a client that misbehaves can be written to rather than
        # simply blocked. Both were previously settable only through
        # environment variables, which meant that in practice almost nobody
        # set them — the LLM keys had sidebar fields and these did not.
        #
        # The API key is genuinely optional: without one NCBI allows three
        # requests a second, with one ten, and the client paces itself to
        # whichever applies rather than assuming the faster limit.
        with st.expander("PubMed / NCBI access", expanded=False):
            ncbi_email = st.text_input(
                "Contact email for NCBI",
                value=os.getenv("NCBI_EMAIL", ""),
                key="ncbi_email",
                help="NCBI asks that requests identify a contact. Falls back "
                     "to $NCBI_EMAIL if blank.",
            ).strip()
            ncbi_api_key = st.text_input(
                "NCBI API key (optional)",
                value=os.getenv("NCBI_API_KEY", ""),
                type="password",
                key="ncbi_api_key",
                help="Free from an NCBI account. Raises the rate limit from 3 "
                     "to 10 requests/second. Falls back to $NCBI_API_KEY.",
            ).strip()
            ncbi_credentials = NCBICredentials(
                email=ncbi_email or None,
                api_key=ncbi_api_key or None,
            )
            resolved_ncbi = NCBICredentials.resolve(ncbi_credentials)
            st.caption(
                f"Searches run at up to {resolved_ncbi.requests_per_second} "
                f"requests/second"
                + ("." if resolved_ncbi.api_key else " — add a key to go faster.")
            )
            if not resolved_ncbi.email:
                st.caption(
                    "⚠️ No contact address set. NCBI asks for one so they can "
                    "reach you before blocking a client."
                )

        if st.button("Test Stage 1 connection", key="s1_test"):
            try:
                LLMConfig(
                    provider=s1_provider_cfg["provider"],
                    model=s1_model.strip(),
                    api_key=(s1_api_key_value or None) if s1_provider_cfg["needs_key"] else None,
                    base_url=s1_api_base_url.strip() or None,
                    max_output_tokens=16,
                ).generate('Reply with only: {"ok":true}')
                st.success("Stage 1 credentials work.")
            except Exception as exc:
                st.error(f"Stage 1 test failed:\n\n{exc}")

        year_start = st.number_input("Year start", min_value=1900, max_value=2100, value=2010, step=1)
        year_end = st.number_input("Year end", min_value=1900, max_value=2100, value=2026, step=1)
        max_records = st.number_input("Max records", min_value=1, max_value=500, value=25, step=1)


if active_stage == STAGE_2:
    with st.sidebar:
        st.subheader("Stage 2 — KER extraction")
        s2_provider_label = st.selectbox(
            "LLM provider (Stage 2)",
            list(PROVIDER_DEFAULTS.keys()),
            index=0,
            key="s2_provider_select",
        )
        s2_provider_cfg = PROVIDER_DEFAULTS[s2_provider_label]

        extraction_model = st.text_input(
            "Model name",
            value=s2_provider_cfg["default_model"],
            key=f"s2_model_{s2_provider_cfg['provider']}",
            help=s2_provider_cfg["model_help"],
        )
        api_base_url = st.text_input(
            "API base URL",
            value=s2_provider_cfg["default_url"],
            key=f"s2_url_{s2_provider_cfg['provider']}",
        )
        api_key_value = ""
        if s2_provider_cfg["needs_key"]:
            s2_env_key = s2_provider_cfg.get("env_key", "")
            api_key_value = st.text_input(
                "API key",
                value=os.getenv(s2_env_key, ""),
                type="password",
                key=f"s2_key_{s2_provider_cfg['provider']}",
                help=f"Used only for this session. Falls back to ${s2_env_key} if blank.",
            ).strip()

        # --- Sending a paper to someone else is a decision -----------------
        #
        # Ollama is the default because local processing does not disclose the
        # paper to a third party. Choosing a hosted provider does, and the
        # application cannot tell whether the user is allowed to: a subscription
        # grants reading, not automatically the right to transmit the text to
        # another company. Licence metadata does not settle it either — much of
        # the literature has none, and free-to-read is not a reuse licence.
        #
        # So this is not a silent default and not a pre-ticked box. It is an
        # explicit act, recorded with the run, so that a later question about a
        # particular extraction has an answer other than nobody's recollection.
        transmission_ack = None
        if s2_provider_cfg["needs_key"]:
            st.warning(
                f"**{s2_provider_label} is a hosted service.** The full text of "
                "every paper you upload will be sent to it. This tool does not "
                "check whether you are permitted to do that — subscription "
                "access grants reading, not necessarily transmission to a third "
                "party. If you are unsure, switch to **Ollama (local)**, which "
                "sends nothing off this machine.",
                icon="⚠️",
            )
            acknowledged = st.checkbox(
                "I have a lawful basis to send these papers to this provider",
                value=False,
                key=f"s2_ack_{s2_provider_cfg['provider']}",
            )
            basis = st.text_input(
                "Basis (recorded with the run)",
                value="",
                placeholder="e.g. institutional TDM agreement, ref. LIB-2026-014",
                key=f"s2_basis_{s2_provider_cfg['provider']}",
                disabled=not acknowledged,
                help="Stored in the run record so the decision can be audited "
                     "later. Free text — the tool does not verify it.",
            ).strip()
            if acknowledged and basis:
                transmission_ack = (
                    f"{s2_provider_label}: {basis} "
                    f"(acknowledged {datetime.date.today().isoformat()})"
                )
            elif acknowledged:
                st.caption("State the basis to continue.")

        # --- What to do when the provider declines -----------------------
        #
        # A safety-classifier refusal is a property of one model, and the tool
        # has always said so — and then told the curator to run the paper
        # through a different provider by hand. That is the correct remedy and
        # doing it manually was the entire cost: reconfigure the sidebar,
        # re-upload the paper, and afterwards remember which of your rows came
        # from which model. Configured here it costs one setting, and the
        # manifest records which rows the fallback produced.
        #
        # Off by default. A fallback that fires without being asked for would
        # silently mix models in a corpus whose whole value is knowing what
        # produced each row.
        with st.expander("If the provider declines"):
            st.caption(
                f"A refusal is decided on the model's reply **as it is being "
                f"written**, not on the paper — so it is sampled, and the same "
                f"paper is often read on the next run. Refused steps are "
                f"therefore re-asked of {extraction_model} automatically at "
                f"rising temperatures "
                f"({len(ker_extractor.REFUSAL_RETRY_TEMPERATURES)} attempts), "
                f"which needs no extra configuration and no second key."
            )
            st.caption(
                "This setting is the last resort after those attempts. **It "
                "does not need a different provider** — a second Claude model "
                "on the same API key is a valid fallback, because classifiers "
                "differ per model."
            )
            # Preselect the provider already in use. The point that was missed
            # the first time is that a *sibling model on the same key* is the
            # normal case; offering a vendor list with no default made this
            # look like it required a second subscription.
            _fb_options = ["Off — fail the paper"] + list(PROVIDER_DEFAULTS.keys())
            _fb_default = next(
                (
                    i for i, label in enumerate(_fb_options)
                    if label != "Off — fail the paper"
                    and PROVIDER_DEFAULTS[label]["provider"]
                    == s2_provider_cfg["provider"]
                ),
                0,
            )
            fb_label = st.selectbox(
                "Fallback provider",
                _fb_options,
                index=_fb_default,
                key="s2_fallback_provider",
                help=(
                    "Usually the same provider you are already using, with a "
                    "different model. A local Ollama model applies no safety "
                    "classifier at all, so it always answers — at whatever "
                    "quality the local model manages."
                ),
            )
            fallback_cfg = None
            if fb_label != "Off — fail the paper":
                fb_provider = PROVIDER_DEFAULTS[fb_label]
                # Default to a sibling that is not the model already in use.
                # Defaulting to the provider's headline model meant that
                # picking "Anthropic" as the fallback for an Anthropic run
                # proposed the primary model back, which is not a fallback at
                # all — the picker then warned about the choice it had just
                # made for you.
                _fb_default_model = fb_provider["default_model"]
                if (
                    fb_provider["provider"] == s2_provider_cfg["provider"]
                    and _fb_default_model.strip() == extraction_model.strip()
                ):
                    _fb_default_model = fb_provider.get(
                        "sibling_model", _fb_default_model
                    )
                fb_model = st.text_input(
                    "Fallback model",
                    value=_fb_default_model,
                    key=f"s2_fb_model_{fb_provider['provider']}",
                    help=fb_provider["model_help"],
                )
                fb_key = ""
                if fb_provider["needs_key"]:
                    fb_env = fb_provider.get("env_key", "")
                    fb_key = st.text_input(
                        "Fallback API key",
                        value=os.getenv(fb_env, ""),
                        type="password",
                        key=f"s2_fb_key_{fb_provider['provider']}",
                        help=f"Falls back to ${fb_env} if blank.",
                    ).strip()
                if (
                    fb_provider["provider"] == s2_provider_cfg["provider"]
                    and fb_model.strip() == extraction_model.strip()
                ):
                    # Retrying the same model with the same prompt is what the
                    # old code did, and it is not a second opinion: identical
                    # input, same temperature, and on Anthropic the paper is
                    # sent under a cache-control marker that makes it the same
                    # bytes by design.
                    st.warning(
                        "That is the same model as the primary. A refusal is "
                        "a decision about this input, so re-asking the same "
                        "model changes nothing — pick a different one.",
                        icon="⚠️",
                    )
                else:
                    fallback_cfg = LLMConfig(
                        provider=fb_provider["provider"],
                        model=fb_model.strip(),
                        api_key=(fb_key or None) if fb_provider["needs_key"] else None,
                        base_url=fb_provider["default_url"] or None,
                        seed=None,
                    )
                    st.caption(
                        f"Refused steps will be retried against "
                        f"**{fb_provider['provider']}/{fb_model.strip()}**. "
                        f"Rows it produces are flagged in the run manifest, "
                        f"because a corpus read by two models is not one "
                        f"population."
                    )
            ker_extractor.set_refusal_fallback(fallback_cfg)

        if st.button("Test Stage 2 connection", key="s2_test"):
            try:
                _stage2_probe = LLMConfig(
                    provider=s2_provider_cfg["provider"],
                    model=extraction_model.strip(),
                    api_key=(api_key_value or None) if s2_provider_cfg["needs_key"] else None,
                    base_url=api_base_url.strip() or None,
                    max_output_tokens=16,
                ).generate('Reply with only: {"ok":true}')
                st.success("Stage 2 credentials work.")
            except Exception as exc:
                st.error(f"Stage 2 test failed:\n\n{exc}")

        st.divider()

        st.subheader("Chunk relevance scoring")
        st.caption(
            "Off by default: the whole paper goes to the model. A full-text "
            "research paper is 8,000–25,000 tokens, which every current cloud "
            "model reads in one pass, and a keyword scorer deciding what the model "
            "may see is the main way relevant evidence gets lost — a passage "
            "reporting NaV1.6 currents in OPCs contains none of the words a Key "
            "Event label is written in. Turn this on only for a local model whose "
            "context window cannot hold a paper."
        )
        use_chunking = st.checkbox(
            "Score and select chunks before extraction",
            value=False,
            help=(
                "Only needed when the context window is too small for a full "
                "paper. Everything the scorer drops is invisible to the model, "
                "and it cannot report evidence it was never shown."
            ),
        )
        chunk_budget = st.slider(
            "Character budget per paper", 10_000, 120_000, 45_000, step=5_000,
            help="Upper bound on the text sent to the model.",
            disabled=not use_chunking,
        )
        chunk_min_score = st.slider(
            "Minimum relevance score", 0.0, 0.8, 0.15, step=0.05,
            help="Lower this if extraction is missing KERs you know are in the paper.",
            disabled=not use_chunking,
        )
        use_llm_triage = st.checkbox(
            "Use the LLM to triage chunks (one extra call)",
            value=False,
            help="Slower and slightly costlier, but better recall on unusual papers.",
            disabled=not use_chunking,
        )

        st.divider()

        st.subheader("Ontology (OLS4)")
        ols4_enabled = st.checkbox(
            "Enrich Key Events with OLS4 terms", value=True,
            help="Attaches canonical labels, CURIEs and IRIs from GO, UBERON, CL and others.",
        )
        ols4_min_score = st.slider(
            "Minimum ontology match score", 0.3, 0.9, 0.45, step=0.05,
            disabled=not ols4_enabled,
        )
        if st.button("Test OLS4 connection", key="ols4_test"):
            reachable, message = ols4_client.check_availability()
            (st.success if reachable else st.error)(message)

        st.divider()

        st.subheader("AOP-Wiki dump")
        local_v = aopwiki_xml.get_local_version()
        if local_v:
            st.caption(f"Local dump: **{local_v}**")
        else:
            # A fresh clone has no dump: it is AOP-Wiki's data, fetched rather
            # than redistributed inside this repository. Enrichment is
            # optional, so this is a prompt and not an error.
            st.warning(
                "No AOP-Wiki dump yet. It is not bundled with the code — it is "
                "AOP-Wiki's data and is downloaded on request. Without it, "
                "extraction and curation still run, but Key Events are not "
                "matched to AOP-Wiki identifiers."
            )
            if st.button("Download the latest dump (~10 MB)", key="aop_fetch_first"):
                with st.spinner("Fetching from aopwiki.org..."):
                    version, error = aopwiki_xml.ensure_dump()
                if version:
                    aopwiki_xml.get_index(force_reload=True)
                    st.success(f"Downloaded dump {version}.")
                    st.rerun()
                else:
                    st.error(error or "Download failed.")

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

        st.subheader("Output token budgets")
        st.caption(
            "Per-step ceilings, not reservations — you pay for tokens generated, "
            "not tokens allowed. Raise this only for a model that reasons before "
            "answering, where the thinking must fit in the same budget as the JSON."
        )
        budget_scale = st.slider(
            "Budget multiplier", 0.5, 4.0, 1.0, step=0.5,
            help=(
                "1.0 gives 5000 tokens for the KER listing step, 5000 for "
                "evidence, 4000 for quantitative, 2500 for classify and "
                "applicability, 1200 for study metadata."
            ),
        )
        if budget_scale > 1.0:
            st.caption(
                f"Evidence step ceiling: "
                f"{ker_extractor.step_budget('evidence', budget_scale):,} tokens."
            )

        st.divider()

        st.subheader("Reproducibility")
        st.caption(REPRODUCIBILITY_NOTE)
        use_seed = st.checkbox(
            "Fix the sampling seed",
            value=False,
            help=(
                "Sent to Ollama and OpenAI, which accept a seed; Anthropic does "
                "not. It narrows run-to-run variation but cannot remove it."
            ),
        )
        run_seed = (
            int(st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1))
            if use_seed
            else None
        )


# Always shown: neither is a per-stage setting, and both are things you go
# looking for when something has gone wrong, whichever stage you are in.
with st.sidebar:
    st.divider()
    with st.expander("Database"):
        # Which database this is, said plainly and first.
        #
        # The two modes differ in what "Clear all extraction data" destroys and
        # in whether closing the tab loses the work. A curator who cannot tell
        # which one they are in cannot judge either button, and the previous
        # version of this panel silently meant "the one shared file" whichever
        # was true.
        if _db_session["persistent"]:
            st.success(
                f"**Persistent database** — `{_db_session['path']}`\n\n"
                f"Set by `{session_db.PERSISTENT_ENV_VAR}`. Your work is kept "
                f"between sessions. Everyone who can reach this app shares "
                f"this file, so use it only when you are the only user.",
                icon="💾",
            )
        else:
            st.info(
                "**This session only.** Your papers, rows and curation live in "
                "a private database that no other session can read, and that is "
                "deleted when the session ends. Download it below to keep it.\n\n"
                f"To work against one lasting corpus instead — the "
                f"single-user case — start the app with "
                f"`{session_db.PERSISTENT_ENV_VAR}=aop_rag.db`.",
                icon="🔒",
            )

        counts = table1_store.table_counts()
        st.caption(f"Schema version {table1_store.SCHEMA_VERSION}")
        st.json(counts)

        # An ephemeral database with no way out is a trap: the curation in it
        # is hours of work and it disappears on a closed tab. The file itself
        # is the export — reopen it later with the environment variable.
        if not _db_session["persistent"]:
            _db_file = Path(_db_session["path"])
            if _db_file.exists():
                st.download_button(
                    "Download this session's database",
                    _db_file.read_bytes(),
                    "aop_rag_session.db",
                    "application/octet-stream",
                    help=(
                        "The whole corpus as a SQLite file — rows, evidence "
                        "spans, curation decisions and run manifests. Put it "
                        "somewhere and point "
                        f"`{session_db.PERSISTENT_ENV_VAR}` at it to carry on "
                        "where you left off."
                    ),
                )

        if st.button("Clear all extraction data", type="secondary"):
            clear_all_table1()
            invalidate_pipeline()
            st.success("Extractions, evidence and canonical KEs cleared.")
            st.rerun()

        st.divider()
        st.markdown("**Reset everything**")
        st.caption(
            "Wipes the whole database — extractions, evidence, runs, curation "
            "decisions, syntheses, approvals and the map layout — plus every "
            "cache: cached ontology lookups, the DOIs detected from uploaded "
            "PDFs, the AOP-Wiki index and anything held in this session. Use "
            "it when the app is showing state you cannot otherwise get rid of. "
            "This cannot be undone."
        )
        confirm_reset = st.checkbox(
            "I understand this deletes everything",
            key="_confirm_full_reset",
        )
        if st.button(
            "Reset everything",
            type="primary",
            disabled=not confirm_reset,
            key="_btn_full_reset",
        ):
            removed = _reset_everything()
            if removed:
                detail = ", ".join(f"{n} {name}" for name, n in removed.items())
                st.success(f"Reset complete — removed {detail}.")
            else:
                st.success("Reset complete — nothing was stored.")
            st.rerun()

    with st.expander("Legal & data handling"):
        st.markdown(LEGAL_SUMMARY_MD)


def _stage2_config() -> LLMConfig:
    return LLMConfig(
        provider=s2_provider_cfg["provider"],
        model=extraction_model.strip(),
        api_key=(api_key_value or None) if s2_provider_cfg["needs_key"] else None,
        base_url=api_base_url.strip() or None,
        seed=run_seed,
    )


def _build_run_manifest(cfg: LLMConfig) -> RunManifest:
    """Capture the conditions of a Stage 2 run before any extraction happens."""
    return RunManifest.from_config(
        cfg,
        stage="extraction",
        budget_scale=float(budget_scale),
        chunking_enabled=bool(use_chunking),
        chunk_char_budget=int(chunk_budget) if use_chunking else None,
        chunk_min_score=float(chunk_min_score) if use_chunking else None,
        llm_triage=bool(use_llm_triage) if use_chunking else False,
        ols4_enabled=bool(ols4_enabled),
        ols4_min_score=float(ols4_min_score),
        # None for a local run, because nothing left the machine and there is
        # nothing to justify. Present only where paper text was transmitted.
        transmission_ack=transmission_ack,
    )


# ---------------------------------------------------------------------------
# Stage 1 — one page, no tabs
# ---------------------------------------------------------------------------

# Stage 1 is a single screen, so it gets no tab row at all. Giving it one tab
# among six made it read as step zero of a six-step sequence; it is a separate
# job that produces the input to the sequence.
if active_stage == STAGE_1:
    tab1 = st.container()

    with tab1:
        st.header("PubMed search & title/abstract screening")
        st.caption(
            "Input a PubMed query and screen the results against your inclusion and "
            "exclusion criteria using the configured LLM provider."
        )

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
            placeholder="e.g. reviews only, non-English, exclusively in vitro",
        )

        conflict_labels = {
            "maybe": "Flag as 'maybe' for human review (recommended)",
            "exclude": "Exclusion wins — reject the record",
            "include": "Inclusion wins — keep the record",
        }
        conflict_policy = st.radio(
            "When a record matches BOTH an inclusion and an exclusion criterion",
            options=list(conflict_labels.keys()),
            format_func=lambda k: conflict_labels[k],
            horizontal=False,
            key="conflict_policy",
            help=(
                "Common in toxicology: an in vivo rodent study with cell-culture "
                "follow-up matches 'mammalian studies' and 'in vitro' at once. "
                "Screening only sees the title and abstract, so a wrong 'no' loses "
                "a paper you never look at again — 'maybe' is the safer default."
            ),
        )

        run_search = st.button("Search and screen", type="primary", key="run_search")

        if "results_df" not in st.session_state:
            st.session_state.results_df = None

        if run_search:
            if not query.strip():
                st.error("Please enter a PubMed query.")
            elif not s1_model.strip():
                st.error("Please enter a model name in the sidebar (e.g. llama3.1:8b).")
            else:
                try:
                    with st.spinner("Fetching PubMed records..."):
                        records = search_pubmed(
                            query=query,
                            year_start=int(year_start),
                            year_end=int(year_end),
                            max_records=int(max_records),
                            credentials=ncbi_credentials,
                        )

                    if not records:
                        st.warning("No PubMed records found.")
                    else:
                        screened_pairs: List[Tuple[PubMedRecord, ScreeningDecision]] = []
                        progress = st.progress(0, text="Screening...")

                        llm_cfg_s1 = LLMConfig(
                            provider=s1_provider_cfg["provider"],
                            model=s1_model.strip(),
                            api_key=(s1_api_key_value or None) if s1_provider_cfg["needs_key"] else None,
                            base_url=s1_api_base_url.strip() or None,
                            # Screening replies are short, but a verbose rationale
                            # can still overflow the default and lose the record.
                            max_output_tokens=1200,
                        )

                        screen_failures: list[str] = []
                        consecutive_failures = 0
                        aborted: Optional[str] = None
                        MAX_CONSECUTIVE = 5

                        for idx, record in enumerate(records, start=1):
                            # One unparseable record must not discard the whole run.
                            # But a bad key or a dead endpoint fails identically for
                            # every record, so those abort instead of burning one
                            # request per record to learn the same thing 71 times.
                            try:
                                decision = screen_record(
                                    record=record,
                                    query=query,
                                    inclusion_criteria=inclusion_criteria,
                                    exclusion_criteria=exclusion_criteria,
                                    llm_cfg=llm_cfg_s1,
                                    conflict_policy=conflict_policy,
                                )
                                consecutive_failures = 0
                            except LLMAuthError as exc:
                                aborted = str(exc)
                                break
                            except Exception as exc:
                                consecutive_failures += 1
                                screen_failures.append(f"PMID {record.pmid}: {exc}")
                                if consecutive_failures >= MAX_CONSECUTIVE:
                                    aborted = (
                                        f"{consecutive_failures} records failed in a row, "
                                        "which points at a configuration problem rather "
                                        f"than difficult abstracts. Last error:\n\n{exc}"
                                    )
                                    break
                                decision = ScreeningDecision(
                                    decision="maybe",
                                    rationale=f"Automatic screening failed — review by hand. ({exc})",
                                    triggered_inclusion_rule=None,
                                    triggered_exclusion_rule=None,
                                    evidence_quote=None,
                                )
                            screened_pairs.append((record, decision))
                            progress.progress(idx / len(records), text=f"Screened {idx}/{len(records)}")
                        progress.empty()

                        if aborted:
                            st.error(
                                f"**Screening stopped after {len(screened_pairs)} of "
                                f"{len(records)} records.** No further requests were sent.\n\n"
                                + aborted
                            )
                        elif screen_failures:
                            ok = len(records) - len(screen_failures)
                            st.warning(
                                f"{ok} of {len(records)} record(s) screened successfully. "
                                f"{len(screen_failures)} failed and were marked 'maybe' "
                                "for manual review."
                            )
                            with st.expander("Screening failures"):
                                for failure in screen_failures:
                                    st.caption(failure)

                        if screened_pairs:
                            st.session_state.results_df = build_export_dataframe(
                                screened_pairs,
                                query=query,
                                inclusion_criteria=inclusion_criteria,
                                exclusion_criteria=exclusion_criteria,
                            )
                except Exception as exc:
                    st.exception(exc)

        if st.session_state.results_df is not None:
            df: pd.DataFrame = st.session_state.results_df
            st.subheader("Screening results")

            counts = df["screening_decision"].value_counts(dropna=False).to_dict()
            n_conflict = int(df.get("criteria_conflict", pd.Series(dtype=bool)).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Yes", counts.get("yes", 0))
            c2.metric("Maybe", counts.get("maybe", 0))
            c3.metric("No", counts.get("no", 0))
            c4.metric("Criteria conflicts", n_conflict)

            if n_conflict:
                st.info(
                    f"{n_conflict} record(s) matched an inclusion **and** an exclusion "
                    "criterion. These are the records most likely to be misfiled — "
                    "the `criteria_conflict` column marks them."
                )
                if st.checkbox("Show only conflicting records", key="show_conflicts"):
                    df = df[df["criteria_conflict"]]

            st.dataframe(df, use_container_width=True, height=500)
            st.download_button(
                "Download CSV",
                data=dataframe_to_csv_bytes(df),
                file_name="aop_rag_screening_results.csv",
                mime="text/csv",
            )
        else:
            st.info("Run a search to see results.")


    # Stage 2 is not merely hidden here, it is not built: constructing its tabs
    # would run every step's queries and widgets for a screen nobody is looking
    # at, and duplicate widget keys across stages would collide.
    st.stop()


# ---------------------------------------------------------------------------
# Stage 2 — five steps, in order
# ---------------------------------------------------------------------------
#
# Step 1 is source-faithful only: everything it shows is something a paper
# said, in that paper's wording, with the quotation that supports it. Nothing
# is merged and no cross-paper judgement is drawn — that is what steps 2 to 4
# are for, and mixing the two is what made it impossible to tell an extracted
# claim from a synthesised one.
#
# Each step depends on the one before it and says so; the last two refuse to
# render anything that has not been approved.
tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1 · Extract evidence",
        "2 · Normalize & curate",
        "3 · Approve",
        "4 · Synthesize evidence",
        "5 · Final AOP",
    ]
)


with tab2:
    ui_common.section_intro(
        "Extract evidence",
        "Extract evidence",
        "Pull statements out of full-text papers, each tied to the words that "
        "support it. Everything here is source-faithful — nothing is merged, "
        "interpreted or scored.",
        (
            "Choose **Targeted KER** and name the two Key Events if you already "
            "have a research question. That is cheaper and avoids most "
            "normalisation work later.",
            "Upload the PDFs that passed screening and run the extraction.",
            "Check the extracted rows below and open a row's **provenance "
            "drawer** to read the quotations behind each claim.",
            "When the extraction looks right, move to **2 · Normalize & "
            "curate**.",
        ),
        caution=(
            "Cross-paper synthesis, confidence bands and weight-of-evidence "
            "assessments do not appear in this section. They come after "
            "curation and approval."
        ),
    )

    # --- Start over -------------------------------------------------------
    # Sited here rather than only in the sidebar because this is where a run
    # goes wrong, and a result you have decided to discard is worth nothing
    # until it is actually gone.
    with st.expander("Start over — clear everything", expanded=False):
        st.caption(
            "Deletes every extraction, evidence quotation, run record, "
            "curation decision, synthesis, approval and map layout, plus all "
            "caches: ontology lookups, Key Event search terms, and the DOIs "
            "remembered for PDFs you have already uploaded. The AOP-Wiki dump "
            "on disk is kept. This cannot be undone."
        )
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            confirm_page_reset = st.checkbox(
                "I understand this deletes everything",
                key="_confirm_page_reset",
            )
        with rc2:
            if st.button(
                "Reset everything",
                type="primary",
                disabled=not confirm_page_reset,
                key="_btn_page_reset",
                use_container_width=True,
            ):
                removed = _reset_everything()
                if removed:
                    st.success(
                        "Reset complete — removed "
                        + ", ".join(f"{n} {name}" for name, n in removed.items())
                        + "."
                    )
                else:
                    st.success("Reset complete — nothing was stored.")
                st.rerun()

    # --- Extraction mode --------------------------------------------------
    # Open discovery lets the model name both events, which is why a corpus
    # ends up with one label per paper per event. When the question is already
    # known, fixing both labels removes that problem at the source rather than
    # trying to repair it afterwards.
    extraction_mode = st.radio(
        "Extraction mode",
        ["Open discovery", "Targeted KER"],
        horizontal=True,
        help=(
            "Open discovery extracts every relationship each paper contains. "
            "Targeted asks one question of every paper — far cheaper, and the "
            "Key Event names are yours, so nothing needs normalising afterwards."
        ),
        key="extraction_mode",
    )
    targeted = extraction_mode == "Targeted KER"

    target_upstream = target_downstream = ""
    directional = True
    target_up_terms: list[str] = []
    target_down_terms: list[str] = []
    if targeted:
        existing_kes: list[str] = []
        try:
            _canon = load_canonical_kes()
            if _canon is not None and not _canon.empty:
                existing_kes = sorted(_canon["canonical_name"].dropna().astype(str))
        except Exception:
            existing_kes = []

        st.caption(
            "Name both events. Every paper is asked the same question, and "
            "every row uses these exact labels — so the answer is one row per "
            "paper for one relationship, not a table to search."
        )

        # Committing to a direction is an assumption, and it is not always one
        # the user wants to make. Asserting a direction means papers get judged
        # for agreeing with it; asserting none means they are only asked what
        # they observed, and a split in the literature reads as a split rather
        # than as disagreement with something nobody claimed.
        direction_choice = st.radio(
            "Direction of the relationship",
            [
                "I know it — test this specific direction",
                "I don't want to assume — report what the papers found",
            ],
            key="direction_mode",
            help=(
                "Agnostic mode asks only what each paper observed at both "
                "ends, then derives whether the events move together or "
                "oppositely. Nothing is proposed, so nothing can contradict it."
            ),
        )
        directional = direction_choice.startswith("I know it")
        if not directional:
            st.caption(
                "Use direction-free labels for this mode — “sodium-channel "
                "activity”, “oligodendrocyte differentiation” — not “decreased "
                "…”. The direction comes from the papers."
            )
        tc1, tc2 = st.columns(2)
        with tc1:
            target_upstream = st.text_input(
                "Upstream Key Event",
                placeholder=(
                    "Decreased voltage-gated sodium-channel activity"
                    if directional
                    else "Voltage-gated sodium-channel activity"
                ),
                key="target_up",
            ).strip()
            if existing_kes:
                pick_up = st.selectbox(
                    "…or reuse an existing Key Event",
                    ["—"] + existing_kes, key="target_up_pick",
                )
                if pick_up != "—" and not target_upstream:
                    target_upstream = pick_up
        with tc2:
            target_downstream = st.text_input(
                "Downstream Key Event",
                placeholder=(
                    "Decreased oligodendrocyte differentiation"
                    if directional
                    else "Oligodendrocyte differentiation"
                ),
                key="target_down",
            ).strip()
            if existing_kes:
                pick_down = st.selectbox(
                    "…or reuse an existing Key Event",
                    ["—"] + existing_kes, key="target_down_pick",
                )
                if pick_down != "—" and not target_downstream:
                    target_downstream = pick_down

        if target_upstream and target_downstream:
            # Two different questions are being asked depending on the mode, and
            # saying "does X lead to Y" in agnostic mode states the very
            # assumption the user just declined to make.
            if directional:
                st.info(
                    f"Each paper is asked for the causal chain it evidences "
                    f"from **{target_upstream}** to **{target_downstream}** — "
                    "including any intermediate events it resolves. Every link "
                    "becomes an edge on the pathway."
                )
            else:
                st.info(
                    f"Each paper is asked what chain it evidences between "
                    f"**{target_upstream}** and **{target_downstream}**, and "
                    "which way each link runs. No direction is proposed; the "
                    "papers supply it, link by link."
                )
            # A direction-free upstream label leaves the question itself
            # ambiguous: "altered X leads to decreased Y" does not say which
            # way X moved, so no paper can be judged consistent or
            # inconsistent with it, and evidence gets misfiled as disagreement.
            _VAGUE = ("altered", "changed", "modified", "affected",
                      "disrupted", "abnormal", "aberrant")
            _DIRECTIONAL = ("increased", "decreased", "reduced", "elevated",
                            "impaired", "enhanced", "loss of", "gain of")

            vague = [w for w in _VAGUE if w in target_upstream.lower()]
            if vague and directional:
                st.warning(
                    f"“{vague[0].capitalize()}” does not say which way the "
                    "upstream event moves, so there is no direction for papers "
                    "to agree or disagree with. Either give it one — e.g. "
                    "**decreased** sodium-channel activity — or switch the "
                    "setting above to let the papers report the direction."
                )

            # The mirror-image mistake, and the one the mode invites: keeping
            # a direction word in a label that is not asserting a direction.
            # It is not merely redundant. Both labels are used as search terms,
            # and "decreased" matches every paper that ever reported a decrease
            # in anything, so it dilutes the vocabulary while adding nothing.
            if not directional:
                stray = [
                    w for w in (*_DIRECTIONAL, *_VAGUE)
                    if w in f"{target_upstream} {target_downstream}".lower()
                ]
                if stray:
                    st.info(
                        f"“{stray[0].capitalize()}” can come out — you are not "
                        "proposing a direction, so the word adds nothing to the "
                        "question and nothing to the search. Name the entity "
                        "and the process; each link comes back with the "
                        "direction the paper reported."
                    )

            st.caption(
                "Both events are expanded into the vocabulary papers actually "
                "use before screening starts — gene and protein symbols and "
                "their isoform families from HGNC, cell types and processes "
                "from the ontologies, and the abbreviations each paper defines "
                "for itself. Nothing to fill in; what was used is recorded "
                "with the run."
            )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="One or more full-text papers. The DOI is detected automatically.",
    )

    # --- Where this paper is about to be sent -----------------------------
    # Shown at the point of upload rather than buried in a policy page: the
    # decision that matters — cloud API or local model — is being made here,
    # and subscription content is the case where it matters most.
    sends_to_cloud = s2_provider_cfg["provider"] != "ollama"
    if uploaded_files:
        if sends_to_cloud:
            st.warning(CLOUD_TRANSFER_WARNING.format(provider=s2_provider_label))
        else:
            st.caption(LOCAL_PROCESSING_NOTE)
        st.caption(UPLOAD_ACKNOWLEDGEMENT)

    doi_overrides: dict[str, str] = {}
    if uploaded_files:
        st.markdown("**Detected DOIs** (edit any field to override):")
        for f in uploaded_files:
            cache_key = f"_auto_doi::{f.name}::{f.size}"
            if cache_key not in st.session_state:
                try:
                    from stage2_extraction.pdf_reader import extract_doi_from_pdf

                    st.session_state[cache_key] = extract_doi_from_pdf(f) or ""
                except Exception:
                    st.session_state[cache_key] = ""
            doi_overrides[f.name] = st.text_input(
                f.name,
                value=st.session_state[cache_key],
                key=f"doi_input::{f.name}",
                placeholder="10.1016/j.tox.2022.01.001 (not found — please enter)",
                help="Auto-extracted from the PDF; edit if it looks wrong.",
            )

    run_extraction = st.button(
        "Test this KER against every paper" if targeted else "Extract KERs",
        type="primary", key="run_extraction",
    )

    if run_extraction:
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        elif targeted and not (target_upstream and target_downstream):
            st.error("Name both the upstream and downstream Key Event.")
        elif any(not (doi_overrides.get(f.name) or "").strip() for f in uploaded_files):
            missing = [f.name for f in uploaded_files if not (doi_overrides.get(f.name) or "").strip()]
            st.error("Missing DOI for: " + ", ".join(missing))
        elif not extraction_model.strip():
            st.error("Please enter a model name in the sidebar (e.g. llama3.1:8b).")
        elif s2_provider_cfg["needs_key"] and not transmission_ack:
            # Refused, not warned. A warning that can be clicked past is how a
            # corpus reaches a third party with nobody having decided anything.
            st.error(
                f"{s2_provider_label} would send the full text of every "
                "uploaded paper to a hosted service. Confirm a lawful basis in "
                "the sidebar, or switch to **Ollama (local)** to process them "
                "on this machine.",
                icon="⚠️",
            )
        else:
            llm_cfg = _stage2_config()

            # Open the run before any work happens, so a crash half way still
            # leaves a record of what was attempted and under what settings.
            manifest = _build_run_manifest(llm_cfg)
            manifest.mode = "targeted" if targeted else "open"
            manifest.target_upstream = target_upstream or None
            manifest.target_downstream = target_downstream or None
            gate_log: list[dict[str, Any]] = []
            # One row per uploaded file, whatever happens to it. Without this a
            # paper that failed to read, was judged irrelevant or crashed simply
            # vanished from the summary, and sixteen papers in could look
            # identical to one paper in.
            paper_log: list[dict[str, Any]] = []

            def _log_paper(
                filename: str,
                doi: str,
                outcome: str,
                detail: str = "",
                n_kers: int = 0,
                category: str = "unknown",
                n_llm_calls: int = 0,
                n_truncated: int = 0,
            ) -> None:
                paper_log.append(
                    {
                        "paper": filename,
                        "doi": doi,
                        "outcome": outcome,
                        "KERs saved": n_kers,
                        "why": table1_store.OUTCOME_CATEGORIES.get(category, ""),
                        "detail": detail,
                    }
                )
                # Also to the database. The in-memory list is gone the moment
                # Streamlit reruns the script, which it does on the next click,
                # so "which papers gave nothing, and why" was answerable for
                # about ten seconds after a run that took twenty minutes.
                table1_store.record_paper_outcome(
                    run_id=active_run_id,
                    source_filename=filename,
                    source_doi=doi,
                    outcome=outcome,
                    category=category,
                    reason=detail,
                    n_kers=n_kers,
                    n_llm_calls=n_llm_calls,
                    n_truncated=n_truncated,
                )

            active_run_id = table1_store.start_run(manifest)
            telemetry = run_manifest.start_run(RunTelemetry())
            st.session_state["_last_run_id"] = active_run_id
            st.caption(f"Run #{active_run_id} — " + " · ".join(manifest.summary_lines()))

            # --- Build the search vocabulary once for the whole run ---------
            # Done here rather than per paper because it is the same question
            # every time and each source is cached: the model expansion, the
            # HGNC family resolution and the ontology synonyms are paid for
            # once and reused across all sixteen papers.
            up_vocab = down_vocab = None
            if targeted:
                with st.spinner(
                    "Working out what these two events are called in the "
                    "literature (gene families, ontologies)..."
                ):
                    up_vocab = ke_synonyms.build_vocabulary(target_upstream, llm_cfg)
                    down_vocab = ke_synonyms.build_vocabulary(target_downstream, llm_cfg)
                target_up_terms = list(up_vocab.terms)
                target_down_terms = list(down_vocab.terms)

                with st.expander(
                    f"Search vocabulary — {len(target_up_terms)} upstream, "
                    f"{len(target_down_terms)} downstream terms",
                    expanded=False,
                ):
                    for vocab in (up_vocab, down_vocab):
                        st.markdown(f"**{vocab.label}** — {vocab.summary()}")
                        rows = [
                            ("label", vocab.from_label),
                            ("model", vocab.from_model),
                            ("HGNC", vocab.from_hgnc),
                            ("ontology", vocab.from_ontology),
                        ]
                        for source, terms in rows:
                            if terms:
                                st.caption(f"{source}: {', '.join(terms)}")
                        for note in vocab.notes:
                            st.caption(f"⚠️ {note}")
                        st.divider()

                # A vocabulary this thin means the external lookups failed and
                # the run is back to matching the label literally — the exact
                # condition that makes relevant papers look irrelevant.
                if len(target_up_terms) < 3 or len(target_down_terms) < 3:
                    st.warning(
                        "Only the labels' own words could be resolved into "
                        "search terms. Passages written in laboratory "
                        "vocabulary may score as off-topic. Check the notes in "
                        "the vocabulary panel above — usually this means HGNC "
                        "or the ontology service was unreachable."
                    )

            paper_progress = st.progress(
                0.0, text=f"0 of {len(uploaded_files)} papers processed"
            )

            for paper_index, uploaded_file in enumerate(uploaded_files):
                paper_doi = doi_overrides[uploaded_file.name].strip()
                st.markdown(
                    f"### `{uploaded_file.name}` — DOI `{paper_doi}`  "
                    f"<span style='opacity:0.6;font-size:0.7em'>"
                    f"paper {paper_index + 1} of {len(uploaded_files)}</span>",
                    unsafe_allow_html=True,
                )
                paper_progress.progress(
                    paper_index / len(uploaded_files),
                    text=(
                        f"{paper_index} of {len(uploaded_files)} papers processed "
                        f"— reading {uploaded_file.name}"
                    ),
                )

                # --- Read the PDF into a page/section-aware document --------
                with st.spinner("Reading PDF and detecting sections..."):
                    try:
                        document = extract_document(uploaded_file, doi=paper_doi)
                    except RuntimeError as exc:
                        st.error(str(exc))
                        _log_paper(
                            uploaded_file.name, paper_doi,
                            "PDF could not be read", str(exc)[:300],
                            category="no_text",
                        )
                        continue

                st.caption(
                    f"{document.n_pages} pages · {len(document.chunks)} chunks · "
                    f"{len(document.full_text):,} characters"
                    + (f" · title: {document.title}" if document.title else "")
                )

                # --- This paper's own abbreviations -------------------------
                # Which shorthand an author picks is that lab's convention:
                # one paper writes OL, the next OPC, a third spells it out.
                # Reading the definitions out of the document turns that from
                # a guess into a fact about the paper in hand.
                paper_up_terms = list(target_up_terms)
                paper_down_terms = list(target_down_terms)
                if targeted:
                    for side_terms, side_name in (
                        (paper_up_terms, "upstream"),
                        (paper_down_terms, "downstream"),
                    ):
                        try:
                            found = ke_synonyms.abbreviations_for_terms(
                                document.full_text, side_terms
                            )
                        except Exception:  # noqa: BLE001
                            found = {}
                        new = [a for a in found if a.lower() not in
                               {t.lower() for t in side_terms}]
                        side_terms.extend(new)
                        if new:
                            st.caption(
                                f"This paper defines {side_name} shorthand: "
                                + ", ".join(
                                    f"**{a}** = {found[a]}" for a in new[:6]
                                )
                            )

                # --- Score and select chunks --------------------------------
                if use_chunking:
                    with st.spinner("Scoring chunks for mechanistic relevance..."):
                        paper_text, selected, report = prepare_paper_text(
                            document.chunks,
                            cfg=llm_cfg,
                            use_llm_triage=use_llm_triage,
                            char_budget=int(chunk_budget),
                            min_score=float(chunk_min_score),
                            target_kes=(
                                [target_upstream, target_downstream]
                                if targeted else None
                            ),
                            # The curator's vocabulary, kept in two groups so a
                            # passage is judged on whether it speaks about both
                            # events rather than on how much of one long word
                            # list it happens to contain.
                            target_term_groups=(
                                [paper_up_terms, paper_down_terms]
                                if targeted and (paper_up_terms or paper_down_terms)
                                else None
                            ),
                        )
                    st.success(
                        f"Selected {report.n_selected} of {report.n_chunks} chunks "
                        f"({report.chars_selected:,} of {report.chars_total:,} chars, "
                        f"{report.reduction_pct:.0f}% reduction) using the "
                        f"{report.method} scorer."
                    )
                    # How much of the paper the model actually saw is part of
                    # the run's conditions: a KER absent from the output may
                    # simply have been in a chunk that was never sent.
                    run_manifest.record(
                        "chunk_selection",
                        n_chunks=report.n_chunks,
                        n_selected=report.n_selected,
                        chars_total=report.chars_total,
                        chars_selected=report.chars_selected,
                    )
                    manifest.chunk_scorer = report.method
                    if report.llm_error:
                        st.warning(f"LLM triage failed, heuristic scores used. {report.llm_error}")
                        run_manifest.record(
                            "note",
                            message=f"LLM chunk triage failed; heuristic scores used ({report.llm_error}).",
                        )

                    with st.expander("Chunk relevance detail", expanded=False):
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "chunk": c.chunk_id,
                                        "section": c.section,
                                        "pages": c.page_label,
                                        "chars": len(c.text),
                                        "score": c.relevance_score,
                                        "selected": c.selected,
                                        "why": c.relevance_reason,
                                    }
                                    for c in sorted(
                                        document.chunks,
                                        key=lambda c: c.relevance_score,
                                        reverse=True,
                                    )
                                ]
                            ),
                            use_container_width=True,
                            height=320,
                        )
                else:
                    paper_text = document.full_text
                    st.caption("Chunk scoring disabled — sending the whole paper.")
                    run_manifest.record(
                        "chunk_selection",
                        n_chunks=len(document.chunks),
                        n_selected=len(document.chunks),
                        chars_total=len(document.full_text),
                        chars_selected=len(paper_text),
                    )

                # Keep the text. A chain that stops at a marker raises the
                # question of what the paper says downstream of it, and that
                # question should not require the PDFs to be uploaded again.
                try:
                    table1_store.store_chunks(
                        document.chunks,
                        source_doi=paper_doi,
                        source_filename=document.filename,
                        run_id=active_run_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Storing evidence is useful, not essential; a failure here
                    # must not cost the extraction that has already been paid for.
                    st.caption(f"Could not store paper text: {exc}")

                # --- Run the extraction pipeline ----------------------------
                debug_container = st.expander(
                    "Per-step LLM debug (prompts + raw responses)", expanded=False
                )
                step_log: list = []

                def _on_step(step_result, _container=debug_container, _log=step_log):
                    _log.append(step_result)
                    status = "OK" if step_result.ok else "FAIL"
                    with _container:
                        quotes = ""
                        if step_result.n_quotes:
                            quotes = (
                                f" — {step_result.n_verified}/{step_result.n_quotes} "
                                "quotations verified"
                            )
                        trunc = " — ⚠️ truncated, partial result recovered" if step_result.truncated else ""
                        st.markdown(f"**[{status}] {step_result.step}**{quotes}{trunc}")
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

                if targeted:
                    with st.spinner(
                        f"Reading the causal chain this paper supports between "
                        f"{target_upstream} and {target_downstream}..."
                    ):
                        try:
                            extractions, pathway, warnings = ker_extractor.extract_pathway_rows(
                                document,
                                target_upstream,
                                target_downstream,
                                cfg=llm_cfg,
                                paper_text=paper_text,
                                on_step=_on_step,
                                budget_scale=float(budget_scale),
                                directional=directional,
                                upstream_aliases=paper_up_terms,
                                downstream_aliases=paper_down_terms,
                            )
                        except ExtractionError as exc:
                            st.error(str(exc))
                            _log_paper(
                                uploaded_file.name, paper_doi,
                                "Extraction failed", str(exc)[:300],
                                category="provider_error",
                                n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            )
                            continue
                        except LLMAuthError:
                            raise  # a bad key will fail every paper; stop now
                        except Exception as exc:  # noqa: BLE001
                            # One malformed PDF or one dropped connection should
                            # cost one paper, not the fifteen queued behind it.
                            st.error(
                                f"Unexpected error on this paper: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            _log_paper(
                                uploaded_file.name, paper_doi,
                                "Unexpected error",
                                f"{type(exc).__name__}: {exc}"[:300],
                                category="error",
                                n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            )
                            continue

                    gate_log.append(
                        {
                            "paper": document.filename or paper_doi,
                            "doi": paper_doi,
                            "steps": pathway.n_steps,
                            "events": " → ".join(pathway.events) if pathway.events else "",
                            "reason": pathway.reason,
                            "problem": pathway.error or "",
                            "calls": sum(max(1, r.attempts) for r in step_log),
                        }
                    )

                    if pathway.steps:
                        st.success(
                            f"{pathway.n_steps} causal link(s) across "
                            f"{len(pathway.events)} event(s) — "
                            + " → ".join(pathway.events)
                        )
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "from": s_.from_event,
                                        "to": s_.to_event,
                                        "direction": s_.direction,
                                        "adjacency": s_.adjacency,
                                        "contradicts": s_.contradicts,
                                        "quoted": sum(1 for q in s_.spans if q.verified),
                                        "what the paper shows": s_.description,
                                    }
                                    for s_ in pathway.steps
                                ]
                            ),
                            use_container_width=True,
                        )
                    else:
                        # `reason` is the model's own account of why it found
                        # nothing, and it is empty whenever the model never
                        # answered. Printing it alone produced "no reason
                        # given" directly above the line stating the reason.
                        st.caption(
                            "No pathway from this paper — "
                            + (
                                pathway.reason
                                or pathway.error
                                or "the model returned no steps and gave no reason"
                            )
                        )

                    if pathway.error and pathway.error != pathway.reason:
                        st.warning(pathway.error)
                    for warning in warnings:
                        st.warning(warning)
                    st.caption(
                        f"Model calls for this paper: "
                        f"{sum(max(1, r.attempts) for r in step_log)}"
                        + (
                            f" ({len(step_log)} steps, some retried after a refusal)"
                            if any(r.attempts > 1 for r in step_log) else ""
                        )
                    )

                    if not extractions:
                        # "No pathway found" is several different results
                        # wearing one label. The model reading the paper and
                        # reporting no link is a finding; a reply cut off at
                        # the token ceiling, or one the safety classifier
                        # declined, is a false negative that looks exactly the
                        # same from here. The step log and the error know which.
                        n_truncated = sum(1 for s in step_log if s.truncated)
                        declined = "declined" in str(pathway.error or "").lower()
                        category = (
                            "refusal" if declined
                            else "truncated" if n_truncated
                            else "parse_failure" if any(not s.ok for s in step_log)
                            else "chunking_dropped" if use_chunking
                            else "no_mechanism"
                        )
                        if declined:
                            # The remedy is now a setting rather than an
                            # instruction, so say which of the two situations
                            # this is. "Run it through another provider" printed
                            # at someone who has already configured one is
                            # advice for a version of the tool they are not
                            # using.
                            _fb = ker_extractor.refusal_fallback()
                            _tries = sum(max(1, s.attempts) for s in step_log) or len(
                                ker_extractor.REFUSAL_RETRY_TEMPERATURES
                            )
                            st.error(
                                f"Declined {_tries} time(s) in a row"
                                + (
                                    f", including by the fallback "
                                    f"{_fb.provider}/{_fb.model}"
                                    if _fb is not None else ""
                                )
                                + ". A refusal is decided on the reply as it is "
                                "written, and the reply is sampled — so this is "
                                "unusual, and **re-running this paper on its own "
                                "will often work**. It is not a judgement about "
                                "the paper."
                                + (
                                    ""
                                    if _fb is not None else
                                    " You can also set a last-resort model under "
                                    "**If the provider declines**; a second "
                                    "Claude model on the same key counts."
                                ),
                                icon="🚫",
                            )
                        _log_paper(
                            uploaded_file.name,
                            paper_doi,
                            "Declined by the provider" if declined
                            else "No pathway found",
                            pathway.error or pathway.reason,
                            category=category,
                            # Refused attempts are calls that were made and
                            # charged for. Counting only the steps that came
                            # back reported a refused paper as costing zero.
                            n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            n_truncated=n_truncated,
                        )
                        continue
                else:
                    with st.spinner(
                        f"Running stepwise extraction with {extraction_model} "
                        f"({s2_provider_label}) — this may take a few minutes..."
                    ):
                        try:
                            extractions, warnings = extract_kers_from_document(
                                document,
                                cfg=llm_cfg,
                                paper_text=paper_text,
                                on_step=_on_step,
                                budget_scale=float(budget_scale),
                            )
                        except ExtractionError as exc:
                            st.error(str(exc))
                            _log_paper(
                                uploaded_file.name, paper_doi,
                                "Extraction failed", str(exc)[:300],
                                category="provider_error",
                                n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            )
                            continue
                        except LLMAuthError:
                            raise  # a bad key will fail every paper; stop now
                        except Exception as exc:  # noqa: BLE001
                            st.error(
                                f"Unexpected error on this paper: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            _log_paper(
                                uploaded_file.name, paper_doi,
                                "Unexpected error",
                                f"{type(exc).__name__}: {exc}"[:300],
                                category="error",
                                n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            )
                            continue

                    n_failed_steps = sum(1 for s in step_log if not s.ok)
                    st.caption(
                        f"Model calls: {sum(max(1, r.attempts) for r in step_log)} "
                        f"across {len(step_log)} steps (failures: {n_failed_steps})"
                    )

                    for warning in warnings:
                        st.warning(warning)

                    if not extractions:
                        n_truncated = sum(1 for s in step_log if s.truncated)
                        category = (
                            "truncated" if n_truncated
                            else "parse_failure" if n_failed_steps
                            else "chunking_dropped" if use_chunking
                            else "no_mechanism"
                        )
                        st.warning(
                            f"No KERs extracted from `{uploaded_file.name}` — "
                            + table1_store.OUTCOME_CATEGORIES.get(category, "")
                        )
                        _log_paper(
                            uploaded_file.name,
                            paper_doi,
                            "No KERs found",
                            (
                                f"{n_failed_steps} of {len(step_log)} model calls "
                                "were unusable — see the debug expander"
                                if n_failed_steps
                                else "; ".join(warnings)[:300]
                                or "The model reported no relationship in the text it was sent."
                            ),
                            category=category,
                            n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                            n_truncated=n_truncated,
                        )
                        continue

                total_spans = sum(len(e.evidence_spans) for e in extractions)
                verified_spans = sum(e.n_verified_spans for e in extractions)
                st.success(
                    f"Extracted {len(extractions)} KER(s) with {total_spans} quotation(s), "
                    f"{verified_spans} located verbatim in the paper. "
                    "Looking up AOP-Wiki IDs..."
                )

                # --- AOP-Wiki enrichment + insert ---------------------------
                inserted = 0
                wiki_progress = st.progress(0)
                for i, extraction in enumerate(extractions):
                    with st.spinner(
                        f"AOP-Wiki lookup {i + 1}/{len(extractions)}: "
                        f"{extraction.ker_name[:60]}..."
                    ):
                        wiki_ids = enrich_ker(
                            upstream_ke_name=extraction.upstream_ke_name,
                            downstream_ke_name=extraction.downstream_ke_name,
                        )
                    insert_table1_row(
                        extraction=extraction,
                        source_doi=paper_doi,
                        wiki_ids=wiki_ids,
                        source_filename=document.filename,
                        source_title=document.title,
                        run_id=active_run_id,
                    )
                    inserted += 1
                    wiki_progress.progress(inserted / len(extractions))

                wiki_progress.empty()
                st.success(f"Saved {inserted} row(s) to Table 1.")
                _log_paper(
                    uploaded_file.name, paper_doi, "KERs saved", "",
                    n_kers=inserted, category="saved",
                    n_llm_calls=sum(max(1, s.attempts) for s in step_log),
                    n_truncated=sum(1 for s in step_log if s.truncated),
                )
                invalidate_pipeline()

            paper_progress.progress(
                1.0, text=f"{len(uploaded_files)} of {len(uploaded_files)} papers processed"
            )

            # --- What happened to every paper -------------------------------
            # Printed before the targeted summary because it is the question a
            # run answers first: of the files handed over, which produced
            # something, and for the rest, why not.
            st.divider()
            st.subheader("Per-paper outcome")
            paper_df = pd.DataFrame(paper_log)
            n_saved = int((paper_df["KERs saved"] > 0).sum()) if not paper_df.empty else 0
            p1, p2, p3 = st.columns(3)
            p1.metric("Papers uploaded", len(uploaded_files))
            p2.metric("Papers yielding KERs", n_saved)
            p3.metric(
                "KERs saved",
                int(paper_df["KERs saved"].sum()) if not paper_df.empty else 0,
            )
            if not paper_df.empty:
                st.dataframe(paper_df, use_container_width=True)
                st.download_button(
                    "Download per-paper outcomes CSV",
                    _csv_bytes(paper_df),
                    "paper_outcomes.csv",
                    "text/csv",
                    key="dl_paper_outcomes",
                )
                n_broken = int(
                    paper_df["outcome"]
                    .isin(["PDF could not be read", "Extraction failed"])
                    .sum()
                ) + int(paper_df["outcome"].str.startswith("Demoted").sum())
                if n_saved <= 1 and len(uploaded_files) > 2:
                    st.warning(
                        f"{len(uploaded_files) - n_saved} of {len(uploaded_files)} "
                        "papers produced nothing. "
                        + (
                            f"{n_broken} of those failed or were demoted rather "
                            "than being judged irrelevant — read the detail "
                            "column before treating this as a result about the "
                            "literature."
                            if n_broken
                            else "The detail column says why for each one."
                        )
                    )

            # --- Pathway summary across all papers -------------------------
            # The point of the run: not how many papers agreed with a
            # proposition, but what chain the corpus as a whole describes.
            if targeted and gate_log:
                st.divider()
                st.subheader(f"{target_upstream} → {target_downstream}")
                gate_df = pd.DataFrame(gate_log)
                contributing = gate_df[gate_df["steps"] > 0]

                # Count distinct events and links across the whole corpus, which
                # is the size of the graph about to be drawn.
                all_events: list[str] = []
                for chain in contributing["events"]:
                    for event in str(chain).split(" → "):
                        event = event.strip()
                        if event and event not in all_events:
                            all_events.append(event)

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Papers read", len(gate_df))
                g2.metric("Papers contributing", len(contributing))
                g3.metric("Causal links", int(gate_df["steps"].sum()))
                g4.metric("Distinct events", len(all_events))

                st.dataframe(gate_df, use_container_width=True)
                st.download_button(
                    "Download pathway summary CSV",
                    _csv_bytes(gate_df),
                    "pathway_summary.csv",
                    "text/csv",
                )

                if contributing.empty:
                    st.warning(
                        "No paper in this set contributed a link. Check the "
                        "wording of both events — the chain is built between "
                        "the two you named, so an event described more "
                        "narrowly than the papers treat it will find nothing."
                    )
                else:
                    st.info(
                        f"{len(all_events)} events are now on the graph. "
                        "Events named identically by different papers are "
                        "already one node; the rest are merged in "
                        "**2 · Normalize & curate**, and the pathway is drawn "
                        "in the AOP map."
                    )

            # --- Close the run --------------------------------------------
            run_manifest.end_run()
            table1_store.finish_run(
                active_run_id,
                telemetry,
                status="completed",
                model_reported=telemetry.model_reported,
                chunk_scorer=manifest.chunk_scorer,
            )
            if telemetry.repair_rate or telemetry.failure_rate:
                st.caption(
                    f"Run #{active_run_id}: {telemetry.llm_calls} model calls, "
                    f"{telemetry.step_failures} unusable replies, "
                    f"{telemetry.json_repairs} repaired. "
                    "See the QC report below before curating."
                )

            st.info(
                "Next: open **Normalization & ontology** to merge equivalent Key "
                "Events before viewing the map."
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Table 1 viewer — raw extraction
    # -----------------------------------------------------------------------

    pipeline = load_pipeline()
    t1_df = pipeline["table1"]

    ui_common.section_heading(
        "Table 1 — raw per-paper extraction",
        "Exactly what each paper yielded, in that paper's own terminology. "
        "Nothing here is merged or interpreted.",
        help_text=(
            "One row per relationship a paper reported. Two papers describing "
            "the same biology in different words produce two rows here, and "
            "they stay two rows until you merge them in step 2 — this table "
            "is the record of what was said, not of what it means.\n\n"
            "**Cite** is the first author and year, fetched from Crossref "
            "using the DOI. A letter is added where the same author published "
            "more than once in that year. Where a DOI could not be resolved "
            "the DOI itself is shown instead."
        ),
    )

    if t1_df.empty:
        st.info("No rows yet. Extract KERs from a paper above.")
    else:
        papers = list_source_papers()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers", len(papers))
        c2.metric("KER claims", len(t1_df),
                  help="One row per relationship per paper. Not a count of Key Events.")
        c3.metric("Quotations", int(t1_df["n_evidence_spans"].fillna(0).sum()))
        verified_total = int(t1_df["n_verified_spans"].fillna(0).sum())
        spans_total = int(t1_df["n_evidence_spans"].fillna(0).sum())
        c4.metric(
            "Verified quotations",
            verified_total,
            f"{100 * verified_total / spans_total:.0f}%" if spans_total else None,
        )

        ui_common.count_chain()

        with st.expander("Papers ingested", expanded=False):
            st.dataframe(papers, use_container_width=True)

        # Papers that gave nothing, and why. Previously this existed only as a
        # table drawn during the run, which Streamlit discards on the next
        # click — so the two papers out of thirteen that produced no rows were
        # unaccounted for the moment the run finished.
        _outcomes = table1_store.load_paper_outcomes(
            st.session_state.get("_last_run_id")
        )
        if not _outcomes.empty:
            _barren = _outcomes[_outcomes["n_kers"].fillna(0) <= 0]
            if not _barren.empty:
                _recoverable = int(_barren["category"].isin(
                    {"truncated", "parse_failure", "provider_error",
                     "chunking_dropped", "no_text", "error"}
                ).sum())
                with st.expander(
                    f"⚠️ {len(_barren)} paper(s) produced no rows"
                    + (f" — {_recoverable} worth re-running" if _recoverable else ""),
                    expanded=bool(_recoverable),
                ):
                    st.caption(
                        "A paper the model read and found no mechanism in is a "
                        "finding. A paper whose reply was truncated or failed "
                        "to parse is a gap in this corpus, and it is invisible "
                        "in every table above."
                    )
                    for _, row in _barren.iterrows():
                        st.markdown(
                            f"**{row.get('source_filename') or row.get('source_doi')}** "
                            f"— {row.get('outcome')}"
                        )
                        st.caption(
                            table1_store.OUTCOME_CATEGORIES.get(
                                str(row.get("category")), ""
                            )
                        )
                        if str(row.get("reason") or "").strip():
                            st.caption(f"Detail: {row['reason']}")

        # --- Add a claim the run missed -------------------------------------
        #
        # Placed here, immediately under the outcome of the run, because this
        # is where the gap becomes visible: the papers are on screen and the
        # reasons some of them produced nothing are on screen. The curator
        # either re-runs them or writes down what they already know is in
        # them. Having to leave for another tab to do the second thing is what
        # made "the extraction missed one" a dead end.
        ui_manual_claim.add_claim_expander(
            key_prefix="extract_add",
            label="✚ Add a relationship the extraction missed",
        )

        # Confidence is the model's self-assessment, so a hand-entered row has
        # no honest value for it and carries a marker instead. That marker
        # still has to be selectable here: filtering the table by confidence
        # and thereby hiding every curator row is how a claim gets entered and
        # then silently disappears from the corpus it was added to.
        _conf_options = ["High", "Medium", "Low"]
        _extra_conf = sorted(
            set(t1_df["extraction_confidence"].dropna().astype(str))
            - set(_conf_options)
        )
        conf_filter = st.multiselect(
            "Filter by extraction confidence",
            options=_conf_options + _extra_conf,
            default=_conf_options + _extra_conf,
            help=(
                "Curator-entered rows have no model confidence, so they carry "
                "their own marker rather than a score nobody assigned."
            ),
        )
        filtered = t1_df[t1_df["extraction_confidence"].isin(conf_filter)]

        # A DOI column identifies a paper without letting anyone recognise it.
        # The citation key goes first, and the DOI stays in the table because
        # it is what resolves.
        _t1_keys = ui_common.citation_keys(filtered["source_doi"].tolist())

        # Two views of the same rows. The reading view puts the two ends of the
        # relationship, what the paper did to each, and the sign of the link
        # first; `SELECT *` put those four columns at positions 41 to 44, past
        # the right-hand edge, behind eight columns of provenance. The full view
        # is still one click away because the fields it holds — applicability,
        # quantitative relationships, run id — are what somebody checking a row
        # needs, just not what somebody reading one does.
        _view = st.radio(
            "Columns",
            ["Reading view", "Every extracted column"],
            horizontal=True,
            key="t1_view_mode",
            help=(
                "**Reading view** is one row per KER claim in the order a KER "
                "is defined: upstream event, what happened to it, the sign of "
                "the link, downstream event, what happened to it, then the "
                "relationship itself.\n\n"
                "**Every extracted column** is the stored row untouched, in "
                "schema order, including provenance and applicability."
            ),
        )

        if _view == "Reading view":
            _display = ui_common.table1_reading_view(filtered)
        else:
            _display = filtered.copy()
        _display.insert(
            0,
            "Cite",
            filtered["source_doi"].map(lambda d: ui_common.cite(d, _t1_keys)),
        )

        st.dataframe(_display, use_container_width=True, height=380)
        st.caption(
            f"{len(filtered)} of {len(t1_df)} rows shown — one row per KER "
            f"claim, in the paper's own words."
        )
        if _view == "Reading view":
            st.caption(
                "⚠️ Two rows naming the same pair of events are not "
                "duplicates. **Upstream change** and **Downstream change** are "
                "what separate them: blocking a channel and activating it are "
                "two claims about one pair, and the next step groups the *event "
                "names*, never the claims."
            )

        _unresolved = sorted(
            {
                str(d).strip().lower()
                for d in filtered["source_doi"]
                if ui_common.cite(d, _t1_keys) == str(d).strip().lower()
            }
        )
        if _unresolved:
            with st.expander(f"{len(_unresolved)} paper(s) have no citation key"):
                st.caption(
                    "Crossref could not resolve these DOIs, so they are shown "
                    "as DOIs. Usually the DOI was misread out of the PDF, or "
                    "the paper is a preprint Crossref does not hold."
                )
                for doi in _unresolved:
                    st.markdown(f"- `{doi}`")
                if st.button("Retry these lookups", key="t1_refresh_citations"):
                    for doi in _unresolved:
                        citations.forget(doi)
                    st.session_state.pop("_citation_keys", None)
                    st.rerun()

        st.download_button(
            "Download Table 1 CSV",
            _csv_bytes(filtered),
            "table1_extractions.csv",
            "text/csv",
        )

        # -------------------------------------------------------------------
        # Run history + QC report
        # -------------------------------------------------------------------
        st.markdown("#### Run provenance & QC")
        st.caption(
            "Every row above was produced under a specific model, prompt "
            "version and chunk budget. Two runs are only comparable when those "
            "match, so each run is recorded and can be reported on."
        )

        runs_df = table1_store.load_runs("extraction")
        if runs_df.empty:
            st.info(
                "No runs recorded yet. Rows extracted before this version of "
                "the app have no run manifest and cannot be attributed to a "
                "model or prompt version."
            )
            qc_run_id = None
        else:
            run_labels = {
                int(r["run_id"]): (
                    f"#{int(r['run_id'])} · {r['started_at']} · "
                    f"{r['provider']}/{r['model']} · "
                    f"{int(r['kers_extracted'] or 0)} KERs · {r['status']}"
                )
                for _, r in runs_df.iterrows()
            }
            options: list = ["All rows"] + list(run_labels.keys())
            default_index = (
                options.index(st.session_state["_last_run_id"])
                if st.session_state.get("_last_run_id") in options
                else 1
            )
            qc_choice = st.selectbox(
                "Report on",
                options=options,
                index=default_index,
                format_func=lambda o: o if isinstance(o, str) else run_labels[o],
                key="qc_run_select",
            )
            qc_run_id = None if qc_choice == "All rows" else int(qc_choice)

            with st.expander("All runs", expanded=False):
                st.dataframe(runs_df, use_container_width=True, height=240)

        if st.button("Build QC report", key="build_qc"):
            with st.spinner("Assembling QC report..."):
                st.session_state["_qc"] = qc_report.build_qc_report(
                    qc_run_id, table2_df=pipeline.get("table2_normalized")
                )

        qc = st.session_state.get("_qc")
        if qc is not None:
            for flag in qc.flags:
                st.warning(flag)
            if not qc.flags:
                st.success("No quality flags raised for this scope.")

            q1, q2, q3 = st.columns(3)
            q1.metric("Quotations", qc.n_spans)
            q2.metric("Verified", qc.n_verified, f"{qc.verification_rate:.0%}")
            q3.metric("Contradicting rows", qc.n_contradicted_rows)

            if not qc.per_paper.empty:
                st.caption("Verification by paper — worst first.")
                st.dataframe(qc.per_paper, use_container_width=True, height=220)

            d1, d2, d3 = st.columns(3)
            stem = f"qc_report_run{qc.run_id}" if qc.run_id else "qc_report_all"
            with d1:
                st.download_button(
                    "QC report (Markdown)",
                    qc_report.report_markdown(qc).encode("utf-8"),
                    f"{stem}.md",
                    "text/markdown",
                )
            with d2:
                st.download_button(
                    "QC report (JSON)",
                    qc_report.report_json(qc).encode("utf-8"),
                    f"{stem}.json",
                    "application/json",
                )
            with d3:
                st.download_button(
                    "Unverified quotations (CSV)",
                    with_disclaimer(qc_report.unverified_quotes_csv(qc)).encode("utf-8"),
                    f"{stem}_unverified.csv",
                    "text/csv",
                )

        # -------------------------------------------------------------------
        # Check a claim against its paper
        #
        # This used to be a section of its own called "Evidence for a single
        # extracted KER", sitting alongside a second evidence display on the
        # AOP map and a third in Table 2. Three views of the same thing that
        # did not agree is worse than one that is imperfect, so the provenance
        # now hangs off the claim it belongs to and exists nowhere else. It
        # was then called the "provenance drawer", which named the mechanism
        # rather than the job and left readers guessing what was in it.
        # -------------------------------------------------------------------
        ui_common.section_heading(
            "Check a claim against its paper",
            "Pick one extracted claim to read what the model wrote about it "
            "and the verbatim sentences from the paper that each part rests on.",
            help_text=(
                "**Where this comes from.** Nothing here is written by hand. "
                "During extraction the model is asked to answer each field — "
                "mechanistic basis, biological plausibility, empirical "
                "evidence, and so on — and, for every answer, to copy out the "
                "sentences in the paper it used. Those quotations are then "
                "searched for in the PDF text character by character.\n\n"
                "**Verified** means the quotation was found in the paper "
                "exactly as the model gave it. **Not located verbatim** means "
                "it was not, which usually means the model paraphrased and "
                "occasionally means it invented the sentence. Either way the "
                "claim above it is unsupported until you have read the paper.\n\n"
                "This is the only place evidence for a single claim is shown. "
                "It used to appear here, on the map and in Table 2, and the "
                "three did not agree."
            ),
            level="subheader",
        )
        _drawer_keys = ui_common.citation_keys(filtered["source_doi"].tolist())
        row_labels = {
            int(r["record_id"]): (
                f"{ui_common.cite(r['source_doi'], _drawer_keys)} · "
                f"{str(r['ker_name'])[:70]}"
            )
            for _, r in filtered.iterrows()
        }
        if row_labels:
            chosen_record = st.selectbox(
                "Extracted claim",
                options=list(row_labels.keys()),
                format_func=lambda rid: row_labels[rid],
                key="t1_record_select",
            )
            spans = load_evidence_spans([chosen_record])
            record = filtered[filtered["record_id"] == chosen_record]
            row = record.iloc[0] if not record.empty else None

            # The extraction IS a written assessment — description, biological
            # plausibility, empirical evidence, essentiality — and the panel
            # used to show none of it, only the quotations underneath. A flat
            # list of sentences pulled from different sections of a paper does
            # not read as anything; the prose is what the quotations support.
            if row is not None:
                st.markdown(f"### {row.get('ker_name', '')}")
                st.caption(
                    f"{row.get('upstream_ke_name')} ({row.get('upstream_ke_level')}) → "
                    f"{row.get('downstream_ke_name')} ({row.get('downstream_ke_level')}) · "
                    f"{row.get('ker_adjacency')} · confidence "
                    f"{row.get('extraction_confidence')}"
                )
                # Where the row itself came from, before where its claim came
                # from. A curator-entered row rendered identically to an
                # extracted one is the single change that would make this panel
                # dishonest, and this panel is the tool's central promise.
                _origin = str(row.get("origin") or "llm")
                if _origin != "llm":
                    _who = str(row.get("entered_by") or "").strip() or "a curator"
                    _label = {
                        "curator": f"✍️ Entered by hand by {_who}",
                        "curator_edited": f"✍️ Extracted by the model, then corrected by {_who}",
                        "imported": "📄 Imported from a spreadsheet",
                    }.get(_origin, f"✍️ {_origin}")
                    st.info(
                        _label
                        + (f" — {row.get('entry_rationale')}"
                           if str(row.get("entry_rationale") or "").strip() else "")
                    )

                # The citation key is what a curator recognises; the DOI is what
                # they need in order to open the paper. Both, in that order.
                if str(row.get("source_doi")) == manual_entry.NO_SOURCE_DOI:
                    st.caption(
                        "**No source** — recorded as the curator's own "
                        "assessment rather than a reading of a paper."
                    )
                else:
                    st.caption(
                        f"**{ui_common.cite(row.get('source_doi'), _drawer_keys)}** · "
                        f"[{row.get('source_doi')}](https://doi.org/{row.get('source_doi')})"
                    )
                if bool(row.get("contradicts_ker")):
                    st.warning(
                        "This paper was recorded as arguing AGAINST the "
                        "relationship."
                    )

            # Each narrative field, then the quotations that were captured for
            # it — so a claim and its support sit together instead of the
            # reader having to match them up by field name.
            _SECTIONS = [
                ("ker_description", "Mechanistic basis"),
                ("biological_plausibility", "Biological plausibility"),
                ("empirical_evidence_summary", "Empirical evidence"),
                ("essentiality_evidence", "Essentiality"),
                ("quantitative_relationships", "Quantitative relationship"),
                ("response_response_relationship", "Response–response"),
                ("time_scale", "Time scale"),
                ("modulating_factors", "Modulating factors"),
                ("ker_link", "Link between the two events"),
            ]
            shown_fields: set[str] = set()

            for field_key, heading in _SECTIONS:
                text = row.get(field_key) if row is not None else None
                field_spans = (
                    spans[spans["field"] == field_key] if not spans.empty else spans
                )
                if (text is None or not str(text).strip() or str(text) == "nan") \
                        and (field_spans is None or field_spans.empty):
                    continue

                shown_fields.add(field_key)
                st.markdown(f"**{heading}**")
                if text is not None and str(text).strip() and str(text) != "nan":
                    st.write(str(text))
                if field_spans is not None and not field_spans.empty:
                    for _, span in field_spans.iterrows():
                        badge = "✅" if span["verified"] else "⚠️ not located verbatim"
                        st.markdown(
                            f"> {span['quote']}  \n"
                            f"<span style='opacity:0.65;font-size:0.85em'>"
                            f"{badge} · {span['citation']}</span>",
                            unsafe_allow_html=True,
                        )
                st.markdown("")

            # Applicability is a set of short values, not prose — a table reads
            # better than four one-line paragraphs.
            if row is not None:
                st.markdown("**Applicability & study**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Field": label, "Value": _fmt(row.get(key))}
                            for key, label in (
                                ("taxonomic_applicability", "Taxonomic"),
                                ("sex_applicability", "Sex"),
                                ("life_stage_applicability", "Life stage"),
                                ("study_design", "Study design"),
                                ("exposure_route", "Exposure route"),
                                ("chemical_stressor", "Stressor"),
                                ("paper_type", "Paper type"),
                            )
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            leftover = (
                spans[~spans["field"].isin(shown_fields)] if not spans.empty else spans
            )
            if leftover is not None and not leftover.empty:
                with st.expander(f"Other quotations ({len(leftover)})"):
                    for _, span in leftover.iterrows():
                        badge = "✅ verified" if span["verified"] else "⚠️ not located verbatim"
                        st.markdown(
                            f"> {span['quote']}\n\n"
                            f"*{badge} · supports: {span['field']} · {span['citation']}*"
                        )

            if spans.empty:
                st.info("No quotations were recorded for this row.")

            # --- Correcting the row ----------------------------------------
            #
            # Adding a claim without being able to fix one is half a loop. A
            # curator who spots a wrong direction and can only add a second row
            # saying the opposite leaves the corpus asserting both, and the
            # synthesis reads it as a genuine disagreement in the literature.
            if row is not None:
                st.divider()
                _edit_col, _del_col = st.columns([3, 1])
                with _edit_col:
                    with st.expander("✎ Correct this claim"):
                        ui_manual_claim.render_form(
                            key_prefix=f"edit_{chosen_record}",
                            record=table1_store.load_record(int(chosen_record)),
                        )
                with _del_col:
                    with st.expander("Remove"):
                        st.caption(
                            "Archived to the record history, not erased."
                        )
                        _why = st.text_input(
                            "Reason", key=f"del_why_{chosen_record}",
                            placeholder="misread figure",
                        )
                        if st.button(
                            "Delete this claim", key=f"del_{chosen_record}",
                            disabled=not _why.strip(),
                        ):
                            table1_store.delete_record(
                                int(chosen_record),
                                curator=ui_common.curator_name(),
                                reason=_why.strip(),
                            )
                            ui_common.invalidate_pipeline()
                            st.success("Removed. Re-run normalization to "
                                       "update the pathway.")
                            st.rerun()

                _history = table1_store.load_record_history(int(chosen_record))
                if not _history.empty:
                    with st.expander(
                        f"Record history ({len(_history)} earlier version(s))"
                    ):
                        st.dataframe(
                            _history[
                                ["archived_at", "action", "curator", "reason"]
                            ],
                            use_container_width=True, hide_index=True,
                        )

    # -----------------------------------------------------------------------
    # Table 2 and the weight-of-evidence assessment used to be rendered here.
    #
    # Both were synthesis: they read across papers, consolidated relationships
    # and printed a confidence band, over Key Events that at this point in the
    # workflow nobody has looked at. That is the ordering the redesign
    # removes — a paragraph reading "the weight of evidence is moderate" over
    # unmerged duplicates is wrong in a way no reader can detect.
    #
    # Consolidation now happens after curation and approval, in
    # "4 · Synthesize evidence", one page per canonical KER.
    # -----------------------------------------------------------------------

    st.divider()
    st.caption(
        "Cross-paper synthesis happens after curation. Once the Key Events "
        "above have been normalised and approved, open **4 · Synthesize "
        "evidence** for the consolidated assessment of each relationship."
    )


# ===========================================================================
# TAB 3 — Normalize & curate
#
# Everything that used to live in "Normalization & ontology", "Explore by Key
# Event" and "Review & curation" now happens in one workspace. Those three
# tabs each had their own merge and rename controls writing to the same
# tables, so a Key Event's history depended on which screen had been used to
# change it.
# ===========================================================================

with tab3:
    ui_curate.render(
        ols4_enabled=bool(ols4_enabled),
        ols4_min_score=float(ols4_min_score),
    )


# ===========================================================================
# TAB 4 — Approve
# ===========================================================================

with tab4:
    ui_approve.render()


# ===========================================================================
# TAB 5 — Synthesize and evaluate evidence
# ===========================================================================

with tab5:
    ui_synthesis.render(llm_config_factory=_stage2_config)


# ===========================================================================
# TAB 6 — Final AOP
# ===========================================================================

with tab6:
    ui_aop_map.render()
