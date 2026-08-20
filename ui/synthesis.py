from __future__ import annotations

"""
The one evidence page per canonical KER.

There used to be three places to read about a relationship: the Table 2 row,
the evidence panel hanging off the AOP map, and the per-record provenance
browser. They disagreed — one showed raw groupings, one showed normalised
ones, and none of them said which was authoritative. A curator comparing two
of them had no way to tell which was wrong.

There is now one page per canonical KER, and it only exists once both of its
Key Events are approved. It lays out, in order:

    1. identity            what relationship this is
    2. study evidence      one block per independent study
    3. biological plausibility
    4. empirical support   concordance across studies
    5. uncertainties and inconsistencies
    6. quantitative understanding
    7. developer assessment

The calculated confidence score appears in section 7 as *decision support*,
labelled as such. The Handbook weight-of-evidence call is the developer's, and
the field where they record it is required rather than pre-filled — a number
the tool computed is not an assessment, and presenting it as one is how an
arithmetic artefact ends up cited as a judgement.

Essentiality is deliberately not a per-KER heading. It is a property of a Key
Event and of the AOP as a whole — whether blocking the event blocks the
outcome — and repeating it under every relationship invited the model to
invent per-KER essentiality claims that no paper made.
"""

from typing import Any, Optional

import pandas as pd
import streamlit as st

import run_manifest
from run_manifest import RunManifest, RunTelemetry
from legal import with_disclaimer
from stage2_extraction import (
    evidence_synthesis,
    table1_store,
    table2_synthesis,
    workflow_state as wf,
)
from stage2_extraction.llm_providers import LLMAuthError, LLMConfig
from ui.common import (
    curator_name,
    csv_bytes,
    fmt,
    has_text,
    quote_block,
    require_curator,
    section_intro,
    state_badge,
)


HOW_TO = (
    "Pick an approved relationship. Only relationships whose Key Events are "
    "both approved appear here.",
    "Read the study evidence first — it is what every later section is built "
    "from.",
    "**Generate synthesis** writes the plausibility, empirical support and "
    "uncertainty sections from those studies.",
    "Record your own **developer assessment** at the bottom. The calculated "
    "score is decision support; the Handbook call is yours.",
)

ASSESSMENT_VALUES = ("Not assessed", "High", "Moderate", "Low")


def render(llm_config_factory=None) -> None:
    section_intro(
        "Synthesize and evaluate evidence",
        "Synthesize evidence",
        "One evidence page per canonical Key Event Relationship, built only "
        "from approved records.",
        HOW_TO,
        caution=(
            "The calculated confidence score is decision support only. The "
            "weight-of-evidence assessment is made and justified by you."
        ),
    )

    table1 = table1_store.load_table1_as_dataframe()
    if table1.empty:
        st.info("No extracted rows yet.")
        return

    available, blocked = _available_kers(table1)

    if blocked:
        st.caption(
            f"{len(blocked)} relationship(s) are not shown because their Key "
            f"Events are not approved yet. Approve them in **Approve**."
        )

    if not available:
        st.warning(
            "No relationship is ready for synthesis. Both Key Events of a "
            "relationship, and the relationship itself, have to be approved "
            "first.",
            icon="🔒",
        )
        return

    _stale_banner()

    choice = st.selectbox(
        "Canonical KER",
        options=list(available.keys()),
        format_func=lambda k: available[k]["label"],
        key="synth_ker_select",
    )
    st.divider()
    _evidence_page(choice, available[choice], table1, llm_config_factory)


# ---------------------------------------------------------------------------
# Which KERs are available
# ---------------------------------------------------------------------------

def _available_kers(table1: pd.DataFrame) -> tuple[dict[str, dict], list[str]]:
    """Split relationships into those cleared for synthesis and those blocked."""
    canonical = table1_store.load_canonical_kes()
    names = {int(r["canonical_id"]): str(r["canonical_name"])
             for _, r in canonical.iterrows()}
    levels = {int(r["canonical_id"]): str(r["level"])
              for _, r in canonical.iterrows()}

    linked = table1.dropna(
        subset=["upstream_ke_canonical_id", "downstream_ke_canonical_id"]
    )
    if linked.empty:
        return {}, []

    available: dict[str, dict] = {}
    blocked: list[str] = []

    grouped = linked.groupby(
        ["upstream_ke_canonical_id", "downstream_ke_canonical_id"]
    )
    for (up, down), rows in grouped:
        up_id, down_id = int(up), int(down)
        key = f"{up_id}->{down_id}"
        label = f"{names.get(up_id, up_id)} → {names.get(down_id, down_id)}"

        gate = wf.gate(ke_ids=[up_id, down_id], ker_keys=[key])
        if not gate.allowed:
            blocked.append(label)
            continue

        # Claims and papers are both worth showing: the gap between them is how
        # much of the apparent support is one paper counted more than once.
        n_papers = evidence_synthesis.n_contributing_papers(rows)
        available[key] = {
            "label": (
                f"{label}  ·  {len(rows)} claim(s) from {n_papers} paper(s)"
                if n_papers != len(rows)
                else f"{label}  ·  {len(rows)} claim(s)"
            ),
            "name": label,
            "upstream_id": up_id,
            "downstream_id": down_id,
            "upstream_name": names.get(up_id, str(up_id)),
            "downstream_name": names.get(down_id, str(down_id)),
            "upstream_level": levels.get(up_id, ""),
            "downstream_level": levels.get(down_id, ""),
            "records": rows,
        }

    return available, blocked


def _stale_banner() -> None:
    stale = wf.stale_syntheses()
    if stale.empty:
        return
    st.error(
        f"{len(stale)} synthesis(es) are stale because a Key Event they were "
        f"built on has changed. Regenerate them before relying on the text.",
        icon="♻️",
    )
    with st.expander("Which ones"):
        st.dataframe(stale, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _evidence_page(
    ker_key: str, meta: dict, table1: pd.DataFrame, llm_config_factory
) -> None:
    rows: pd.DataFrame = meta["records"]
    stored = table1_store.load_synthesis(ker_key) or {}

    _identity(ker_key, meta, rows, stored)
    st.divider()
    _study_evidence(rows)
    st.divider()
    _generate_controls(ker_key, meta, rows, stored, llm_config_factory)
    _plausibility(stored)
    _empirical_support(rows, stored)
    _uncertainties(rows, stored)
    _quantitative(stored)
    st.divider()
    _developer_assessment(ker_key, meta, rows, stored)


# --- 1. identity -----------------------------------------------------------

def _identity(ker_key: str, meta: dict, rows: pd.DataFrame, stored: dict) -> None:
    st.subheader("1 · KER identity")

    adjacency = (
        "Adjacent" if (rows["ker_adjacency"] == "Adjacent").any() else "Non-adjacent"
    )
    status = wf.get_status("ker", ker_key)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"### {meta['upstream_name']} → {meta['downstream_name']}"
        )
        st.caption(
            f"Upstream: **{meta['upstream_name']}** ({meta['upstream_level']}) · "
            f"Downstream: **{meta['downstream_name']}** ({meta['downstream_level']}) · "
            f"{adjacency}"
        )
    with c2:
        st.markdown(state_badge(status.effective_state.label, status.drifted),
                    unsafe_allow_html=True)

    description = stored.get("mechanistic_basis") or _first_text(rows, "ker_description")
    if has_text(description):
        st.markdown("**Canonical description**")
        st.write(description)

    st.markdown("**Applicability domain**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Field": "Taxa", "Value": _joined(rows, "taxonomic_applicability")},
                {"Field": "Sex", "Value": _joined(rows, "sex_applicability")},
                {"Field": "Life stage", "Value": _joined(rows, "life_stage_applicability")},
                {"Field": "Study designs", "Value": _joined(rows, "study_design")},
                {"Field": "Stressors", "Value": _joined(rows, "chemical_stressor")},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# --- 2. study evidence -----------------------------------------------------

def _study_evidence(rows: pd.DataFrame) -> None:
    st.subheader("2 · Study evidence")
    st.caption(
        "One block per independent study, in that paper's own terms. Everything "
        "below this section is derived from these blocks."
    )

    supporting = rows[~rows["contradicts_ker"].astype(bool)]
    contradicting = rows[rows["contradicts_ker"].astype(bool)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Independent studies", int(rows["source_doi"].nunique()))
    c2.metric("Supporting", len(supporting))
    c3.metric("Contradicting", len(contradicting))

    for _, row in rows.iterrows():
        verdict = "Contradicting" if bool(row["contradicts_ker"]) else "Supporting"
        icon = "🔴" if bool(row["contradicts_ker"]) else "🟢"
        title = f"{icon} {fmt(row['source_doi'])} — {verdict}"

        with st.expander(title):
            st.markdown(f"**{fmt(row.get('source_title'))}**")
            st.caption(
                f"{fmt(row.get('paper_type'))} · extracted "
                f"{fmt(row.get('extraction_date'))} · confidence "
                f"{fmt(row.get('extraction_confidence'))}"
            )

            st.markdown("**Stressor and model**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": "Stressor", "Value": fmt(row.get("chemical_stressor"))},
                        {"Field": "Exposure route", "Value": fmt(row.get("exposure_route"))},
                        {"Field": "Study design", "Value": fmt(row.get("study_design"))},
                        {"Field": "Taxon", "Value": fmt(row.get("taxonomic_applicability"))},
                        {"Field": "Sex", "Value": fmt(row.get("sex_applicability"))},
                        {"Field": "Life stage", "Value": fmt(row.get("life_stage_applicability"))},
                        {"Field": "Dose and time", "Value": fmt(row.get("time_scale"))},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Upstream and downstream measurements**")
            st.write(
                f"Upstream — {fmt(row.get('upstream_ke_name'))} "
                f"({fmt(row.get('upstream_ke_level'))})"
            )
            st.write(
                f"Downstream — {fmt(row.get('downstream_ke_name'))} "
                f"({fmt(row.get('downstream_ke_level'))})"
            )

            for label, key in (
                ("Temporal concordance", "time_scale"),
                ("Dose–response concordance", "quantitative_relationships"),
                ("Response–response", "response_response_relationship"),
                ("Empirical evidence", "empirical_evidence_summary"),
                ("Modulating factors", "modulating_factors"),
            ):
                if has_text(row.get(key)):
                    st.markdown(f"**{label}**")
                    st.write(str(row.get(key)))

            spans = table1_store.load_evidence_spans([int(row["record_id"])])
            if not spans.empty:
                st.markdown(f"**Quotations ({len(spans)})**")
                for _, span in spans.iterrows():
                    quote_block(span["quote"], span["citation"], bool(span["verified"]))
            else:
                st.caption("No quotations were captured for this study.")


# --- generation ------------------------------------------------------------

def _generate_controls(
    ker_key: str, meta: dict, rows: pd.DataFrame, stored: dict, llm_config_factory
) -> None:
    is_stale = bool(stored.get("stale"))
    generated = fmt(stored.get("generated_at"), "")

    c1, c2 = st.columns([3, 1])
    with c1:
        if stored and not is_stale:
            st.caption(
                f"Synthesis generated {generated} with "
                f"{fmt(stored.get('model'))}."
            )
        elif is_stale:
            st.warning(
                f"This synthesis is stale — {fmt(stored.get('stale_reason'))}. "
                f"The text below is the superseded version.",
                icon="♻️",
            )
        else:
            st.caption("No synthesis generated yet.")

    with c2:
        label = "Regenerate" if stored else "Generate synthesis"
        if st.button(label, type="primary", use_container_width=True,
                     key=f"gen_{ker_key}"):
            _generate(ker_key, meta, rows, llm_config_factory)

    if stored and not stored.get("stale"):
        history = wf.synthesis_history(ker_key)
        if not history.empty:
            with st.expander(f"Previous versions ({len(history)})"):
                st.dataframe(
                    history[["archived_at", "reason"]],
                    use_container_width=True, hide_index=True,
                )


def _generate(ker_key: str, meta: dict, rows: pd.DataFrame, llm_config_factory) -> None:
    """Run the synthesis, refusing if the gate has closed since the page loaded."""
    gate = wf.gate(ke_ids=[meta["upstream_id"], meta["downstream_id"]],
                   ker_keys=[ker_key])
    if not gate.allowed:
        st.error(f"Synthesis refused. {gate.summary}", icon="🔒")
        return

    if llm_config_factory is None:
        st.error("No model is configured. Set one up in the sidebar.")
        return

    try:
        config: LLMConfig = llm_config_factory()
    except Exception as exc:
        st.error(f"Could not build the model configuration: {exc}")
        return

    # The synthesis gets its own run record. Until it did, this was the only
    # model call in the pipeline whose conditions were not written down — and
    # it is the one whose output gets read as the assessment. A manifest here
    # also gives `run_manifest.record(...)` inside evidence_synthesis an active
    # run to report to, so JSON repairs, provider retries and refusals during
    # synthesis are counted instead of discarded.
    manifest = RunManifest.from_config(
        config,
        stage="synthesis",
        mode="targeted",
        target_upstream=meta.get("upstream_name"),
        target_downstream=meta.get("downstream_name"),
        schema_version=table1_store.SCHEMA_VERSION,
    )
    run_id = table1_store.start_run(manifest)
    run_manifest.start_run(RunTelemetry())
    status = "failed"
    result = None

    try:
        with st.spinner("Synthesising across studies…"):
            try:
                result = evidence_synthesis.synthesise_ker(
                    meta["name"],
                    rows,
                    config,
                    ker_key=ker_key,
                )
            except LLMAuthError as exc:
                st.error(f"Authentication failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Synthesis failed: {exc}")
                return

        if result.error:
            st.error(result.error)
            return
        status = "completed"
    finally:
        # Closed on every path, including the early returns above. A run left
        # open would keep collecting telemetry from whatever the user did next
        # and sit in the runs table as permanently "running" — and a failed
        # synthesis is exactly the run worth being able to look up later.
        telemetry = run_manifest.end_run()
        table1_store.finish_run(
            run_id,
            telemetry,
            status=status,
            model_reported=getattr(result, "model", None),
        )

    table1_store.save_synthesis(result, run_id=run_id)
    _clear_stale(ker_key)
    wf.set_state("ker", ker_key, wf.State.SYNTHESIZED, curator=curator_name() or None)
    st.success("Synthesis written.")
    st.rerun()


def _clear_stale(ker_key: str) -> None:
    with table1_store.connect() as conn:
        conn.execute(
            "UPDATE ker_synthesis SET stale = 0, stale_reason = NULL "
            "WHERE ker_key = ?",
            (ker_key,),
        )
        conn.commit()


# --- 3-6. derived sections -------------------------------------------------

def _plausibility(stored: dict) -> None:
    st.subheader("3 · Biological plausibility")
    if has_text(stored.get("biological_plausibility")):
        rating = fmt(stored.get("biological_plausibility_rating"), "")
        if rating:
            st.caption(f"Rated: {rating}")
        st.write(stored["biological_plausibility"])
    else:
        st.caption("Not yet synthesised.")


def _empirical_support(rows: pd.DataFrame, stored: dict) -> None:
    st.subheader("4 · Empirical support")
    st.caption(
        "Temporal, dose–response and incidence concordance across the studies "
        "above."
    )

    concordance = pd.DataFrame(
        [
            {
                "Study": fmt(r["source_doi"]),
                "Direction": ("Contradicting" if bool(r["contradicts_ker"])
                              else "Supporting"),
                "Temporal": fmt(r.get("time_scale")),
                "Dose–response": fmt(r.get("quantitative_relationships")),
                "Response–response": fmt(r.get("response_response_relationship")),
            }
            for _, r in rows.iterrows()
        ]
    )
    st.dataframe(concordance, use_container_width=True, hide_index=True)

    if has_text(stored.get("empirical_evidence")):
        rating = fmt(stored.get("empirical_evidence_rating"), "")
        if rating:
            st.caption(f"Rated: {rating}")
        st.write(stored["empirical_evidence"])


def _sign_disagreement(rows: pd.DataFrame) -> None:
    """
    Papers that report this relationship running in opposite directions.

    Distinct from a contradiction: a contradicting paper says the link is not
    there, whereas these papers all say it is and disagree about which way it
    goes. Pooled into one KER the disagreement vanishes and the row reads as
    well supported, which is how a knockdown result and a gain-of-function
    result in a different cell type came to share an edge.
    """
    if "direction" not in rows.columns:
        return

    signs = rows["direction"].astype(str).str.strip().str.lower()
    positive = rows[signs == "positive"]
    negative = rows[signs == "negative"]
    unsigned = rows[~signs.isin(["positive", "negative"])]

    if not positive.empty and not negative.empty:
        st.error(
            f"{len(positive)} paper(s) report these events moving together and "
            f"{len(negative)} report them moving oppositely. Until that is "
            f"resolved this is two findings, not one relationship.",
            icon="🔀",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Same direction**")
            for _, row in positive.iterrows():
                cells = fmt(row.get("upstream_cell_type")) or "cell type not stated"
                st.markdown(f"- {fmt(row['source_doi'])} — {cells}")
        with c2:
            st.markdown("**Opposite directions**")
            for _, row in negative.iterrows():
                cells = fmt(row.get("upstream_cell_type")) or "cell type not stated"
                st.markdown(f"- {fmt(row['source_doi'])} — {cells}")

    if not unsigned.empty:
        st.warning(
            f"{len(unsigned)} of {len(rows)} paper(s) gave no direction for "
            "this relationship. The arrow on the map does not mean the "
            "evidence had one.",
            icon="➡️",
        )


def _uncertainties(rows: pd.DataFrame, stored: dict) -> None:
    st.subheader("5 · Uncertainties and inconsistencies")

    _sign_disagreement(rows)

    contradicting = rows[rows["contradicts_ker"].astype(bool)]
    if not contradicting.empty:
        st.warning(
            f"{len(contradicting)} study(ies) argue against this relationship.",
            icon="⚠️",
        )
        for _, row in contradicting.iterrows():
            st.markdown(f"- **{fmt(row['source_doi'])}** — {fmt(row.get('ker_description'))}")

    if has_text(stored.get("uncertainties")):
        st.write(stored["uncertainties"])
    elif contradicting.empty:
        st.caption("No contradicting evidence recorded, and nothing synthesised yet.")


def _quantitative(stored: dict) -> None:
    st.subheader("6 · Quantitative understanding")
    st.caption(
        "How accurately the downstream Key Event can be predicted from the "
        "upstream one."
    )
    if has_text(stored.get("quantitative_understanding")):
        st.write(stored["quantitative_understanding"])
    else:
        st.caption("Not yet synthesised.")


# --- 7. developer assessment ----------------------------------------------

def _developer_assessment(
    ker_key: str, meta: dict, rows: pd.DataFrame, stored: dict
) -> None:
    st.subheader("7 · Developer assessment")

    _decision_support(rows, stored)

    st.markdown("**Your assessment**")
    st.caption(
        "This is the Handbook weight-of-evidence call. It is not derived from "
        "the score above and you have to write the rationale yourself."
    )

    if not require_curator("Enter your name in the sidebar to record an assessment."):
        return

    current = fmt(stored.get("developer_assessment"), "Not assessed")
    with st.form(f"assess_{ker_key}"):
        assessment = st.selectbox(
            "Weight of evidence",
            ASSESSMENT_VALUES,
            index=(ASSESSMENT_VALUES.index(current)
                   if current in ASSESSMENT_VALUES else 0),
        )
        rationale = st.text_area(
            "Rationale",
            value=fmt(stored.get("developer_rationale"), ""),
            height=140,
            placeholder=(
                "Why this rating? Reference the studies above, including the "
                "ones that disagree."
            ),
        )
        submitted = st.form_submit_button("Record assessment", type="primary")

    if submitted:
        if assessment != "Not assessed" and not rationale.strip():
            st.error("An assessment needs a written rationale.")
            return
        _save_assessment(ker_key, assessment, rationale.strip(), curator_name())
        st.success("Assessment recorded.")
        st.rerun()

    if has_text(stored.get("developer_assessment")):
        st.caption(
            f"Recorded by {fmt(stored.get('developer_curator'))}"
            + (f" · approved by {stored['approved_by']} on {stored['approved_at']}"
               if stored.get("approved_by") else "")
        )

    _export(ker_key, meta, rows, stored)


def _decision_support(rows: pd.DataFrame, stored: dict) -> None:
    """The calculated score, labelled as an input rather than a verdict."""
    n_papers = int(rows["source_doi"].nunique())
    n_contra = int(rows["contradicts_ker"].astype(bool).sum())
    spans = int(rows["n_evidence_spans"].fillna(0).sum())
    verified = int(rows["n_verified_spans"].fillna(0).sum())

    with st.container(border=True):
        st.markdown("**Decision support — calculated, not an assessment**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Independent studies", n_papers)
        c2.metric("Contradicting", n_contra)
        c3.metric("Quotations", spans)
        c4.metric(
            "Verified verbatim",
            verified,
            f"{100 * verified / spans:.0f}%" if spans else None,
        )
        if has_text(stored.get("overall_confidence")):
            st.caption(
                f"Model's overall confidence: **{stored['overall_confidence']}**. "
                f"This is the model's reading of the same evidence, offered for "
                f"comparison with your own."
            )


def _save_assessment(ker_key: str, assessment: str, rationale: str, curator: str) -> None:
    with table1_store.connect() as conn:
        existing = conn.execute(
            "SELECT ker_key FROM ker_synthesis WHERE ker_key = ?", (ker_key,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO ker_synthesis (ker_key, ker_name, developer_assessment, "
                "developer_rationale, developer_curator) VALUES (?, ?, ?, ?, ?)",
                (ker_key, ker_key, assessment, rationale, curator),
            )
        else:
            conn.execute(
                "UPDATE ker_synthesis SET developer_assessment = ?, "
                "developer_rationale = ?, developer_curator = ? WHERE ker_key = ?",
                (assessment, rationale, curator, ker_key),
            )
        conn.commit()


def _export(ker_key: str, meta: dict, rows: pd.DataFrame, stored: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Evidence page (Markdown)",
            _markdown(ker_key, meta, rows, stored).encode("utf-8"),
            f"ker_{ker_key.replace('->', '_to_')}.md",
            "text/markdown",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Contributing studies (CSV)",
            csv_bytes(rows),
            f"ker_{ker_key.replace('->', '_to_')}_studies.csv",
            "text/csv",
            use_container_width=True,
        )


def _markdown(ker_key: str, meta: dict, rows: pd.DataFrame, stored: dict) -> str:
    """The page as a document, in the same order it appears on screen."""
    lines = [
        f"# {meta['name']}",
        "",
        f"- Upstream: {meta['upstream_name']} ({meta['upstream_level']})",
        f"- Downstream: {meta['downstream_name']} ({meta['downstream_level']})",
        f"- Independent studies: {rows['source_doi'].nunique()}",
        f"- Contradicting studies: {int(rows['contradicts_ker'].astype(bool).sum())}",
        "",
        "## Study evidence",
        "",
    ]
    for _, row in rows.iterrows():
        verdict = "contradicting" if bool(row["contradicts_ker"]) else "supporting"
        lines += [
            f"### {fmt(row['source_doi'])} ({verdict})",
            "",
            f"- Stressor: {fmt(row.get('chemical_stressor'))}",
            f"- Design: {fmt(row.get('study_design'))}",
            f"- Taxon / sex / life stage: {fmt(row.get('taxonomic_applicability'))} / "
            f"{fmt(row.get('sex_applicability'))} / {fmt(row.get('life_stage_applicability'))}",
            "",
        ]
        if has_text(row.get("empirical_evidence_summary")):
            lines += [str(row["empirical_evidence_summary"]), ""]

    for heading, key in (
        ("Biological plausibility", "biological_plausibility"),
        ("Empirical support", "empirical_evidence"),
        ("Uncertainties and inconsistencies", "uncertainties"),
        ("Quantitative understanding", "quantitative_understanding"),
    ):
        lines += [f"## {heading}", "", fmt(stored.get(key), "_Not yet synthesised._"), ""]

    lines += [
        "## Developer assessment",
        "",
        f"**{fmt(stored.get('developer_assessment'), 'Not assessed')}** — "
        f"{fmt(stored.get('developer_rationale'), 'No rationale recorded.')}",
        "",
        f"Recorded by {fmt(stored.get('developer_curator'))}.",
        "",
    ]
    return with_disclaimer("\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_text(rows: pd.DataFrame, column: str) -> str:
    for value in rows[column]:
        if has_text(value):
            return str(value)
    return ""


def _joined(rows: pd.DataFrame, column: str, limit: int = 8) -> str:
    values = sorted({
        str(v).strip() for v in rows[column]
        if has_text(v) and str(v).strip().lower() != "not specified"
    })
    if not values:
        return "—"
    shown = values[:limit]
    return "; ".join(shown) + ("…" if len(values) > limit else "")
