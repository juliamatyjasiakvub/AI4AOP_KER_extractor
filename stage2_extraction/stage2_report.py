from __future__ import annotations

"""
One document describing everything Stage 2 did and every decision a curator made.

Why this exists
---------------
Stage 2 already exports plenty: Table 1 as CSV, Table 2 as CSV, the graph as
SVG, nodes and edges as JSON, a QC report on the extraction. What none of them
answers is the question an assessor, a reviewer or a co-author actually asks —
*how did this AOP come to look like this?*

That answer is spread across nine tables. Which papers were read and which gave
nothing. Which wordings were folded into which Key Event, and on whose
authority. Which merges a curator accepted, which they refused, and why. Who
approved what, and when. Which relationships were synthesised, under which
model, and what the developer concluded independently of the calculated score.
What is still unresolved. Assembling that by hand from the exports is a day's
work, and it is exactly the material that makes the difference between a figure
a reader must take on trust and one they can audit.

So this module walks the same stores the UI reads and writes one report. It
adds no interpretation of its own: every number is counted from the database
and every rationale is quoted as the curator typed it.

Deliberately included even when empty. A section reading "no merges were
recorded" is a finding — it says the canonical Key Events came from the
proposer untouched — and silently omitting the heading would hide that.
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from stage2_extraction import (
    canonical_groups,
    evidence_synthesis,
    table1_store,
    table2_synthesis,
    workflow_state as wf,
)

__all__ = ["Stage2Report", "build_stage2_report", "report_markdown", "report_csv"]

#: Approval-log rows printed in full. The log grows a row per state change per
#: object, so a small corpus reaches three figures quickly — the Markdown is
#: for reading, and the CSV export carries every row for auditing.
_MAX_LOG_ROWS = 25


def _safe(fn, default):
    """Run a store call, tolerating a database that predates the feature."""
    try:
        result = fn()
    except Exception:
        return default
    return default if result is None else result


def _text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    # "NA" and "n/a" are what a curator types into a box they have nothing to
    # say in. Rendering them verbatim makes an empty rationale look answered.
    if not text or text.lower() in ("nan", "none", "null", "na", "n/a", "-"):
        return fallback
    # Pipes would break the Markdown table they land in.
    return text.replace("|", "\\|").replace("\n", " ")


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "—"


@dataclass
class Stage2Report:
    """Everything Stage 2 produced, and every decision behind it."""

    generated_at: str = ""
    schema_version: Optional[int] = None

    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    papers: pd.DataFrame = field(default_factory=pd.DataFrame)

    table1: pd.DataFrame = field(default_factory=pd.DataFrame)
    table2: pd.DataFrame = field(default_factory=pd.DataFrame)
    spans: pd.DataFrame = field(default_factory=pd.DataFrame)

    canonical: pd.DataFrame = field(default_factory=pd.DataFrame)
    crosswalk: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    mappings: pd.DataFrame = field(default_factory=pd.DataFrame)
    relations: pd.DataFrame = field(default_factory=pd.DataFrame)

    ke_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    ker_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    approvals: pd.DataFrame = field(default_factory=pd.DataFrame)
    roles: pd.DataFrame = field(default_factory=pd.DataFrame)

    syntheses: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: Gathered here rather than looked up while rendering. Rendering used to
    #: call `wf.counts()` and `wf.stale_syntheses()` directly, which meant the
    #: document could describe a different database from the one it was built
    #: from — and made `report_markdown` open a connection, so merely
    #: formatting a report created an `aop_rag.db` in the working directory.
    ke_counts: dict[str, int] = field(default_factory=dict)
    ker_counts: dict[str, int] = field(default_factory=dict)
    stale_syntheses: pd.DataFrame = field(default_factory=pd.DataFrame)


def _roles() -> pd.DataFrame:
    """MIE / AO assignments, which live in a table with no loader of its own."""
    try:
        with table1_store.connect() as conn:
            return pd.read_sql_query(
                "SELECT k.canonical_name AS key_event, r.role, r.curator, "
                "       r.rationale, r.assigned_at "
                "FROM ke_role r JOIN ke_canonical k "
                "  ON k.canonical_id = r.canonical_id "
                "ORDER BY r.role, k.canonical_name",
                conn,
            )
    except Exception:
        return pd.DataFrame()


def build_stage2_report() -> Stage2Report:
    """Gather every Stage 2 artefact from the database."""
    table1 = _safe(table1_store.load_table1_as_dataframe, pd.DataFrame())

    table2 = pd.DataFrame()
    if not table1.empty:
        try:
            table2 = table2_synthesis.compute_table2(table1, normalized=True)
        except Exception:
            table2 = pd.DataFrame()

    return Stage2Report(
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        schema_version=_safe(table1_store.get_schema_version, None),
        runs=_safe(lambda: table1_store.load_runs(), pd.DataFrame()),
        papers=_safe(table1_store.list_source_papers, pd.DataFrame()),
        table1=table1,
        table2=table2,
        spans=_safe(table1_store.load_evidence_spans, pd.DataFrame()),
        canonical=_safe(table1_store.load_canonical_kes, pd.DataFrame()),
        crosswalk=_safe(table1_store.load_alias_crosswalk, pd.DataFrame()),
        decisions=_safe(lambda: canonical_groups.decision_log(limit=10_000), pd.DataFrame()),
        mappings=_safe(canonical_groups.ontology_mappings, pd.DataFrame()),
        relations=_safe(canonical_groups.ke_relations, pd.DataFrame()),
        ke_states=_safe(lambda: wf.state_frame("ke"), pd.DataFrame()),
        ker_states=_safe(lambda: wf.state_frame("ker"), pd.DataFrame()),
        approvals=_safe(lambda: wf.approval_log(limit=10_000), pd.DataFrame()),
        roles=_roles(),
        syntheses=_safe(table1_store.load_all_syntheses, pd.DataFrame()),
        ke_counts=_safe(lambda: wf.counts("ke"), {}),
        ker_counts=_safe(lambda: wf.counts("ker"), {}),
        stale_syntheses=_safe(wf.stale_syntheses, pd.DataFrame()),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def report_markdown(report: Stage2Report) -> str:
    out: list[str] = []
    w = out.append

    w("# Stage 2 — extraction, curation and synthesis record\n")
    w(
        f"Generated {report.generated_at}"
        + (f" · database schema v{report.schema_version}" if report.schema_version else "")
        + "\n"
    )
    w(
        "> Every figure below is counted from the database and every rationale "
        "is quoted as the curator entered it. This document records what was "
        "done; it does not assess whether it was right.\n"
    )

    _section_corpus(report, w)
    _section_extraction(report, w)
    _section_normalisation(report, w)
    _section_decisions(report, w)
    _section_approval(report, w)
    _section_relationships(report, w)
    _section_synthesis(report, w)
    _section_outstanding(report, w)

    return "\n".join(out)


def _section_corpus(r: Stage2Report, w) -> None:
    w("\n## 1 · Corpus and runs\n")
    if r.runs.empty:
        w("No extraction run was recorded. Rows in this database predate run "
          "manifests and cannot be attributed to a model or prompt version.\n")
    else:
        w(f"{len(r.runs)} run(s) recorded.\n")
        w("| Run | Stage | Mode | Provider / model | Temp | Seed | Prompt | Code | Papers | Rows | Status |")
        w("| ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |")
        for _, run in r.runs.iterrows():
            w(
                f"| {_text(run.get('run_id'))} | {_text(run.get('stage'))} "
                f"| {_text(run.get('mode'))} "
                f"| {_text(run.get('provider'))}/{_text(run.get('model'))} "
                f"| {_text(run.get('temperature'))} | {_text(run.get('seed'))} "
                f"| {_text(run.get('prompt_fingerprint'))} "
                f"| {_text(run.get('code_version'))} "
                f"| {_text(run.get('papers_attempted'))} "
                f"| {_text(run.get('kers_extracted'))} "
                f"| {_text(run.get('status'))} |"
            )
        w("")
        targeted = r.runs[r.runs.get("mode").eq("targeted")] if "mode" in r.runs else pd.DataFrame()
        if not targeted.empty:
            w("Targeted runs asked about one relationship each:\n")
            for _, run in targeted.iterrows():
                w(f"- run {_text(run.get('run_id'))}: "
                  f"**{_text(run.get('target_upstream'))}** → "
                  f"**{_text(run.get('target_downstream'))}**")
            w("")
        w(
            "> Two runs are only comparable when provider, model, temperature, "
            "prompt fingerprint and code version all match. Language-model "
            "output is not deterministic even when they do.\n"
        )

        if "transmission_ack" in r.runs.columns:
            hosted = r.runs[r.runs["transmission_ack"].notna()]
            if not hosted.empty:
                w("\nRuns that sent paper text to a hosted provider, and the "
                  "basis given:\n")
                w("| Run | Provider | Basis recorded |")
                w("| ---: | --- | --- |")
                for _, run in hosted.iterrows():
                    w(f"| {_text(run.get('run_id'))} | "
                      f"{_text(run.get('provider'))} | "
                      f"{_text(run.get('transmission_ack'))} |")
                w("")
                w("> The tool records this basis; it does not verify it.\n")
            else:
                w("\nNo run sent paper text to a hosted provider — all "
                  "processing was local.\n")

    if not r.papers.empty:
        w(f"\n**{len(r.papers)} source paper(s).**\n")


def _section_extraction(r: Stage2Report, w) -> None:
    w("\n## 2 · What was extracted\n")
    if r.table1.empty:
        w("Table 1 is empty — nothing was extracted.\n")
        return

    t1 = r.table1
    n_rows = len(t1)
    n_papers = evidence_synthesis.n_contributing_papers(t1)
    curator_rows = 0
    if "origin" in t1.columns:
        curator_rows = int(
            t1["origin"].fillna(table1_store.LLM_ORIGIN)
            .isin(table1_store.CURATOR_ORIGINS).sum()
        )

    n_spans = len(r.spans)
    verified = int(r.spans["verified"].sum()) if not r.spans.empty else 0
    contradicting = int(t1["contradicts_ker"].astype(bool).sum()) if "contradicts_ker" in t1 else 0

    w("| Measure | Value |")
    w("| --- | ---: |")
    w(f"| Claims (Table 1 rows) | {n_rows} |")
    w(f"| Distinct papers behind them | {n_papers} |")
    w(f"| Claims entered or corrected by a curator | {curator_rows} |")
    w(f"| Claims recorded as contradicting their relationship | {contradicting} |")
    w(f"| Quotations captured | {n_spans} |")
    w(f"| Quotations located verbatim in the source | {verified} ({_pct(verified, n_spans)}) |")
    w("")
    w(
        "> Claims and papers are different counts. One paper often yields "
        "several claims, and how many it yields varies between runs — so a "
        "count of claims is not a count of evidence.\n"
    )
    if n_spans and verified / n_spans < 0.4:
        w(
            f"**Only {_pct(verified, n_spans)} of quotations could be located in "
            "the source text.** The rest are paraphrases or misattributions and "
            "are flagged in the evidence panel. Treat the extraction as "
            "provisional until they are checked.\n"
        )


def _section_normalisation(r: Stage2Report, w) -> None:
    w("\n## 3 · From raw wording to Key Events\n")
    if r.canonical.empty:
        w(
            "Normalisation has not been run: there are no canonical Key Events, "
            "so relationships are still grouped on the literal strings each "
            "paper used.\n"
        )
        return

    n_canon = len(r.canonical)
    n_alias = len(r.crosswalk)
    w(
        f"{n_alias} raw wording(s) were resolved into **{n_canon} canonical Key "
        f"Event(s)**.\n"
    )

    basis_col = "merge_basis" if "merge_basis" in r.crosswalk.columns else None
    if basis_col:
        w("Authorising rule, one row per wording:\n")
        w("| Rule | Wordings |")
        w("| --- | ---: |")
        counts = r.crosswalk[basis_col].fillna("unrecorded").value_counts()
        labels = getattr(table1_store, "ALIAS_BASIS_LABELS", {}) or {}
        for basis, n in counts.items():
            w(f"| {_text(labels.get(basis, basis))} | {int(n)} |")
        w("")
        w(
            "> The rule matters as much as the result: an identical AOP-Wiki "
            "identifier and a lexical similarity score are not equally strong "
            "grounds for calling two wordings the same event.\n"
        )

    w("\nCanonical Key Events:\n")
    w("| Key Event | Level | Ontology | Claims |")
    w("| --- | --- | --- | ---: |")
    for _, ke in r.canonical.iterrows():
        w(
            f"| {_text(ke.get('canonical_name'))} | {_text(ke.get('level'))} "
            f"| {_text(ke.get('ontology_curie'))} | {_text(ke.get('n_source_rows'))} |"
        )
    w("")


def _section_decisions(r: Stage2Report, w) -> None:
    w("\n## 4 · Curator decisions\n")
    if r.decisions.empty:
        w(
            "No merge, split, mapping or rejection was recorded. The canonical "
            "Key Events above are the proposer's grouping, unreviewed — which "
            "is itself worth stating, since nothing here has had a second "
            "opinion.\n"
        )
    else:
        reverted = int(r.decisions.get("reverted").fillna(0).astype(bool).sum()) \
            if "reverted" in r.decisions else 0
        w(f"{len(r.decisions)} decision(s) recorded"
          + (f", {reverted} since reverted" if reverted else "") + ".\n")

        w("| Action | Count |")
        w("| --- | ---: |")
        for action, n in r.decisions["action_label"].fillna("—").value_counts().items():
            w(f"| {_text(action)} | {int(n)} |")
        w("")

        w("Every decision, with the classification it was made against:\n")
        w("| When | Action | Classification | Curator | Rationale | Reverted |")
        w("| --- | --- | --- | --- | --- | --- |")
        for _, d in r.decisions.iterrows():
            w(
                f"| {_text(d.get('created_at'))} "
                f"| {_text(d.get('action_label') or d.get('action'))} "
                f"| {_text(d.get('relationship'))} "
                f"| {_text(d.get('curator'))} "
                f"| {_text(d.get('curator_rationale'))} "
                f"| {'yes' if d.get('reverted') else 'no'} |"
            )
        w("")
        w(
            "> Only pairs classified *equivalent* may be merged. Where a "
            "curator's action disagrees with the classification, that "
            "disagreement is the record — it is not an error.\n"
        )

    if not r.mappings.empty:
        w(f"\n**{len(r.mappings)} ontology mapping(s).** A mapping attaches a "
          "broader concept without pooling evidence into it.\n")
    if not r.relations.empty:
        w(f"\n**{len(r.relations)} recorded biological relationship(s)** between "
          "Key Events kept deliberately separate.\n")


def _section_approval(r: Stage2Report, w) -> None:
    w("\n## 5 · Approval\n")
    ke_counts, ker_counts = r.ke_counts, r.ker_counts
    if not ke_counts and not ker_counts:
        w("No workflow state is recorded; nothing has been approved.\n")
    else:
        w("| Object | State | Count |")
        w("| --- | --- | ---: |")
        for label, counts in (("Key Event", ke_counts), ("Relationship", ker_counts)):
            for state, n in sorted((counts or {}).items()):
                w(f"| {label} | {_text(state)} | {int(n)} |")
        w("")

    if not r.roles.empty:
        w("Pathway endpoints declared by the curator:\n")
        w("| Role | Key Event | Curator | Rationale |")
        w("| --- | --- | --- | --- |")
        for _, role in r.roles.iterrows():
            w(
                f"| {_text(role.get('role'))} | {_text(role.get('key_event'))} "
                f"| {_text(role.get('curator'))} | {_text(role.get('rationale'))} |"
            )
        w("")
        w(
            "> No endpoint is inferred from graph position. An event with "
            "nothing upstream of it means only that no paper in this corpus "
            "reported an earlier step.\n"
        )
    else:
        w(
            "\nNo molecular initiating event or adverse outcome has been "
            "declared. Every event is drawn as an ordinary Key Event.\n"
        )

    if not r.approvals.empty:
        # A corpus of nineteen Key Events generated a hundred and fifty log
        # entries, because approving, editing and re-approving each one writes
        # a row every time. Printing them all buries the report; printing none
        # loses the audit trail. So: the shape of the whole log, then the most
        # recent entries in full, and a pointer to the CSV for the rest.
        log = r.approvals
        retractions = int(log["to_state"].ne("approved").sum()) if "to_state" in log else 0
        w(f"\n{len(log)} state change(s) logged"
          + (f", including {retractions} retraction(s) or downgrade(s)" if retractions else "")
          + ".\n")

        if "curator" in log.columns:
            w("| Curator | Changes |")
            w("| --- | ---: |")
            for curator, n in log["curator"].fillna("—").value_counts().items():
                w(f"| {_text(curator)} | {int(n)} |")
            w("")

        recent = log.head(_MAX_LOG_ROWS)
        w(f"Most recent {len(recent)} change(s):\n")
        w("| When | Object | From | To | Curator | Note |")
        w("| --- | --- | --- | --- | --- | --- |")
        for _, a in recent.iterrows():
            w(
                f"| {_text(a.get('created_at'))} "
                f"| {_text(a.get('target_type'))} {_text(a.get('target_key'))} "
                f"| {_text(a.get('from_state'))} | {_text(a.get('to_state'))} "
                f"| {_text(a.get('curator'))} | {_text(a.get('note'), '')} |"
            )
        w("")
        if len(log) > len(recent):
            w(f"> {len(log) - len(recent)} earlier change(s) are omitted here "
              "and included in full in the CSV export.\n")


def _section_relationships(r: Stage2Report, w) -> None:
    w("\n## 6 · Relationships\n")
    if r.table2.empty:
        w("No relationships could be assembled.\n")
        return

    t2 = r.table2
    w(f"{len(t2)} consolidated relationship(s).\n")
    w("| Relationship | Papers | Claims | Adjacency | Direction | Evidence | Score | Band |")
    w("| --- | ---: | ---: | --- | --- | --- | ---: | --- |")
    for _, row in t2.iterrows():
        w(
            f"| {_text(row.get('upstream_ke_name'))} → "
            f"{_text(row.get('downstream_ke_name'))} "
            f"| {_text(row.get('n_papers_total'))} "
            f"| {_text(row.get('n_source_rows'))} "
            f"| {_text(row.get('ker_adjacency'))} "
            f"| {_text(row.get('direction'))} "
            f"| {_text(row.get('evidence_type'))} "
            f"| {_text(row.get('confidence_score'))} "
            f"| {_text(row.get('confidence_band'))} |"
        )
    w("")
    w(
        "> The score is a transparent heuristic for ranking, not a "
        "weight-of-evidence assessment. The Handbook call is the developer's, "
        "and appears in the next section where one has been made.\n"
    )

    conflicted = t2[t2.get("sign_conflict").fillna(False)] if "sign_conflict" in t2 else pd.DataFrame()
    if not conflicted.empty:
        w(f"\n**{len(conflicted)} relationship(s) have papers disagreeing about "
          "direction.** The disagreement is a finding about the corpus and is "
          "carried into the figure rather than averaged away:\n")
        for _, row in conflicted.iterrows():
            w(
                f"- {_text(row.get('upstream_ke_name'))} → "
                f"{_text(row.get('downstream_ke_name'))} — "
                f"{_text(row.get('n_positive'))} positive, "
                f"{_text(row.get('n_negative'))} negative"
            )
        w("")


def _section_synthesis(r: Stage2Report, w) -> None:
    w("\n## 7 · Evidence syntheses and developer assessments\n")
    if r.syntheses.empty:
        w("No synthesis has been generated.\n")
        return

    w(f"{len(r.syntheses)} synthesis(es).\n")
    w("| Relationship | Papers | Claims | Plausibility | Empirical | Essentiality | Calculated | Developer | Stale |")
    w("| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |")
    for _, s in r.syntheses.iterrows():
        w(
            f"| {_text(s.get('ker_name'))} | {_text(s.get('n_papers'))} "
            f"| {_text(s.get('n_rows'))} "
            f"| {_text(s.get('biological_plausibility_rating'))} "
            f"| {_text(s.get('empirical_evidence_rating'))} "
            f"| {_text(s.get('essentiality_rating'))} "
            f"| {_text(s.get('overall_confidence'))} "
            f"| {_text(s.get('developer_assessment'), 'not assessed')} "
            f"| {'yes' if s.get('stale') else 'no'} |"
        )
    w("")

    rationales = [
        s for _, s in r.syntheses.iterrows()
        if _text(s.get("developer_rationale"), "") not in ("", "—")
    ]
    if rationales:
        w("Developer rationales, as written:\n")
        for s in rationales:
            w(f"**{_text(s.get('ker_name'))}** — "
              f"{_text(s.get('developer_assessment'), 'not assessed')} "
              f"({_text(s.get('developer_curator'))})\n")
            w(f"> {_text(s.get('developer_rationale'))}\n")

    undocumented = int((r.syntheses.get("run_id").isna()).sum()) if "run_id" in r.syntheses else 0
    if undocumented:
        w(
            f"\n**{undocumented} synthesis(es) carry no run record**, so the "
            "provider, temperature and prompt version behind that text are "
            "unknown and it cannot be compared with a later one on equal "
            "terms.\n"
        )


def _section_outstanding(r: Stage2Report, w) -> None:
    w("\n## 8 · Outstanding\n")
    items: list[str] = []

    stale = r.stale_syntheses
    if not stale.empty:
        items.append(
            f"{len(stale)} synthesis(es) are stale — a Key Event they were "
            "built on has changed since they were written."
        )

    if not r.spans.empty:
        unverified = int((~r.spans["verified"].astype(bool)).sum())
        if unverified:
            items.append(
                f"{unverified} quotation(s) could not be located in their "
                "source and remain unverified."
            )

    if not r.syntheses.empty and "developer_assessment" in r.syntheses:
        unassessed = int(
            r.syntheses["developer_assessment"].isna().sum()
            + (r.syntheses["developer_assessment"] == "Not assessed").sum()
        )
        if unassessed:
            items.append(
                f"{unassessed} synthesis(es) have no developer assessment. The "
                "calculated score is decision support only, so these "
                "relationships have not yet been judged by anyone."
            )

    if not r.table2.empty and r.canonical.empty:
        items.append(
            "Normalisation has not been run, so relationships are grouped on "
            "raw wording and the same event may appear under several names."
        )

    # Two canonical Key Events sharing a name are two nodes the figure will
    # draw separately and a reader will read as one. Nothing else in the tool
    # looks for this, and it is invisible in the map itself.
    if not r.canonical.empty and "canonical_name" in r.canonical:
        names = r.canonical["canonical_name"].astype(str).str.strip().str.lower()
        duplicated = sorted(set(names[names.duplicated()]))
        if duplicated:
            items.append(
                f"{len(duplicated)} canonical Key Event name(s) are used by more "
                f"than one Key Event: {', '.join(duplicated)}. They will appear "
                "as separate nodes carrying the same label — merge them or "
                "rename them before the figure is used."
            )

    if not r.crosswalk.empty and "merge_basis" in r.crosswalk:
        unrecorded = int(r.crosswalk["merge_basis"].isna().sum())
        if unrecorded:
            items.append(
                f"{unrecorded} raw wording(s) have no recorded authorising rule, "
                "so the step from that wording to its Key Event cannot be "
                "audited — only the result is known, not the grounds."
            )

    if not items:
        w("Nothing outstanding: no stale syntheses, no unverified quotations, "
          "and every synthesis carries a developer assessment.\n")
        return

    for item in items:
        w(f"- {item}")
    w("")


def report_csv(report: Stage2Report) -> str:
    """A flat one-row-per-decision export, for spreadsheets and appendices."""
    rows: list[dict[str, Any]] = []

    for _, d in report.decisions.iterrows():
        rows.append({
            "section": "curation decision",
            "subject": _text(d.get("member_ids")),
            "detail": _text(d.get("action_label") or d.get("action")),
            "classification": _text(d.get("relationship")),
            "curator": _text(d.get("curator")),
            "rationale": _text(d.get("curator_rationale")),
            "when": _text(d.get("created_at")),
        })
    for _, a in report.approvals.iterrows():
        rows.append({
            "section": "approval",
            "subject": f"{_text(a.get('target_type'))} {_text(a.get('target_key'))}",
            "detail": f"{_text(a.get('from_state'))} -> {_text(a.get('to_state'))}",
            "classification": "",
            "curator": _text(a.get("curator")),
            "rationale": _text(a.get("note")),
            "when": _text(a.get("created_at")),
        })
    for _, s in report.syntheses.iterrows():
        rows.append({
            "section": "developer assessment",
            "subject": _text(s.get("ker_name")),
            "detail": _text(s.get("developer_assessment"), "not assessed"),
            "classification": _text(s.get("overall_confidence")),
            "curator": _text(s.get("developer_curator")),
            "rationale": _text(s.get("developer_rationale")),
            "when": _text(s.get("generated_at")),
        })
    for _, role in report.roles.iterrows():
        rows.append({
            "section": "pathway endpoint",
            "subject": _text(role.get("key_event")),
            "detail": _text(role.get("role")),
            "classification": "",
            "curator": _text(role.get("curator")),
            "rationale": _text(role.get("rationale")),
            "when": _text(role.get("assigned_at")),
        })

    if not rows:
        return "section,subject,detail,classification,curator,rationale,when\n"
    return pd.DataFrame(rows).to_csv(index=False)
