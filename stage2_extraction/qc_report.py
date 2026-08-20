from __future__ import annotations

"""
Per-run quality-control report.

The pipeline already computes everything an auditor would ask for — how many
quotations were located verbatim, which ones were not, which edges are
contradicted, how confident each edge is — but it computes them for the screen
and then forgets them. A curator who returns a week later, or a reviewer who
was never in the room, has no way to ask "how good was that run?".

This module assembles the answer from the run manifest, the robustness
counters and the evidence spans, and renders it as Markdown, JSON or CSV so it
can be filed alongside the extraction it describes.

It also states plainly where a run should not be trusted. A verification rate
below 40 %, a handful of repaired JSON replies or a temperature the provider
silently rejected are all individually survivable and collectively decisive,
and none of them are visible in Table 1.
"""

import datetime
import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from run_manifest import REPRODUCIBILITY_NOTE
from stage2_extraction import table1_store
from stage2_extraction.pdf_reader import strip_control_chars

__all__ = [
    "QCReport",
    "build_qc_report",
    "report_markdown",
    "report_json",
    "unverified_quotes_csv",
]


#: Verification rate below which the run's provenance is not fit to curate.
#: Matches the threshold `ker_extractor` already warns at, so the QC report and
#: the live warning cannot disagree.
_MIN_VERIFICATION_RATE = 0.40

#: Share of model calls needing JSON repair above which the output is partial
#: often enough to matter.
_MAX_REPAIR_RATE = 0.10


@dataclass
class QCReport:
    """Everything needed to judge one extraction run."""

    run_id: Optional[int]
    generated_at: str
    manifest: dict[str, Any] = field(default_factory=dict)
    scope: str = ""                       # what the numbers cover

    n_papers: int = 0
    n_kers: int = 0
    n_spans: int = 0
    n_verified: int = 0

    per_paper: pd.DataFrame = field(default_factory=pd.DataFrame)
    unverified: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence_counts: dict[str, int] = field(default_factory=dict)
    n_contradicted_rows: int = 0
    flags: list[str] = field(default_factory=list)

    #: Papers handed to the run that produced no rows, and why.
    #:
    #: The report used to be built entirely from `table1_extractions`, which
    #: means it could only ever describe papers that yielded something. A run
    #: over thirteen papers that extracted from eleven produced a QC report
    #: about eleven papers, and the two that gave nothing — the two most worth
    #: looking at — were not mentioned anywhere.
    barren: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_papers_attempted: int = 0

    #: One entry per stored KER synthesis, with the conditions of the model
    #: call that wrote it.
    #:
    #: Included because a synthesis is the prose a reader treats as the
    #: assessment, and until it carried a `run_id` it was the least documented
    #: output the tool produces — a model name and a timestamp, nothing else.
    #: A QC report that certifies the extraction while saying nothing about the
    #: text built on top of it describes the wrong half of the pipeline.
    syntheses: list[dict[str, Any]] = field(default_factory=list)

    #: Rows a curator entered or corrected by hand, counted and then set aside.
    #:
    #: This report answers "did the model do its job on this corpus?", and a
    #: hand-typed row has no model output in it to verify, no self-assessed
    #: confidence and no reply to repair. Counting one would move the
    #: verification rate for a reason that has nothing to do with the
    #: extraction — a curator who added ten well-quoted claims would appear to
    #: have improved the model. They are reported on their own line instead,
    #: because leaving them out entirely would understate the corpus.
    n_curator_rows: int = 0
    n_curator_edited_rows: int = 0
    n_curator_spans: int = 0
    n_curator_verified: int = 0

    @property
    def verification_rate(self) -> float:
        return self.n_verified / self.n_spans if self.n_spans else 0.0

    @property
    def curator_verification_rate(self) -> float:
        return (
            self.n_curator_verified / self.n_curator_spans
            if self.n_curator_spans
            else 0.0
        )

    @property
    def n_barren(self) -> int:
        return int(len(self.barren))

    @property
    def n_recoverable(self) -> int:
        """
        Barren papers whose failure was mechanical rather than a finding.

        A paper the model read and reported nothing about is a result. A paper
        whose reply was truncated, failed to parse, or never reached the model
        is a gap, and re-running it would probably close it.
        """
        if self.barren.empty or "category" not in self.barren.columns:
            return 0
        # `refusal` counts as recoverable: nothing was learned about the
        # paper, and another model will usually read it. It is not recoverable
        # by re-running the same configuration, which is why it carries its
        # own category and its own advice.
        recoverable = {"truncated", "parse_failure", "provider_error",
                       "chunking_dropped", "no_text", "error", "refusal"}
        return int(self.barren["category"].isin(recoverable).sum())

    @property
    def n_refused(self) -> int:
        if self.barren.empty or "category" not in self.barren.columns:
            return 0
        return int((self.barren["category"] == "refusal").sum())


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _repair_rate(manifest: dict[str, Any]) -> float:
    calls = int(manifest.get("llm_calls") or 0)
    if not calls:
        return 0.0
    repairs = int(manifest.get("json_repairs") or 0)
    failures = int(manifest.get("json_failures") or 0)
    return (repairs + failures) / calls


def _collect_flags(report: QCReport) -> list[str]:
    """
    Reasons to distrust this run, worst first.

    Each entry names the observation and what it implies, so the reader does
    not have to know the pipeline internals to act on it.
    """
    flags: list[str] = []
    m = report.manifest

    # First, because it is the only flag about evidence that is not in the
    # table at all. Everything else here judges what was extracted; this
    # judges what was not, and a corpus can look clean precisely because the
    # papers that would have complicated it produced nothing.
    if report.n_recoverable:
        flags.append(
            f"{report.n_recoverable} of {report.n_papers_attempted} paper(s) "
            f"produced no rows for a mechanical reason — a truncated reply, a "
            f"reply that would not parse, a provider error, or text that never "
            f"reached the model. Those papers are gaps in this corpus, not "
            f"evidence of absence, and the gaps are invisible in every table "
            f"below. See “Papers that produced nothing”."
        )
    elif report.n_barren:
        flags.append(
            f"{report.n_barren} of {report.n_papers_attempted} paper(s) "
            f"produced no rows, each because the model read the paper and "
            f"reported no mechanistic link. That is a finding about those "
            f"papers rather than a defect in the run."
        )

    if report.n_refused:
        flags.append(
            f"{report.n_refused} paper(s) were declined by the provider's "
            f"safety classifier and contributed nothing. On peer-reviewed "
            f"toxicology this is a false positive and a property of the model "
            f"rather than of the paper — re-running the same configuration "
            f"will not help. Send those papers through a different model or "
            f"provider, and say so in the methods: a corpus that silently "
            f"excludes the papers one classifier disliked is not the corpus "
            f"you described."
        )

    if report.n_spans == 0 and report.n_kers:
        flags.append(
            "No quotations were captured at all, so none of these KERs has "
            "provenance. Nothing in this run can be validated against the "
            "source text."
        )
    elif report.n_spans and report.verification_rate < _MIN_VERIFICATION_RATE:
        flags.append(
            f"Only {report.verification_rate:.0%} of quotations were located "
            "verbatim in the source. The rest are probably paraphrases the "
            "model composed, and each one needs checking by hand before the "
            "KER it supports is accepted."
        )

    repair_rate = _repair_rate(m)
    if repair_rate > _MAX_REPAIR_RATE:
        flags.append(
            f"{repair_rate:.0%} of model replies were truncated and repaired. "
            "Repaired replies are missing whatever came after the cut, so "
            "fields may be absent here that the paper does actually report. "
            "Raise the output-token budget and re-run."
        )

    if int(m.get("step_failures") or 0):
        flags.append(
            f"{m['step_failures']} extraction step(s) returned nothing usable. "
            "The affected KERs are incomplete rather than negative — an empty "
            "field here does not mean the paper was silent."
        )

    if int(m.get("empty_replies") or 0):
        flags.append(
            f"The provider returned an empty reply {m['empty_replies']} time(s), "
            "usually a token budget exhausted by reasoning output or a context "
            "window overrun."
        )

    dropped = (m.get("dropped_params") or "").strip()
    if dropped:
        flags.append(
            f"The provider rejected and the pipeline dropped: {dropped}. Those "
            "settings reverted to the provider's defaults, so this run was not "
            "conducted under the configuration shown above."
        )

    if m.get("model_reported") and m.get("model") and m["model_reported"] != m["model"]:
        flags.append(
            f"The model alias {m['model']!r} resolved to {m['model_reported']!r}. "
            "Aliases are re-pointed without notice; quote the resolved name "
            "when reporting this run."
        )

    code_version = m.get("code_version") or ""
    if code_version.endswith("+dirty"):
        flags.append(
            "The code had uncommitted changes when this run executed, so the "
            "commit hash alone will not reproduce it."
        )
    elif not code_version:
        flags.append(
            "No code version was recorded (not a git checkout), so the exact "
            "pipeline behind this run cannot be identified later."
        )

    temperature = m.get("temperature")
    if temperature is not None and float(temperature) > 0:
        flags.append(
            f"Sampling temperature was {temperature}. Re-running will not "
            "reproduce these KERs exactly; treat single-run output as one "
            "sample, not as the paper's content."
        )

    return flags


def _synthesis_records() -> list[dict[str, Any]]:
    """
    Every stored synthesis, joined to the run that produced it.

    Left-joined deliberately: syntheses written before synthesis runs were
    recorded have `run_id` NULL, and they must still be listed. Omitting them
    would make an undocumented assessment look like no assessment, which is the
    more misleading of the two.
    """
    try:
        stored = table1_store.load_all_syntheses()
    except Exception:
        return []
    if stored is None or stored.empty:
        return []

    runs: dict[int, dict[str, Any]] = {}
    try:
        runs_df = table1_store.load_runs("synthesis")
        if runs_df is not None and not runs_df.empty:
            runs = {
                int(r["run_id"]): dict(r) for _, r in runs_df.iterrows()
            }
    except Exception:
        runs = {}

    out: list[dict[str, Any]] = []
    for _, row in stored.iterrows():
        run_id = row.get("run_id")
        run = runs.get(int(run_id)) if pd.notna(run_id) else None
        out.append(
            {
                "ker_name": row.get("ker_name"),
                "n_papers": row.get("n_papers"),
                "n_rows": row.get("n_rows"),
                "generated_at": row.get("generated_at"),
                "stale": bool(row.get("stale") or 0),
                "overall_confidence": row.get("overall_confidence"),
                "model": (run or {}).get("model") or row.get("model"),
                "provider": (run or {}).get("provider"),
                "temperature": (run or {}).get("temperature"),
                "seed": (run or {}).get("seed"),
                "prompt_fingerprint": (run or {}).get("prompt_fingerprint"),
                "code_version": (run or {}).get("code_version"),
                "json_repairs": (run or {}).get("json_repairs"),
                "documented": run is not None,
            }
        )
    return out


def build_qc_report(
    run_id: Optional[int] = None,
    *,
    table2_df: Optional[pd.DataFrame] = None,
) -> QCReport:
    """
    Assemble the QC report for one run, or for the whole database.

    `run_id=None` reports across every row, which is what you want when asking
    "is this corpus fit to curate?" rather than "was that run sound?".
    `table2_df` is accepted rather than recomputed so the report describes the
    same synthesis the user is looking at.
    """
    manifest = table1_store.get_run(run_id) if run_id is not None else {}
    manifest = manifest or {}
    synthesis_records = _synthesis_records()

    t1 = table1_store.load_table1_as_dataframe()
    if run_id is not None and not t1.empty and "run_id" in t1.columns:
        t1 = t1[t1["run_id"] == run_id]

    # Split curator rows out before any model quality measure is computed. See
    # `QCReport.n_curator_rows` for why they are counted separately rather than
    # either included or dropped.
    curator_t1 = pd.DataFrame()
    if not t1.empty and "origin" in t1.columns:
        is_curator = t1["origin"].fillna(table1_store.LLM_ORIGIN).isin(
            table1_store.CURATOR_ORIGINS
        )
        curator_t1 = t1[is_curator]
        t1 = t1[~is_curator]

    spans = table1_store.load_evidence_spans(
        t1["record_id"].tolist() if not t1.empty else []
    )
    curator_spans = table1_store.load_evidence_spans(
        curator_t1["record_id"].tolist() if not curator_t1.empty else []
    )

    report = QCReport(
        run_id=run_id,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        manifest=manifest,
        syntheses=synthesis_records,
        scope=(
            f"run {run_id}" if run_id is not None else "all rows in the database"
        ),
        n_papers=int(t1["source_doi"].nunique()) if not t1.empty else 0,
        n_kers=int(len(t1)),
        n_spans=int(len(spans)),
        n_verified=int(spans["verified"].sum()) if not spans.empty else 0,
        n_curator_rows=int(
            (curator_t1["origin"] == "curator").sum()
        ) if not curator_t1.empty else 0,
        n_curator_edited_rows=int(
            (curator_t1["origin"] == "curator_edited").sum()
        ) if not curator_t1.empty else 0,
        n_curator_spans=int(len(curator_spans)),
        n_curator_verified=(
            int(curator_spans["verified"].sum()) if not curator_spans.empty else 0
        ),
    )

    # --- per paper --------------------------------------------------------
    if not t1.empty:
        grouped = t1.groupby("source_doi", dropna=False).agg(
            paper=("source_filename", "first"),
            kers=("record_id", "count"),
            quotations=("n_evidence_spans", "sum"),
            verified=("n_verified_spans", "sum"),
            contradicting=("contradicts_ker", "sum"),
        )
        grouped = grouped.reset_index().rename(columns={"source_doi": "doi"})
        grouped["verified_pct"] = (
            grouped["verified"] / grouped["quotations"].replace(0, pd.NA) * 100
        ).round(0)
        report.per_paper = grouped.sort_values("verified_pct", na_position="first")
        report.n_contradicted_rows = int(t1["contradicts_ker"].fillna(0).sum())

        if "extraction_confidence" in t1.columns:
            report.confidence_counts = {
                str(k): int(v)
                for k, v in t1["extraction_confidence"].value_counts().items()
            }

    # --- papers that produced nothing -------------------------------------
    outcomes = table1_store.load_paper_outcomes(run_id)
    if outcomes.empty and run_id is not None:
        # An outcome whose run link could not be written still describes this
        # corpus, and dropping it here would reintroduce exactly the blind
        # spot this section exists to remove.
        unlinked = table1_store.load_paper_outcomes()
        if not unlinked.empty and "run_id" in unlinked.columns:
            outcomes = unlinked[unlinked["run_id"].isna()]
    if not outcomes.empty:
        report.n_papers_attempted = int(len(outcomes))
        barren = outcomes[outcomes["n_kers"].fillna(0) <= 0].copy()
        if not barren.empty:
            barren["why"] = barren["category"].map(
                lambda c: table1_store.OUTCOME_CATEGORIES.get(str(c), "")
            )
            cols = [
                c for c in (
                    "source_filename", "source_doi", "outcome", "category",
                    "why", "reason", "n_llm_calls", "n_truncated",
                )
                if c in barren.columns
            ]
            report.barren = barren[cols].reset_index(drop=True)
    elif manifest.get("papers_attempted"):
        # A run recorded before per-paper outcomes were stored. The count is
        # still known, so say how many are unaccounted for rather than
        # implying every paper is described below.
        report.n_papers_attempted = int(manifest.get("papers_attempted") or 0)

    # --- unverified quotations -------------------------------------------
    if not spans.empty:
        unverified = spans[~spans["verified"].astype(bool)].copy()
        cols = [
            c
            for c in (
                "record_id", "source_doi", "field", "match_ratio",
                "section", "page_start", "quote",
            )
            if c in unverified.columns
        ]
        unverified = unverified[cols].sort_values(
            "match_ratio", ascending=False
        ) if cols else unverified
        report.unverified = unverified

    # --- table 2 confidence distribution ---------------------------------
    if table2_df is not None and not table2_df.empty:
        if "confidence_band" in table2_df.columns:
            report.confidence_counts = {
                f"band:{k}": int(v)
                for k, v in table2_df["confidence_band"].value_counts().items()
            } | report.confidence_counts

    report.flags = _collect_flags(report)
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_MANIFEST_LABELS: list[tuple[str, str]] = [
    ("mode", "Extraction mode"),
    ("target_upstream", "Target upstream KE"),
    ("target_downstream", "Target downstream KE"),
    ("provider", "Provider"),
    ("model", "Model requested"),
    ("model_reported", "Model reported by API"),
    ("endpoint_host", "Endpoint"),
    ("temperature", "Temperature"),
    ("top_p", "top_p"),
    ("seed", "Seed"),
    ("max_output_tokens", "Max output tokens"),
    ("num_ctx", "Context window (Ollama)"),
    ("transmission_ack", "Hosted-processing basis"),
    ("prompt_fingerprint", "Prompt fingerprint"),
    ("budget_scale", "Output budget multiplier"),
    ("code_version", "Code version"),
    ("schema_version", "Database schema"),
    ("aopwiki_version", "AOP-Wiki dump"),
    ("chunking_enabled", "Chunk selection"),
    ("chunk_char_budget", "Chunk char budget"),
    ("chunk_min_score", "Chunk min score"),
    ("chunk_scorer", "Chunk scorer"),
    ("llm_triage", "LLM chunk triage"),
    ("ols4_enabled", "OLS4 enrichment"),
    ("python_version", "Python"),
    ("platform", "Platform"),
    ("started_at", "Started"),
    ("finished_at", "Finished"),
    ("status", "Status"),
]

_TELEMETRY_LABELS: list[tuple[str, str]] = [
    ("llm_calls", "Model calls"),
    ("step_failures", "Steps with unusable replies"),
    ("json_repairs", "Truncated replies repaired"),
    ("json_failures", "Replies that could not be parsed"),
    ("truncated_steps", "Steps that hit the token ceiling"),
    ("empty_replies", "Empty replies"),
    ("refusals", "Calls the provider declined"),
    ("provider_errors", "Provider/network errors"),
    ("provider_retries", "Payload retries after rejected parameters"),
    ("dropped_params", "Parameters dropped"),
    ("renamed_params", "Parameters renamed"),
    ("papers_attempted", "Papers attempted"),
    ("papers_with_kers", "Papers yielding KERs"),
    ("chunks_selected", "Chunks sent"),
    ("chunks_total", "Chunks available"),
    ("chars_sent", "Characters sent"),
    ("chars_total", "Characters available"),
]


def _markdown_table(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a Markdown table.

    Written out rather than using `DataFrame.to_markdown`, which needs
    `tabulate` — an undeclared dependency would turn a QC report into an
    ImportError on a fresh deployment.
    """
    if df.empty:
        return ""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


#: Manifest columns stored as SQLite integers that mean yes/no. Rendering them
#: as "1" would make the report harder to read than the settings panel it is
#: meant to replace.
_BOOLEAN_KEYS = {"chunking_enabled", "llm_triage", "ols4_enabled"}


def _fmt(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass  # arrays and the like are not missing values
    if value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and abs(value) >= 1000:
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def report_markdown(report: QCReport) -> str:
    """Render the report as Markdown, suitable for filing with the results."""
    out = io.StringIO()
    w = out.write

    w(f"# Extraction QC report — {report.scope}\n\n")
    w(f"Generated {report.generated_at}\n\n")

    if report.flags:
        w("## Read this first\n\n")
        for flag in report.flags:
            w(f"- {flag}\n")
        w("\n")
    else:
        w("No quality flags were raised for this run.\n\n")

    w("## Summary\n\n")
    w("| Measure | Value |\n| --- | --- |\n")
    if report.n_papers_attempted:
        w(f"| Papers submitted | {report.n_papers_attempted} |\n")
    w(f"| Papers that yielded rows | {report.n_papers} |\n")
    if report.n_barren:
        w(f"| Papers that yielded nothing | {report.n_barren} |\n")
    w(f"| KER rows | {report.n_kers} |\n")
    w(f"| Quotations | {report.n_spans} |\n")
    w(
        f"| Located verbatim | {report.n_verified} "
        f"({report.verification_rate:.0%}) |\n"
    )
    w(f"| Rows marked as contradicting | {report.n_contradicted_rows} |\n\n")

    if report.n_curator_rows or report.n_curator_edited_rows:
        w("## Curator-entered rows\n\n")
        w(
            "Excluded from every figure above. Nothing in them came from a "
            "model, so a verification rate, a repair rate and a self-assessed "
            "confidence are all measures of something that did not happen — "
            "including them would change the run's apparent quality for a "
            "reason unrelated to the extraction.\n\n"
        )
        w("| Measure | Value |\n| --- | --- |\n")
        w(f"| Claims entered by hand | {report.n_curator_rows} |\n")
        w(f"| Extracted rows since corrected | {report.n_curator_edited_rows} |\n")
        w(f"| Quotations on those rows | {report.n_curator_spans} |\n")
        w(
            f"| Located verbatim | {report.n_curator_verified} "
            f"({report.curator_verification_rate:.0%}) |\n\n"
        )
        w(
            "A hand-entered claim whose quotation was located in the source "
            "is as traceable as an extracted one. One with no quotation is an "
            "assertion, and the final map draws it as such.\n\n"
        )

    if not report.barren.empty:
        w("## Papers that produced nothing\n\n")
        w(
            "Every paper below was submitted and returned no rows. The "
            "distinction that matters is between a paper the model read and "
            "found no mechanism in — a finding — and one that failed "
            "mechanically, which is a gap in this corpus and is invisible "
            "everywhere else in this report.\n\n"
        )
        w(_markdown_table(report.barren))
        w("\n")
        if report.n_recoverable:
            w(
                f"**{report.n_recoverable} of these are worth re-running.** "
                "Truncated replies need a higher output-token budget; parse "
                "failures usually need a larger model; papers dropped by chunk "
                "scoring need a lower relevance threshold or a bigger "
                "character budget.\n\n"
            )
    elif report.n_papers_attempted > report.n_papers:
        w("## Papers that produced nothing\n\n")
        w(
            f"{report.n_papers_attempted - report.n_papers} paper(s) yielded "
            "no rows, but this run predates per-paper outcome recording, so "
            "the reason was not kept. Re-running will capture it.\n\n"
        )

    if report.syntheses:
        w("## Evidence syntheses\n\n")
        undocumented = [s for s in report.syntheses if not s["documented"]]
        stale = [s for s in report.syntheses if s["stale"]]
        w(f"{len(report.syntheses)} stored synthesis(es).\n\n")
        w("| KER | Papers | Claims | Confidence | Model | Temp | Prompt | Stale |\n")
        w("| --- | ---: | ---: | --- | --- | ---: | --- | --- |\n")
        for s in report.syntheses:
            w(
                f"| {_fmt(s['ker_name'])} | {_fmt(s['n_papers'])} | "
                f"{_fmt(s['n_rows'])} | {_fmt(s['overall_confidence'])} | "
                f"{_fmt(s['model'])} | {_fmt(s['temperature'])} | "
                f"{_fmt(s['prompt_fingerprint'])} | "
                f"{'yes' if s['stale'] else 'no'} |\n"
            )
        w("\n")
        if undocumented:
            w(
                f"**{len(undocumented)} synthesis(es) carry no run record.** They "
                "were written before the synthesis step recorded a manifest, so "
                "the provider, temperature, seed and prompt version behind that "
                "text are not known and it cannot be compared with a later one "
                "on equal terms. Regenerate them if the text is going to be "
                "relied on.\n\n"
            )
        if stale:
            w(
                f"**{len(stale)} synthesis(es) are stale** — a Key Event they "
                "were built on has changed since. The text shown is the "
                "superseded version.\n\n"
            )

    if report.manifest:
        w("## Run conditions\n\n")
        w("| Setting | Value |\n| --- | --- |\n")
        for key, label in _MANIFEST_LABELS:
            if key not in report.manifest:
                continue
            value = report.manifest.get(key)
            if key in _BOOLEAN_KEYS and value is not None:
                value = bool(value)
            w(f"| {label} | {_fmt(value)} |\n")
        w("\n")

        w("## What the pipeline absorbed\n\n")
        w("| Event | Count |\n| --- | --- |\n")
        for key, label in _TELEMETRY_LABELS:
            if key in report.manifest:
                w(f"| {label} | {_fmt(report.manifest.get(key))} |\n")
        w("\n")

        notes = report.manifest.get("notes")
        if notes:
            w("### Notes recorded during the run\n\n")
            for line in str(notes).splitlines():
                w(f"- {line}\n")
            w("\n")

    if report.confidence_counts:
        w("## Confidence distribution\n\n")
        w("| Level | Rows |\n| --- | --- |\n")
        for key, count in sorted(report.confidence_counts.items()):
            w(f"| {key} | {count} |\n")
        w("\n")

    if not report.per_paper.empty:
        w("## Verification by paper\n\n")
        w("Lowest verification rate first — start curation here.\n\n")
        w(_markdown_table(report.per_paper))
        w("\n\n")

    if not report.unverified.empty:
        w(f"## Unverified quotations ({len(report.unverified)})\n\n")
        w(
            "These sentences were returned as verbatim quotations but could "
            "not be found in the source document. Each one is either a "
            "paraphrase or a fabrication, and the field it supports is "
            "unsupported until a human checks it.\n\n"
        )
        for _, row in report.unverified.head(50).iterrows():
            ratio = row.get("match_ratio")
            ratio_txt = f"{float(ratio):.0%}" if pd.notna(ratio) else "—"
            w(
                f"- **{row.get('source_doi', '?')}** "
                f"(field `{row.get('field', '?')}`, best match {ratio_txt})\n"
                f"  > {str(row.get('quote', '')).strip()}\n"
            )
        if len(report.unverified) > 50:
            w(f"\n_{len(report.unverified) - 50} further rows in the CSV export._\n")
        w("\n")

    w("## On reproducibility\n\n")
    w(REPRODUCIBILITY_NOTE + "\n\n")
    w(
        "This report describes machine extraction only. It says whether a "
        "quotation exists in the source document — not whether the Key Event "
        "Relationship inferred from it is correct. Expert review remains "
        "required.\n"
    )

    return out.getvalue()


def report_json(report: QCReport) -> str:
    payload = {
        "run_id": report.run_id,
        "generated_at": report.generated_at,
        "scope": report.scope,
        "manifest": report.manifest,
        "summary": {
            "papers": report.n_papers,
            "ker_rows": report.n_kers,
            "quotations": report.n_spans,
            "verified": report.n_verified,
            "verification_rate": round(report.verification_rate, 4),
            "contradicting_rows": report.n_contradicted_rows,
        },
        "syntheses": report.syntheses,
        "undocumented_syntheses": sum(
            1 for s in report.syntheses if not s["documented"]
        ),
        "curator_entered": {
            "rows": report.n_curator_rows,
            "edited_rows": report.n_curator_edited_rows,
            "quotations": report.n_curator_spans,
            "verified": report.n_curator_verified,
            "verification_rate": round(report.curator_verification_rate, 4),
            "note": (
                "Excluded from the summary figures: no model produced these "
                "rows, so model quality measures do not apply to them."
            ),
        },
        "confidence_counts": report.confidence_counts,
        "flags": report.flags,
        "per_paper": (
            report.per_paper.to_dict(orient="records")
            if not report.per_paper.empty
            else []
        ),
        "unverified_quotations": (
            report.unverified.to_dict(orient="records")
            if not report.unverified.empty
            else []
        ),
        "reproducibility_note": REPRODUCIBILITY_NOTE,
    }
    return json.dumps(payload, indent=2, default=str)


def unverified_quotes_csv(report: QCReport) -> str:
    if report.unverified.empty:
        return "source_doi,field,match_ratio,section,page_start,quote\n"
    cleaned = report.unverified.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(strip_control_chars)
    return cleaned.to_csv(index=False)
