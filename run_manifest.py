from __future__ import annotations

"""
Run provenance and robustness telemetry.

Why this exists
---------------
An LLM extraction is not reproducible the way a database query is. The same
paper, the same prompts and the same model can yield different KERs on
different days: sampling is stochastic, hosted models are re-versioned without
notice, batching changes numerics even at temperature 0, and this pipeline
deliberately absorbs a lot of trouble — it repairs truncated JSON, drops
parameters a provider rejects, and carries on when a step fails.

None of that is a defect. But it means an extraction is only interpretable
alongside the conditions that produced it. Two Table 1 rows that disagree are
not evidence of anything until you know whether they came from the same model,
the same prompts and the same chunk budget.

This module records those conditions — one `RunManifest` per run — and counts
what went quietly wrong along the way, so a run can be judged, compared with an
earlier one, and reproduced as closely as the provider allows.

The manifest is persisted by `table1_store` in the `extraction_runs` table, and
every Table 1 row and evidence span carries its `run_id`.

Telemetry is collected through an ambient *active run*, so the low-level call
sites (`llm_providers._post_json`, `json_repair.extract_json`) can report an
event without every intermediate function having to pass a recorder down.
Recording never raises: instrumentation must not be able to break an
extraction.

That recorder is held per *thread*, not per process. Streamlit serves every
browser session from one process, so a process-wide recorder meant two people
extracting at the same time shared one set of counters and each run's manifest
attributed the other's calls, refusals and repairs to itself.
"""

import datetime
import hashlib
import inspect
import platform
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

__all__ = [
    "RunTelemetry",
    "RunManifest",
    "start_run",
    "active_run",
    "record",
    "end_run",
    "prompt_fingerprint",
    "code_version",
    "REPRODUCIBILITY_NOTE",
]


#: Shown in the UI and written into the QC report. Users reasonably assume that
#: a low temperature means a fixed answer; it does not, and saying so once
#: plainly is better than letting them discover it from a confusing diff.
REPRODUCIBILITY_NOTE = (
    "Identical inputs do not guarantee identical output. Sampling is "
    "stochastic, hosted models are re-versioned without notice, and even at "
    "temperature 0 server-side batching makes results non-deterministic. This "
    "manifest records the conditions of the run so results can be compared and "
    "approximately reproduced — it is not a guarantee that they will be."
)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class RunTelemetry:
    """
    Counters for everything the pipeline recovers from silently.

    A run where a third of the steps needed JSON repair is not comparable with
    a clean one, and right now that difference is invisible once the run ends.
    """

    llm_calls: int = 0              # steps that reached a provider
    step_failures: int = 0          # steps whose reply could not be parsed
    provider_errors: int = 0        # network / HTTP failures surfaced to the pipeline
    provider_retries: int = 0       # payload resent after a rejected parameter
    json_repairs: int = 0           # truncated JSON salvaged by json_repair
    json_failures: int = 0          # JSON that could not be salvaged
    truncated_steps: int = 0        # replies that hit the output-token ceiling
    empty_replies: int = 0          # provider returned no text at all
    #: Calls the provider's safety classifier declined.
    #:
    #: Counted separately from `provider_errors` because it is not an
    #: error in the pipeline or the network: the call succeeded and the
    #: answer was withheld. Folding it into the error count made a run
    #: that lost a paper to a classifier look like a run with a flaky
    #: connection, which points at the wrong fix.
    refusals: int = 0

    #: Steps a refusal cost the primary model and the fallback then answered,
    #: and which models those were. A run where this is non-zero produced rows
    #: from more than one model, so the manifest cannot describe it with a
    #: single model name — the whole purpose of recording provider and model is
    #: to say what conditions the rows were produced under, and "mostly
    #: claude-sonnet-5" is not those conditions.
    refusal_fallbacks: int = 0
    fallback_models: set[str] = field(default_factory=set)

    #: Refused steps re-asked of the same model at a raised temperature.
    #:
    #: Separate from `provider_retries`, which counts payloads resent after the
    #: provider rejected a *parameter*. A refusal was previously recorded as
    #: `provider_retry(param="refusal")`, which put the string "refusal" into
    #: the dropped-parameter set — so a run that lost a paper to a classifier
    #: reported that the provider had rejected a request parameter of that
    #: name. Two different events, and conflating them made the field that
    #: records what the provider would not accept unreadable.
    refusal_retries: int = 0

    papers_attempted: int = 0
    papers_with_kers: int = 0
    kers_extracted: int = 0

    chunks_total: int = 0
    chunks_selected: int = 0
    chars_total: int = 0            # characters in the source documents
    chars_sent: int = 0             # characters actually sent to the model

    dropped_params: set[str] = field(default_factory=set)
    renamed_params: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    #: The model string the API echoed back. Hosted aliases resolve to dated
    #: snapshots — "gpt-4o" was not the same weights in March as in November —
    #: so the resolved name is what a later comparison actually needs.
    model_reported: Optional[str] = None

    # -- event handling ----------------------------------------------------

    def record(self, event: str, **detail: Any) -> None:
        """Apply one telemetry event. Unknown events are ignored, not raised."""
        if event == "llm_call":
            self.llm_calls += 1
        elif event == "step_failure":
            self.step_failures += 1
        elif event == "provider_error":
            self.provider_errors += 1
        elif event == "provider_retry":
            self.provider_retries += 1
            param = detail.get("param")
            if param:
                if detail.get("replacement"):
                    self.renamed_params.add(str(param))
                else:
                    self.dropped_params.add(str(param))
        elif event == "json_repair":
            self.json_repairs += 1
        elif event == "json_failure":
            self.json_failures += 1
        elif event == "truncated_step":
            self.truncated_steps += 1
        elif event == "empty_reply":
            self.empty_replies += 1
        elif event == "refusal":
            self.refusals += 1
        elif event == "refusal_retry":
            self.refusal_retries += 1
        elif event == "refusal_fallback":
            self.refusal_fallbacks += 1
            name = str(detail.get("model") or "").strip()
            if name:
                self.fallback_models.add(name)
                self.record(
                    "note",
                    message=(
                        f"A refusal on step {detail.get('step')!r} was answered "
                        f"by {name} instead of the run's model. Rows from this "
                        f"run were produced by more than one model."
                    ),
                )
        elif event == "paper_attempted":
            self.papers_attempted += 1
        elif event == "paper_extracted":
            self.papers_with_kers += 1
            self.kers_extracted += int(detail.get("n_kers") or 0)
        elif event == "pathway_extracted":
            # The targeted extractor reports a whole chain rather than one
            # relationship, and reports it even when the chain is empty. A
            # paper that yielded no link is attempted but not contributing,
            # so it must not be counted here — that undercount is what made
            # a finished run claim zero papers with KERs while Table 1 held
            # eighteen rows.
            n_steps = int(detail.get("n_steps") or 0)
            if n_steps:
                self.papers_with_kers += 1
                self.kers_extracted += n_steps
        elif event == "chunk_selection":
            self.chunks_total += int(detail.get("n_chunks") or 0)
            self.chunks_selected += int(detail.get("n_selected") or 0)
            self.chars_total += int(detail.get("chars_total") or 0)
            self.chars_sent += int(detail.get("chars_selected") or 0)
        elif event == "model_reported":
            name = str(detail.get("model") or "").strip()
            if name and name != self.model_reported:
                if self.model_reported:
                    # Two different resolved models inside one run means the
                    # rows it produced are not a single population.
                    self.record(
                        "note",
                        message=(
                            f"Provider reported more than one model during this "
                            f"run: {self.model_reported} then {name}."
                        ),
                    )
                self.model_reported = name
        elif event == "note":
            message = str(detail.get("message") or "").strip()
            if message and message not in self.notes:
                self.notes.append(message)

    # -- derived -----------------------------------------------------------

    @property
    def repair_rate(self) -> float:
        """Share of model calls whose reply had to be repaired or failed."""
        if not self.llm_calls:
            return 0.0
        return (self.json_repairs + self.json_failures) / self.llm_calls

    @property
    def failure_rate(self) -> float:
        if not self.llm_calls:
            return 0.0
        return self.step_failures / self.llm_calls

    def as_row(self) -> dict[str, Any]:
        """Flatten to database columns."""
        return {
            "llm_calls": self.llm_calls,
            "step_failures": self.step_failures,
            "provider_errors": self.provider_errors,
            "provider_retries": self.provider_retries,
            "json_repairs": self.json_repairs,
            "json_failures": self.json_failures,
            "truncated_steps": self.truncated_steps,
            "empty_replies": self.empty_replies,
            "papers_attempted": self.papers_attempted,
            "papers_with_kers": self.papers_with_kers,
            "kers_extracted": self.kers_extracted,
            "chunks_total": self.chunks_total,
            "chunks_selected": self.chunks_selected,
            "chars_total": self.chars_total,
            "chars_sent": self.chars_sent,
            "dropped_params": ", ".join(sorted(self.dropped_params)) or None,
            "renamed_params": ", ".join(sorted(self.renamed_params)) or None,
            # `finish_run` drops keys the runs table has no column for, so
            # these are safe to emit now and become persistent the day the
            # columns exist. Until then the fact still survives, because a
            # fallback also writes a note and `notes` is a column.
            "refusals": self.refusals,
            "refusal_retries": self.refusal_retries,
            "refusal_fallbacks": self.refusal_fallbacks,
            "fallback_models": ", ".join(sorted(self.fallback_models)) or None,
            "notes": "\n".join(self.notes) or None,
            "model_reported": self.model_reported,
        }


# ---------------------------------------------------------------------------
# Active run
# ---------------------------------------------------------------------------

#: The active recorder, per thread.
#:
#: This was a module global, which meant one recorder for the whole process.
#: Streamlit serves every browser session from that one process, so two people
#: extracting at the same time incremented the same counters and each run's
#: manifest described the other's calls, refusals and repairs as its own. The
#: manifest exists precisely to say what conditions produced a given set of
#: rows; a shared one states that wrongly rather than not at all.
#:
#: Per-thread is per-session for work done during a script run, which is where
#: extraction happens.
_LOCAL = threading.local()


def start_run(telemetry: Optional[RunTelemetry] = None) -> RunTelemetry:
    """Make `telemetry` the active recorder for this thread and return it."""
    _LOCAL.active = telemetry or RunTelemetry()
    return _LOCAL.active


def active_run() -> Optional[RunTelemetry]:
    return getattr(_LOCAL, "active", None)


def end_run() -> Optional[RunTelemetry]:
    """Detach the active recorder and return it for persisting."""
    finished = getattr(_LOCAL, "active", None)
    _LOCAL.active = None
    return finished


def record(event: str, **detail: Any) -> None:
    """
    Report a telemetry event to the active run, if there is one.

    Deliberately silent when no run is active — the pipeline is importable and
    testable outside the app, and instrumentation should never be a
    precondition for it working.
    """
    recorder = active_run()
    if recorder is None:
        return
    try:
        recorder.record(event, **detail)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent


def code_version() -> Optional[str]:
    """
    Short git commit for the running code, suffixed `+dirty` if the working
    tree has uncommitted changes.

    A result produced from an edited working tree cannot be reproduced from a
    commit hash alone, so that distinction belongs in the manifest.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return None
    dirty = _git("status", "--porcelain")
    return f"{sha}+dirty" if dirty else sha


def prompt_fingerprint() -> Optional[str]:
    """
    Hash of the persona and every task prompt in `ker_extractor`.

    Editing a prompt changes the output as surely as changing the model does,
    so a run recorded without this cannot be compared with an earlier one.
    """
    try:
        from stage2_extraction import ker_extractor as ke

        parts: list[str] = [getattr(ke, "_PERSONA", "")]
        for name in sorted(dir(ke)):
            if not name.startswith("_task_"):
                continue
            fn = getattr(ke, name)
            if callable(fn):
                try:
                    parts.append(inspect.getsource(fn))
                except (OSError, TypeError):
                    parts.append(name)
        blob = "\n".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _endpoint_host(base_url: Optional[str]) -> Optional[str]:
    """Host only — the full URL may carry a key in a proxied setup."""
    if not base_url:
        return None
    try:
        parsed = urlparse(base_url)
        return parsed.netloc or base_url[:80]
    except Exception:
        return None


def _aopwiki_version() -> Optional[str]:
    try:
        from stage2_extraction import aopwiki_xml

        return aopwiki_xml.get_local_version()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Everything needed to interpret, compare or re-attempt a run."""

    #: 'extraction' | 'screening' | 'synthesis'.
    #:
    #: Synthesis was missing until it was noticed that it is the only model
    #: call in the pipeline with no manifest at all. It runs from a button in
    #: its own Streamlit rerun, outside the extraction page's start_run /
    #: end_run pair, so `active_run()` was None and every telemetry event it
    #: reported went nowhere. That left the consolidated assessment — the prose
    #: a reader treats as the finding — recording its model and nothing else:
    #: no provider, temperature, seed, prompt fingerprint or code version. Two
    #: syntheses of one KER that disagreed could not be told apart.
    stage: str = "extraction"

    #: 'open' discovers whatever the papers contain and lets the model name the
    #: events; 'targeted' answers one specified question with the labels fixed
    #: by the user. Rows from the two modes are not comparable — an open run's
    #: silence about a relationship means nobody looked for it, a targeted
    #: run's silence means it was looked for and not found.
    mode: str = "open"                           # 'open' | 'targeted'
    target_upstream: Optional[str] = None
    target_downstream: Optional[str] = None
    started_at: str = ""
    finished_at: Optional[str] = None
    status: str = "running"                      # running | completed | failed

    # Model configuration
    provider: Optional[str] = None
    model: Optional[str] = None                  # as requested by the user
    model_reported: Optional[str] = None         # as echoed back by the API
    endpoint_host: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    num_ctx: Optional[int] = None
    seed: Optional[int] = None

    # Pipeline configuration
    prompt_fingerprint: Optional[str] = None
    budget_scale: Optional[float] = None
    chunking_enabled: Optional[bool] = None
    chunk_char_budget: Optional[int] = None
    chunk_min_score: Optional[float] = None
    chunk_scorer: Optional[str] = None
    llm_triage: Optional[bool] = None
    ols4_enabled: Optional[bool] = None
    ols4_min_score: Optional[float] = None
    max_kers: Optional[int] = None

    # Environment
    aopwiki_version: Optional[str] = None
    code_version: Optional[str] = None
    schema_version: Optional[int] = None
    python_version: str = ""
    platform: str = ""

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        if not self.python_version:
            self.python_version = sys.version.split()[0]
        if not self.platform:
            self.platform = platform.platform()

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        stage: str = "extraction",
        **overrides: Any,
    ) -> "RunManifest":
        """Build a manifest from an `LLMConfig` plus UI settings."""
        manifest = cls(
            stage=stage,
            provider=getattr(cfg, "provider", None),
            model=getattr(cfg, "model", None),
            endpoint_host=_endpoint_host(getattr(cfg, "base_url", None)),
            temperature=getattr(cfg, "temperature", None),
            top_p=getattr(cfg, "top_p", None),
            max_output_tokens=getattr(cfg, "max_output_tokens", None),
            num_ctx=getattr(cfg, "num_ctx", None),
            seed=getattr(cfg, "seed", None),
            prompt_fingerprint=prompt_fingerprint(),
            aopwiki_version=_aopwiki_version(),
            code_version=code_version(),
        )
        for key, value in overrides.items():
            if hasattr(manifest, key):
                setattr(manifest, key, value)
        return manifest

    # -- serialisation -----------------------------------------------------

    def finish(self, status: str = "completed") -> "RunManifest":
        self.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.status = status
        return self

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("chunking_enabled", "llm_triage", "ols4_enabled"):
            if row.get(key) is not None:
                row[key] = int(bool(row[key]))
        return row

    def summary_lines(self) -> list[str]:
        """Human-readable one-per-line summary for the UI and QC report."""
        bits = [
            f"Provider: {self.provider or '—'}",
            f"Model requested: {self.model or '—'}",
        ]
        if self.model_reported and self.model_reported != self.model:
            bits.append(f"Model reported by API: {self.model_reported}")
        bits += [
            f"Temperature: {self.temperature if self.temperature is not None else '—'}",
            f"Max output tokens: {self.max_output_tokens or '—'}",
            f"Prompt fingerprint: {self.prompt_fingerprint or '—'}",
            f"Code version: {self.code_version or 'not a git checkout'}",
            f"AOP-Wiki dump: {self.aopwiki_version or '—'}",
        ]
        if self.chunking_enabled is not None:
            bits.append(
                f"Chunking: {'on' if self.chunking_enabled else 'off'}"
                + (
                    f" (budget {self.chunk_char_budget:,} chars, "
                    f"min score {self.chunk_min_score})"
                    if self.chunking_enabled and self.chunk_char_budget
                    else ""
                )
            )
        return bits
