from __future__ import annotations

"""
Stepwise KER extraction pipeline.

Instead of asking the LLM to produce one giant JSON object covering every field
of every KER in a single call, we break the work into small focused steps:

    Step 1 — list_ker_pairs        : identify upstream/downstream pairs
    Step 2 — classify_ker          : levels, adjacency, name, description
    Step 3 — assess_evidence       : paper_type, plausibility, contradicts, ...
    Step 4 — applicability         : taxa, sex, life stage
    Step 5 — quantitative          : modulating factors, time scale, ...
    Step 6 — study_meta            : study_design, exposure_route, confidence, ...

Each step is a separate Ollama call with its own prompt. A StepResult is
captured for every call (prompt, raw response, parsed value, error) so the
caller can show exactly what happened at each step — making debugging much
easier than the previous one-shot prompt.

Public entry point:

    extract_kers_from_text(paper_text, model, ollama_url=None, on_step=None)
        returns (extractions, warnings)

Pass `on_step=lambda s: ...` to receive each StepResult as it completes (e.g.
for live streaming into the Streamlit UI).
"""

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from schemas import KERExtraction
from stage2_extraction.llm_providers import LLMConfig, LLMProviderError

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

KE_LEVELS         = {"MIE", "Molecular", "Cellular", "Tissue", "Organ", "Individual", "Population"}
KER_ADJACENCY     = {"Adjacent", "Non-adjacent"}
PAPER_TYPES       = {"Primary study", "Review / meta-analysis", "In silico"}
STUDY_DESIGNS     = {"In vivo", "In vitro", "In silico", "Ex vivo", "Epidemiological", "Review / meta-analysis"}
SEX_VALUES        = {"Male", "Female", "Mixed", "Not specified"}
CONFIDENCE_VALUES = {"High", "Medium", "Low"}


# ---------------------------------------------------------------------------
# Errors + step result
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    """Raised when the pipeline cannot continue (e.g. Ollama unreachable)."""


class ExtractionValidationError(ValueError):
    """Raised when a single KER fails schema validation — others can still proceed."""


@dataclass
class StepResult:
    """Outcome of one LLM call in the stepwise pipeline."""
    step: str                       # short id e.g. 'list_ker_pairs', 'classify_ker[1]'
    ok: bool                        # whether parsing succeeded
    prompt: str                     # the exact prompt sent to Ollama
    raw_response: str               # the exact raw text from Ollama
    parsed: Optional[Any] = None    # parsed dict/list, or None if parsing failed
    error: Optional[str] = None     # error message if ok=False
    ker_index: Optional[int] = None # which KER this step applies to (None for step 1)


StepCallback = Callable[[StepResult], None]


# ---------------------------------------------------------------------------
# Low-level provider call + JSON parsing
# ---------------------------------------------------------------------------

def _call_llm(
    cfg: LLMConfig,
    prompt: str,
    num_predict: int,
    cached_prefix: Optional[str] = None,
) -> str:
    """Invoke the configured provider with a per-call output-token budget.

    `cached_prefix` is forwarded to the provider so that on Anthropic /
    OpenAI / Ollama the persona + paper text are billed (or KV-cached) once
    instead of on every step.
    """
    call_cfg = replace(cfg, max_output_tokens=num_predict)
    try:
        return call_cfg.generate(prompt, cached_prefix=cached_prefix)
    except LLMProviderError as exc:
        raise ExtractionError(str(exc)) from exc


def _extract_json(raw: str) -> Any:
    """Best-effort JSON extraction from a possibly-noisy model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    obj_start, obj_end = text.find("{"), text.rfind("}")
    arr_start, arr_end = text.find("["), text.rfind("]")
    candidates: list[str] = []
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(text[arr_start : arr_end + 1])
    if not candidates:
        raise ValueError(f"No JSON found in response. First 300 chars:\n{raw[:300]}")
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
    raise ValueError(f"Invalid JSON: {last_err}\nRaw (500 chars):\n{raw[:500]}")


def _run_step(
    step_id: str,
    prompt: str,
    cfg: LLMConfig,
    on_step: Optional[StepCallback],
    ker_index: Optional[int] = None,
    num_predict: int = 1024,
    cached_prefix: Optional[str] = None,
) -> StepResult:
    """Run one LLM call and capture everything as a StepResult."""
    raw = _call_llm(cfg, prompt, num_predict=num_predict, cached_prefix=cached_prefix)
    try:
        parsed = _extract_json(raw)
        result = StepResult(
            step=step_id, ok=True, prompt=prompt, raw_response=raw,
            parsed=parsed, ker_index=ker_index,
        )
    except Exception as exc:
        result = StepResult(
            step=step_id, ok=False, prompt=prompt, raw_response=raw,
            parsed=None, error=str(exc), ker_index=ker_index,
        )
    if on_step is not None:
        try:
            on_step(result)
        except Exception:
            # Never let a UI callback break the pipeline.
            pass
    return result


# ---------------------------------------------------------------------------
# Step prompts
#
# Every call shares the same long prefix (persona + paper text). We send that
# prefix once via the provider's prompt-cache mechanism, and only the small
# variable "task" string in the user message. That cuts input tokens for the
# 30-ish per-paper calls from O(N × paper_size) to O(paper_size + N × task).
# ---------------------------------------------------------------------------

_PERSONA = (
    "You are a specialist in adverse outcome pathway (AOP) toxicology and the "
    "AOP-Wiki data model. A Key Event Relationship (KER) is any causal or "
    "mechanistic link the paper describes between an upstream biological event "
    "(e.g. receptor activation, oxidative stress, DNA damage) and a downstream "
    "event (e.g. apoptosis, inflammation, organ dysfunction, disease).\n"
)


def _build_cached_prefix(paper_text: str) -> str:
    """Return the static text shared by every step call for one paper."""
    return f"{_PERSONA}\nPAPER:\n{paper_text}"


def _task_list_pairs() -> str:
    return (
        "TASK: List every KER described or supported by the paper provided in "
        "the system context.\n"
        "Return ONLY JSON of the form:\n"
        '  {"pairs": [{"upstream": "<upstream KE name>", "downstream": "<downstream KE name>"}, ...]}\n'
        "Rules:\n"
        "- Aim for at least one pair if any mechanistic link is mentioned.\n"
        "- Use short specific KE names taken from or paraphrased from the paper.\n"
        "- Return {\"pairs\": []} ONLY if the paper has no mechanistic content "
        "(e.g. a pure exposure-assessment or analytical-method paper).\n"
        "JSON:"
    )


def _task_classify(upstream: str, downstream: str) -> str:
    return (
        f"For the KER below, classify the events using the paper in the system context.\n"
        f"  Upstream KE:   {upstream}\n"
        f"  Downstream KE: {downstream}\n\n"
        "Return ONLY JSON with these keys (no extra text):\n"
        '  {\n'
        '    "upstream_ke_level":   "MIE|Molecular|Cellular|Tissue|Organ|Individual|Population",\n'
        '    "downstream_ke_level": "same enum",\n'
        '    "ker_adjacency":       "Adjacent|Non-adjacent",\n'
        '    "ker_name":            "<upstream> leads to <downstream>",\n'
        '    "ker_description":     "1-3 sentences on the mechanistic basis, grounded in the paper"\n'
        '  }\n'
        "JSON:"
    )


def _task_evidence(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"summarise the evidence in the paper provided in the system context.\n\n"
        "Return ONLY JSON with these keys:\n"
        '  {\n'
        '    "paper_type":                 "Primary study|Review / meta-analysis|In silico",\n'
        '    "cited_evidence_dois":        "semicolon-separated DOIs from references, or null",\n'
        '    "biological_plausibility":    "short string or null",\n'
        '    "empirical_evidence_summary": "key data points / measurements supporting the link, or null",\n'
        '    "essentiality_evidence":      "knockout / antagonist / blocker evidence, or null",\n'
        '    "contradicts_ker":            true_or_false  // true if the paper argues AGAINST the KER\n'
        '  }\n'
        "JSON:"
    )


def _task_applicability(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"describe applicability based on what the paper in the system context studied.\n\n"
        "Return ONLY JSON with these keys:\n"
        '  {\n'
        '    "taxonomic_applicability":   "NCBI species name(s) e.g. \\"Mus musculus\\"; or \\"Not specified\\"",\n'
        '    "sex_applicability":         "Male|Female|Mixed|Not specified",\n'
        '    "life_stage_applicability":  "e.g. Adult, Embryo, Juvenile, or Not specified"\n'
        '  }\n'
        "JSON:"
    )


def _task_quantitative(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"extract any quantitative or temporal information from the paper in the system context.\n\n"
        "Return ONLY JSON with these keys (use null if the paper does not say):\n"
        '  {\n'
        '    "modulating_factors":             "string or null",\n'
        '    "quantitative_relationships":     "string or null",\n'
        '    "response_response_relationship": "string or null",\n'
        '    "time_scale":                     "string or null",\n'
        '    "feedforward_feedback_loops":     "string or null"\n'
        '  }\n'
        "JSON:"
    )


def _task_study_meta(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"describe the study design and your confidence in the extraction, "
        f"using the paper in the system context.\n\n"
        "Return ONLY JSON with these keys:\n"
        '  {\n'
        '    "study_design":          "In vivo|In vitro|In silico|Ex vivo|Epidemiological|Review / meta-analysis",\n'
        '    "exposure_route":        "e.g. Oral gavage, IP, Inhalation; or null",\n'
        '    "chemical_stressor":     "chemical(s) tested, or null",\n'
        '    "extraction_confidence": "High|Medium|Low"\n'
        '  }\n'
        "JSON:"
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _coerce_enum(value, allowed: set, fallback: Optional[str] = None) -> Optional[str]:
    if value is None:
        return fallback
    s = str(value).strip()
    if s in allowed:
        return s
    for candidate in allowed:
        if candidate.lower() == s.lower():
            return candidate
    return fallback


def _opt_str(v) -> Optional[str]:
    if v is None or str(v).strip().lower() in ("null", "none", ""):
        return None
    return str(v).strip()


def _req_str(v, field_name: str) -> str:
    if not v or not str(v).strip():
        raise ExtractionValidationError(f"Required field '{field_name}' missing or empty.")
    return str(v).strip()


def _req_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _assemble(
    upstream: str,
    downstream: str,
    classify: dict,
    evidence: dict,
    applicability: dict,
    quantitative: dict,
    study_meta: dict,
) -> KERExtraction:
    return KERExtraction(
        upstream_ke_name               = _req_str(upstream, "upstream_ke_name"),
        upstream_ke_level              = _coerce_enum(classify.get("upstream_ke_level"), KE_LEVELS, "Molecular"),
        downstream_ke_name             = _req_str(downstream, "downstream_ke_name"),
        downstream_ke_level            = _coerce_enum(classify.get("downstream_ke_level"), KE_LEVELS, "Molecular"),
        ker_name                       = _req_str(classify.get("ker_name") or f"{upstream} leads to {downstream}", "ker_name"),
        ker_description                = _req_str(classify.get("ker_description"), "ker_description"),
        ker_adjacency                  = _coerce_enum(classify.get("ker_adjacency"), KER_ADJACENCY, "Adjacent"),
        paper_type                     = _coerce_enum(evidence.get("paper_type"), PAPER_TYPES, "Primary study"),
        cited_evidence_dois            = _opt_str(evidence.get("cited_evidence_dois")),
        biological_plausibility        = _opt_str(evidence.get("biological_plausibility")),
        empirical_evidence_summary     = _opt_str(evidence.get("empirical_evidence_summary")),
        essentiality_evidence          = _opt_str(evidence.get("essentiality_evidence")),
        contradicts_ker                = _req_bool(evidence.get("contradicts_ker")),
        taxonomic_applicability        = applicability.get("taxonomic_applicability") or "Not specified",
        sex_applicability              = _coerce_enum(applicability.get("sex_applicability"), SEX_VALUES, "Not specified"),
        life_stage_applicability       = applicability.get("life_stage_applicability") or "Not specified",
        modulating_factors             = _opt_str(quantitative.get("modulating_factors")),
        quantitative_relationships     = _opt_str(quantitative.get("quantitative_relationships")),
        response_response_relationship = _opt_str(quantitative.get("response_response_relationship")),
        time_scale                     = _opt_str(quantitative.get("time_scale")),
        feedforward_feedback_loops     = _opt_str(quantitative.get("feedforward_feedback_loops")),
        study_design                   = _coerce_enum(study_meta.get("study_design"), STUDY_DESIGNS, "In vivo"),
        exposure_route                 = _opt_str(study_meta.get("exposure_route")),
        chemical_stressor              = _opt_str(study_meta.get("chemical_stressor")),
        extraction_confidence          = _coerce_enum(study_meta.get("extraction_confidence"), CONFIDENCE_VALUES, "Low"),
    )


# ---------------------------------------------------------------------------
# Public step functions — each one runs ONE LLM call against `cfg`
# ---------------------------------------------------------------------------

def list_ker_pairs(
    paper_text: str,
    cfg: LLMConfig,
    on_step: Optional[StepCallback] = None,
) -> StepResult:
    """Step 1 — return a StepResult whose `parsed` is `{'pairs': [...]}`."""
    return _run_step(
        "list_ker_pairs",
        _task_list_pairs(),
        cfg=cfg, on_step=on_step, num_predict=1024,
        cached_prefix=_build_cached_prefix(paper_text),
    )


def classify_ker(paper_text, upstream, downstream, cfg, idx, on_step=None) -> StepResult:
    return _run_step(
        f"classify_ker[{idx}]",
        _task_classify(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx, num_predict=512,
        cached_prefix=_build_cached_prefix(paper_text),
    )


def assess_evidence(paper_text, upstream, downstream, cfg, idx, on_step=None) -> StepResult:
    return _run_step(
        f"assess_evidence[{idx}]",
        _task_evidence(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx, num_predict=768,
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_applicability(paper_text, upstream, downstream, cfg, idx, on_step=None) -> StepResult:
    return _run_step(
        f"applicability[{idx}]",
        _task_applicability(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx, num_predict=256,
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_quantitative(paper_text, upstream, downstream, cfg, idx, on_step=None) -> StepResult:
    return _run_step(
        f"quantitative[{idx}]",
        _task_quantitative(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx, num_predict=512,
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_study_meta(paper_text, upstream, downstream, cfg, idx, on_step=None) -> StepResult:
    return _run_step(
        f"study_meta[{idx}]",
        _task_study_meta(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx, num_predict=256,
        cached_prefix=_build_cached_prefix(paper_text),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_kers_from_text(
    paper_text: str,
    cfg: Optional[LLMConfig] = None,
    *,
    model: str = "llama3.1:8b",
    ollama_url: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    max_kers: int = 20,
) -> tuple[list[KERExtraction], list[str]]:
    """
    Run the stepwise extraction pipeline and return (extractions, warnings).

    Pass an `LLMConfig` to use any provider (Ollama, Anthropic, OpenAI).
    The legacy `model` / `ollama_url` keyword arguments still work and build
    an Ollama config automatically.

    `on_step` is an optional callback invoked once for every LLM call with a
    fully populated StepResult, allowing the UI to display each step's prompt
    + raw response live.
    """
    if cfg is None:
        cfg = LLMConfig(
            provider="ollama",
            model=model,
            base_url=ollama_url or OLLAMA_URL,
        )

    warnings: list[str] = []
    extractions: list[KERExtraction] = []

    # Step 1 — list KER pairs
    step1 = list_ker_pairs(paper_text, cfg=cfg, on_step=on_step)

    if not step1.ok:
        warnings.append(
            "Step 1 (list_ker_pairs) failed to return valid JSON.\n"
            f"Error: {step1.error}\n"
            f"Raw (500 chars): {step1.raw_response[:500]}"
        )
        return extractions, warnings

    pairs_payload = step1.parsed or {}
    pairs = pairs_payload.get("pairs") if isinstance(pairs_payload, dict) else None
    if not isinstance(pairs, list) or not pairs:
        warnings.append(
            "Step 1 returned no KER pairs. Possible causes:\n"
            "• Paper truly lacks mechanistic content.\n"
            "• Model is too small — try llama3.1:70b or qwen2.5:14b.\n"
            f"Step 1 raw (500 chars): {step1.raw_response[:500]}"
        )
        return extractions, warnings

    pairs = pairs[:max_kers]

    # Steps 2-6 — per KER
    for i, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            warnings.append(f"KER {i}: pair is not a JSON object — skipped.")
            continue
        upstream   = (pair.get("upstream") or "").strip()
        downstream = (pair.get("downstream") or "").strip()
        if not upstream or not downstream:
            warnings.append(f"KER {i}: missing upstream/downstream name — skipped.")
            continue

        per_ker_steps = [
            ("classify",      classify_ker),
            ("evidence",      assess_evidence),
            ("applicability", extract_applicability),
            ("quantitative",  extract_quantitative),
            ("study_meta",    extract_study_meta),
        ]
        results: dict[str, dict] = {}

        for label, fn in per_ker_steps:
            step = fn(paper_text, upstream, downstream, cfg, i, on_step)
            if not step.ok or not isinstance(step.parsed, dict):
                warnings.append(
                    f"KER {i} ({upstream} → {downstream}): step '{label}' failed.\n"
                    f"Error: {step.error}\nRaw (300 chars): {step.raw_response[:300]}"
                )
                results[label] = {}
            else:
                results[label] = step.parsed

        try:
            extractions.append(_assemble(
                upstream      = upstream,
                downstream    = downstream,
                classify      = results["classify"],
                evidence      = results["evidence"],
                applicability = results["applicability"],
                quantitative  = results["quantitative"],
                study_meta    = results["study_meta"],
            ))
        except ExtractionValidationError as exc:
            warnings.append(f"KER {i} ({upstream} → {downstream}): assembly failed — {exc}")

    if not extractions and not warnings:
        warnings.append("No KERs assembled. See step results for details.")

    return extractions, warnings
