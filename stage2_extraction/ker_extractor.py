from __future__ import annotations

"""
Stepwise, quote-grounded KER extraction pipeline.

The work is broken into these steps:

    Step 1 — list_ker_pairs        : identify upstream/downstream pairs
    Step 2 — classify_ker          : levels, adjacency, name, description
    Step 3 — assess_evidence       : paper_type, plausibility, contradicts, ...
    Step 4 — applicability         : taxa, sex, life stage
    Step 5 — quantitative          : modulating factors, time scale, ...
    Step 6 — study_meta            : study_design, exposure_route, confidence, ...

Every step now also asks for **verbatim supporting quotations**. Each returned
quote is looked up in the source document with `pdf_reader.locate_quote`, which
resolves it to a page number, a section heading and a chunk id, and records
whether the text was actually found. The result is that no claim in the final
AOP is untraceable: a curator can always ask "where does this come from?" and
get an exact answer, or see plainly that the model could not support it.

A StepResult is captured for every call (prompt, raw response, parsed value,
error) so the caller can show exactly what happened at each step.

Public entry point:

    extract_kers_from_document(document, cfg, ...) -> (extractions, warnings)
    extract_kers_from_text(paper_text, cfg, ...)   -> (extractions, warnings)

Pass `on_step=lambda s: ...` to receive each StepResult as it completes.
"""

import json
import os
from collections import deque
import re
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

from schemas import (
    CONFIDENCE_VALUES,
    DIRECTION_VALUES,
    RELATION_KIND_VALUES,
    EVIDENCE_TYPE_VALUES,
    CAUSAL_EVIDENCE,
    EvidenceSpan,
    KE_LEVEL_ORDER,
    KER_ADJACENCY_VALUES,
    KERExtraction,
    PAPER_TYPE_VALUES,
    PaperDocument,
    SEX_VALUES,
    STUDY_DESIGN_VALUES,
)
from stage2_extraction import cell_lineage
import run_manifest
from json_repair import TRUNCATED_KEY, extract_json
from stage2_extraction.llm_providers import LLMAuthError, LLMConfig, LLMProviderError
from stage2_extraction.pdf_reader import locate_quote

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Enums (kept as sets for the coercion helpers)
# ---------------------------------------------------------------------------

KE_LEVELS = set(KE_LEVEL_ORDER)
KER_ADJACENCY = set(KER_ADJACENCY_VALUES)
PAPER_TYPES = set(PAPER_TYPE_VALUES)
STUDY_DESIGNS = set(STUDY_DESIGN_VALUES)
SEX_SET = set(SEX_VALUES)
CONFIDENCE_SET = set(CONFIDENCE_VALUES)
DIRECTIONS = set(DIRECTION_VALUES)
RELATION_KINDS = set(RELATION_KIND_VALUES)
EVIDENCE_TYPES = set(EVIDENCE_TYPE_VALUES)

#: Minimum characters for a quotation to be worth storing. Anything shorter
#: cannot be located unambiguously in the source.
MIN_QUOTE_CHARS = 25


# ---------------------------------------------------------------------------
# Errors + step result
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    """Raised when the pipeline cannot continue (e.g. the provider is unreachable)."""


class ExtractionValidationError(ValueError):
    """Raised when a single KER fails schema validation — others can still proceed."""


class ExtractionAuthError(ExtractionError):
    """
    Credentials were rejected.

    Never absorbed by the per-step error handling: a bad key fails identically
    on every subsequent call, so retrying just wastes the user's time and quota.
    """


@dataclass
class StepResult:
    """Outcome of one LLM call in the stepwise pipeline."""

    step: str                       # short id e.g. 'list_ker_pairs', 'classify_ker[1]'
    ok: bool                        # whether parsing succeeded
    prompt: str                     # the exact prompt sent to the provider
    raw_response: str               # the exact raw text returned
    parsed: Optional[Any] = None    # parsed dict/list, or None if parsing failed
    error: Optional[str] = None     # error message if ok=False
    ker_index: Optional[int] = None # which KER this step applies to (None for step 1)
    n_quotes: int = 0               # quotations returned by this step
    n_verified: int = 0             # quotations located in the source document
    truncated: bool = False         # response hit the token ceiling and was repaired

    #: The model that actually answered, set only when it was not the one
    #: configured for the run — i.e. when the primary refused and the fallback
    #: took over. A paper read partly by one model and partly by another is not
    #: a single population, and the row has to say so rather than inheriting the
    #: run's model by default.
    answered_by: Optional[str] = None

    #: How many provider calls this step cost, including refused ones. A
    #: refused paper used to report zero LLM calls, because the counter was
    #: only incremented once a call had returned — so the requests that were
    #: actually made, and paid for, were invisible in the run record.
    attempts: int = 1


StepCallback = Callable[[StepResult], None]


#: Temperatures for successive attempts at a step the classifier refused.
#:
#: `None` means "whatever the run is configured for" — the first attempt must
#: not be perturbed, because it is the one that produces almost every row and
#: its sampling settings are what the manifest promises. The retries climb,
#: because the only way to get a different verdict from an output-side
#: classifier is to generate a different output. Higher temperature costs some
#: JSON tidiness; that is worth paying on an attempt whose alternative is
#: losing the paper, and not worth paying anywhere else.
REFUSAL_RETRY_TEMPERATURES: tuple[Optional[float], ...] = (None, 0.6, 1.0)


# ---------------------------------------------------------------------------
# Refusal fallback
#
# A safety-classifier refusal is a property of one model. The tool has always
# said so and then told the curator to re-run the paper against a different
# provider by hand — which meant reconfiguring the sidebar, re-uploading, and
# remembering afterwards which paper came from which model. The advice was
# right and doing it manually was the whole cost.
#
# So the fallback is configured once and applied automatically. It is held at
# module level for the same reason `run_manifest` holds its recorder there:
# it is a property of the run rather than of any one call, and threading it
# through `extract_pathway_rows` → `extract_pathway` → `_run_step` and the
# seven stepwise call sites would add a parameter to every signature to carry
# something none of them decide.
# ---------------------------------------------------------------------------

#: Thread-local, and that is not a detail. An `LLMConfig` carries an
#: `api_key`. Held in a module global, one browser session's fallback key would
#: be picked up and billed by every other session in the process the moment
#: their primary model declined — a credential crossing between users, in a
#: codebase whose whole discipline is knowing which conditions produced which
#: row. Streamlit runs each session's script in its own thread, so per-thread
#: is per-session for anything set during a script run.
_LOCAL = threading.local()


def set_refusal_fallback(cfg: Optional[LLMConfig]) -> None:
    """Configure the model to try when the primary provider declines."""
    _LOCAL.refusal_fallback = cfg


def refusal_fallback() -> Optional[LLMConfig]:
    """The configured fallback, or None if refusals should simply fail."""
    return getattr(_LOCAL, "refusal_fallback", None)


def _is_distinct_model(primary: LLMConfig, fallback: Optional[LLMConfig]) -> bool:
    """
    Whether the fallback is actually a different model from the primary.

    Falling back to the same model is what the old retry did: identical bytes,
    same temperature, and on Anthropic the prefix is marked
    `cache_control: ephemeral` so it is deliberately the same cached input. A
    classifier verdict on identical input is not an independent second draw,
    so that retry cost a call and an input-token charge to learn nothing, then
    reported one verdict as two. A fallback that is not distinct is no fallback.
    """
    if fallback is None:
        return False
    return (
        (fallback.provider or "").lower(),
        (fallback.model or "").strip(),
    ) != (
        (primary.provider or "").lower(),
        (primary.model or "").strip(),
    )


# ---------------------------------------------------------------------------
# Low-level provider call + JSON parsing
# ---------------------------------------------------------------------------

def _call_llm(
    cfg: LLMConfig,
    prompt: str,
    num_predict: int,
    cached_prefix: Optional[str] = None,
) -> str:
    """Invoke the configured provider with a per-call output-token budget."""
    call_cfg = replace(cfg, max_output_tokens=num_predict)
    try:
        return call_cfg.generate(prompt, cached_prefix=cached_prefix)
    except LLMAuthError as exc:
        raise ExtractionAuthError(str(exc)) from exc
    except LLMProviderError as exc:
        raise ExtractionError(str(exc)) from exc


#: JSON parsing, including recovery of responses truncated by the token limit,
#: is shared with Stage 1 screening — both hit the same failure mode.
def _extract_json(raw: str) -> Any:
    """Parse the model's JSON reply, recovering truncated responses."""
    return extract_json(raw, context="extraction response")


def _report_refused_step(
    step_id: str,
    prompt: str,
    ker_index: Optional[int],
    attempts: int,
    on_step: Optional[StepCallback],
    message: str,
) -> None:
    """
    Log a step the classifier would not answer, then raise.

    The logging is the point. A refusal used to raise straight out of
    `_run_step` without ever building a `StepResult`, so the step never reached
    the callback, never entered the step log, and the paper was recorded as
    having cost **zero** model calls — while two requests had been sent and
    charged for. A step that failed is still a step that happened.
    """
    result = StepResult(
        step=step_id, ok=False, prompt=prompt, raw_response="",
        parsed=None, error=message, ker_index=ker_index, attempts=attempts,
    )
    if on_step is not None:
        try:
            on_step(result)
        except Exception:  # noqa: BLE001 - a broken UI callback is not an extraction failure
            pass
    raise ExtractionError(message)


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
    answered_by: Optional[str] = None
    attempts = 0
    refusal: Optional[ExtractionError] = None

    # --- Same model, different draw -------------------------------------
    #
    # `stop_reason: "refusal"` is decided on the model's *generated output*,
    # not on the prompt: streaming classifiers watch the reply as it is being
    # written. So it is not a verdict on the paper computed in advance — it is
    # an event that may or may not happen depending on what the model starts
    # writing, and generation is sampled. On Anthropic nothing pins that
    # sampling: `seed` is only sent to Ollama and OpenAI, and the temperature
    # is 0.1 rather than 0.
    #
    # Which is why the same paper is read on one run and refused on the next.
    # It is a biased coin, not a fixed verdict — and it is one call, so the
    # coin decides whether the paper enters the corpus at all.
    #
    # The old retry re-sent the identical request. Anthropic lists that under
    # common pitfalls: re-sending a refused request to the same model usually
    # earns another refusal. Usually, not always — so the answer is to make
    # each attempt a genuinely different draw by raising the temperature,
    # rather than either re-sending the same one or giving up after it.
    # Raising temperature does not break the prompt cache, which is keyed on
    # the prefix, so the retries stay cheap on input.
    for temperature in REFUSAL_RETRY_TEMPERATURES:
        call_cfg = cfg if temperature is None else replace(cfg, temperature=temperature)
        attempts += 1
        try:
            raw = _call_llm(
                call_cfg, prompt, num_predict=num_predict, cached_prefix=cached_prefix
            )
            refusal = None
            break
        except ExtractionError as exc:
            if "declined" not in str(exc).lower():
                raise
            refusal = exc
            run_manifest.record("refusal", step=step_id, attempt=attempts)
            if attempts > 1:
                # Counted as its own thing. It used to be recorded as
                # `provider_retry(param="refusal")`, which lands in the
                # manifest's dropped-parameter set — so a run that lost a paper
                # to a classifier reported that the provider had rejected a
                # request parameter named "refusal".
                run_manifest.record("refusal_retry", step=step_id, attempt=attempts)

    # --- Still refused: ask a different model ---------------------------
    if refusal is not None:
        fallback = refusal_fallback()
        if _is_distinct_model(cfg, fallback):
            attempts += 1
            try:
                raw = _call_llm(
                    fallback, prompt,
                    num_predict=num_predict, cached_prefix=cached_prefix,
                )
            except ExtractionError as fallback_exc:
                _report_refused_step(
                    step_id, prompt, ker_index, attempts, on_step,
                    f"Refused on the “{step_id}” step {attempts} times. "
                    f"{cfg.provider}/{cfg.model} declined "
                    f"{attempts - 1} time(s) across rising temperatures, and "
                    f"the fallback {fallback.provider}/{fallback.model} also "
                    f"declined: {fallback_exc}",
                )
            else:
                # It answered. The row it produces did not come from the run's
                # model, and every consumer that compares runs by model has to
                # be able to see that.
                answered_by = f"{fallback.provider}/{fallback.model}"
                run_manifest.record(
                    "refusal_fallback", step=step_id, model=answered_by
                )
                refusal = None

    if refusal is not None:
        _report_refused_step(
            step_id, prompt, ker_index, attempts, on_step,
            f"{refusal} Refused on the “{step_id}” step {attempts} time(s) in "
            f"a row, at temperatures "
            f"{', '.join(str(t if t is not None else cfg.temperature) for t in REFUSAL_RETRY_TEMPERATURES[:attempts])}. "
            f"That is unusual — the same paper often goes through on a later "
            f"run, because the refusal is decided on the reply as it is "
            f"written and the reply is sampled. Re-run this paper, or set a "
            f"different model under **If the provider declines** in the "
            f"sidebar; a second Claude model on the same key counts.",
        )

    run_manifest.record("llm_call", step=step_id)
    try:
        parsed = _extract_json(raw)
        truncated = isinstance(parsed, dict) and bool(parsed.pop(TRUNCATED_KEY, False))
        if truncated:
            run_manifest.record("truncated_step", step=step_id)
        result = StepResult(
            step=step_id, ok=True, prompt=prompt, raw_response=raw,
            parsed=parsed, ker_index=ker_index, truncated=truncated,
            answered_by=answered_by, attempts=attempts,
        )
    except Exception as exc:
        run_manifest.record("step_failure", step=step_id)
        result = StepResult(
            step=step_id, ok=False, prompt=prompt, raw_response=raw,
            parsed=None, error=str(exc), ker_index=ker_index,
            answered_by=answered_by, attempts=attempts,
        )
    if on_step is not None:
        try:
            on_step(result)
        except Exception:
            # Never let a UI callback break the pipeline.
            pass
    return result


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

_QUOTE_KEYS = ("supporting_quotes", "quotes", "evidence_quotes", "quote")


def _coerce_quotes(payload: Any) -> list[str]:
    """
    Pull quotation strings out of whatever shape the model returned.

    Models are inconsistent here: some return a list of strings, some a list of
    {"quote": ...} objects, some a single string. All three are accepted.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        for key in _QUOTE_KEYS:
            if key in payload:
                return _coerce_quotes(payload[key])
        return []
    if isinstance(payload, list):
        out: list[str] = []
        for item in payload:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("quote", "text", "sentence", "excerpt"):
                    if isinstance(item.get(key), str):
                        out.append(item[key])
                        break
        return out
    return []


def _clean_quote(quote: str) -> str:
    text = (quote or "").strip()
    text = text.strip('"“”‘’')
    text = re.sub(r"^\s*\[[^\]]{0,60}\]\s*", "", text)   # drop a leading [c007 | ...] header
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_evidence_spans(
    quotes: Sequence[str],
    field: str,
    document: Optional[PaperDocument],
    *,
    max_spans: int = 3,
) -> list[EvidenceSpan]:
    """
    Turn raw quotation strings into located, verified `EvidenceSpan` objects.

    Quotes that cannot be found in the document are still returned, flagged
    `verified=False`. That is deliberate: an unlocatable quote is a signal the
    curator needs to see, not something to hide.
    """
    spans: list[EvidenceSpan] = []
    seen: set[str] = set()

    for raw_quote in quotes:
        quote = _clean_quote(raw_quote)
        if len(quote) < MIN_QUOTE_CHARS:
            continue
        fingerprint = _dedup_key(quote)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        span = EvidenceSpan(quote=quote, field=field)
        if document is not None:
            located = locate_quote(quote, document)
            span.verified = bool(located["verified"])
            span.match_ratio = float(located["match_ratio"])
            span.section = located["section"]
            span.section_kind = located["section_kind"]
            span.page_start = located["page_start"]
            span.page_end = located["page_end"]
            span.chunk_id = located["chunk_id"]
            span.char_start = located["char_start"]
            span.char_end = located["char_end"]
            span.source_doi = document.doi
            span.source_filename = document.filename

        spans.append(span)
        if len(spans) >= max_spans:
            break

    return spans


# ---------------------------------------------------------------------------
# Step prompts
#
# Every call shares the same long prefix (persona + paper text). We send that
# prefix once via the provider's prompt-cache mechanism, and only the small
# variable "task" string in the user message.
# ---------------------------------------------------------------------------

_PERSONA = (
    # ---------------------------------------------------------------
    # What this task is, stated before anything else.
    #
    # The prompt used to open on the persona and go straight into the paper,
    # which meant the request arrived as a bare instruction to extract
    # mechanism of action and dose-response from thirty thousand characters
    # about a neurotoxin, with nothing saying where the text came from or what
    # the output was for. That framing is missing regardless of how any
    # provider responds to it: a reader who does not know the source is
    # peer-reviewed literature cannot judge how much to trust the text, and
    # the naming and quotation rules below only make sense once the output is
    # known to be an evidence table rather than prose.
    #
    # It is also the honest description of the work. Nothing here is written
    # to produce a particular reaction from a classifier — every sentence is
    # true of every run this tool performs, and it would belong here if
    # refusals had never happened.
    # ---------------------------------------------------------------
    "CONTEXT FOR THIS TASK\n"
    "The text below is a peer-reviewed, published scientific paper, supplied "
    "by a toxicologist who is building an Adverse Outcome Pathway. An AOP is "
    "the standard OECD framework for organising mechanistic evidence about "
    "how a chemical or other stressor causes harm, and it is used by "
    "regulators — ECHA, EFSA, the US EPA — to assess hazard and reduce animal "
    "testing.\n\n"
    "Your job is bibliographic, not advisory. You read what this paper "
    "reports and record it as structured rows in an evidence table, each tied "
    "to a verbatim quotation from the paper so a human curator can check it "
    "against the source. You are not designing an experiment, not "
    "recommending an exposure, and not producing instructions for anything. "
    "Every claim you output is already in the published literature and is "
    "attributed to the paper that made it.\n\n"
    "Papers in this field routinely concern potent neurotoxins, metals, "
    "pesticides and drugs of abuse, and describe how they damage tissue. "
    "That is the subject matter of regulatory toxicology, and characterising "
    "such mechanisms is how exposure limits get set and vulnerable "
    "populations get protected. Read such papers exactly as you would any "
    "other: report what the authors measured and concluded.\n\n"
    "You are a specialist in adverse outcome pathway (AOP) toxicology and the "
    "AOP-Wiki data model. A Key Event Relationship (KER) is any causal or "
    "mechanistic link the paper describes between an upstream biological event "
    "(e.g. receptor activation, oxidative stress, DNA damage) and a downstream "
    "event (e.g. apoptosis, inflammation, organ dysfunction, disease).\n\n"
    "The paper below may be presented as excerpts, each preceded by a header of "
    "the form [chunk-id | section | pages]. Those headers are metadata, not "
    "content — never quote them.\n\n"
    # ---------------------------------------------------------------
    # Naming rules. Without these the model reports study findings —
    # "Sox10 inactivation in Oln93 cells", "Scn2a deletion in
    # oligodendrocytes" — which are experimental manipulations in a named
    # test system, not Key Events. Every paper then contributes its own
    # untransferable vocabulary, no two papers agree, and the normalizer is
    # asked to merge labels that were never describing the same thing at the
    # same level of abstraction.
    # ---------------------------------------------------------------
    "HOW TO NAME A KEY EVENT\n"
    "A Key Event is a measurable change in biological state that generalises "
    "beyond the experiment that revealed it. Another laboratory, using a "
    "different model system, must be able to measure the same event.\n\n"
    "1. Name the biological state, never the manipulation used to produce it. "
    "A gene deletion, knockdown, knockout, transfection, silencing or "
    "overexpression is a STRESSOR, not a Key Event. If the paper deletes a "
    "gene, the Key Event is the resulting change — 'decreased Nav1.2 channel "
    "function', not 'Scn2a deletion in oligodendrocytes'.\n"
    "2. Never put the test system in the name. No cell lines (Oln93, SH-SY5Y), "
    "no species, strain, sex, age or maturity, no 'in vitro'/'in vivo', no "
    "'in mature X cells'. That information is captured separately in the "
    "study-design and applicability fields, and repeating it in the name makes "
    "the same event look like several.\n"
    "3. Exception — a molecular initiating event IS the direct interaction of "
    "a stressor with its molecular target, so 'blockade of voltage-gated "
    "sodium channels' or 'agonist binding to the aryl hydrocarbon receptor' "
    "are correct MIE names.\n"
    # ---------------------------------------------------------------
    # Rules 4 and 5 are the ones a real corpus broke. Rule 4 used to read
    # "use the form '<direction> <entity>' WHERE DIRECTION APPLIES", and that
    # escape hatch was taken every time: twelve rows came back named
    # "Voltage-gated sodium channels", a thing rather than a change, which is
    # not a Key Event and cannot be a node. Rule 5 is the other half of the
    # same failure — those twelve rows mixed patch-clamp current density,
    # Nav1.6 immunolabeling and RNA-seq transcript counts under one name, so
    # one node claimed function, protein and message at once and disagreed
    # with itself about direction.
    # ---------------------------------------------------------------
    "4. A Key Event is a CHANGE, so its name must say which way it went. Use "
    "'<direction> <entity or process>': 'decreased myelin gene expression', "
    "'increased reactive oxygen species'. A bare entity is NEVER a Key Event. "
    "'Voltage-gated sodium channels', 'synaptic input', 'mitochondria' name "
    "things, not events — write 'decreased voltage-gated sodium current' "
    "instead. If the paper does not say which way the quantity moved, that is "
    "not a Key Event and you should not report it as one.\n"
    "5. Say WHAT was measured, because function, protein and transcript are "
    "different Key Events even for the same molecule. 'Decreased sodium "
    "current density' (electrophysiology), 'decreased Nav1.6 protein at nodes' "
    "(immunolabeling) and 'decreased SCN2A transcript' (RNA-seq or qPCR) are "
    "three events, not one. Never merge them under a single name.\n"
    "6. Use ONE wording per event throughout your answer. If you have already "
    "called something 'decreased myelin gene expression', do not later call it "
    "'downregulation of myelin-related genes'.\n"
    "7. Do not create a separate Key Event for each gene, transcript or marker "
    "measured. Report the biological event the panel demonstrates — 'decreased "
    "myelin gene expression' — not one event per gene. This does not override "
    "rule 5: one event per panel, but still separate events for function, "
    "protein and transcript.\n\n"
    "Before returning any Key Event name, check it against rule 4: does the "
    "name state a direction of change? If it names only a thing, rewrite it.\n\n"
    "Whenever you are asked for a quotation you MUST copy the sentence "
    "word-for-word from the paper. Do not paraphrase, do not summarise, do not "
    "correct grammar, and do not invent text. If the paper contains no sentence "
    "supporting a claim, return an empty list rather than an approximation.\n\n"
    "Keep every quotation to ONE sentence of at most 40 words, and keep every "
    "other string field under 60 words. Your reply is truncated at a fixed token "
    "limit, so a long answer loses its final fields entirely — brevity keeps the "
    "whole record intact.\n"
)


def _build_cached_prefix(paper_text: str) -> str:
    """Return the static text shared by every step call for one paper."""
    return f"{_PERSONA}\nPAPER:\n{paper_text}"


def _task_list_pairs() -> str:
    return (
        "TASK: List every KER described or supported by the paper provided in "
        "the system context.\n"
        "Return ONLY JSON of the form:\n"
        '  {"pairs": [{"upstream": "<upstream KE name>", '
        '"downstream": "<downstream KE name>", '
        '"quote": "<one verbatim sentence from the paper stating this link>"}, ...]}\n'
        "Rules:\n"
        "- Aim for at least one pair if any mechanistic link is mentioned.\n"
        "- Name both events using the KEY EVENT NAMING rules in the system "
        "context: a generalisable biological state, no manipulations, no cell "
        "lines, no maturity or model-system qualifiers, one wording per event.\n"
        "- Every name must state a DIRECTION of change and distinguish "
        "function from protein from transcript. \"Voltage-gated sodium "
        "channels\" is not a Key Event name; \"decreased voltage-gated sodium "
        "current\" is. Reject your own bare-entity names before returning.\n"
        "- Before returning, re-read your list: if two pairs differ only in "
        "wording, merge them into one. Do not emit the same relationship twice "
        "under different phrasing.\n"
        "- The `quote` must be copied verbatim from the paper text.\n"
        '- Return {"pairs": []} ONLY if the paper has no mechanistic content '
        "(e.g. a pure exposure-assessment or analytical-method paper).\n"
        "JSON:"
    )


def _task_relevance_gate_agnostic(
    upstream: str,
    downstream: str,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> str:
    """
    Ask what a paper observed between two events, without proposing a direction.

    For the case where the user does not want to assume which way the
    relationship runs. Nothing is asserted for the paper to agree or disagree
    with: it is asked only whether the two events are causally linked and what
    happened to each. The sign of the coupling is then computed from the two
    observations, so a split in the literature shows up as a split rather than
    as papers "contradicting" a direction nobody committed to.
    """
    return (
        "TASK: Report what the paper in the system context observed about the "
        "relationship between two biological events. No direction is being "
        "proposed — do not judge the paper against one.\n"
        f"  Event A: {upstream}\n"
        f"  Event B: {downstream}\n"
        + _alias_block(upstream, downstream, upstream_aliases, downstream_aliases)
        + "\n"
        "Return ONLY JSON:\n"
        "  {\n"
        '    "linked": "direct|indirect|no_link|not_addressed",\n'
        '    "observed_upstream_change": "increased|decreased|not stated",\n'
        '    "observed_downstream_change": "increased|decreased|no change|not stated",\n'
        '    "intermediate_events": ["<event between A and B, if indirect>"],\n'
        '    "reason": "<at most 25 words>",\n'
        '    "supporting_quotes": ["<verbatim sentence, required unless not_addressed>"]\n'
        "  }\n"
        "Meanings:\n"
        "- direct: the paper presents evidence that A causally affects B.\n"
        "- indirect: A affects B through named intermediate events.\n"
        "- no_link: the paper perturbed A and found B did not change, or "
        "argues the two are not causally linked.\n"
        "- not_addressed: the paper does not examine this relationship.\n\n"
        "For the two change fields, report the direction the paper actually "
        "observed or manipulated — whichever way round the experiment ran. If "
        "the paper increased A and B rose, say increased and increased. If it "
        "removed A and B fell, say decreased and decreased. Both are ordinary "
        "results; neither is better or worse.\n\n"
        + _GATE_STANDARD_OF_PROOF.replace('"none"', '"not_addressed"')
        + "Every answer other than \"not_addressed\" needs at least one "
        "verbatim quotation from the paper. Copy it exactly.\n"
        "JSON:"
    )


def _alias_block(
    upstream: str,
    downstream: str,
    upstream_aliases: Optional[Sequence[str]],
    downstream_aliases: Optional[Sequence[str]],
) -> str:
    """
    Tell the model what else each event is called.

    A Key Event is named in AOP wording; papers are written in laboratory
    wording. "Altered voltage-gated sodium channel kinetics" is NaV1.6, VGSC,
    SCN8A or a TTX-sensitive current in the papers that actually measured it,
    and "oligodendrocyte differentiation" is OPC maturation, myelination or MBP
    expression. Matching on the label alone makes a paper that is entirely
    about the relationship look like a paper that never mentions it — and the
    model has no way to know the two vocabularies refer to the same event
    unless it is told.
    """
    up = [a for a in (upstream_aliases or []) if a and a.strip()]
    down = [a for a in (downstream_aliases or []) if a and a.strip()]
    if not up and not down:
        return ""

    lines = [
        "\nTHE SAME EVENT UNDER OTHER NAMES\n",
        "Papers rarely use the formal wording above. Treat any of the "
        "following as naming the same event, and search the text for them "
        "rather than for the formal label:\n",
    ]
    if up:
        lines.append(f"  Upstream ({upstream}): {', '.join(up)}\n")
    if down:
        lines.append(f"  Downstream ({downstream}): {', '.join(down)}\n")
    lines.append(
        "A quotation using one of these names counts as being about the event. "
        "Measuring a standard readout for an event — a marker gene, a current, "
        "a stain — counts as measuring that event.\n"
    )
    return "".join(lines)


#: What to say about the base rate. The corpus reaching this gate has already
#: been screened on title and abstract for bearing on the research question, so
#: telling the model that most papers are irrelevant is not calibration — it is
#: a false prior, and it costs recall on exactly the papers the screening was
#: meant to keep. What the gate still must refuse is a connection the paper
#: does not make, which is a different thing and is stated separately.
_GATE_STANDARD_OF_PROOF = (
    "STANDARD OF PROOF\n"
    "These papers were pre-screened as plausibly bearing on this "
    "relationship, so relevance is expected rather than unusual. Judge this "
    "paper on what it reports, not on a prior about how often papers are "
    "relevant.\n"
    "What still does not count: mentioning both events without connecting "
    "them; citing someone else's finding in the introduction or discussion "
    "without testing it; speculating about a link in the discussion. The "
    "paper's own data must bear on the relationship.\n"
    "The evidence does not have to sit in one sentence. If the paper "
    "establishes one event and the other in nearby text, figures or a results "
    "paragraph, that is evidence — quote the sentence that carries the most "
    "of it.\n"
)


def _task_relevance_gate(
    upstream: str,
    downstream: str,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> str:
    """
    Ask whether the paper bears on one specified relationship.

    Deliberately worded so that "this paper says nothing about it" is an
    ordinary answer rather than a failure. Asking a model "does this paper
    support X?" reliably produces support; the verdict list therefore puts
    `none` and `contradicts` on equal footing with `direct` and demands a
    verbatim quote for any positive verdict.

    The aliases matter as much as the wording. Without them the gate is asking
    whether the paper uses the AOP's vocabulary, not whether it studied the
    relationship.
    """
    return (
        "TASK: Decide whether the paper in the system context provides evidence "
        "about ONE specific relationship.\n"
        f"  Upstream event:   {upstream}\n"
        f"  Downstream event: {downstream}\n"
        + _alias_block(upstream, downstream, upstream_aliases, downstream_aliases)
        + "\n"
        "Return ONLY JSON:\n"
        "  {\n"
        '    "verdict": "direct|indirect|contradicts|none",\n'
        '    "observed_upstream_change": "increased|decreased|altered|not stated",\n'
        '    "observed_downstream_change": "increased|decreased|no change|not stated",\n'
        '    "intermediate_events": ["<event between upstream and downstream, if indirect>"],\n'
        '    "reason": "<at most 25 words>",\n'
        '    "supporting_quotes": ["<verbatim sentence, required unless verdict is none>"]\n'
        "  }\n"
        "Verdict meanings:\n"
        "- direct: the paper presents evidence that the upstream event causes, "
        "produces or leads to the downstream event.\n"
        "- indirect: the paper supports a chain from upstream to downstream "
        "through one or more intermediate events. Name them.\n"
        "- contradicts: the paper presents evidence that this causal link does "
        "NOT hold.\n"
        "- none: the paper does not address this relationship.\n\n"
        # -------------------------------------------------------------
        # The rule below is the one models get wrong, and getting it wrong
        # is expensive: it files the strongest evidence for a KER — the
        # gain-of-function and rescue experiments that establish
        # essentiality — as evidence against it.
        # -------------------------------------------------------------
        "PERTURBING IN THE OPPOSITE DIRECTION IS STILL SUPPORT\n"
        "A causal relationship can be demonstrated from either end. If the "
        "relationship is that losing X reduces Y, then a paper showing that "
        "adding or increasing X raises Y is demonstrating the SAME "
        "relationship, and the verdict is direct — not contradicts.\n"
        "Worked example. Proposed: 'reduced sodium-channel activity leads to "
        "decreased oligodendrocyte differentiation'. A paper reporting that "
        "sodium-channel expression ENHANCES differentiation supports this: it "
        "establishes that the channel drives differentiation, which is why "
        "losing it impairs differentiation. Verdict: direct.\n"
        "Rescue, knock-in, gain-of-function and dose-response experiments all "
        "fall under this rule.\n\n"
        "Use \"contradicts\" ONLY when the paper genuinely disagrees:\n"
        "- it perturbs the upstream event and the downstream event does not "
        "change; or\n"
        "- it reports the downstream effect moving the way the proposed "
        "relationship says it should NOT, for the same direction of "
        "perturbation; or\n"
        "- it argues explicitly that the two events are not causally linked.\n"
        "Record what the paper actually observed in "
        "`observed_upstream_change` and `observed_downstream_change` so a "
        "curator can check this judgement.\n\n"
        + _GATE_STANDARD_OF_PROOF
        + "Every verdict other than \"none\" needs at least one verbatim "
        "quotation from the paper. Copy it exactly, including its numbers.\n"
        "JSON:"
    )


def _task_classify(upstream: str, downstream: str) -> str:
    return (
        f"For the KER below, classify the events using the paper in the system context.\n"
        f"  Upstream KE:   {upstream}\n"
        f"  Downstream KE: {downstream}\n\n"
        "Return ONLY JSON with these keys (no extra text):\n"
        "  {\n"
        '    "upstream_ke_level":   "MIE|Molecular|Cellular|Tissue|Organ|Individual|Population",\n'
        '    "downstream_ke_level": "same enum",\n'
        '    "ker_adjacency":       "Adjacent|Non-adjacent",\n'
        '    "ker_name":            "<upstream> leads to <downstream>",\n'
        '    "ker_description":     "1-3 sentences on the mechanistic basis, grounded in the paper",\n'
        '    "supporting_quotes":   ["<verbatim sentence establishing this mechanism>"]\n'
        "  }\n"
        "Mark the KER `Adjacent` only if no intermediate key event lies between "
        "the two; otherwise `Non-adjacent`.\n"
        "JSON:"
    )


def _task_evidence(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"summarise the evidence in the paper provided in the system context.\n\n"
        "Return ONLY JSON with these keys:\n"
        "  {\n"
        '    "paper_type":                 "Primary study|Review / meta-analysis|In silico",\n'
        '    "cited_evidence_dois":        "semicolon-separated DOIs from references, or null",\n'
        '    "biological_plausibility":    "short string or null",\n'
        '    "empirical_evidence_summary": "key data points / measurements supporting the link, or null",\n'
        '    "essentiality_evidence":      "knockout / antagonist / blocker evidence, or null",\n'
        '    "contradicts_ker":            true_or_false,  // true if the paper argues AGAINST the KER\n'
        '    "supporting_quotes":          ["<up to 3 verbatim sentences reporting the evidence>"]\n'
        "  }\n"
        "The quotes should be the sentences that report the actual measurements "
        "or experimental results, not the introduction or the conclusions.\n"
        "JSON:"
    )


def _task_applicability(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"describe applicability based on what the paper in the system context studied.\n\n"
        "Return ONLY JSON with these keys:\n"
        "  {\n"
        '    "taxonomic_applicability":   "NCBI species name(s) e.g. \\"Mus musculus\\"; or \\"Not specified\\"",\n'
        '    "sex_applicability":         "Male|Female|Mixed|Not specified",\n'
        '    "life_stage_applicability":  "e.g. Adult, Embryo, Juvenile, or Not specified",\n'
        '    "supporting_quotes":         ["<verbatim sentence naming the test system>"]\n'
        "  }\n"
        "JSON:"
    )


def _task_quantitative(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"extract any quantitative or temporal information from the paper in the system context.\n\n"
        "Return ONLY JSON with these keys (use null if the paper does not say):\n"
        "  {\n"
        '    "modulating_factors":             "string or null",\n'
        '    "quantitative_relationships":     "string or null",\n'
        '    "response_response_relationship": "string or null",\n'
        '    "time_scale":                     "string or null",\n'
        '    "feedforward_feedback_loops":     "string or null",\n'
        '    "supporting_quotes":              ["<verbatim sentence carrying a dose, time or effect size>"]\n'
        "  }\n"
        "JSON:"
    )


def _task_study_meta(upstream: str, downstream: str) -> str:
    return (
        f"For this KER (Upstream: {upstream}; Downstream: {downstream}) "
        f"describe the study design and your confidence in the extraction, "
        f"using the paper in the system context.\n\n"
        "Return ONLY JSON with these keys:\n"
        "  {\n"
        '    "study_design":          "In vivo|In vitro|In silico|Ex vivo|Epidemiological|Review / meta-analysis",\n'
        '    "study_context":         "<the biological situation studied>",\n'
        '    "exposure_route":        "e.g. Oral gavage, IP, Inhalation; or null",\n'
        '    "chemical_stressor":     "chemical(s) tested, or null",\n'
        '    "extraction_confidence": "High|Medium|Low"\n'
        "  }\n"
        "\"study_context\" is the situation the findings belong to — \"normal "
        "postnatal development\", \"spinal cord injury and remyelination\", "
        "\"conditional knockout\", \"demyelinating lesion\", \"cell culture\". "
        "Be specific and short. Findings from an injury model and from normal "
        "development describe different biology even when they use identical "
        "words for the events, and chaining one onto the other builds a "
        "pathway that neither paper reports.\n"
        "Report `Low` confidence when the link is inferred rather than measured.\n"
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
    evidence_spans: Optional[list[EvidenceSpan]] = None,
    context: Optional[dict] = None,
) -> KERExtraction:
    context = context or {}
    return KERExtraction(
        upstream_ke_name               = _req_str(upstream, "upstream_ke_name"),
        upstream_ke_level              = _coerce_enum(classify.get("upstream_ke_level"), KE_LEVELS, "Molecular"),
        downstream_ke_name             = _req_str(downstream, "downstream_ke_name"),
        downstream_ke_level            = _coerce_enum(classify.get("downstream_ke_level"), KE_LEVELS, "Molecular"),
        ker_name                       = _req_str(classify.get("ker_name") or f"{upstream} leads to {downstream}", "ker_name"),
        ker_description                = _req_str(
            classify.get("ker_description") or f"Relationship between {upstream} and {downstream}.",
            "ker_description",
        ),
        ker_adjacency                  = _coerce_enum(classify.get("ker_adjacency"), KER_ADJACENCY, "Adjacent"),
        paper_type                     = _coerce_enum(evidence.get("paper_type"), PAPER_TYPES, "Primary study"),
        cited_evidence_dois            = _opt_str(evidence.get("cited_evidence_dois")),
        biological_plausibility        = _opt_str(evidence.get("biological_plausibility")),
        empirical_evidence_summary     = _opt_str(evidence.get("empirical_evidence_summary")),
        essentiality_evidence          = _opt_str(evidence.get("essentiality_evidence")),
        contradicts_ker                = _req_bool(evidence.get("contradicts_ker")),
        taxonomic_applicability        = applicability.get("taxonomic_applicability") or "Not specified",
        sex_applicability              = _coerce_enum(applicability.get("sex_applicability"), SEX_SET, "Not specified"),
        life_stage_applicability       = applicability.get("life_stage_applicability") or "Not specified",
        modulating_factors             = _opt_str(quantitative.get("modulating_factors")),
        quantitative_relationships     = _opt_str(quantitative.get("quantitative_relationships")),
        response_response_relationship = _opt_str(quantitative.get("response_response_relationship")),
        time_scale                     = _opt_str(quantitative.get("time_scale")),
        feedforward_feedback_loops     = _opt_str(quantitative.get("feedforward_feedback_loops")),
        study_design                   = _coerce_enum(study_meta.get("study_design"), STUDY_DESIGNS, "In vivo"),
        exposure_route                 = _opt_str(study_meta.get("exposure_route")),
        chemical_stressor              = _opt_str(study_meta.get("chemical_stressor")),
        extraction_confidence          = _coerce_enum(study_meta.get("extraction_confidence"), CONFIDENCE_SET, "Low"),
        evidence_spans                 = evidence_spans or [],
        # Sign and cell type travel with the row from here on. They were being
        # extracted and then dropped, which is how a knockdown result and an
        # overexpression result ended up indistinguishable on one edge.
        direction                      = _coerce_enum(context.get("direction"), DIRECTIONS, "unclear"),
        upstream_change                = _opt_str(context.get("upstream_change")),
        downstream_change              = _opt_str(context.get("downstream_change")),
        upstream_cell_type             = _opt_str(context.get("upstream_cell_type")),
        downstream_cell_type           = _opt_str(context.get("downstream_cell_type")),
        relation_kind                  = _coerce_enum(context.get("relation_kind"), RELATION_KINDS, "causal"),
        evidence_type                  = _coerce_enum(context.get("evidence_type"), EVIDENCE_TYPES, "not_stated"),
        measured_as                    = _opt_str(context.get("measured_as")),
        upstream_target                = _opt_str(context.get("upstream_target")),
        downstream_target              = _opt_str(context.get("downstream_target")),
        null_findings                  = _opt_str(context.get("null_findings")),
        study_context                  = _opt_str(context.get("study_context")),
    )


# ---------------------------------------------------------------------------
# Public step functions — each one runs ONE LLM call against `cfg`
# ---------------------------------------------------------------------------

#: Per-step output ceilings, sized to the JSON each step is asked for plus
#: comfortable headroom. These are ceilings, not reservations: you are billed
#: for the tokens actually generated, so raising them costs nothing on a model
#: that answers directly. It costs real money only on a model that reasons
#: internally, because those tokens are generated and billed even though the
#: pipeline discards them.
#:
#: The values below were doubled from their original sizes after truncation
#: was observed on papers reporting many quantitative relationships — the
#: fields at the end of a step's JSON were being cut off, which reads as
#: "the paper didn't say" rather than "the reply ran out".
_STEP_BUDGETS: dict[str, int] = {
    "ke_synonyms":     600,
    "relevance_gate": 800,
    #: A chain of several steps, each with a description and a verbatim quote,
    #: is the largest JSON any step produces. Sized so a five-step pathway is
    #: not truncated into a three-step one — which would read as the paper
    #: having less to say rather than the reply having run out.
    #:
    #: Raised from 6000 after two papers in a thirteen-paper run came back
    #: completely empty: on a reasoning model the ceiling covers the thinking
    #: AND the answer, and the thinking now has more to do — cell type,
    #: isoform, evidence type and null findings per link. The model spent the
    #: whole budget reasoning and had nothing left to write with, which the
    #: pipeline sees as "this paper says nothing about the relationship".
    #: A silent false negative is the worst failure this tool has, and 16000
    #: costs nothing on a model that answers directly: you are billed for
    #: tokens generated, not for the ceiling.
    "pathway":      16000,
    "list_ker_pairs": 5000,
    "classify":       2500,
    "evidence":       5000,
    "applicability":  2500,
    "quantitative":   4000,
    "study_meta":     1200,
}


def step_budget(step: str, scale: float = 1.0) -> int:
    """
    Output ceiling for `step`, multiplied by `scale`.

    `scale` exists for models that reason before answering: the visible JSON
    needs the same room as always, but the thinking has to fit in the same
    budget. Raising it is the alternative to switching model, not a quality
    setting — a direct-answering model produces the same output at scale 1.0.
    """
    base = _STEP_BUDGETS.get(step, 1024)
    return max(256, int(round(base * max(0.1, float(scale)))))


def screen_for_ker(
    paper_text: str,
    upstream: str,
    downstream: str,
    cfg: LLMConfig,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
    directional: bool = True,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> StepResult:
    """One call deciding whether a paper bears on a specified relationship."""
    task = (
        _task_relevance_gate(
            upstream, downstream, upstream_aliases, downstream_aliases
        )
        if directional
        else _task_relevance_gate_agnostic(
            upstream, downstream, upstream_aliases, downstream_aliases
        )
    )
    return _run_step(
        "relevance_gate",
        task,
        cfg=cfg, on_step=on_step,
        num_predict=step_budget("relevance_gate", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def list_ker_pairs(
    paper_text: str,
    cfg: LLMConfig,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
) -> StepResult:
    """Step 1 — return a StepResult whose `parsed` is `{'pairs': [...]}`."""
    return _run_step(
        "list_ker_pairs",
        _task_list_pairs(),
        cfg=cfg, on_step=on_step,
        num_predict=step_budget("list_ker_pairs", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def classify_ker(paper_text, upstream, downstream, cfg, idx, on_step=None, budget_scale=1.0) -> StepResult:
    return _run_step(
        f"classify_ker[{idx}]",
        _task_classify(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx,
        num_predict=step_budget("classify", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def assess_evidence(paper_text, upstream, downstream, cfg, idx, on_step=None, budget_scale=1.0) -> StepResult:
    return _run_step(
        f"assess_evidence[{idx}]",
        _task_evidence(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx,
        num_predict=step_budget("evidence", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_applicability(paper_text, upstream, downstream, cfg, idx, on_step=None, budget_scale=1.0) -> StepResult:
    return _run_step(
        f"applicability[{idx}]",
        _task_applicability(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx,
        num_predict=step_budget("applicability", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_quantitative(paper_text, upstream, downstream, cfg, idx, on_step=None, budget_scale=1.0) -> StepResult:
    return _run_step(
        f"quantitative[{idx}]",
        _task_quantitative(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx,
        num_predict=step_budget("quantitative", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


def extract_study_meta(paper_text, upstream, downstream, cfg, idx, on_step=None, budget_scale=1.0) -> StepResult:
    return _run_step(
        f"study_meta[{idx}]",
        _task_study_meta(upstream, downstream),
        cfg=cfg, on_step=on_step, ker_index=idx,
        num_predict=step_budget("study_meta", budget_scale),
        cached_prefix=_build_cached_prefix(paper_text),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

#: How many model calls may fail back-to-back before we stop trying. One flaky
#: call is worth absorbing; ten in a row means something systemic (bad key,
#: wrong model name, service outage) and continuing just wastes time.
_MAX_CONSECUTIVE_PROVIDER_ERRORS = 4

#: Which KER field each step's quotations are taken to support.
_STEP_EVIDENCE_FIELD = {
    "pair": "ker_link",
    "classify": "ker_description",
    "evidence": "empirical_evidence_summary",
    "applicability": "taxonomic_applicability",
    "quantitative": "quantitative_relationships",
}


def extract_kers_from_document(
    document: PaperDocument,
    cfg: Optional[LLMConfig] = None,
    *,
    paper_text: Optional[str] = None,
    model: str = "llama3.1:8b",
    ollama_url: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    max_kers: int = 20,
    budget_scale: float = 1.0,
) -> tuple[list[KERExtraction], list[str]]:
    """
    Run the stepwise extraction pipeline against a `PaperDocument`.

    `paper_text` is the text actually sent to the model — normally the output
    of `chunk_scorer.prepare_paper_text()`, i.e. only the mechanistically
    relevant chunks. Quotations are located against the FULL document, so a
    quote is still resolvable even if the model recalled it from the abstract.

    Returns (extractions, warnings).
    """
    if cfg is None:
        cfg = LLMConfig(provider="ollama", model=model, base_url=ollama_url or OLLAMA_URL)

    text_for_model = paper_text if paper_text is not None else document.full_text

    warnings: list[str] = []
    extractions: list[KERExtraction] = []
    consecutive_provider_errors = 0

    run_manifest.record("paper_attempted", doi=document.doi)

    # --- Step 1 — list KER pairs ------------------------------------------
    step1 = list_ker_pairs(
        text_for_model, cfg=cfg, on_step=on_step, budget_scale=budget_scale
    )

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
            "• Chunk selection was too aggressive — lower the relevance "
            "threshold or raise the character budget.\n"
            "• Model is too small — try llama3.1:70b or qwen2.5:14b.\n"
            f"Step 1 raw (500 chars): {step1.raw_response[:500]}"
        )
        return extractions, warnings

    pairs = pairs[:max_kers]

    # --- Steps 2-6 — per KER ----------------------------------------------
    for i, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            warnings.append(f"KER {i}: pair is not a JSON object — skipped.")
            continue
        upstream = (pair.get("upstream") or "").strip()
        downstream = (pair.get("downstream") or "").strip()
        if not upstream or not downstream:
            warnings.append(f"KER {i}: missing upstream/downstream name — skipped.")
            continue

        for problem in name_problems(upstream, downstream):
            warnings.append(f"KER {i}: {problem}")

        spans: list[EvidenceSpan] = build_evidence_spans(
            _coerce_quotes(pair.get("quote")),
            _STEP_EVIDENCE_FIELD["pair"],
            document,
            max_spans=1,
        )

        per_ker_steps = [
            ("classify",      classify_ker),
            ("evidence",      assess_evidence),
            ("applicability", extract_applicability),
            ("quantitative",  extract_quantitative),
            ("study_meta",    extract_study_meta),
        ]
        results: dict[str, dict] = {}

        for label, fn in per_ker_steps:
            # A provider error on one step should cost that step, not every KER
            # extracted so far. Persistent failures (auth, wrong model name)
            # still abort, via the consecutive-error breaker below.
            try:
                step = fn(
                    text_for_model, upstream, downstream, cfg, i, on_step,
                    budget_scale=budget_scale,
                )
                consecutive_provider_errors = 0
            except ExtractionAuthError:
                raise
            except ExtractionError as exc:
                consecutive_provider_errors += 1
                warnings.append(
                    f"KER {i} ({upstream} → {downstream}): step '{label}' could not "
                    f"reach the model — {exc}"
                )
                results[label] = {}
                if consecutive_provider_errors >= _MAX_CONSECUTIVE_PROVIDER_ERRORS:
                    warnings.append(
                        f"Aborting: {consecutive_provider_errors} model calls failed "
                        f"in a row. {len(extractions)} KER(s) extracted before the "
                        "failure are returned and can still be saved."
                    )
                    return extractions, warnings
                continue

            if not step.ok or not isinstance(step.parsed, dict):
                warnings.append(
                    f"KER {i} ({upstream} → {downstream}): step '{label}' failed.\n"
                    f"Error: {step.error}\nRaw (300 chars): {step.raw_response[:300]}"
                )
                results[label] = {}
                continue

            results[label] = step.parsed

            if step.truncated:
                warnings.append(
                    f"KER {i} ({upstream} → {downstream}): step '{label}' hit the "
                    "output-token limit. The fields returned before the cut were "
                    "kept; any later fields are missing."
                )

            field = _STEP_EVIDENCE_FIELD.get(label)
            if field:
                step_spans = build_evidence_spans(
                    _coerce_quotes(step.parsed.get("supporting_quotes")),
                    field,
                    document,
                    max_spans=3,
                )
                spans.extend(step_spans)
                step.n_quotes = len(step_spans)
                step.n_verified = sum(1 for s in step_spans if s.verified)

        spans = _deduplicate_spans(spans)

        try:
            extractions.append(
                _assemble(
                    upstream      = upstream,
                    downstream    = downstream,
                    classify      = results.get("classify", {}),
                    evidence      = results.get("evidence", {}),
                    applicability = results.get("applicability", {}),
                    quantitative  = results.get("quantitative", {}),
                    study_meta    = results.get("study_meta", {}),
                    evidence_spans= spans,
                )
            )
        except ExtractionValidationError as exc:
            warnings.append(f"KER {i} ({upstream} → {downstream}): assembly failed — {exc}")

    # --- Provenance quality warning ---------------------------------------
    if extractions:
        run_manifest.record("paper_extracted", n_kers=len(extractions))
        total_spans = sum(len(e.evidence_spans) for e in extractions)
        verified = sum(e.n_verified_spans for e in extractions)
        if total_spans == 0:
            warnings.append(
                "No supporting quotations were returned, so these KERs have no "
                "provenance. Consider a larger model — quote extraction is "
                "noticeably harder for small local models."
            )
        elif verified / total_spans < 0.4:
            warnings.append(
                f"Only {verified} of {total_spans} quotations could be located "
                "verbatim in the paper. The rest are likely paraphrases and are "
                "flagged unverified in the evidence panel."
            )

    if not extractions and not warnings:
        warnings.append("No KERs assembled. See step results for details.")

    return extractions, warnings


# ---------------------------------------------------------------------------
# Targeted extraction
#
# Open extraction asks "what relationships does this paper contain?", and the
# model invents both event names every time. Across a corpus that produces one
# label per paper per event, which is why consolidation is hard: the pipeline
# spends its effort trying to recognise afterwards that twelve phrasings meant
# one thing.
#
# When the user already knows which relationship they care about, that whole
# problem is avoidable. Fixing both labels up front means every paper's row
# uses identical wording by construction — nothing to normalise, nothing to
# merge — and the run only pays for papers that actually bear on the question.
# ---------------------------------------------------------------------------

_GATE_VERDICTS = {"direct", "indirect", "contradicts", "none"}

#: Changes that count as a real movement in one direction.
_UP_CHANGES = {"increased"}
_DOWN_CHANGES = {"decreased"}


def derive_coupling(upstream_change: str, downstream_change: str) -> str:
    """
    Sign of the coupling between two events, from what a paper observed.

    Derived in code rather than asked of the model, because it is arithmetic
    and the model is bad at it: "increased X, increased Y" and "decreased X,
    decreased Y" are the same finding, and asking for a verdict on whether
    that agrees with a stated direction is where misfiling happens.

    Returns 'positive' (they move together), 'negative' (they move oppositely),
    'none' (upstream moved, downstream did not) or 'unclear'.
    """
    up = (upstream_change or "").strip().lower()
    down = (downstream_change or "").strip().lower()

    if up in _UP_CHANGES | _DOWN_CHANGES and down == "no change":
        return "none"
    if up not in _UP_CHANGES | _DOWN_CHANGES:
        return "unclear"
    if down not in _UP_CHANGES | _DOWN_CHANGES:
        return "unclear"

    same = (up in _UP_CHANGES) == (down in _UP_CHANGES)
    return "positive" if same else "negative"


@dataclass
class GateResult:
    """Outcome of the relevance gate for one paper."""

    verdict: str = "none"               # direct | indirect | contradicts | none
    reason: str = ""
    intermediate_events: list[str] = field(default_factory=list)
    spans: list[EvidenceSpan] = field(default_factory=list)
    error: Optional[str] = None

    #: Set when the model returned a positive verdict that this code overruled.
    #: A demoted paper and a paper the model never thought relevant both end up
    #: as "none", and they are not the same finding — one means the literature
    #: is silent, the other means the reply was unusable. Keeping the original
    #: verdict here lets the UI tell them apart.
    downgraded_from: Optional[str] = None

    #: What the paper actually observed, recorded separately from the verdict.
    #: A verdict is a judgement and can be wrong; these are closer to the
    #: reported result, and they let a curator see WHY a paper was filed as
    #: supporting or contradicting instead of having to take it on trust.
    observed_upstream_change: str = "not stated"
    observed_downstream_change: str = "not stated"

    #: 'positive' (events move together), 'negative' (oppositely), 'none'
    #: (upstream moved, downstream did not) or 'unclear'. Derived from the two
    #: observations above, not asked of the model.
    coupling: str = "unclear"

    @property
    def is_relevant(self) -> bool:
        """Whether the full evidence steps are worth running for this paper."""
        return self.verdict in ("direct", "indirect", "contradicts")


def screen_document_for_ker(
    document: PaperDocument,
    upstream: str,
    downstream: str,
    cfg: LLMConfig,
    *,
    paper_text: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
    directional: bool = True,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> GateResult:
    """
    Run the relevance gate against one document.

    `directional=False` asks what the paper observed without proposing a
    direction, for exploratory questions where the user does not yet want to
    commit to which way the relationship runs.
    """
    text_for_model = paper_text if paper_text is not None else document.full_text

    try:
        step = screen_for_ker(
            text_for_model, upstream, downstream, cfg, on_step, budget_scale,
            directional=directional,
            upstream_aliases=upstream_aliases,
            downstream_aliases=downstream_aliases,
        )
    except ExtractionAuthError:
        raise
    except ExtractionError as exc:
        return GateResult(verdict="none", error=str(exc))

    if not step.ok or not isinstance(step.parsed, dict):
        return GateResult(
            verdict="none",
            error=f"Gate reply could not be parsed: {step.error}",
        )

    parsed = step.parsed
    if directional:
        verdict = str(parsed.get("verdict", "none")).strip().lower()
    else:
        # The agnostic prompt reports linkage, not agreement. Map it onto the
        # same vocabulary so everything downstream — storage, Table 2, the UI —
        # stays identical between the two modes.
        verdict = {
            "direct": "direct",
            "indirect": "indirect",
            "no_link": "contradicts",
            "not_addressed": "none",
        }.get(str(parsed.get("linked", "not_addressed")).strip().lower(), "none")
    if verdict not in _GATE_VERDICTS:
        verdict = "none"

    spans = build_evidence_spans(
        _coerce_quotes(parsed.get("supporting_quotes")),
        "ker_link",
        document,
        max_spans=2,
    )

    # A positive verdict with no locatable quotation is the exact shape of a
    # strained connection, and this gate decides whether the expensive steps
    # run at all. Demote it rather than let it through.
    if verdict in ("direct", "indirect") and not spans:
        return GateResult(
            verdict="none",
            reason=str(parsed.get("reason") or "")[:300],
            downgraded_from=verdict,
            error=(
                f"The model called this '{verdict}' but supplied no usable "
                "quotation, so it was demoted to 'none'. Either the paper does "
                "not say it in so many words, or the quotation fell outside the "
                "text that was sent — check the chunk selection above."
            ),
        )

    intermediates = [
        str(e).strip()
        for e in (parsed.get("intermediate_events") or [])
        if isinstance(e, str) and e.strip()
    ]

    step.n_quotes = len(spans)
    step.n_verified = sum(1 for s in spans if s.verified)

    up_change = str(parsed.get("observed_upstream_change") or "not stated").strip().lower()
    down_change = str(parsed.get("observed_downstream_change") or "not stated").strip().lower()

    # A "contradicts" verdict where the two observations move together is the
    # signature of the inverse-perturbation mistake: the paper drove the
    # upstream event one way and saw the downstream event follow, which is the
    # relationship holding, not failing. Flag it rather than silently
    # overriding — the model saw the text and this check has not.
    # Only meaningful when a direction was proposed. In agnostic mode nothing
    # was asserted, so there is no direction to be inconsistent with.
    inverse_confusion = (
        directional
        and verdict == "contradicts"
        and {up_change, down_change} <= {"increased", "decreased"}
        and up_change == down_change
    )

    return GateResult(
        verdict=verdict,
        reason=str(parsed.get("reason") or "")[:300],
        intermediate_events=intermediates[:5],
        spans=spans,
        observed_upstream_change=up_change,
        observed_downstream_change=down_change,
        coupling=derive_coupling(up_change, down_change),
        error=(
            "Filed as contradicting, but the paper reports both events moving "
            f"the same way ({up_change} upstream, {down_change} downstream), "
            "which is usually the relationship holding. Check this one by hand."
            if inverse_confusion else None
        ),
    )


def name_problems(*names: Optional[str]) -> list[str]:
    """
    Key Event names the extractor returned that are not Key Events.

    The prompt now forbids bare entities, and prompts are not a guarantee: a
    rule the model complies with on fifteen papers is a rule it can drop on
    the sixteenth, and the failure is silent. "Voltage-gated sodium channels"
    reads like a Key Event, sits on the map like one, and is a noun phrase
    naming a thing — so twelve rows measuring current density, protein at
    nodes and transcript counts all landed on one node that then disagreed
    with itself about direction.

    `semantic_merge.object_type` already draws this distinction and was only
    ever consulted during curation, which is after the run has been paid for.
    Consulting it here turns a defect nobody could see into a warning attached
    to the paper that produced it.

    Returns one message per problem name. Deliberately a warning rather than a
    rejection: the extraction is still evidence, and discarding a paper's
    findings over its phrasing would cost more than it saves.
    """
    from stage2_extraction.semantic_merge import ObjectType, object_type

    problems: list[str] = []
    for name in names:
        label = str(name or "").strip()
        if not label:
            continue
        kind = object_type(label)
        if kind is ObjectType.ENTITY:
            # No auto-suggested rewrite. Prefixing "decreased" to an arbitrary
            # label produces "decreased shortened internodes" as readily as it
            # produces something sensible, and a wrong suggestion next to a
            # correct diagnosis discredits the diagnosis.
            problems.append(
                f"“{label}” names a thing rather than a change, so it is not a "
                f"Key Event. A name that does not state a direction cannot "
                f"distinguish the papers reporting this quantity rising from "
                f"those reporting it falling, nor function from protein from "
                f"transcript — all of them will land on this one node. Rename "
                f"it in Normalize & curate before approving it."
            )
        elif kind is ObjectType.OBSERVATION:
            problems.append(
                f"“{label}” states that a measurement did not change. That is "
                f"a study observation about the evidence, not a Key Event."
            )
    return problems


def _task_pathway(
    upstream: str,
    downstream: str,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
    directional: bool = True,
) -> str:
    """
    Ask for the causal chain a paper supports, not a verdict on one link.

    The difference matters for what comes out the other end. Asking "does this
    paper support A → B?" can only ever produce one edge, and every paper
    produces the same one, so sixteen papers make a two-node graph. Papers do
    not work that way: one reports that the channel carries calcium, another
    that calcium drives the transcription factor, a third that the factor is
    needed for the cell to mature. Each is a different link in the same chain,
    and the chain is the thing being reconstructed.

    So the model is asked to lay out the steps THIS paper evidences, with the
    query's two events as the anchors. Steps that other papers also report
    become shared nodes, and the graph assembles itself.

    The anchoring rule is deliberately narrower than it first was. Anchoring
    everything that resembles the question's wording does make the graph
    connect, but connection is not the goal — a *correct* graph is, and the
    two are in tension. A corpus on sodium channels in the brain contains
    oligodendroglial Nav1.2 and microglial Na+ current, which are different
    events in different cells with opposite consequences; collapsing both onto
    "voltage-gated sodium channel" because both resemble the question produced
    a single pathway running from microglial inflammation into oligodendrocyte
    maturation that no paper supports. So an event is only anchored when the
    cell type agrees, and the cell type is asked for explicitly.
    """
    anchor_rule = (
        "ANCHORING — this is what lets separate papers join into one pathway.\n"
        f"When a step's event IS the upstream event of the question, AND it is "
        f"observed in the same cell type, write it EXACTLY as: {upstream}\n"
        f"When a step's event IS the downstream event of the question, AND it "
        f"is observed in the same cell type, write it EXACTLY as: {downstream}\n"
        "If the event resembles one of those two but was observed in a "
        "DIFFERENT cell type, do NOT use the anchor wording. Name it with its "
        "cell type — \"sodium current in activated microglia\" is not the same "
        "event as \"Nav1.2 in oligodendrocytes\", and merging them invents a "
        "pathway that no paper reports.\n"
        "Do not paraphrase the anchors otherwise. For every other event, name "
        "it in the paper's own terms but as a standalone biological state or "
        "process — "
        "\"increased intracellular calcium\", not \"calcium was measured\", "
        "and not \"intracellular calcium\". Every non-anchor event name must "
        "state which way the quantity went, and must distinguish function "
        "from protein from transcript (rules 4 and 5 above).\n"
    )

    return (
        "TASK: Reconstruct the causal chain that the paper in the system "
        "context provides evidence for, between these two events:\n"
        f"  Upstream event:   {upstream}\n"
        f"  Downstream event: {downstream}\n"
        + _alias_block(upstream, downstream, upstream_aliases, downstream_aliases)
        + "\n"
        + anchor_rule
        + "\n"
        "Return ONLY JSON:\n"
        "  {\n"
        '    "bears_on_question": true,\n'
        '    "reason": "<at most 25 words>",\n'
        '    "steps": [\n'
        "      {\n"
        '        "from_event": "<upstream end of this link>",\n'
        '        "from_level": "MIE|Molecular|Cellular|Tissue|Organ|Individual|Population",\n'
        '        "from_cell_type": "<cell type this event was observed in, or null>",\n'
        '        "from_target": "<gene, isoform or protein this event is about, or null>",\n'
        '        "from_change": "<increased|decreased|lost|abolished|unchanged|not stated>",\n'
        '        "to_event": "<downstream end of this link>",\n'
        '        "to_level": "MIE|Molecular|Cellular|Tissue|Organ|Individual|Population",\n'
        '        "to_cell_type": "<cell type this event was observed in, or null>",\n'
        '        "to_target": "<gene, isoform or protein this event is about, or null>",\n'
        '        "to_change": "<increased|decreased|lost|abolished|unchanged|not stated>",\n'
        '        "description": "<what the paper shows about this link, 1-2 sentences>",\n'
        '        "relation_kind": "causal|marker|definitional",\n'
        '        "evidence_type": "rescue|perturbation|common_stressor|correlation|reverse_only|not_stated",\n'
        '        "measured_as": "<the assay behind this claim>",\n'
        '        "null_findings": "<what was measured here and did NOT change, or null>",\n'
        '        "direction": "positive|negative|none|unclear",\n'
        '        "adjacency": "Adjacent|Non-adjacent",\n'
        '        "contradicts": false,\n'
        '        "quote": "<verbatim sentence from the paper evidencing THIS link>"\n'
        "      }\n"
        "    ]\n"
        "  }\n\n"
        "How to build the chain:\n"
        "- Include a step ONLY where this paper's own data speak to that link. "
        "A chain of three well-evidenced steps is worth more than a chain of "
        "eight that is mostly inference.\n"
        "- A STRESSOR STUDY BEARS ON THE QUESTION. If the paper applies a "
        "chemical or physical exposure — a metal, a toxin, a drug, irradiation "
        "— and reports that both the upstream and the downstream event "
        "changed, that is a step. Report it with evidence_type "
        "\"common_stressor\" and name the exposure in \"measured_as\". Do NOT "
        "answer \"the paper reports these as parallel effects rather than a "
        "causal link\" and return nothing: parallel effects of one exposure "
        "are exactly what an AOP records, and the framework exists to "
        "assemble them. Saying so in the description is right; withholding "
        "the step is not.\n"
        "- Intermediate events are the point. If the paper shows the upstream "
        "event acts through calcium influx, and calcium influx through a "
        "transcription factor, that is three steps, not one.\n"
        "- If the paper evidences only part of the chain — say it establishes "
        "the upstream event's effect on an intermediate and goes no further — "
        "return just those steps. A partial chain is a normal and useful "
        "result.\n"
        "- If the paper evidences the two events connecting with nothing "
        "identified in between, return a single step from the upstream event "
        "to the downstream event.\n"
        "- DO NOT STOP AT THE DOWNSTREAM EVENT. If the paper also shows what "
        "follows from it — impaired myelination, conduction failure, a "
        "behavioural or functional deficit, cell death — include those steps "
        "too, up to the most organism-level consequence the paper actually "
        "measures. The question names a starting point and a waypoint, not "
        "the end of the pathway, and an AOP that stops at a cellular event "
        "has no adverse outcome.\n"
        "- \"evidence_type\" is the most important field here, because an AOP "
        "is a claim about causation and every link below looks identical once "
        "written as \"A leads to B\":\n"
        "    rescue       — the from_event was removed AND restored (or "
        "blocked and rescued), and the to_event followed both ways.\n"
        "    perturbation — the from_event was manipulated (knockout, "
        "knockdown, blocker, overexpression) and the to_event was measured.\n"
        "    common_stressor — a chemical or physical exposure was applied "
        "and BOTH events responded to it. The paper has not shown that one "
        "causes the other, but a stressor acting on the upstream event also "
        "produced the downstream one in the same animals. That is evidence "
        "for the pathway and it is how most toxicological AOPs are built.\n"
        "    correlation  — both were measured, neither was manipulated and "
        "no exposure was applied. Two things declining together is this.\n"
        "    reverse_only — only the TO_event was manipulated. If the paper "
        "shows glutamatergic input triggering sodium-channel spikes, that is "
        "evidence input causes spiking; it is NOT evidence that losing the "
        "channel raises input. Use this and keep the step; the direction is "
        "corrected later rather than lost.\n"
        "    not_stated   — the paper asserts the link without showing it.\n"
        "  Do not upgrade a correlation to a perturbation because the "
        "conclusion is stated confidently in the abstract. Judge it on what "
        "was done in the experiment.\n"
        "- \"null_findings\": anything measured on THIS link that did not "
        "change. A null result is a finding — \"basal mEPSC amplitude was not "
        "significantly reduced\" belongs on the record next to what did "
        "change, and dropping it overstates the effect.\n"
        "- \"relation_kind\": \"causal\" if the from_event brings the "
        "to_event about. \"marker\" if the to_event is how the from_event was "
        "MEASURED — myelin basic protein staining is a readout of "
        "oligodendrocyte maturation, not a consequence of it. "
        "\"definitional\" if the to_event is part of what the from_event "
        "means. Marker and definitional links are still worth reporting; "
        "they are just not steps in the pathway, and mislabelling one as "
        "causal adds a fake terminal event to the AOP.\n"
        "- \"from_change\" and \"to_change\": what the paper's experiment did "
        "to the upstream event and what it observed at the downstream one. "
        "Knocking a channel down and seeing maturation fail is "
        "decreased/decreased; blocking it and seeing inflammation fall is "
        "decreased/decreased too, but in a different cell type — which is why "
        "both fields are asked for on every step.\n"
        "- \"from_target\" and \"to_target\": the specific gene, isoform or "
        "protein the event concerns, as the paper names it — \"SCN2A/Nav1.2\", "
        "not \"sodium channel\". The question is asked in general terms "
        "because the curator does not yet know which isoform each paper "
        "studied; that is precisely what this field recovers. Null only when "
        "the paper really does mean the whole class.\n"
        "- \"from_cell_type\" and \"to_cell_type\": the cell type each event "
        "was observed in, in the paper's own words (\"oligodendrocyte "
        "precursor cells\", \"activated microglia\"). Use null only when the "
        "paper genuinely does not localise the event. This is not optional "
        "detail: the same molecule in two cell types is two Key Events.\n"
        "- \"direction\": positive if the two events of that step move "
        "together, negative if oppositely. Report what the paper observed, "
        "whichever way round the experiment was run — driving an event up and "
        "seeing the next one rise is the same relationship as removing it and "
        "seeing the next one fall. Use \"unclear\" rather than guessing; an "
        "unsigned link is treated as unproven downstream, which is the "
        "honest outcome.\n"
        "- \"contradicts\": true only where the paper perturbed the from_event "
        "and the to_event did not follow, or it argues the two are not linked.\n"
        "- Every step needs a verbatim quote from the paper. Copy it exactly, "
        "including numbers. A step you cannot quote does not belong in the "
        "chain.\n\n"
        + (
            ""
            if directional
            else "No direction is being proposed for the overall question — "
            "report the chain and the per-step directions the paper observed, "
            "without judging them against an expectation.\n"
        )
        + "If the paper's own data bear on none of this, return "
        '"bears_on_question": false and an empty "steps" list. That is an '
        "ordinary answer. Citing someone else's finding in the introduction, "
        "or speculating in the discussion, is not the paper's own data.\n"
        "JSON:"
    )


@dataclass
class PathwayStep:
    """One link in the chain a single paper supports."""

    from_event: str
    to_event: str
    from_level: str = "Molecular"
    to_level: str = "Molecular"
    description: str = ""
    direction: str = "unclear"
    adjacency: str = "Adjacent"
    contradicts: bool = False
    spans: list[EvidenceSpan] = field(default_factory=list)

    #: What the experiment did to each end, and in which cell type it was
    #: observed. Without these a knockdown and an overexpression describing
    #: opposite results produce the same link.
    from_change: Optional[str] = None
    to_change: Optional[str] = None
    from_cell_type: Optional[str] = None
    to_cell_type: Optional[str] = None

    #: causal | marker | definitional. A marker link is how the upstream event
    #: was measured, not an event downstream of it.
    relation_kind: str = "causal"

    #: How the direction was established — see schemas.EVIDENCE_TYPE_VALUES.
    #: Everything downstream that judges a link (adjacency, confidence,
    #: whether it should be drawn at all) reads this rather than guessing
    #: from how many papers mentioned it.
    evidence_type: str = "not_stated"
    measured_as: Optional[str] = None
    from_target: Optional[str] = None
    to_target: Optional[str] = None
    null_findings: Optional[str] = None

    @property
    def is_anchored(self) -> bool:
        """Whether this step touches one of the question's two events."""
        return bool(self.from_event and self.to_event)


@dataclass
class PathwayResult:
    """Everything one paper contributed to the pathway."""

    bears_on_question: bool = False
    reason: str = ""
    steps: list[PathwayStep] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def events(self) -> list[str]:
        """Distinct event names this paper put on the graph, in chain order."""
        seen: list[str] = []
        for step in self.steps:
            for event in (step.from_event, step.to_event):
                if event and event not in seen:
                    seen.append(event)
        return seen


def _canonical_anchor(
    event: str,
    upstream: str,
    downstream: str,
    cell_type: Optional[str] = None,
    anchor_cell_types: Optional[dict[str, str]] = None,
) -> str:
    """
    Snap an event back onto the question's wording when it is plainly the same.

    The prompt asks the model to reuse the anchor labels verbatim, and models
    mostly comply, but "decreased oligodendrocyte differentiation" coming back
    as "Decreased oligodendrocyte differentiation." with a capital and a full
    stop would create a second node that never merges with the first, so near
    misses are pulled back onto the anchor here.

    What this must NOT do is treat string identity as event identity. The
    original version snapped on the normalised string alone, reasoning that a
    disconnected graph was the failure to avoid. That is true only up to the
    point where connecting things starts inventing biology: in a sodium-channel
    corpus it merged the channel in oligodendrocytes with the channel in
    activated microglia, and the resulting single pathway ran from microglial
    inflammation into oligodendrocyte maturation — an assembly no paper
    supports and a reviewer will reject on sight.

    So a step is only anchored when the cell type does not contradict the one
    the anchor has already been seen in. `anchor_cell_types` accumulates the
    first cell type observed for each anchor within a paper; a later step in a
    different cell type keeps its own name and stays a separate node, which is
    a disconnection the curator can inspect rather than a merge nobody sees.
    """
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    target = norm(event)
    if not target:
        return event

    anchor_cell_types = anchor_cell_types if anchor_cell_types is not None else {}

    # Compared by LINEAGE, not by the words the paper happened to use. The
    # first version compared normalised strings, and a single paper calling
    # its cells "oligodendrocyte lineage cells" in one sentence and
    # "oligodendrocyte lineage cells (group 1 NG2+, group 2 pre-OLs, group 3
    # OLs)" in the next produced two Key Events for one event — and a node
    # label with a parenthetical inside a parenthetical. Nine spellings of
    # one lineage are one cell; an oligodendrocyte and a microglial cell are
    # two, and only that distinction should ever split an anchor.
    observed = cell_lineage.lineage(cell_type)
    if observed == cell_lineage.UNSPECIFIED:
        observed = ""

    for anchor in (upstream, downstream):
        if target != norm(anchor):
            continue
        known = anchor_cell_types.get(anchor)
        if observed and known and observed != known:
            # Same words, genuinely different cell. Keep them apart, and say
            # which is which so the curator is not comparing identical labels.
            return f"{event.strip()} {cell_lineage.suffix_for(observed)}"
        if observed and not known:
            anchor_cell_types[anchor] = observed
        return anchor

    return event.strip()


def extract_pathway(
    document: PaperDocument,
    upstream: str,
    downstream: str,
    cfg: LLMConfig,
    *,
    paper_text: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
    directional: bool = True,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> PathwayResult:
    """One call returning the causal chain this paper evidences."""
    text_for_model = paper_text if paper_text is not None else document.full_text

    try:
        step = _run_step(
            "pathway",
            _task_pathway(
                upstream, downstream, upstream_aliases, downstream_aliases,
                directional=directional,
            ),
            cfg=cfg,
            on_step=on_step,
            num_predict=step_budget("pathway", budget_scale),
            cached_prefix=_build_cached_prefix(text_for_model),
        )
    except ExtractionAuthError:
        raise
    except ExtractionError as exc:
        return PathwayResult(error=str(exc))

    if not step.ok or not isinstance(step.parsed, dict):
        return PathwayResult(
            error=f"Pathway reply could not be parsed: {step.error}"
        )

    parsed = step.parsed
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[PathwayStep] = []
    seen_edges: set[tuple[str, str]] = set()
    # Shared across every step of this paper, so the second mention of an
    # anchor is checked against the cell type the first one established.
    anchor_cell_types: dict[str, str] = {}
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        from_cell = _opt_str(raw.get("from_cell_type"))
        to_cell = _opt_str(raw.get("to_cell_type"))
        from_event = _canonical_anchor(
            str(raw.get("from_event") or "").strip(), upstream, downstream,
            from_cell, anchor_cell_types,
        )
        to_event = _canonical_anchor(
            str(raw.get("to_event") or "").strip(), upstream, downstream,
            to_cell, anchor_cell_types,
        )
        if not from_event or not to_event or from_event == to_event:
            continue
        edge = (from_event.lower(), to_event.lower())
        if edge in seen_edges:
            continue
        seen_edges.add(edge)

        spans = build_evidence_spans(
            _coerce_quotes(raw.get("quote") or raw.get("supporting_quotes")),
            "ker_link",
            document,
            max_spans=2,
        )
        # A step with no quotation is the shape of an inferred link rather than
        # an observed one. The chain is the output here, so a weak link is
        # dropped instead of being allowed to connect two nodes on nothing.
        if not spans:
            continue

        steps.append(
            PathwayStep(
                from_event=from_event,
                to_event=to_event,
                from_level=_coerce_enum(raw.get("from_level"), KE_LEVELS, "Molecular"),
                to_level=_coerce_enum(raw.get("to_level"), KE_LEVELS, "Molecular"),
                description=str(raw.get("description") or "").strip(),
                direction=_coerce_enum(
                    str(raw.get("direction") or "").strip().lower(),
                    DIRECTIONS,
                    "unclear",
                ),
                adjacency=_coerce_enum(
                    raw.get("adjacency"), KER_ADJACENCY, "Adjacent"
                ),
                contradicts=bool(raw.get("contradicts")),
                spans=spans,
                from_change=_opt_str(raw.get("from_change")),
                to_change=_opt_str(raw.get("to_change")),
                from_cell_type=_opt_str(raw.get("from_cell_type")),
                to_cell_type=_opt_str(raw.get("to_cell_type")),
                relation_kind=_coerce_enum(
                    str(raw.get("relation_kind") or "").strip().lower(),
                    RELATION_KINDS,
                    "causal",
                ),
                evidence_type=_coerce_enum(
                    str(raw.get("evidence_type") or "").strip().lower(),
                    EVIDENCE_TYPES,
                    "not_stated",
                ),
                measured_as=_opt_str(raw.get("measured_as")),
                from_target=_opt_str(raw.get("from_target")),
                to_target=_opt_str(raw.get("to_target")),
                null_findings=_opt_str(raw.get("null_findings")),
            )
        )

    step.n_quotes = sum(len(s.spans) for s in steps)
    step.n_verified = sum(1 for s in steps for span in s.spans if span.verified)

    bears = bool(parsed.get("bears_on_question")) and bool(steps)
    return PathwayResult(
        bears_on_question=bears,
        reason=str(parsed.get("reason") or "")[:300],
        steps=steps,
        error=(
            "The model said this paper bears on the question but produced no "
            "step it could quote, so nothing was added to the pathway."
            if parsed.get("bears_on_question") and not steps
            else None
        ),
    )


def extract_pathway_rows(
    document: PaperDocument,
    upstream: str,
    downstream: str,
    cfg: LLMConfig,
    *,
    paper_text: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
    directional: bool = True,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> tuple[list[KERExtraction], PathwayResult, list[str]]:
    """
    One paper in, one Table 1 row per link of the chain it supports.

    Two model calls, not six. The pathway call carries everything that varies
    per link — the two events, what the paper shows about that link, its
    quotation. Applicability, study design and stressor are properties of the
    study, identical for every link it contributes, so they are asked once and
    shared. That makes a five-step paper cheaper than the old single-edge
    extraction while producing five edges instead of one.
    """
    warnings: list[str] = []
    text_for_model = paper_text if paper_text is not None else document.full_text

    run_manifest.record("paper_attempted", doi=document.doi)

    pathway = extract_pathway(
        document, upstream, downstream, cfg,
        paper_text=text_for_model, on_step=on_step, budget_scale=budget_scale,
        directional=directional,
        upstream_aliases=upstream_aliases,
        downstream_aliases=downstream_aliases,
    )

    if pathway.error:
        warnings.append(f"{document.filename or document.doi}: {pathway.error}")
    if not pathway.steps:
        return [], pathway, warnings

    # The anchors are the curator's own wording and are not the model's to get
    # wrong, so only the events the model invented are checked. Reported once
    # per name rather than once per step: a chain of five links naming the same
    # bare entity is one problem, not five.
    anchors = {upstream.strip().lower(), downstream.strip().lower()}
    invented: list[str] = []
    for pstep in pathway.steps:
        for event in (pstep.from_event, pstep.to_event):
            label = str(event or "").strip()
            if label and label.lower() not in anchors and label not in invented:
                invented.append(label)
    for problem in name_problems(*invented):
        warnings.append(problem)

    # --- Study-level fields, asked once ------------------------------------
    study: dict = {}
    applicability: dict = {}
    for label, fn, target in (
        ("study_meta", extract_study_meta, "study"),
        ("applicability", extract_applicability, "applicability"),
    ):
        try:
            step = fn(
                text_for_model, upstream, downstream, cfg, 1, on_step,
                budget_scale=budget_scale,
            )
        except ExtractionAuthError:
            raise
        except ExtractionError as exc:
            warnings.append(f"Step '{label}' could not reach the model — {exc}")
            continue
        if step.ok and isinstance(step.parsed, dict):
            if target == "study":
                study = step.parsed
            else:
                applicability = step.parsed
        else:
            warnings.append(f"Step '{label}' failed: {step.error}")

    rows: list[KERExtraction] = []
    for index, pstep in enumerate(pathway.steps, start=1):
        # Adjacency follows the evidence rather than the model's opinion of
        # it. "Adjacent" asserts that nothing is known to sit between two
        # events, which a correlation cannot establish — and every one of the
        # 31 links in the last run was given an adjacency with no criteria
        # behind it at all.
        adjacency = (
            "Adjacent" if pstep.evidence_type in CAUSAL_EVIDENCE
            else "Non-adjacent"
        )
        classify = {
            "upstream_ke_level": pstep.from_level,
            "downstream_ke_level": pstep.to_level,
            "ker_name": f"{pstep.from_event} leads to {pstep.to_event}",
            "ker_description": pstep.description
            or f"Link {index} of the pathway reported by this paper.",
            "ker_adjacency": adjacency,
        }

        # Essentiality was empty on every row a pathway run produced, because
        # nothing asked. It is exactly what `evidence_type` now records, so
        # the field is filled from it rather than left for a curator to
        # reconstruct from the description.
        essentiality = None
        if pstep.evidence_type == "rescue":
            essentiality = (
                f"Rescue or bidirectional test reported"
                + (f" ({pstep.measured_as})" if pstep.measured_as else "")
                + "."
            )
        elif pstep.evidence_type == "perturbation":
            essentiality = (
                f"Upstream event manipulated and the downstream event measured"
                + (f" ({pstep.measured_as})" if pstep.measured_as else "")
                + "."
            )

        evidence = {
            "paper_type": study.get("paper_type"),
            "empirical_evidence_summary": pstep.description or None,
            "essentiality_evidence": essentiality,
            "contradicts_ker": pstep.contradicts,
        }
        # One link that fails validation must cost that link and no more. This
        # was unguarded, so a single malformed step raised out of the whole
        # function and the paper was discarded after its model calls had
        # already been paid for — the other four links went with it.
        try:
            rows.append(
                _assemble(
                    pstep.from_event,
                    pstep.to_event,
                    classify,
                    evidence,
                    applicability,
                    {},      # quantitative detail is per-link and not asked for here
                    study,
                    evidence_spans=_deduplicate_spans(list(pstep.spans)),
                    context={
                        "direction": pstep.direction,
                        "upstream_change": pstep.from_change,
                        "downstream_change": pstep.to_change,
                        "upstream_cell_type": pstep.from_cell_type,
                        "downstream_cell_type": pstep.to_cell_type,
                        "relation_kind": pstep.relation_kind,
                        "evidence_type": pstep.evidence_type,
                        "measured_as": pstep.measured_as,
                        "upstream_target": pstep.from_target,
                        "downstream_target": pstep.to_target,
                        "null_findings": pstep.null_findings,
                        "study_context": study.get("study_context"),
                    },
                )
            )
        except ExtractionValidationError as exc:
            warnings.append(
                f"Link {index} ({pstep.from_event} → {pstep.to_event}) could "
                f"not be assembled and was dropped: {exc}"
            )

    # --- The relationship the curator actually asked about -------------------
    #
    # Every row above is one LINK of the chain. If the model wrote the chain as
    # A -> x -> y -> B, not one of those rows is about A -> B, so the paper
    # contributes nothing to the target relationship — even though its chain
    # demonstrates exactly that relationship.
    #
    # Whether the target edge appeared at all therefore depended on whether the
    # model also emitted a redundant A -> B shortcut alongside the chain, which
    # it does inconsistently: measured over replicate runs of one 13-paper
    # corpus, 5 of 13 papers changed that answer between two identical runs,
    # while the underlying chains stayed the same. The support count moved
    # because of a stylistic choice about summarising, not because of evidence.
    #
    # So support is now decided by REACHABILITY: if the paper's chain connects
    # the two anchors, the paper supports the relationship and gets a row for
    # it. The row is marked Non-adjacent and carries the path, because a link
    # through four intermediates is real support and is not the same claim as a
    # demonstrated direct step. Over the same corpus this takes per-paper
    # agreement from 54% to 92%.
    path = _anchor_path(pathway.steps, upstream, downstream)
    already_direct = any(
        _norm_event(s.from_event) == _norm_event(upstream)
        and _norm_event(s.to_event) == _norm_event(downstream)
        for s in pathway.steps
    )
    if path and not already_direct:
        try:
            rows.append(
                _assemble_indirect_target_row(
                    upstream, downstream, path, pathway.steps,
                    study, applicability,
                )
            )
        except ExtractionValidationError as exc:
            warnings.append(
                f"The chain connects {upstream} to {downstream} via "
                f"{len(path) - 2} intermediate event(s), but the summary row "
                f"could not be assembled: {exc}"
            )

    run_manifest.record(
        "pathway_extracted",
        doi=document.doi,
        n_steps=len(rows),
        n_events=len(pathway.events),
    )
    return rows, pathway, warnings


def _norm_event(text: Optional[str]) -> str:
    """Event-name comparison key, matching `_canonical_anchor`'s notion."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _anchor_path(
    steps: Sequence["PathwayStep"], upstream: str, downstream: str
) -> list[str]:
    """
    The shortest chain of events linking the two anchors, or [] if none.

    Breadth-first, so the path returned is the fewest intermediates the paper
    supports rather than the first one stumbled upon. Cycles are common in
    these chains — a paper reporting remyelination restoring channel clustering
    genuinely closes a loop — so visited events are tracked.
    """
    start, goal = _norm_event(upstream), _norm_event(downstream)
    if not start or not goal or start == goal:
        return []

    successors: dict[str, list[str]] = {}
    label: dict[str, str] = {}
    for step in steps:
        a, b = _norm_event(step.from_event), _norm_event(step.to_event)
        if not a or not b:
            continue
        label.setdefault(a, str(step.from_event).strip())
        label.setdefault(b, str(step.to_event).strip())
        successors.setdefault(a, []).append(b)

    previous: dict[str, Optional[str]] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            chain: list[str] = []
            cursor: Optional[str] = node
            while cursor is not None:
                chain.append(label.get(cursor, cursor))
                cursor = previous[cursor]
            return list(reversed(chain))
        for nxt in successors.get(node, ()):
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    return []


def _assemble_indirect_target_row(
    upstream: str,
    downstream: str,
    path: Sequence[str],
    steps: Sequence["PathwayStep"],
    study: dict,
    applicability: dict,
) -> KERExtraction:
    """
    One Table 1 row recording that this paper links the two target events.

    The evidence is the chain, so the row carries the weakest link's evidence
    type rather than the strongest: a path is only as good as its flimsiest
    step, and taking the best would let one knockout launder three
    correlations into a demonstrated relationship.
    """
    on_path = {(_norm_event(a), _norm_event(b))
               for a, b in zip(path, path[1:])}
    path_steps = [
        s for s in steps
        if (_norm_event(s.from_event), _norm_event(s.to_event)) in on_path
    ]

    ranked = sorted(
        path_steps,
        key=lambda s: _EVIDENCE_STRENGTH.get(s.evidence_type, 0),
    )
    weakest = ranked[0] if ranked else None
    contradicts = any(s.contradicts for s in path_steps)

    route = " → ".join(path)
    n_intermediates = max(0, len(path) - 2)

    classify = {
        "upstream_ke_level": path_steps[0].from_level if path_steps else "Molecular",
        "downstream_ke_level": path_steps[-1].to_level if path_steps else "Cellular",
        "ker_name": f"{upstream} leads to {downstream}",
        "ker_description": (
            f"This paper links {upstream} to {downstream} through "
            f"{n_intermediates} intermediate event(s): {route}. The link is "
            f"assembled from the paper's own chain rather than asserted "
            f"directly by it."
        ),
        # Never "Adjacent": the paper itself put events in between.
        "ker_adjacency": "Non-adjacent",
    }
    evidence = {
        "paper_type": study.get("paper_type"),
        "empirical_evidence_summary": (
            f"Indirect support via {n_intermediates} intermediate event(s). "
            f"Weakest link on the path: "
            f"{weakest.evidence_type if weakest else 'not_stated'}."
        ),
        "essentiality_evidence": None,
        "contradicts_ker": contradicts,
    }
    return _assemble(
        upstream,
        downstream,
        classify,
        evidence,
        applicability,
        # `_assemble` reads modulating_factors out of the quantitative block,
        # not out of context. Passing the route in context put it nowhere, and
        # the row reached the curator with no way to see which path it rests
        # on — which is the one thing an indirect row must show.
        {"modulating_factors": f"Indirect path via: {route}."},
        study,
        evidence_spans=_deduplicate_spans(
            [span for s in path_steps for span in s.spans]
        ),
        context={
            "direction": _path_direction(path_steps),
            "upstream_change": path_steps[0].from_change if path_steps else None,
            "downstream_change": path_steps[-1].to_change if path_steps else None,
            "upstream_cell_type": path_steps[0].from_cell_type if path_steps else None,
            "downstream_cell_type": path_steps[-1].to_cell_type if path_steps else None,
            "relation_kind": "causal",
            "evidence_type": weakest.evidence_type if weakest else "not_stated",
            "measured_as": None,
            "null_findings": None,
            "study_context": study.get("study_context"),
        },
    )


#: Ordering used to find the weakest link on an indirect path.
_EVIDENCE_STRENGTH = {
    "rescue": 5, "perturbation": 4, "common_stressor": 3,
    "correlation": 2, "reverse_only": 1, "not_stated": 0,
}


def _path_direction(path_steps: Sequence["PathwayStep"]) -> str:
    """
    Sign of an indirect link: the product of the signs along the path.

    Two negative steps compose to a positive relationship, which is the whole
    reason this cannot be taken from either endpoint alone. Any unsigned or
    conflicting step makes the composition unknowable.
    """
    sign = 1
    for step in path_steps:
        if step.direction == "positive":
            continue
        if step.direction == "negative":
            sign = -sign
        else:
            return "unclear"
    return "positive" if sign > 0 else "negative"


def extract_targeted_ker(
    document: PaperDocument,
    upstream: str,
    downstream: str,
    cfg: LLMConfig,
    *,
    paper_text: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    budget_scale: float = 1.0,
    gate: Optional[GateResult] = None,
    directional: bool = True,
    upstream_aliases: Optional[Sequence[str]] = None,
    downstream_aliases: Optional[Sequence[str]] = None,
) -> tuple[Optional[KERExtraction], GateResult, list[str]]:
    """
    Characterise ONE specified relationship in one paper.

    Returns (extraction | None, gate result, warnings). The extraction is None
    when the paper does not bear on the relationship — which is a normal
    outcome, not a failure, and costs a single model call.

    `upstream` and `downstream` are used verbatim as the Key Event names, so
    rows from different papers describing this relationship are identical by
    construction.
    """
    warnings: list[str] = []
    text_for_model = paper_text if paper_text is not None else document.full_text

    run_manifest.record("paper_attempted", doi=document.doi)

    if gate is None:
        gate = screen_document_for_ker(
            document, upstream, downstream, cfg,
            paper_text=text_for_model, on_step=on_step, budget_scale=budget_scale,
            directional=directional,
            upstream_aliases=upstream_aliases,
            downstream_aliases=downstream_aliases,
        )

    if gate.error:
        warnings.append(f"{document.filename or document.doi}: {gate.error}")
    if not gate.is_relevant:
        return None, gate, warnings

    per_ker_steps = [
        ("classify",      classify_ker),
        ("evidence",      assess_evidence),
        ("applicability", extract_applicability),
        ("quantitative",  extract_quantitative),
        ("study_meta",    extract_study_meta),
    ]
    results: dict[str, dict] = {}
    spans: list[EvidenceSpan] = list(gate.spans)

    for label, fn in per_ker_steps:
        try:
            step = fn(
                text_for_model, upstream, downstream, cfg, 1, on_step,
                budget_scale=budget_scale,
            )
        except ExtractionAuthError:
            raise
        except ExtractionError as exc:
            warnings.append(f"Step '{label}' could not reach the model — {exc}")
            results[label] = {}
            continue

        if not step.ok or not isinstance(step.parsed, dict):
            warnings.append(f"Step '{label}' failed: {step.error}")
            results[label] = {}
            continue

        results[label] = step.parsed
        if step.truncated:
            warnings.append(
                f"Step '{label}' hit the output-token limit; later fields are missing."
            )

        field_name = _STEP_EVIDENCE_FIELD.get(label)
        if field_name:
            step_spans = build_evidence_spans(
                _coerce_quotes(step.parsed.get("supporting_quotes")),
                field_name,
                document,
                max_spans=3,
            )
            spans.extend(step_spans)
            step.n_quotes = len(step_spans)
            step.n_verified = sum(1 for s in step_spans if s.verified)

    # The mechanistic description comes from the classify step, and `_assemble`
    # falls back to "Relationship between <upstream> and <downstream>." when it
    # is missing. In open mode that placeholder is merely thin; here it is
    # worse than useless, because upstream and downstream are the user's own
    # words — the row would echo the question back as though it were an answer.
    # So the step gets a second attempt, and a failure is reported rather than
    # papered over.
    if not str(results.get("classify", {}).get("ker_description") or "").strip():
        run_manifest.record(
            "note", message="Classify step returned no KER description; retrying."
        )
        try:
            retry = classify_ker(
                text_for_model, upstream, downstream, cfg, 1, on_step,
                budget_scale=max(budget_scale, 1.5),
            )
            if retry.ok and isinstance(retry.parsed, dict):
                merged = dict(results.get("classify", {}))
                merged.update({k: v for k, v in retry.parsed.items() if v})
                results["classify"] = merged
        except ExtractionAuthError:
            raise
        except ExtractionError as exc:
            warnings.append(f"Retry of step 'classify' could not reach the model — {exc}")

    if not str(results.get("classify", {}).get("ker_description") or "").strip():
        warnings.append(
            "No mechanistic description could be generated for this paper: the "
            "classify step returned nothing usable twice. The row carries a "
            "placeholder that restates the two Key Event names — treat the "
            "description as missing, not as the paper's account."
        )

    spans = _deduplicate_spans(spans)

    # The gate already established how the paper connects the two events, and
    # it looked at the same text the classify step did. Its verdict is the
    # better authority on adjacency, so it overrides.
    classify = dict(results.get("classify", {}))
    if gate.verdict == "indirect":
        classify["ker_adjacency"] = "Non-adjacent"
    elif gate.verdict == "direct":
        classify.setdefault("ker_adjacency", "Adjacent")

    try:
        extraction = _assemble(
            upstream      = upstream,
            downstream    = downstream,
            classify      = classify,
            evidence      = results.get("evidence", {}),
            applicability = results.get("applicability", {}),
            quantitative  = results.get("quantitative", {}),
            study_meta    = results.get("study_meta", {}),
            evidence_spans= spans,
        )
    except ExtractionValidationError as exc:
        warnings.append(f"Assembly failed for {upstream} → {downstream}: {exc}")
        return None, gate, warnings

    # The gate saw the contradiction; the evidence step is not asked about it
    # directly, so carry the verdict across rather than lose it.
    if gate.verdict == "contradicts":
        extraction.contradicts_ker = True

    # The gate already worked out which way the experiment ran. Until v8 that
    # went into a prose field, where nothing downstream could read it, and the
    # edge reached the graph unsigned. It is now a field on the row.
    extraction.direction = _coerce_enum(gate.coupling, DIRECTIONS, "unclear")
    extraction.upstream_change = _opt_str(gate.observed_upstream_change)
    extraction.downstream_change = _opt_str(gate.observed_downstream_change)

    notes: list[str] = []
    if gate.intermediate_events:
        notes.append("Indirect path via: " + " → ".join(gate.intermediate_events) + ".")
    if notes:
        existing = extraction.modulating_factors or ""
        extraction.modulating_factors = f"{existing} {' '.join(notes)}".strip()

    run_manifest.record("paper_extracted", n_kers=1)
    return extraction, gate, warnings


#: Discourse markers a model adds or drops when quoting the same sentence
#: twice. "However, the spiking mESC-OPCs showed…" and "The spiking mESC-OPCs
#: showed…" are one sentence, but keying on the raw prefix filed them as two
#: separate pieces of evidence — which inflates the apparent support for a KER
#: with the same quotation counted twice.
_LEAD_IN_RE = re.compile(
    r"^(?:however|moreover|furthermore|additionally|in addition|notably|"
    r"importantly|interestingly|therefore|thus|indeed|finally|overall|"
    r"in contrast|by contrast|conversely|similarly|consistent with this)"
    r"[,;:\s]+",
    re.I,
)


def _dedup_key(quote: str) -> str:
    """
    Fingerprint identifying a quotation regardless of its lead-in.

    Strips a leading discourse marker, drops punctuation and collapses
    whitespace, so trivially different renderings of one sentence collapse.
    """
    text = (quote or "").strip()
    # Applied twice: "However, notably, ..." does occur.
    for _ in range(2):
        text = _LEAD_IN_RE.sub("", text.strip())
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()[:160]


def _deduplicate_spans(spans: Sequence[EvidenceSpan]) -> list[EvidenceSpan]:
    """
    Collapse quotations repeated across steps.

    The same sentence often supports several fields; we keep one span and
    record the union of the fields it supports, preferring verified copies.
    """
    by_quote: dict[str, EvidenceSpan] = {}
    fields: dict[str, list[str]] = {}

    for span in spans:
        key = _dedup_key(span.quote)
        if key not in by_quote:
            by_quote[key] = span
            fields[key] = [span.field]
            continue
        fields[key].append(span.field)
        if span.verified and not by_quote[key].verified:
            by_quote[key] = span

    out: list[EvidenceSpan] = []
    for key, span in by_quote.items():
        unique_fields = list(dict.fromkeys(fields[key]))
        span.field = ", ".join(unique_fields)
        out.append(span)
    return out


def extract_kers_from_text(
    paper_text: str,
    cfg: Optional[LLMConfig] = None,
    *,
    model: str = "llama3.1:8b",
    ollama_url: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
    max_kers: int = 20,
    document: Optional[PaperDocument] = None,
) -> tuple[list[KERExtraction], list[str]]:
    """
    Text-only entry point, kept for compatibility.

    When no `document` is supplied a minimal one is synthesised from the text so
    that quote verification still works — though without page numbers, since a
    bare string carries no pagination.
    """
    if document is None:
        document = PaperDocument(
            filename="(text input)",
            doi=None,
            full_text=paper_text,
            pages=[],
            chunks=[],
        )
    return extract_kers_from_document(
        document,
        cfg,
        paper_text=paper_text,
        model=model,
        ollama_url=ollama_url,
        on_step=on_step,
        max_kers=max_kers,
    )


__all__ = [
    "ExtractionError",
    "ExtractionAuthError",
    "ExtractionValidationError",
    "StepResult",
    "extract_kers_from_document",
    "extract_kers_from_text",
    "build_evidence_spans",
    "GateResult",
    "screen_for_ker",
    "screen_document_for_ker",
    "extract_targeted_ker",
    "extract_pathway",
    "extract_pathway_rows",
    "PathwayStep",
    "PathwayResult",
    "step_budget",
    "list_ker_pairs",
    "classify_ker",
    "assess_evidence",
    "extract_applicability",
    "extract_quantitative",
    "extract_study_meta",
]
