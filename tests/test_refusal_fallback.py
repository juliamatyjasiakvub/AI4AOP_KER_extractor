"""
What happens when a safety classifier declines a paper.

`stop_reason: "refusal"` is decided on the model's generated output as it is
being written, not on the prompt. It is therefore sampled, and on Anthropic
nothing pins that sampling — `seed` is only sent to Ollama and OpenAI, and the
temperature is 0.1 rather than 0. The same paper is read on one run and refused
on the next, which is exactly what a curator sees and what makes the failure
look arbitrary.

That has two consequences these tests encode. Retrying is worth doing, because
the refusal is a biased coin and not a verdict — but only if each attempt is a
different draw, which means raising the temperature rather than re-sending the
identical request. And a different *model* is the last resort, not the first,
because it needs configuration and changes what produced the rows.

The rest holds the implementation to its claims: refused attempts must be
counted and visible, a row produced by another model must say so, and a network
error must never be mistaken for a refusal.
"""

from __future__ import annotations

import pytest

from stage2_extraction import ker_extractor
from stage2_extraction.ker_extractor import ExtractionError
from stage2_extraction.llm_providers import LLMConfig


REFUSAL = (
    "anthropic declined to answer for model 'claude-sonnet-5'. This is a "
    "safety-classifier false positive rather than a judgement about the work."
)


@pytest.fixture
def calls(monkeypatch):
    """Record every provider call and script its outcome per model."""
    seen: list[tuple[str, str]] = []
    behaviour: dict[str, str] = {}

    def fake_call(cfg, prompt, num_predict=1024, cached_prefix=None):
        key = f"{cfg.provider}/{cfg.model}"
        seen.append((key, prompt, cfg.temperature))
        outcome = behaviour.get(key, "ok")
        if callable(outcome):
            outcome = outcome(len([s for s in seen if s[0] == key]))
        if outcome == "refuse":
            raise ExtractionError(REFUSAL)
        if outcome == "error":
            raise ExtractionError("connection reset")
        return '{"pairs": []}'

    monkeypatch.setattr(ker_extractor, "_call_llm", fake_call)
    # The fallback moved from a module global to thread-local state, and this
    # fixture kept patching the old `_REFUSAL_FALLBACK` name with
    # `raising=False` — so it silently stopped isolating anything. A test that
    # configured a fallback leaked it into every test that ran afterwards, and
    # the one asserting that a refusal with NO fallback raises was failing
    # against a fallback set two tests earlier. Cleared through the accessor
    # that actually owns the state.
    ker_extractor.set_refusal_fallback(None)
    return seen, behaviour


PRIMARY = LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key="x")
FALLBACK = LLMConfig(provider="openai", model="gpt-4o", api_key="y")


# ---------------------------------------------------------------------------
# The fallback has to be a different model
# ---------------------------------------------------------------------------

def test_same_model_is_not_a_fallback():
    """
    Nominating the primary as its own fallback is a no-op: the same-model
    retries have already happened by then, so it would add a fourth identical
    attempt and nothing else.
    """
    same = LLMConfig(provider="anthropic", model="claude-sonnet-5")
    assert not ker_extractor._is_distinct_model(PRIMARY, same)


def test_no_fallback_configured_is_not_a_fallback():
    assert not ker_extractor._is_distinct_model(PRIMARY, None)


def test_a_sibling_model_on_the_same_key_is_a_fallback():
    """
    The objection that made the first version of this useless: it looked as
    though it required a second subscription. It does not — classifiers differ
    per model, so another Claude on the same key is a valid fallback, and that
    is what the sidebar now proposes.
    """
    assert ker_extractor._is_distinct_model(
        PRIMARY, LLMConfig(provider="anthropic", model="claude-haiku-4-5")
    )
    assert ker_extractor._is_distinct_model(PRIMARY, FALLBACK)


# ---------------------------------------------------------------------------
# What happens on a refusal
# ---------------------------------------------------------------------------

def test_the_same_model_is_re_asked_at_a_higher_temperature(calls):
    """
    The core fix. The refusal is decided on the sampled output, so a second
    draw is a real second chance — but only if it is actually a different
    draw. The first attempt must keep the run's configured temperature,
    because it is the one that produces almost every row.
    """
    seen, behaviour = calls
    # Refuses once, then answers — the pattern the curator sees across runs,
    # compressed into one call site.
    behaviour["anthropic/claude-sonnet-5"] = lambda n: "refuse" if n == 1 else "ok"

    result = ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    assert result.ok
    assert len(seen) == 2, "refused once, retried once, answered"
    assert seen[0][2] == PRIMARY.temperature, "first attempt is unperturbed"
    assert seen[1][2] > seen[0][2], "the retry has to be a different draw"
    assert result.answered_by is None, "the run's own model answered"
    assert result.attempts == 2, "both calls were made and must be counted"


def test_every_same_model_attempt_is_tried_before_the_fallback(calls):
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(FALLBACK)

    result = ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    n_same_model = len(ker_extractor.REFUSAL_RETRY_TEMPERATURES)
    assert [model for model, _, _ in seen] == (
        ["anthropic/claude-sonnet-5"] * n_same_model + ["openai/gpt-4o"]
    ), "the fallback is the last resort, not the first"
    assert result.ok
    assert result.answered_by == "openai/gpt-4o", (
        "a row produced by the fallback has to say which model produced it"
    )


def test_temperatures_rise_across_attempts(calls):
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"

    with pytest.raises(ExtractionError):
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    temperatures = [t for _, _, t in seen]
    assert temperatures == sorted(temperatures), "each attempt is a wider draw"
    assert len(set(temperatures)) == len(temperatures), "no attempt repeats another"


def test_the_fallback_gets_the_same_prompt(calls):
    """The question does not change — only the model being asked."""
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(FALLBACK)

    ker_extractor._run_step("pathway", "THE TASK", cfg=PRIMARY, on_step=None)

    assert {prompt for _, prompt, _ in seen} == {"THE TASK"}


def test_no_fallback_still_gets_the_retry_ladder(calls):
    """
    The retry needs no configuration and no second key, so it must happen
    whether or not a fallback model is set. That is the whole point: most
    curators have one provider.
    """
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(None)

    with pytest.raises(ExtractionError) as caught:
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    assert len(seen) == len(ker_extractor.REFUSAL_RETRY_TEMPERATURES)
    message = str(caught.value)
    assert "pathway" in message, "say which step was refused"
    assert "re-run" in message.lower(), "say that another run often works"
    assert "sampled" in message.lower(), "say why it is not about the paper"


def test_both_refusing_names_both_models(calls):
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    behaviour["openai/gpt-4o"] = "refuse"
    ker_extractor.set_refusal_fallback(FALLBACK)

    with pytest.raises(ExtractionError) as caught:
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    message = str(caught.value)
    assert len(seen) == len(ker_extractor.REFUSAL_RETRY_TEMPERATURES) + 1
    assert "claude-sonnet-5" in message and "gpt-4o" in message


# ---------------------------------------------------------------------------
# Everything else is untouched
# ---------------------------------------------------------------------------

def test_a_network_error_does_not_reach_the_fallback(calls):
    """
    The fallback exists for refusals. A dropped connection is not a refusal,
    and quietly answering it with a second model would swap the corpus's model
    on a flaky wifi connection.
    """
    seen, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "error"
    ker_extractor.set_refusal_fallback(FALLBACK)

    with pytest.raises(ExtractionError):
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    assert [model for model, _, _ in seen] == ["anthropic/claude-sonnet-5"], (
        "a dropped connection is not a refusal and gets no temperature ladder"
    )


def test_a_successful_call_does_not_record_a_model(calls):
    """`answered_by` marks the exception, so it must stay None on the normal path."""
    seen, _ = calls
    ker_extractor.set_refusal_fallback(FALLBACK)

    result = ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)

    assert len(seen) == 1
    assert result.answered_by is None
    assert result.ok


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------

def test_the_manifest_records_a_mixed_run(calls):
    """
    The point of recording provider and model is to say what conditions the
    rows were produced under. A run whose rows came from two models cannot be
    described by one name, so the fallback has to leave a mark.
    """
    import run_manifest

    _, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(FALLBACK)

    telemetry = run_manifest.start_run()
    try:
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)
    finally:
        run_manifest.end_run()

    assert telemetry.refusals == len(ker_extractor.REFUSAL_RETRY_TEMPERATURES), (
        "every refused attempt is a refusal, not just the first"
    )
    assert telemetry.refusal_fallbacks == 1
    assert telemetry.fallback_models == {"openai/gpt-4o"}
    assert any("more than one model" in note for note in telemetry.notes)

    row = telemetry.as_row()
    assert row["fallback_models"] == "openai/gpt-4o"


def test_llm_calls_counts_answered_steps_not_requests(calls):
    """
    `llm_calls` is a count of steps that produced a usable reply, and it must
    stay that even when a step took four requests to get one. The cost of the
    refused requests is carried by `refusals` and by `StepResult.attempts`,
    which is what the per-paper figure in the UI now sums.
    """
    import run_manifest

    _, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(FALLBACK)

    telemetry = run_manifest.start_run()
    try:
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)
    finally:
        run_manifest.end_run()

    # One step reached a provider once and produced one usable reply.
    assert telemetry.llm_calls == 1, (
        "llm_calls counts steps that yielded an answer, not HTTP requests"
    )


# ---------------------------------------------------------------------------
# Refused calls are calls
# ---------------------------------------------------------------------------

def test_a_refused_step_reaches_the_step_log(calls):
    """
    A refused paper used to report zero model calls. `_run_step` raised before
    ever building a StepResult, so the callback never fired, the step log
    stayed empty, and the requests that were actually sent and charged for were
    invisible in the run record.
    """
    _, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(None)

    logged: list = []
    with pytest.raises(ExtractionError):
        ker_extractor._run_step(
            "pathway", "TASK", cfg=PRIMARY, on_step=logged.append
        )

    assert len(logged) == 1, "the failed step still happened and must be logged"
    step = logged[0]
    assert not step.ok
    assert step.step == "pathway"
    assert step.attempts == len(ker_extractor.REFUSAL_RETRY_TEMPERATURES)
    assert sum(max(1, s.attempts) for s in logged) == step.attempts, (
        "this is the expression app.py uses to report cost; it must not be zero"
    )


def test_a_broken_callback_does_not_mask_the_refusal(calls):
    """Instrumentation must not be able to change what the pipeline reports."""
    _, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(None)

    def explode(_step):
        raise RuntimeError("the debug panel blew up")

    with pytest.raises(ExtractionError):
        ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=explode)


def test_retries_are_not_recorded_as_dropped_parameters(calls):
    """
    The manifest bug. A refusal was recorded as `provider_retry(param=...)`,
    which lands in the dropped-parameter set — so a run that lost a paper to a
    classifier claimed the provider had rejected a request parameter called
    "refusal", and the field that says what the provider would not accept
    became unreadable.
    """
    import run_manifest

    _, behaviour = calls
    behaviour["anthropic/claude-sonnet-5"] = "refuse"
    ker_extractor.set_refusal_fallback(None)

    telemetry = run_manifest.start_run()
    try:
        with pytest.raises(ExtractionError):
            ker_extractor._run_step("pathway", "TASK", cfg=PRIMARY, on_step=None)
    finally:
        run_manifest.end_run()

    assert "refusal" not in telemetry.dropped_params
    assert telemetry.dropped_params == set()
    assert telemetry.refusals == len(ker_extractor.REFUSAL_RETRY_TEMPERATURES)
    assert telemetry.refusal_retries == len(
        ker_extractor.REFUSAL_RETRY_TEMPERATURES
    ) - 1, "the first attempt is not a retry"
    assert telemetry.as_row()["refusal_retries"] == telemetry.refusal_retries
