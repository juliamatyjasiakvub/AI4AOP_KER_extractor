from __future__ import annotations

"""
Pluggable LLM backend for the KER extraction pipeline.

Supports three providers behind a single `LLMConfig.generate(prompt) -> str`
contract that returns the model's raw text (always JSON when possible):

    * Ollama    — local, default. Same behaviour as before.
    * Anthropic — Claude (e.g. claude-sonnet-4-5, claude-opus-4-5). 200k context.
    * OpenAI    — GPT (e.g. gpt-4o, gpt-4o-mini, gpt-4.1). 128k+ context.

No vendor SDKs are required — every provider is reached via plain HTTPS with
the `requests` library that is already in the project.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests

import run_manifest


@dataclass
class LLMConfig:
    """Provider-agnostic configuration for a single chat/completion call."""

    provider: str = "ollama"          # "ollama" | "anthropic" | "openai"
    model: str = "llama3.1:8b"
    api_key: Optional[str] = None     # required for anthropic / openai
    base_url: Optional[str] = None    # ollama: http://localhost:11434
    temperature: float = 0.1
    max_output_tokens: int = 1024
    num_ctx: int = 65536              # ollama only — ignored elsewhere
    request_timeout: int = 1200

    #: Sampling seed, sent where the provider accepts one (Ollama, OpenAI).
    #: It narrows run-to-run variation but does not remove it: hosted models
    #: are re-versioned and server-side batching perturbs results regardless.
    #: Recorded in the run manifest either way, so a later run can be compared
    #: against this one on equal terms.
    seed: Optional[int] = None
    top_p: Optional[float] = None

    def generate(self, prompt: str, cached_prefix: Optional[str] = None) -> str:
        """
        Send `prompt` to the configured provider and return the raw text reply.

        `cached_prefix` is a long, stable block (e.g. persona + paper text) that
        will be REUSED across many calls. Each provider lays it out so its
        prompt-caching mechanism can charge it once instead of per-call:

            * Ollama    — concatenated; the in-process KV cache reuses tokens
                          when consecutive calls share the same prefix.
            * Anthropic — placed in a system block with
                          `cache_control: {"type": "ephemeral"}`. First call
                          writes the cache; subsequent calls within ~5 min
                          read from it at ~10 % of the input cost.
            * OpenAI    — placed at the start of a stable `system` message.
                          The platform's automatic prefix-cache (≥1024 tokens)
                          discounts repeated prefixes by ~50 %.

        Pipeline-level errors (network, auth, 5xx) bubble up as
        `LLMProviderError` so the orchestrator can surface them clearly.
        """
        provider = (self.provider or "ollama").lower()
        if provider == "ollama":
            return _generate_ollama(self, prompt, cached_prefix)
        if provider == "anthropic":
            return _generate_anthropic(self, prompt, cached_prefix)
        if provider == "openai":
            return _generate_openai(self, prompt, cached_prefix)
        raise LLMProviderError(f"Unknown provider: {self.provider!r}")


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot fulfil a request."""


class LLMAuthError(LLMProviderError):
    """
    Raised when the provider rejects the credentials.

    Kept distinct from the generic error because it is *permanent*: retrying
    with the same key fails identically every time. Callers looping over many
    records should abort on this rather than burning one request per record to
    learn the same thing 71 times.
    """


# ---------------------------------------------------------------------------
# Unsupported-parameter handling
#
# Providers retire request parameters over time — newer Claude and OpenAI
# models reject `temperature` outright, and OpenAI renamed `max_tokens` to
# `max_completion_tokens`. Hard-coding a list of which models accept what goes
# stale the moment a new model ships, so instead we send the parameter, read
# the 400 the API returns, drop the offending field and retry once.
#
# The result is remembered per (provider, model) for the life of the process,
# so a session pays the extra round-trip once rather than on all ~30 calls per
# paper.
# ---------------------------------------------------------------------------

#: (provider, model) -> set of parameter names that provider rejected.
_UNSUPPORTED_PARAMS: dict[tuple[str, str], set[str]] = {}

#: Parameters we are willing to drop automatically. Anything not listed here
#: is essential, so a 400 mentioning it is a real error and must surface.
_DROPPABLE_PARAMS = ("temperature", "top_p", "top_k")

#: Parameters that were renamed rather than removed. OpenAI's newer models
#: reject `max_tokens` and want `max_completion_tokens` instead; dropping it
#: would leave the response length uncapped, so we rename rather than drop.
_RENAMED_PARAMS: dict[str, str] = {
    "max_tokens": "max_completion_tokens",
}


def _remembered_unsupported(provider: str, model: str) -> set[str]:
    return _UNSUPPORTED_PARAMS.setdefault((provider, model), set())


_REJECTION_MARKERS = (
    "deprecated",
    "unsupported",
    "not supported",
    "unrecognized",
    "unknown parameter",
    "is not permitted",
    "use `max_completion_tokens`",
)


def _mentions(error_body: str, param: str) -> bool:
    """True if the error text names `param`, in any of the usual quotings."""
    return any(
        token in error_body
        for token in (f"`{param}`", f"'{param}'", f'"{param}"', f" {param} ")
    )


def _offending_param(error_body: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Identify which parameter an API rejected and what to do about it.

    Returns (param, replacement_name) where replacement_name is None if the
    parameter should simply be dropped. Matching is on the error text rather
    than the status code alone, so we only touch a parameter the provider
    actually complained about.
    """
    lowered = (error_body or "").lower()
    if not any(marker in lowered for marker in _REJECTION_MARKERS):
        return None

    for param, replacement in _RENAMED_PARAMS.items():
        if _mentions(lowered, param):
            return param, replacement

    for param in _DROPPABLE_PARAMS:
        if _mentions(lowered, param):
            return param, None

    return None


def _post_json(
    url: str,
    payload: dict,
    headers: dict,
    timeout: int,
    provider: str,
    model: str,
) -> dict:
    """
    POST `payload`, retrying without any parameter the provider rejects.

    Raises LLMProviderError with the provider's own message for anything we
    cannot resolve by dropping a parameter.
    """
    known_bad = _remembered_unsupported(provider, model)

    body = dict(payload)
    for param in known_bad:
        if param in body:
            replacement = _RENAMED_PARAMS.get(param)
            value = body.pop(param)
            if replacement:
                body[replacement] = value

    max_attempts = len(_DROPPABLE_PARAMS) + len(_RENAMED_PARAMS) + 1
    for _ in range(max_attempts):
        try:
            response = requests.post(url, json=body, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            text = exc.response.text if exc.response is not None else ""

            if status == 400:
                found = _offending_param(text)
                if found is not None:
                    param, replacement = found
                    if param in body:
                        known_bad.add(param)
                        value = body.pop(param)
                        if replacement and replacement not in body:
                            body[replacement] = value
                        # A dropped parameter changes the run's conditions —
                        # temperature silently reverting to the provider's
                        # default is exactly the kind of thing that makes two
                        # runs incomparable, so it is recorded.
                        run_manifest.record(
                            "provider_retry", param=param, replacement=replacement
                        )
                        continue  # retry with the corrected payload

            run_manifest.record("provider_error", status=status)

            if status in (401, 403):
                raise LLMAuthError(
                    f"{provider.title()} rejected the API key (HTTP {status}).\n\n"
                    "Check the key for this stage in the sidebar — Stage 1 and "
                    "Stage 2 have separate key fields, so a key that works for "
                    "extraction is not automatically used for screening.\n"
                    "Common causes: a trailing space or newline from pasting, a "
                    "key from a different organisation, or a revoked key.\n\n"
                    f"Provider said: {text[:300]}"
                ) from exc

            raise LLMProviderError(
                f"{provider.title()} HTTP {status}: {text[:500]}"
            ) from exc
        except requests.RequestException as exc:
            run_manifest.record("provider_error", status=0)
            raise LLMProviderError(f"{provider.title()} request failed: {exc}") from exc

    raise LLMProviderError(
        f"{provider.title()} rejected every parameter combination tried for model "
        f"{model!r}. Last payload keys: {sorted(body)}"
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _generate_ollama(cfg: LLMConfig, prompt: str, cached_prefix: Optional[str] = None) -> str:
    base = (cfg.base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    # Concatenate so the prefix appears verbatim at the start of every call
    # — Ollama's KV cache will reuse the matching tokens automatically.
    full_prompt = f"{cached_prefix}\n\n{prompt}" if cached_prefix else prompt
    payload = {
        "model": cfg.model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.temperature,
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.max_output_tokens,
        },
    }
    if cfg.seed is not None:
        payload["options"]["seed"] = int(cfg.seed)
    if cfg.top_p is not None:
        payload["options"]["top_p"] = float(cfg.top_p)
    try:
        r = requests.post(f"{base}/api/generate", json=payload, timeout=cfg.request_timeout)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise LLMProviderError(
            f"Could not reach Ollama at {base}. Make sure Ollama is running "
            f"(`ollama serve`).\nDetail: {exc}"
        ) from exc
    data = r.json()
    run_manifest.record("model_reported", model=data.get("model"))
    text = (data.get("response") or "").strip()
    if text:
        return text

    run_manifest.record("empty_reply", provider="ollama")
    raise LLMProviderError(
        f"Ollama returned an empty response for model {cfg.model!r} "
        f"(done_reason={data.get('done_reason')!r}). "
        "Usually this means the model is not pulled yet (`ollama pull "
        f"{cfg.model}`), or the paper text exceeded the model's context window "
        f"(num_ctx={cfg.num_ctx}) — lower the chunk character budget in the sidebar."
    )


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

_ANTHROPIC_URL_DEFAULT = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION     = "2023-06-01"


def _generate_anthropic(cfg: LLMConfig, prompt: str, cached_prefix: Optional[str] = None) -> str:
    api_key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMAuthError(
            "Anthropic API key missing. Provide one in the sidebar (Stage 1 and "
            "Stage 2 have separate fields) or set ANTHROPIC_API_KEY."
        )
    url = (cfg.base_url or _ANTHROPIC_URL_DEFAULT).rstrip("/")

    # Always include a short JSON-only instruction at the top of the system
    # block. If a cached_prefix is provided, append it as a second system
    # block marked for caching — Anthropic dedupes by exact bytes, so the
    # same prefix on subsequent calls hits the cache.
    system_blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are a JSON API. Respond with ONLY a single valid JSON value "
                "(object or array). No prose, no markdown fences, no explanation."
            ),
        }
    ]
    if cached_prefix:
        system_blocks.append({
            "type": "text",
            "text": cached_prefix,
            "cache_control": {"type": "ephemeral"},
        })

    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_output_tokens,
        "temperature": cfg.temperature,
        "system": system_blocks,
        "messages": [{"role": "user", "content": prompt}],
    }

    data = _post_json(
        url,
        payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        timeout=cfg.request_timeout,
        provider="anthropic",
        model=cfg.model,
    )
    run_manifest.record("model_reported", model=data.get("model"))

    blocks = data.get("content") or []
    parts: list[str] = []
    block_types: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_types.append(str(block.get("type")))
        if block.get("type") == "text":
            parts.append(block.get("text") or "")

    text = "".join(parts).strip()
    if text:
        return text

    # An empty reply used to be returned as "", which surfaced downstream as
    # the baffling "No JSON found in response" against a blank body. Explain
    # what actually happened instead.
    raise LLMProviderError(
        _describe_empty_reply(
            provider="Anthropic",
            model=cfg.model,
            stop_reason=data.get("stop_reason"),
            block_types=block_types,
            max_tokens=cfg.max_output_tokens,
            usage=data.get("usage") or {},
        )
    )


def _describe_empty_reply(
    *,
    provider: str,
    model: str,
    stop_reason: Optional[str],
    block_types: list[str],
    max_tokens: int,
    usage: dict,
) -> str:
    """Turn an empty model reply into a message that says what to do about it."""
    run_manifest.record("empty_reply", provider=provider, stop_reason=stop_reason)
    reasoning_blocks = [b for b in block_types if "thinking" in b or "reasoning" in b]

    if stop_reason in ("max_tokens", "length"):
        detail = (
            f"{provider} returned no text: the model used its entire "
            f"{max_tokens}-token output budget before writing any answer"
        )
        if reasoning_blocks:
            detail += " (it was all spent on internal reasoning)"
        return (
            detail + ".\n\n"
            "Fixes, in order of preference:\n"
            "  • Use a non-reasoning model for extraction — this pipeline asks "
            "for short structured JSON, which reasoning models are poor value for.\n"
            "  • Raise the per-step token budgets in ker_extractor.py.\n"
            "  • Reduce the chunk character budget in the sidebar so each call "
            "has less to consider."
        )

    if reasoning_blocks:
        return (
            f"{provider} returned only reasoning blocks ({', '.join(sorted(set(reasoning_blocks)))}) "
            f"and no answer text for model {model!r}. Extended thinking leaves no "
            "room for the JSON reply at these token budgets — switch to a "
            "non-reasoning model, or raise the budgets substantially."
        )

    if stop_reason == "refusal":
        # The old text sent the reader to check the PDF, which is the wrong
        # place: a refusal is returned on text that parsed perfectly well, and
        # the corpus that produced this one had six clean pages and 32,000
        # characters. Saying so saves an hour spent re-extracting a file that
        # was never the problem.
        #
        # It said "twice", and this function cannot know that. It is called on
        # every refusal including the first, so the first refusal of a run
        # announced itself as the second. Attempts are the orchestrator's to
        # count, and so is the remedy — it is the layer that knows whether a
        # fallback model is configured. This states the one fact available
        # here and stops.
        return (
            f"{provider} declined to answer for model {model!r}. This is a "
            "safety-classifier false positive rather than a judgement about "
            "the work — peer-reviewed toxicology on neurotoxins trips it "
            "occasionally, and the same model reads the rest of a corpus "
            "without complaint. It is a property of the model, not of the "
            "paper or of the extracted text."
        )

    return (
        f"{provider} returned an empty response for model {model!r} "
        f"(stop_reason={stop_reason!r}, blocks={block_types or 'none'}, usage={usage}). "
        "Verify the model name is correct and currently available."
    )


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------

_OPENAI_URL_DEFAULT = "https://api.openai.com/v1/chat/completions"


def _generate_openai(cfg: LLMConfig, prompt: str, cached_prefix: Optional[str] = None) -> str:
    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMAuthError(
            "OpenAI API key missing. Provide one in the sidebar (Stage 1 and "
            "Stage 2 have separate fields) or set OPENAI_API_KEY."
        )
    url = (cfg.base_url or _OPENAI_URL_DEFAULT).rstrip("/")

    # Build a single stable system message: JSON-only instruction first, then
    # the long cached_prefix (paper text). OpenAI's automatic prefix cache
    # keys on the byte-identical leading tokens of the request, so a stable
    # message order is what triggers cache hits across the 6 step calls.
    system_text = (
        "You are a JSON API. Respond with ONLY a single valid JSON object. "
        "No prose, no markdown fences, no explanation."
    )
    if cached_prefix:
        system_text = f"{system_text}\n\n{cached_prefix}"

    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
    }
    if cfg.seed is not None:
        payload["seed"] = int(cfg.seed)
    if cfg.top_p is not None:
        payload["top_p"] = float(cfg.top_p)
    data = _post_json(
        url,
        payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=cfg.request_timeout,
        provider="openai",
        model=cfg.model,
    )
    run_manifest.record("model_reported", model=data.get("model"))

    choices = data.get("choices") or []
    if not choices:
        raise LLMProviderError(
            f"OpenAI returned no choices for model {cfg.model!r}. "
            "Verify the model name is correct and available on your account."
        )

    choice = choices[0]
    msg = choice.get("message") or {}

    refusal = msg.get("refusal")
    if refusal:
        raise LLMProviderError(f"OpenAI declined to answer: {refusal}")

    content = (msg.get("content") or "").strip()
    if content:
        return content

    raise LLMProviderError(
        _describe_empty_reply(
            provider="OpenAI",
            model=cfg.model,
            stop_reason=choice.get("finish_reason"),
            block_types=["reasoning"] if (data.get("usage") or {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens") else [],
            max_tokens=cfg.max_output_tokens,
            usage=data.get("usage") or {},
        )
    )


__all__ = ["LLMConfig", "LLMProviderError", "LLMAuthError"]
