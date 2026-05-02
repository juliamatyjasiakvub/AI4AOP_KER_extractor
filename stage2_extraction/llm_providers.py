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
    try:
        r = requests.post(f"{base}/api/generate", json=payload, timeout=cfg.request_timeout)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise LLMProviderError(
            f"Could not reach Ollama at {base}. Make sure Ollama is running "
            f"(`ollama serve`).\nDetail: {exc}"
        ) from exc
    return r.json().get("response", "")


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

_ANTHROPIC_URL_DEFAULT = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION     = "2023-06-01"


def _generate_anthropic(cfg: LLMConfig, prompt: str, cached_prefix: Optional[str] = None) -> str:
    api_key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "Anthropic API key missing. Provide one in the sidebar or set ANTHROPIC_API_KEY."
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
    try:
        r = requests.post(
            url,
            json=payload,
            timeout=cfg.request_timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        r.raise_for_status()
    except requests.HTTPError as exc:
        # Surface Anthropic's error body for clearer debugging.
        body = exc.response.text if exc.response is not None else ""
        raise LLMProviderError(f"Anthropic HTTP {exc.response.status_code}: {body[:500]}") from exc
    except requests.RequestException as exc:
        raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

    data = r.json()
    blocks = data.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------

_OPENAI_URL_DEFAULT = "https://api.openai.com/v1/chat/completions"


def _generate_openai(cfg: LLMConfig, prompt: str, cached_prefix: Optional[str] = None) -> str:
    api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderError(
            "OpenAI API key missing. Provide one in the sidebar or set OPENAI_API_KEY."
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
    try:
        r = requests.post(
            url,
            json=payload,
            timeout=cfg.request_timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        raise LLMProviderError(f"OpenAI HTTP {exc.response.status_code}: {body[:500]}") from exc
    except requests.RequestException as exc:
        raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


__all__ = ["LLMConfig", "LLMProviderError"]
