from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import requests

from schemas import PubMedRecord, ScreeningDecision

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class ScreeningError(RuntimeError):
    pass


def _criteria_text(criteria: Optional[str]) -> str:
    text = (criteria or "").strip()
    return text if text else "None provided"


def _build_prompt(record: PubMedRecord, query: str, inclusion_criteria: Optional[str], exclusion_criteria: Optional[str]) -> str:
    return f"""
You are screening biomedical literature for downstream AOP-related review.

User PubMed query:
{query}

Inclusion criteria:
{_criteria_text(inclusion_criteria)}

Exclusion criteria:
{_criteria_text(exclusion_criteria)}

Decision rules:
- yes = clearly relevant to the user query and likely useful for downstream toxicity, homeostasis, or mechanistic evidence review.
- no = clearly irrelevant or excluded.
- maybe = partially relevant, uncertain, too broad, or missing enough detail.
- If no criteria are provided, decide from the semantic relevance of the title and abstract to the query and to toxicity/homeostasis/mechanistic biology.
- Use only the title and abstract. Do not invent facts.
- The evidence_quote must be a short verbatim quote copied from the title or abstract when possible.

Title:
{record.title}

Abstract:
{record.abstract or '[No abstract available]'}

Return ONLY valid JSON matching this schema:
{{
  "decision": "yes|no|maybe",
  "rationale": "short explanation",
  "triggered_inclusion_rule": "string or null",
  "triggered_exclusion_rule": "string or null",
  "evidence_quote": "short quote or null"
}}
""".strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def screen_record(
    record: PubMedRecord,
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
    model: Optional[str] = None,
) -> ScreeningDecision:
    prompt = _build_prompt(record, query, inclusion_criteria, exclusion_criteria)
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()
    raw = result.get("response", "")

    try:
        parsed = _parse_json(raw)
    except Exception as e:
        raise ScreeningError(f"Could not parse Ollama JSON response: {e}\nRaw response: {raw[:500]}") from e

    decision = str(parsed.get("decision", "maybe")).strip().lower()
    if decision not in {"yes", "no", "maybe"}:
        decision = "maybe"

    return ScreeningDecision(
        decision=decision,
        rationale=str(parsed.get("rationale", "")).strip(),
        triggered_inclusion_rule=(str(parsed.get("triggered_inclusion_rule")).strip() if parsed.get("triggered_inclusion_rule") not in (None, "", "null") else None),
        triggered_exclusion_rule=(str(parsed.get("triggered_exclusion_rule")).strip() if parsed.get("triggered_exclusion_rule") not in (None, "", "null") else None),
        evidence_quote=(str(parsed.get("evidence_quote")).strip() if parsed.get("evidence_quote") not in (None, "", "null") else None),
    )


def screen_records(
    records: Iterable[PubMedRecord],
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
    model: Optional[str] = None,
):
    for record in records:
        yield record, screen_record(
            record=record,
            query=query,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            model=model,
        )
