from __future__ import annotations

import json
import os
from typing import Iterable

from openai import OpenAI

from schemas import PubMedRecord, ScreenedRecord, ScreeningDecision

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class ScreeningError(RuntimeError):
    pass


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_screening_prompt(
    query: str,
    title: str,
    abstract: str,
    inclusion_criteria: str | None = None,
    exclusion_criteria: str | None = None,
) -> str:
    inclusion = inclusion_criteria.strip() if inclusion_criteria else "None provided"
    exclusion = exclusion_criteria.strip() if exclusion_criteria else "None provided"
    abstract = abstract.strip() if abstract else "No abstract provided. Base the decision on the title alone and be conservative."

    return f"""
You are screening biomedical literature for downstream toxicology and AOP-related review.

User PubMed query:
{query}

Inclusion criteria:
{inclusion}

Exclusion criteria:
{exclusion}

Title:
{title}

Abstract:
{abstract}

Task:
Classify this paper as one of:
- yes = clearly relevant
- no = clearly irrelevant
- maybe = partially relevant, ambiguous, or insufficient detail

Decision rules:
1. Apply user-defined inclusion and exclusion criteria if they are provided.
2. If no criteria are provided, judge whether the paper is genuinely relevant to the PubMed query and to toxicity, homeostasis, mechanistic biology, perturbation, or event-level evidence.
3. Do not invent facts not present in the title or abstract.
4. Keep the rationale concise.
5. The evidence_quote must be a short exact quote copied from the title or abstract when possible.

Return valid JSON only with this exact shape:
{{
  "decision": "yes|no|maybe",
  "rationale": "...",
  "triggered_inclusion_rule": "... or null",
  "triggered_exclusion_rule": "... or null",
  "evidence_quote": "... or null"
}}
""".strip()


def screen_record(
    record: PubMedRecord,
    query: str,
    inclusion_criteria: str | None = None,
    exclusion_criteria: str | None = None,
    model: str = DEFAULT_MODEL,
) -> ScreeningDecision:
    prompt = build_screening_prompt(
        query=query,
        title=record.title,
        abstract=record.abstract,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
    )

    try:
        response = _client().responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
    except Exception as exc:
        raise ScreeningError(f"LLM screening request failed for PMID {record.pmid}: {exc}") from exc

    text = getattr(response, "output_text", "") or ""
    if not text:
        raise ScreeningError(f"Empty LLM response for PMID {record.pmid}.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_block(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScreeningError(f"Could not parse JSON for PMID {record.pmid}: {text}") from exc

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"yes", "no", "maybe"}:
        raise ScreeningError(f"Invalid decision '{decision}' for PMID {record.pmid}.")

    return ScreeningDecision(
        decision=decision,  # type: ignore[arg-type]
        rationale=_clean_optional_text(payload.get("rationale")) or "No rationale provided.",
        triggered_inclusion_rule=_clean_optional_text(payload.get("triggered_inclusion_rule")),
        triggered_exclusion_rule=_clean_optional_text(payload.get("triggered_exclusion_rule")),
        evidence_quote=_clean_optional_text(payload.get("evidence_quote")),
    )


def screen_records(
    records: Iterable[PubMedRecord],
    query: str,
    inclusion_criteria: str | None = None,
    exclusion_criteria: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[ScreenedRecord]:
    screened: list[ScreenedRecord] = []
    for record in records:
        decision = screen_record(
            record=record,
            query=query,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            model=model,
        )
        screened.append(ScreenedRecord.from_parts(record, decision))
    return screened


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text
