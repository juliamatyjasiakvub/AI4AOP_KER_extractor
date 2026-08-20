from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import requests

from json_repair import TRUNCATED_KEY, extract_json
from schemas import PubMedRecord, ScreeningDecision
from stage2_extraction.llm_providers import LLMConfig

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class ScreeningError(RuntimeError):
    pass


#: What to do when a record satisfies an inclusion criterion AND an exclusion
#: criterion at once. This happens constantly in toxicology — an in vivo rodent
#: study with cell-culture mechanistic follow-up matches "mammalian studies"
#: and "in vitro data" simultaneously — and leaving it unspecified means the
#: model decides differently from one record to the next.
CONFLICT_POLICIES: dict[str, str] = {
    "maybe": (
        "If the record satisfies BOTH an inclusion criterion and an exclusion "
        "criterion, you MUST return \"maybe\" and name both rules in "
        "triggered_inclusion_rule and triggered_exclusion_rule. Do not pick a "
        "side. A human will adjudicate."
    ),
    "exclude": (
        "If the record satisfies BOTH an inclusion criterion and an exclusion "
        "criterion, exclusion takes precedence: return \"no\", and name both "
        "rules in triggered_inclusion_rule and triggered_exclusion_rule."
    ),
    "include": (
        "If the record satisfies BOTH an inclusion criterion and an exclusion "
        "criterion, inclusion takes precedence: return \"yes\", and name both "
        "rules in triggered_inclusion_rule and triggered_exclusion_rule."
    ),
}

DEFAULT_CONFLICT_POLICY = "maybe"


def _criteria_text(criteria: Optional[str]) -> str:
    text = (criteria or "").strip()
    return text if text else "None provided"


def _build_prompt(
    record: PubMedRecord,
    query: str,
    inclusion_criteria: Optional[str],
    exclusion_criteria: Optional[str],
    conflict_policy: str = DEFAULT_CONFLICT_POLICY,
) -> str:
    conflict_rule = CONFLICT_POLICIES.get(conflict_policy, CONFLICT_POLICIES[DEFAULT_CONFLICT_POLICY])

    # Same framing as the Stage 2 prefix, for the same reason: the task
    # arrives as a bare instruction otherwise. Screening sees only titles and
    # abstracts, but those abstracts are drawn from the same toxicology
    # literature, and a screener that does not know it is doing systematic
    # review cannot weigh "relevant" properly either.
    return f"""
CONTEXT FOR THIS TASK
You are performing title-and-abstract screening for a systematic literature
review, the first stage of building an Adverse Outcome Pathway. An AOP is the
OECD framework for organising mechanistic evidence about how a stressor causes
harm; regulators use it to assess hazard and to reduce animal testing.

The records below are PubMed search results — published, peer-reviewed
biomedical literature. Your job is to decide whether each one is in scope for
the review. You are recording a screening decision about a citation, not
advising on any substance. This literature routinely concerns neurotoxins,
metals, pesticides and drugs, because characterising how those cause harm is
what the field is for; screen such records exactly as you would any other.

User PubMed query:
{query}

Inclusion criteria:
{_criteria_text(inclusion_criteria)}

Exclusion criteria:
{_criteria_text(exclusion_criteria)}

Decision rules:
- yes = clearly relevant to the user query and likely useful for downstream toxicity, homeostasis, or mechanistic evidence review.
- no = clearly irrelevant, or excluded with no offsetting inclusion criterion.
- maybe = partially relevant, uncertain, too broad, or missing enough detail.
- If no criteria are provided, decide from the semantic relevance of the title and abstract to the query and to toxicity/homeostasis/mechanistic biology.

Handling conflicts between criteria:
- {conflict_rule}
- Set criteria_conflict to true whenever both an inclusion and an exclusion criterion apply, whatever decision you reach.

Judging from limited text:
- You see ONLY the title and abstract, not the full paper. An abstract that does not mention a study type is NOT evidence that the study type is absent.
- Mark a criterion as triggered only when the title or abstract gives positive evidence for it. Never infer a criterion is met from silence.
- If an exclusion criterion might apply but the abstract is too thin to tell, return "maybe" rather than "no".
- Do not invent facts.
- The evidence_quote must be a short verbatim quote copied from the title or abstract when possible.

Length limits (your reply is cut off at a fixed token budget — exceed these and
the answer is lost):
- rationale: at most 40 words. State the verdict's reason, not a full appraisal.
- triggered_inclusion_rule / triggered_exclusion_rule: quote the criterion itself, not an argument about it.
- evidence_quote: one sentence, at most 30 words.

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
  "criteria_conflict": true_or_false,
  "evidence_quote": "short quote or null"
}}
""".strip()


def _parse_json(text: str) -> dict:
    """
    Parse the screening reply, recovering responses cut off by the token limit.

    A verbose rationale can exhaust the output budget mid-sentence, leaving the
    JSON unterminated. The decision itself is written first, so the useful part
    has almost always arrived by the time the text is cut — discarding it over a
    missing brace would throw away a perfectly good verdict.
    """
    parsed = extract_json(text, context="screening response")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _opt(parsed: dict, key: str) -> Optional[str]:
    value = parsed.get(key)
    if value in (None, "", "null"):
        return None
    text = str(value).strip()
    return text or None


#: Decision each policy forces when both kinds of criteria are satisfied.
_POLICY_DECISION = {"maybe": "maybe", "exclude": "no", "include": "yes"}


def screen_record(
    record: PubMedRecord,
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
    llm_cfg: Optional[LLMConfig] = None,
    conflict_policy: str = DEFAULT_CONFLICT_POLICY,
) -> ScreeningDecision:
    if conflict_policy not in CONFLICT_POLICIES:
        conflict_policy = DEFAULT_CONFLICT_POLICY

    prompt = _build_prompt(
        record, query, inclusion_criteria, exclusion_criteria, conflict_policy
    )

    # Use provided LLMConfig or create default Ollama config
    if llm_cfg is None:
        llm_cfg = LLMConfig(
            provider="ollama",
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        )

    raw = llm_cfg.generate(prompt)

    try:
        parsed = _parse_json(raw)
    except Exception as e:
        raise ScreeningError(f"Could not parse LLM JSON response: {e}\nRaw response: {raw[:500]}") from e

    decision = str(parsed.get("decision", "maybe")).strip().lower()
    if decision not in {"yes", "no", "maybe"}:
        decision = "maybe"

    inclusion_rule = _opt(parsed, "triggered_inclusion_rule")
    exclusion_rule = _opt(parsed, "triggered_exclusion_rule")
    rationale = str(parsed.get("rationale", "")).strip()

    if parsed.pop(TRUNCATED_KEY, False):
        rationale = (rationale + " […reply truncated at the token limit]").strip()

    # A conflict is established by the evidence — both rule slots filled —
    # rather than by the model's own boolean, which it often forgets to set.
    conflict = bool(inclusion_rule and exclusion_rule) or bool(
        parsed.get("criteria_conflict") is True and inclusion_rule and exclusion_rule
    )

    # Enforce the policy here rather than relying on the prompt. The whole
    # point of the setting is that identical evidence produces an identical
    # verdict every time, and instruction-following alone does not guarantee
    # that across models or across runs.
    if conflict:
        forced = _POLICY_DECISION[conflict_policy]
        if decision != forced:
            rationale = (
                f"{rationale} [Both criteria apply; resolved to '{forced}' by the "
                f"'{conflict_policy}' conflict policy.]"
            ).strip()
            decision = forced

    return ScreeningDecision(
        decision=decision,
        rationale=rationale,
        triggered_inclusion_rule=inclusion_rule,
        triggered_exclusion_rule=exclusion_rule,
        evidence_quote=_opt(parsed, "evidence_quote"),
        criteria_conflict=conflict,
        conflict_policy=conflict_policy if conflict else None,
    )


def screen_records(
    records: Iterable[PubMedRecord],
    query: str,
    inclusion_criteria: Optional[str] = None,
    exclusion_criteria: Optional[str] = None,
    llm_cfg: Optional[LLMConfig] = None,
    conflict_policy: str = DEFAULT_CONFLICT_POLICY,
):
    for record in records:
        yield record, screen_record(
            record=record,
            query=query,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            llm_cfg=llm_cfg,
            conflict_policy=conflict_policy,
        )
