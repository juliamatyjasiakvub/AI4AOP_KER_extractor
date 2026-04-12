from __future__ import annotations

"""
Send full paper text to the Anthropic API and parse the returned KER JSON.

This module is intentionally kept thin — prompt logic lives in
EXTRACTION_SYSTEM_PROMPT / _build_user_prompt(), and all schema validation
logic lives in _validate_and_coerce().
"""

import json
import os
from typing import Optional

import anthropic

from schemas import KERExtraction

# ---------------------------------------------------------------------------
# Enums — must match the prompt and schemas exactly
# ---------------------------------------------------------------------------

KE_LEVELS = {"MIE", "Molecular", "Cellular", "Tissue", "Organ", "Individual", "Population"}
KER_ADJACENCY = {"Adjacent", "Non-adjacent"}
PAPER_TYPES = {"Primary study", "Review / meta-analysis", "In silico"}
STUDY_DESIGNS = {"In vivo", "In vitro", "In silico", "Ex vivo", "Epidemiological", "Review / meta-analysis"}
SEX_VALUES = {"Male", "Female", "Mixed", "Not specified"}
CONFIDENCE_VALUES = {"High", "Medium", "Low"}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a specialist in adverse outcome pathway (AOP) toxicology and the AOP-Wiki \
data model. Your task is to read scientific papers and extract structured data about \
Key Event Relationships (KERs) according to the OECD AOP Developer's Handbook schema.

OUTPUT RULES — follow these exactly:
1. Return ONLY a valid JSON object. No preamble, no explanation, no markdown fences.
2. The JSON must have exactly one top-level key: "kers", whose value is an array.
3. Each element of the array is one KER found in the paper. Return ALL KERs.
4. If no KERs are extractable, return: {"kers": []}
5. Every field listed below MUST appear in every KER object. Use null for fields \
that cannot be determined — never omit a field.
6. Use the EXACT enum strings specified — do not paraphrase or abbreviate.
7. Do not invent data. If unsupported by the paper, return null.
"""


def _build_user_prompt(paper_text: str) -> str:
    return f"""Extract all Key Event Relationships (KERs) from the paper below.

<paper>
{paper_text}
</paper>

Return a JSON object with this exact structure. Every KER object must contain \
ALL of the following fields (use null where the paper does not provide information):

{{
  "kers": [
    {{
      "upstream_ke_name": "string — canonical KE name",
      "upstream_ke_level": "one of: MIE | Molecular | Cellular | Tissue | Organ | Individual | Population",
      "downstream_ke_name": "string — canonical KE name",
      "downstream_ke_level": "one of: MIE | Molecular | Cellular | Tissue | Organ | Individual | Population",
      "ker_name": "string — '<upstream> leads to <downstream>'",
      "ker_description": "string — 1-3 sentence mechanistic description",
      "ker_adjacency": "one of: Adjacent | Non-adjacent",
      "paper_type": "one of: Primary study | Review / meta-analysis | In silico",
      "cited_evidence_dois": "semicolon-separated DOIs or null (null for primary studies)",
      "biological_plausibility": "string or null",
      "empirical_evidence_summary": "string or null",
      "essentiality_evidence": "string or null",
      "contradicts_ker": true or false (never null),
      "taxonomic_applicability": "NCBI scientific name(s), semicolon-separated, or 'Not specified'",
      "sex_applicability": "one of: Male | Female | Mixed | Not specified",
      "life_stage_applicability": "string e.g. Adult, Embryonic, Not specified",
      "modulating_factors": "string or null",
      "quantitative_relationships": "string or null",
      "response_response_relationship": "string or null",
      "time_scale": "string or null",
      "feedforward_feedback_loops": "string or null",
      "study_design": "one of: In vivo | In vitro | In silico | Ex vivo | Epidemiological | Review / meta-analysis",
      "exposure_route": "string or null",
      "chemical_stressor": "string or null",
      "extraction_confidence": "one of: High | Medium | Low"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ExtractionValidationError(ValueError):
    pass


def _coerce_enum(value: object, allowed: set[str], field: str, fallback: Optional[str] = None) -> Optional[str]:
    """Return value if valid, fallback if provided, else raise."""
    if value is None:
        return fallback
    s = str(value).strip()
    if s in allowed:
        return s
    # case-insensitive match
    for candidate in allowed:
        if candidate.lower() == s.lower():
            return candidate
    if fallback is not None:
        return fallback
    raise ExtractionValidationError(f"Field '{field}' has invalid value '{s}'. Allowed: {allowed}")


def _validate_and_coerce(raw: dict) -> KERExtraction:
    """Validate one raw KER dict from LLM output and return a KERExtraction."""

    def req_str(key: str) -> str:
        v = raw.get(key)
        if not v or not str(v).strip():
            raise ExtractionValidationError(f"Required field '{key}' is missing or empty.")
        return str(v).strip()

    def opt_str(key: str) -> Optional[str]:
        v = raw.get(key)
        if v is None or str(v).strip().lower() in ("null", "none", ""):
            return None
        return str(v).strip()

    def req_bool(key: str) -> bool:
        v = raw.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.strip().lower() in ("true", "1", "yes"):
                return True
            if v.strip().lower() in ("false", "0", "no"):
                return False
        raise ExtractionValidationError(f"Field '{key}' must be true or false, got: {v!r}")

    return KERExtraction(
        upstream_ke_name=req_str("upstream_ke_name"),
        upstream_ke_level=_coerce_enum(req_str("upstream_ke_level"), KE_LEVELS, "upstream_ke_level"),
        downstream_ke_name=req_str("downstream_ke_name"),
        downstream_ke_level=_coerce_enum(req_str("downstream_ke_level"), KE_LEVELS, "downstream_ke_level"),
        ker_name=req_str("ker_name"),
        ker_description=req_str("ker_description"),
        ker_adjacency=_coerce_enum(req_str("ker_adjacency"), KER_ADJACENCY, "ker_adjacency", fallback="Adjacent"),
        paper_type=_coerce_enum(req_str("paper_type"), PAPER_TYPES, "paper_type", fallback="Primary study"),
        cited_evidence_dois=opt_str("cited_evidence_dois"),
        biological_plausibility=opt_str("biological_plausibility"),
        empirical_evidence_summary=opt_str("empirical_evidence_summary"),
        essentiality_evidence=opt_str("essentiality_evidence"),
        contradicts_ker=req_bool("contradicts_ker"),
        taxonomic_applicability=req_str("taxonomic_applicability"),
        sex_applicability=_coerce_enum(raw.get("sex_applicability"), SEX_VALUES, "sex_applicability", fallback="Not specified"),
        life_stage_applicability=req_str("life_stage_applicability"),
        modulating_factors=opt_str("modulating_factors"),
        quantitative_relationships=opt_str("quantitative_relationships"),
        response_response_relationship=opt_str("response_response_relationship"),
        time_scale=opt_str("time_scale"),
        feedforward_feedback_loops=opt_str("feedforward_feedback_loops"),
        study_design=_coerce_enum(req_str("study_design"), STUDY_DESIGNS, "study_design", fallback="In vivo"),
        exposure_route=opt_str("exposure_route"),
        chemical_stressor=opt_str("chemical_stressor"),
        extraction_confidence=_coerce_enum(raw.get("extraction_confidence"), CONFIDENCE_VALUES, "extraction_confidence", fallback="Low"),
    )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    pass


def extract_kers_from_text(
    paper_text: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[list[KERExtraction], list[str]]:
    """
    Send paper text to Claude and return validated KERExtraction objects.

    Returns
    -------
    extractions : list[KERExtraction]
        Successfully parsed and validated KER objects.
    warnings : list[str]
        Any per-KER validation errors (skipped KERs) or soft warnings.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ExtractionError(
            "ANTHROPIC_API_KEY not set. Add it to your environment or enter it in the sidebar."
        )

    client = anthropic.Anthropic(api_key=key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(paper_text)}],
        )
    except anthropic.APIError as exc:
        raise ExtractionError(f"Anthropic API error: {exc}") from exc

    raw_text = message.content[0].text.strip()

    # Strip markdown fences if model wraps output despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"LLM did not return valid JSON.\nError: {exc}\nRaw (first 500 chars): {raw_text[:500]}"
        ) from exc

    raw_kers = payload.get("kers", [])
    if not isinstance(raw_kers, list):
        raise ExtractionError(f"Expected 'kers' to be a list, got: {type(raw_kers)}")

    extractions: list[KERExtraction] = []
    warnings: list[str] = []

    for i, raw_ker in enumerate(raw_kers):
        try:
            extractions.append(_validate_and_coerce(raw_ker))
        except ExtractionValidationError as exc:
            warnings.append(f"KER {i+1} skipped — validation error: {exc}")

    return extractions, warnings
