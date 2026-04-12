from __future__ import annotations

"""
Send full paper text to a local Ollama model and parse the returned KER JSON.

Uses the same /api/generate endpoint as screening.py — no new dependencies.

Swapping to Anthropic later: replace _call_ollama() with _call_anthropic()
and update extract_kers_from_text() to call it instead.
"""

import json
import os
from typing import Optional

import requests

from schemas import KERExtraction

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
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a specialist in adverse outcome pathway (AOP) toxicology and the AOP-Wiki "
    "data model. Extract Key Event Relationships (KERs) from scientific papers.\n\n"
    "STRICT OUTPUT RULES:\n"
    "1. Return ONLY valid JSON. No explanation, no markdown fences, no extra text.\n"
    "2. Top-level key must be 'kers' whose value is an array.\n"
    "3. Every field must appear in every KER object. Use JSON null for unknown fields.\n"
    "4. Use EXACT enum strings listed below.\n"
    "5. Never invent data not present in the paper.\n"
)

_FIELD_SPEC = (
    "Each KER object must have ALL these fields:\n"
    "upstream_ke_name            : string\n"
    "upstream_ke_level           : MIE|Molecular|Cellular|Tissue|Organ|Individual|Population\n"
    "downstream_ke_name          : string\n"
    "downstream_ke_level         : same enum\n"
    "ker_name                    : string ('<upstream> leads to <downstream>')\n"
    "ker_description             : string (1-3 sentences mechanistic basis)\n"
    "ker_adjacency               : Adjacent|Non-adjacent\n"
    "paper_type                  : 'Primary study'|'Review / meta-analysis'|'In silico'\n"
    "cited_evidence_dois         : semicolon-separated DOIs or null\n"
    "biological_plausibility     : string or null\n"
    "empirical_evidence_summary  : string or null\n"
    "essentiality_evidence       : string or null\n"
    "contradicts_ker             : true or false (NEVER null)\n"
    "taxonomic_applicability     : NCBI species name(s) or 'Not specified'\n"
    "sex_applicability           : Male|Female|Mixed|'Not specified'\n"
    "life_stage_applicability    : string\n"
    "modulating_factors          : string or null\n"
    "quantitative_relationships  : string or null\n"
    "response_response_relationship : string or null\n"
    "time_scale                  : string or null\n"
    "feedforward_feedback_loops  : string or null\n"
    "study_design                : 'In vivo'|'In vitro'|'In silico'|'Ex vivo'|Epidemiological|'Review / meta-analysis'\n"
    "exposure_route              : string or null\n"
    "chemical_stressor           : string or null\n"
    "extraction_confidence       : High|Medium|Low\n"
)


def _build_prompt(paper_text: str) -> str:
    return (
        f"{_SYSTEM}\n{_FIELD_SPEC}\n"
        f"Extract all KERs from this paper and return JSON object "
        f'with key "kers" containing an array:\n\nPAPER:\n{paper_text}\n\nJSON:'
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    pass


def _call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 16384},
    }
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        raise ExtractionError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Make sure Ollama is running (`ollama serve`).\nDetail: {exc}"
        ) from exc
    return r.json().get("response", "")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ExtractionError(
            f"No JSON object in model response. First 300 chars:\n{raw[:300]}"
        )
    text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Invalid JSON from model: {exc}\nRaw (500 chars):\n{raw[:500]}"
        ) from exc
    kers = payload.get("kers", [])
    if not isinstance(kers, list):
        raise ExtractionError(f"'kers' should be a list, got {type(kers)}")
    return kers


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ExtractionValidationError(ValueError):
    pass


def _coerce_enum(value, allowed: set, field: str, fallback: Optional[str] = None) -> Optional[str]:
    if value is None:
        return fallback
    s = str(value).strip()
    if s in allowed:
        return s
    for candidate in allowed:
        if candidate.lower() == s.lower():
            return candidate
    return fallback


def _validate_and_coerce(raw: dict) -> KERExtraction:
    def req_str(key: str) -> str:
        v = raw.get(key)
        if not v or not str(v).strip():
            raise ExtractionValidationError(f"Required field '{key}' missing or empty.")
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
        if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
            return True
        return False

    return KERExtraction(
        upstream_ke_name               = req_str("upstream_ke_name"),
        upstream_ke_level              = _coerce_enum(raw.get("upstream_ke_level"), KE_LEVELS, "upstream_ke_level", "Molecular"),
        downstream_ke_name             = req_str("downstream_ke_name"),
        downstream_ke_level            = _coerce_enum(raw.get("downstream_ke_level"), KE_LEVELS, "downstream_ke_level", "Molecular"),
        ker_name                       = req_str("ker_name"),
        ker_description                = req_str("ker_description"),
        ker_adjacency                  = _coerce_enum(raw.get("ker_adjacency"), KER_ADJACENCY, "ker_adjacency", "Adjacent"),
        paper_type                     = _coerce_enum(raw.get("paper_type"), PAPER_TYPES, "paper_type", "Primary study"),
        cited_evidence_dois            = opt_str("cited_evidence_dois"),
        biological_plausibility        = opt_str("biological_plausibility"),
        empirical_evidence_summary     = opt_str("empirical_evidence_summary"),
        essentiality_evidence          = opt_str("essentiality_evidence"),
        contradicts_ker                = req_bool("contradicts_ker"),
        taxonomic_applicability        = raw.get("taxonomic_applicability") or "Not specified",
        sex_applicability              = _coerce_enum(raw.get("sex_applicability"), SEX_VALUES, "sex_applicability", "Not specified"),
        life_stage_applicability       = raw.get("life_stage_applicability") or "Not specified",
        modulating_factors             = opt_str("modulating_factors"),
        quantitative_relationships     = opt_str("quantitative_relationships"),
        response_response_relationship = opt_str("response_response_relationship"),
        time_scale                     = opt_str("time_scale"),
        feedforward_feedback_loops     = opt_str("feedforward_feedback_loops"),
        study_design                   = _coerce_enum(raw.get("study_design"), STUDY_DESIGNS, "study_design", "In vivo"),
        exposure_route                 = opt_str("exposure_route"),
        chemical_stressor              = opt_str("chemical_stressor"),
        extraction_confidence          = _coerce_enum(raw.get("extraction_confidence"), CONFIDENCE_VALUES, "extraction_confidence", "Low"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_kers_from_text(
    paper_text: str,
    model: str = "llama3.1:8b",
    ollama_url: Optional[str] = None,
) -> tuple[list[KERExtraction], list[str]]:
    """
    Extract KERs from paper text using a local Ollama model.

    Returns (extractions, warnings) where warnings are skipped-KER messages.
    """
    if ollama_url:
        global OLLAMA_URL
        OLLAMA_URL = ollama_url

    raw_text = _call_ollama(_build_prompt(paper_text), model)
    raw_kers = _parse_response(raw_text)

    extractions: list[KERExtraction] = []
    warnings:    list[str]           = []

    for i, raw_ker in enumerate(raw_kers):
        try:
            extractions.append(_validate_and_coerce(raw_ker))
        except ExtractionValidationError as exc:
            warnings.append(f"KER {i+1} skipped — {exc}")

    if not extractions and not warnings:
        warnings.append(
            "Model returned an empty KER list. "
            "The paper may lack mechanistic content, or try a larger model (e.g. llama3.1:70b)."
        )

    return extractions, warnings
