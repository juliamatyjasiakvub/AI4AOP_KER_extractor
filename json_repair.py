from __future__ import annotations

"""
Tolerant JSON extraction from LLM responses.

Models return JSON wrapped in markdown fences, prefixed with prose, and — most
awkwardly — cut off partway through when they hit the output-token ceiling.
A response truncated mid-string has no closing brace, so a plain `json.loads`
fails and the entire result is discarded even though most of the fields
arrived intact.

This module recovers what it can:

    extract_json(raw)              -> parsed value, or raises ValueError
    repair_truncated_json(text)    -> repaired JSON text, or None

When a value is recovered from a truncated response, dict results carry a
`_truncated: True` marker so the caller can warn the user that later fields are
missing rather than genuinely absent from the source.

Shared by Stage 1 screening and Stage 2 extraction — both hit exactly the same
failure mode, and a fix in one is worthless if the other keeps its own parser.
"""

import json
from typing import Any, Optional

import run_manifest

__all__ = ["extract_json", "repair_truncated_json", "scan_json", "TRUNCATED_KEY"]

#: Key injected into recovered dicts to mark them as incomplete.
TRUNCATED_KEY = "_truncated"


def _scan(text: str) -> tuple[list[str], bool, list[int], list[int]]:
    """
    Walk `text` tracking bracket depth and string state.

    Returns (open_closers, ended_inside_string, safe_cuts, container_cuts).
    `safe_cuts` are every index at which the text could be truncated without
    splitting a value in half. `container_cuts` are the subset that fall just
    past a *complete* object or array — the only places where cutting cannot
    leave a half-built record behind.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    safe_cuts: list[int] = []
    container_cuts: list[int] = []

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                safe_cuts.append(i + 1)   # just past a completed string
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe_cuts.append(i + 1)       # just past a completed container
            container_cuts.append(i + 1)
        elif ch == ",":
            safe_cuts.append(i)           # cutting before a comma is always safe

    return stack, in_string, safe_cuts, container_cuts


def scan_json(text: str) -> tuple[list[str], bool, list[int]]:
    """Backwards-compatible view of `_scan`."""
    stack, in_string, safe_cuts, _ = _scan(text)
    return stack, in_string, safe_cuts


def repair_truncated_json(
    fragment: str, *, error_pos: Optional[int] = None
) -> Optional[str]:
    """
    Rebuild parseable JSON from a response that was cut off — or broken — mid-flight.

    Everything before the cut is good data, so rather than discarding the whole
    response we truncate back to a complete value and close the open brackets.

    Choosing the cut point by pattern-matching is unreliable — a trailing
    `"rationale"` might be a key awaiting a value or a completed string value,
    and the two need opposite treatment. So instead of guessing, we walk the
    candidate cut points from latest to earliest and return the first that
    actually parses. Correctness is decided by the JSON parser, not a heuristic.

    `error_pos` extends the same machinery to a *complete but malformed* reply,
    which is the other way a model breaks JSON and the one that used to cost a
    whole paper. A single unescaped quote inside a value —

        "to_event": "Nav1.6 at the so-called "nodal" domain"

    — leaves the brackets perfectly balanced, so the truncation check below
    concluded there was nothing to repair and the reply was discarded whole:
    every step before the bad one, and every step after it, thrown away for one
    stray character in the middle. The decoder consumed everything up to
    `error_pos` successfully, which makes that prefix a valid JSON *prefix* and
    exactly the input this function already knows how to close.

    Returns the repaired text, or None if nothing salvageable remains.
    """
    if error_pos is not None:
        # Only the part the decoder actually accepted. Beyond it the text is
        # being read under a misparse — after an unescaped quote the scanner's
        # idea of what is inside a string is wrong — so nothing there can be
        # trusted to contribute a cut point.
        fragment = fragment[:error_pos]

    stack, in_string, safe_cuts, container_cuts = _scan(fragment)

    if not stack and not in_string and error_pos is None:
        return None  # not actually truncated; nothing to repair

    if error_pos is not None:
        # Whole records only. The damage is *inside* a value, so the record
        # containing it is half-read by definition: cutting at the last
        # complete string would keep it, with that value silently shortened to
        # whatever preceded the stray quote. That is worse than losing the
        # record — a Key Event named "Nav1.6 at the so-called" is a plausible
        # string that no paper contains, and it would be merged, mapped and
        # drawn like any other. Truncation is different and keeps the wider
        # rule: there the cut is where the model stopped, not inside a value it
        # was mid-way through describing correctly.
        safe_cuts = container_cuts

    for cut in reversed(safe_cuts):
        head = fragment[:cut].rstrip().rstrip(",").rstrip()
        if not head:
            continue

        remaining, still_in_string, _ = scan_json(head)
        if still_in_string:
            continue

        candidate = head + "".join(reversed(remaining))
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate

    return None


def extract_json(raw: str, *, context: str = "response") -> Any:
    """
    Best-effort JSON extraction from a possibly-noisy model response.

    Handles markdown fences, leading prose, and responses truncated by the
    output-token limit.

    Raises ValueError with an actionable message when nothing can be recovered.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    obj_start = text.find("{")
    arr_start = text.find("[")

    starts = [i for i in (obj_start, arr_start) if i != -1]
    if not starts:
        raise ValueError(
            f"No JSON found in {context}. First 300 chars:\n{raw[:300]}"
        )

    # Only ever consider the OUTERMOST container. Falling back to a nested one
    # — say, grabbing an inner array out of a truncated object — would return a
    # fragment that parses cleanly while meaning something entirely different
    # from what the model was asked for.
    first = min(starts)
    fragment = text[first:]
    closer = "}" if text[first] == "{" else "]"
    end = fragment.rfind(closer)

    last_err: Optional[Exception] = None
    if end > 0:
        try:
            return json.loads(fragment[: end + 1])
        except json.JSONDecodeError as exc:
            last_err = exc

    # Two shapes of damage, one recovery. First the classic: cut off by the
    # token ceiling, brackets left open. Then the complete-but-malformed reply,
    # salvaged back to wherever the decoder stopped being happy — which is only
    # attempted if we have a position to cut at, because without one there is
    # no principled place to stop.
    attempts: list[tuple[Optional[str], str]] = [
        (repair_truncated_json(fragment), "truncated")
    ]
    error_pos = getattr(last_err, "pos", None)
    if attempts[0][0] is None and isinstance(error_pos, int) and error_pos > 0:
        attempts.append(
            (repair_truncated_json(fragment, error_pos=error_pos), "malformed")
        )

    for repaired, kind in attempts:
        if repaired is None:
            continue
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue

        if isinstance(parsed, dict):
            parsed[TRUNCATED_KEY] = True
        # A salvaged reply is a partial one: the fields after the cut are
        # simply absent. That is worth knowing when judging a run, so the
        # repair is counted rather than passing silently. `kind` separates the
        # two causes because they call for different responses — a truncated
        # reply says raise the token budget, a malformed one says the model
        # wrote invalid JSON and there is nothing to configure.
        run_manifest.record("json_repair", context=context, kind=kind)
        return parsed

    run_manifest.record("json_failure", context=context)

    if end <= 0:
        raise ValueError(
            f"The {context} was truncated before any complete JSON value — the "
            "output-token limit was almost certainly hit. Raise the token "
            f"budget or ask for shorter text.\nFirst 300 chars:\n{raw[:300]}"
        )
    raise ValueError(f"Invalid JSON in {context}: {last_err}\nRaw (500 chars):\n{raw[:500]}")
