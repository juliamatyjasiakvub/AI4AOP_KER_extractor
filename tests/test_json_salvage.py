"""
A model that writes one bad character must not cost a whole paper.

`repair_truncated_json` recovered replies cut off by the output-token ceiling,
which leave brackets open. It refused to look at the other way a model breaks
JSON: a *complete* reply with a stray quote in the middle —

    "to_event": "Nav1.6 at the so-called "nodal" domain"

The brackets balance, so the function returned None at its "not actually
truncated" guard and the whole reply was discarded — every step before the bad
one and every step after it, thrown away for one character. That is how
`40037549.pdf` produced nothing from a reply whose first step was complete and
correct.

The recovery rule differs between the two cases, and the difference matters
more than the recovery. A truncated reply stops where the model stopped, so
cutting at the last complete value keeps good data. A malformed reply is broken
*inside* a value, so the record holding it is half-read: cutting at the last
complete string would keep it with that value silently shortened, and a Key
Event named "Nav1.6 at the so-called" is a plausible-looking string that
appears in no paper and would be merged, mapped and drawn like any other.
Whole records only.
"""

from __future__ import annotations

import json

import pytest

from json_repair import TRUNCATED_KEY, extract_json, repair_truncated_json


MALFORMED = '''{
  "bears_on_question": true,
  "reason": "Nav1.6 co-expression in NG2+ OPCs; observational only.",
  "steps": [
    {"from_event": "Voltage-gated sodium channel", "to_event": "Oligodendrocyte differentiation"},
    {"from_event": "Nav1.6 at the so-called "nodal" domain", "to_event": "Myelination"},
    {"from_event": "Third step", "to_event": "Fourth"}
  ]
}'''


def test_the_reply_really_is_unparseable():
    """Guard the premise: this is a decoder failure, not a test artefact."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(MALFORMED)


def test_a_complete_but_malformed_reply_is_not_thrown_away():
    recovered = extract_json(MALFORMED, context="extraction response")

    assert recovered["bears_on_question"] is True
    assert "observational only" in recovered["reason"]
    assert len(recovered["steps"]) == 1
    assert recovered["steps"][0]["to_event"] == "Oligodendrocyte differentiation"


def test_the_half_read_record_is_dropped_not_salvaged():
    """
    The important half. Keeping the broken record would invent an event name.
    """
    recovered = extract_json(MALFORMED, context="extraction response")
    events = {
        value
        for step in recovered["steps"]
        for value in step.values()
    }
    assert not any("so-called" in e for e in events), (
        "a value cut at the stray quote must never reach the corpus"
    )
    assert "Myelination" not in events


def test_a_salvaged_reply_is_marked_partial():
    """
    Downstream treats this marker as "later fields are absent, not denied by
    the paper". A silent partial parse is the worst outcome of the three.
    """
    recovered = extract_json(MALFORMED, context="extraction response")
    assert recovered.get(TRUNCATED_KEY) is True


def test_truncation_recovery_is_unchanged():
    """
    The original behaviour has to survive. Here the model stopped mid-value
    rather than corrupting one, so the wider cut rule still applies and the
    fields that did arrive are kept.
    """
    cut_off = '{"steps": [{"from_event": "A", "to_event": "B"}, {"from_event": "C", "to_ev'
    recovered = extract_json(cut_off, context="extraction response")

    assert recovered["steps"][0] == {"from_event": "A", "to_event": "B"}
    assert recovered["steps"][1] == {"from_event": "C"}
    assert recovered.get(TRUNCATED_KEY) is True


def test_a_well_formed_reply_is_untouched():
    """No marker, no salvage, no repair counted."""
    clean = '{"steps": [{"from_event": "A", "to_event": "B"}]}'
    recovered = extract_json(clean, context="extraction response")

    assert recovered == {"steps": [{"from_event": "A", "to_event": "B"}]}
    assert TRUNCATED_KEY not in recovered


def test_repair_still_declines_a_healthy_fragment():
    """
    Without an error position there is no principled place to cut, so the
    guard has to stay: salvaging a reply that parses fine would truncate it
    for no reason.
    """
    assert repair_truncated_json('{"a": 1}') is None


def test_damage_before_any_complete_record_fails_rather_than_returning_nothing():
    """
    When the very first record is the broken one there is nothing to keep, and
    the right answer is to fail — not to return an empty list.

    The distinction is the whole reason the paper-outcome categories exist. An
    empty `steps` array means "the model read this paper and found no
    mechanism", which is a finding worth recording. A reply that could not be
    parsed is a gap in the corpus. Salvaging the second into the shape of the
    first would file a parse failure as a scientific result.
    """
    broken_early = '{"steps": [{"from_event": "the "bad" one", "to_event": "X"}]}'
    with pytest.raises(ValueError):
        extract_json(broken_early, context="extraction response")


def test_unrecoverable_input_still_raises():
    with pytest.raises(ValueError):
        extract_json("not json at all, no braces here", context="extraction response")


def test_the_two_repair_causes_are_recorded_separately():
    """
    A truncated reply says "raise the token budget". A malformed one says the
    model wrote invalid JSON and there is nothing to configure. One counter for
    both would send a curator to the wrong setting.
    """
    import run_manifest

    telemetry = run_manifest.start_run()
    try:
        extract_json(MALFORMED, context="extraction response")
    finally:
        run_manifest.end_run()

    assert telemetry.json_repairs == 1
    assert telemetry.json_failures == 0
