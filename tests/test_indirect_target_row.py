"""
A paper that connects the two target events through intermediates supports the
target relationship.

Why this exists. `extract_pathway_rows` writes one Table 1 row per link of the
chain a paper describes. When the model writes that chain as
A -> x -> y -> B, none of those rows is about A -> B, so the paper contributed
nothing at all to the relationship the curator asked about — while its chain
demonstrated exactly that relationship.

Whether the target edge existed therefore came down to whether the model also
emitted a redundant A -> B shortcut next to the chain, which it does
inconsistently. Measured over replicate runs of one 13-paper corpus, 5 of 13
papers changed that answer between two identical runs while their chains stayed
the same; the supporting-paper count moved because of a stylistic choice about
summarising, not because of evidence. Deciding support by reachability instead
took per-paper agreement from 54% to 92% on that corpus.

The row is always Non-adjacent and always carries its route, because support
through four intermediates is real support and is not the same claim as a
demonstrated direct step.
"""

from __future__ import annotations

import pytest

from stage2_extraction.ker_extractor import (
    PathwayStep,
    _anchor_path,
    _path_direction,
)


UP = "Voltage-gated sodium channel"
DOWN = "Oligodendrocyte differentiation"


def _step(a: str, b: str, **kwargs) -> PathwayStep:
    return PathwayStep(from_event=a, to_event=b, **kwargs)


# ---------------------------------------------------------------------------
# Finding the path
# ---------------------------------------------------------------------------

def test_a_direct_step_is_a_path_of_length_two():
    path = _anchor_path([_step(UP, DOWN)], UP, DOWN)
    assert path == [UP, DOWN]


def test_a_chain_through_intermediates_is_found():
    """The real shape from 23940003: four links, no shortcut edge."""
    steps = [
        _step(UP, "increased TTX-sensitive inward sodium current"),
        _step("increased TTX-sensitive inward sodium current",
              "increased action potential firing"),
        _step("increased action potential firing",
              "increased glutamatergic postsynaptic currents"),
        _step("increased glutamatergic postsynaptic currents", DOWN),
    ]
    path = _anchor_path(steps, UP, DOWN)
    assert path[0] == UP and path[-1] == DOWN
    assert len(path) == 5, "three intermediates"


def test_the_shortest_path_wins():
    """A paper stating both the chain and the shortcut reports 0 intermediates."""
    steps = [
        _step(UP, "increased sodium current"),
        _step("increased sodium current", DOWN),
        _step(UP, DOWN),
    ]
    assert _anchor_path(steps, UP, DOWN) == [UP, DOWN]


def test_an_unconnected_chain_yields_no_path():
    """
    24038428 in the real corpus: a microglial paper whose downstream event is
    oligodendrocyte apoptosis, not differentiation. It must stay a true
    negative — reachability has to be able to say no.
    """
    steps = [
        _step(UP, "increased NF-kB nuclear translocation"),
        _step("increased NF-kB nuclear translocation", "increased microglial activation"),
        _step("increased microglial activation", "increased oligodendrocyte apoptosis"),
    ]
    assert _anchor_path(steps, UP, DOWN) == []


def test_a_cycle_does_not_hang():
    """
    Remyelination restoring channel clustering genuinely closes a loop, and
    37208933 produced exactly that. A naive walk never terminates.
    """
    steps = [
        _step(UP, "increased demyelination"),
        _step("increased demyelination", DOWN),
        _step(DOWN, "increased new myelin formation"),
        _step("increased new myelin formation", UP),
    ]
    assert _anchor_path(steps, UP, DOWN) == [UP, "increased demyelination", DOWN]


def test_direction_of_travel_is_respected():
    """B -> A is not evidence that A leads to B."""
    assert _anchor_path([_step(DOWN, UP)], UP, DOWN) == []


def test_anchor_matching_ignores_case_and_punctuation():
    steps = [_step("voltage-gated sodium channel!", "  Oligodendrocyte Differentiation ")]
    assert _anchor_path(steps, UP, DOWN) != []


# ---------------------------------------------------------------------------
# Sign composition
# ---------------------------------------------------------------------------

def test_two_negative_steps_compose_to_a_positive_relationship():
    steps = [
        _step(UP, "x", direction="negative"),
        _step("x", DOWN, direction="negative"),
    ]
    assert _path_direction(steps) == "positive"


def test_one_negative_step_makes_the_path_negative():
    steps = [
        _step(UP, "x", direction="positive"),
        _step("x", DOWN, direction="negative"),
    ]
    assert _path_direction(steps) == "negative"


@pytest.mark.parametrize("unsigned", ["unclear", "none", ""])
def test_any_unsigned_step_makes_the_whole_path_unclear(unsigned):
    """
    An unsigned link breaks the product. Guessing a sign here would put a
    direction on the map that no paper stated.
    """
    steps = [
        _step(UP, "x", direction="negative"),
        _step("x", DOWN, direction=unsigned),
    ]
    assert _path_direction(steps) == "unclear"
