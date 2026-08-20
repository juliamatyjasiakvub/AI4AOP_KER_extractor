from __future__ import annotations

"""
The left-to-right axis is evidence, not preference.

The property under test throughout: an edge never points leftwards. If a Key
Event is drawn to the left of something that points at it, the picture is
making a causal claim backwards — and a figure is the most quotable thing this
tool produces, so it has to be right without anyone checking.
"""

import networkx as nx
import pytest

from stage2_extraction import causal_layout


def chain(*names: str) -> nx.DiGraph:
    g = nx.DiGraph()
    for a, b in zip(names, names[1:]):
        g.add_edge(a, b, n_papers=1)
    return g


class TestCausalOrder:

    def test_a_chain_runs_left_to_right(self):
        g = chain("MIE", "early", "middle", "late", "AO")
        placements = causal_layout.layout(g)

        columns = [placements[n].column for n in ("MIE", "early", "middle", "late", "AO")]
        assert columns == sorted(columns)
        assert columns[0] < columns[-1]
        assert causal_layout.validate(placements, g) == []

    def test_longest_path_decides_depth(self):
        """
        A node reachable by both a short and a long route belongs after the
        long one, or it is drawn before something it depends on.
        """
        g = nx.DiGraph()
        g.add_edge("a", "b", n_papers=1)
        g.add_edge("b", "c", n_papers=1)
        g.add_edge("c", "d", n_papers=1)
        g.add_edge("a", "d", n_papers=1)   # the shortcut

        depths = causal_layout.causal_depth(g)
        assert depths["d"] == 3, "the shortcut must not pull d forward"

    def test_no_edge_points_left(self):
        g = nx.DiGraph()
        for a, b in [("m", "x"), ("m", "y"), ("x", "z"), ("y", "z"), ("z", "ao")]:
            g.add_edge(a, b, n_papers=1)
        placements = causal_layout.layout(g)
        assert causal_layout.validate(placements, g) == []

    def test_feedback_loop_does_not_break_the_layout(self):
        g = chain("a", "b", "c")
        g.add_edge("c", "a", n_papers=1)      # feedback
        placements = causal_layout.layout(g)
        assert len(placements) == 3
        assert placements["a"].column < placements["c"].column


class TestRoles:
    """
    Position in the graph is not a role.

    These tests once asserted the opposite — that a source node is the MIE and
    a sink is the AO — and the assertions were correct about the code and wrong
    about the biology. A node with nothing upstream of it says that no paper in
    *this corpus* reported an earlier step, which is the ordinary state of a
    collection assembled around the middle of a pathway. Promoting it to
    molecular initiating event turns a gap in the literature into a claim, and
    prints that claim in the leftmost column of the figure people quote.

    So both ends are now declarations. What the graph knows, it still offers as
    a suggestion.
    """

    def test_nothing_is_a_role_until_it_is_declared(self):
        g = chain("start", "middle", "end")
        roles = causal_layout.assign_roles(g)
        assert roles == {"start": "KE", "middle": "KE", "end": "KE"}

    def test_declared_roles_win(self):
        g = chain("start", "middle", "end")
        roles = causal_layout.assign_roles(g, declared={"middle": "AO"})
        assert roles["middle"] == "AO"

    def test_a_declared_mie_is_pinned_to_the_first_column(self):
        g = chain("a", "b", "c", "d")
        placements = causal_layout.layout(
            g, declared_roles={"a": "MIE", "d": "AO"}
        )
        assert placements["a"].column == 0
        assert placements["a"].band == "MIE"
        assert placements["d"].band == "AO"

    def test_an_undeclared_pathway_leaves_no_empty_first_column(self):
        """
        Column 0 is reserved for the MIE. With none declared — now the common
        case — reserving it anyway drew every map with an empty leftmost
        column, which reads as a missing initiating event rather than an
        undeclared one.
        """
        g = chain("a", "b", "c", "d")
        placements = causal_layout.layout(g)

        assert min(p.column for p in placements.values()) == 0
        assert placements["a"].column == 0
        assert placements["a"].band != "MIE"

    def test_bands_span_the_pathway(self):
        g = chain("a", "b", "c", "d", "e", "f", "g")
        placements = causal_layout.layout(
            g, declared_roles={"a": "MIE", "g": "AO"}
        )
        bands = {p.band for p in placements.values()}
        assert "MIE" in bands and "AO" in bands
        assert bands & {"Early", "Intermediate", "Late"}

    def test_the_graph_still_suggests_where_an_mie_could_go(self):
        """Not inferring a role is not the same as having no opinion."""
        g = nx.DiGraph()
        g.add_edge("receptor binding", "cell death", n_papers=1)
        g.nodes["receptor binding"]["level"] = "Molecular"

        suggestions = causal_layout.mie_candidates(g)
        assert [c["ke_name"] for c in suggestions] == ["receptor binding"]
        assert suggestions[0]["plausible"] is True
        assert "not collected" in suggestions[0]["why"]


class TestSavedPositions:

    def test_vertical_offsets_are_honoured(self):
        g = chain("a", "b")
        placements = causal_layout.layout(g, y_offsets={"b": 640.0})
        assert placements["b"].y == 640.0
        assert placements["b"].y_from_saved is True

    def test_a_saved_position_cannot_change_the_column(self):
        """
        The guarantee the redesign asks for. There is no x in the offsets at
        all, so no stored layout can move a node into the wrong causal column.
        """
        g = chain("a", "b", "c")
        without = causal_layout.layout(g)
        with_offsets = causal_layout.layout(
            g, y_offsets={"a": 900.0, "b": 0.0, "c": 450.0}
        )
        for node in ("a", "b", "c"):
            assert with_offsets[node].column == without[node].column
            assert with_offsets[node].x == without[node].x

    def test_offsets_never_produce_a_backwards_edge(self):
        g = chain("a", "b", "c", "d")
        placements = causal_layout.layout(
            g, y_offsets={n: 1000.0 for n in g.nodes}
        )
        assert causal_layout.validate(placements, g) == []

    def test_layout_takes_no_x_offsets(self):
        """A horizontal override must not be reachable through the API."""
        import inspect
        signature = inspect.signature(causal_layout.layout)
        assert "x_offsets" not in signature.parameters
        assert "y_offsets" in signature.parameters


class TestEmptyAndTrivial:

    def test_empty_graph(self):
        assert causal_layout.layout(nx.DiGraph()) == {}

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("only")
        placements = causal_layout.layout(g)
        assert placements["only"].column == 0
