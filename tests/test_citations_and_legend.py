from __future__ import annotations

"""
Two things a reader has to be able to trust on sight.

A citation key and a legend entry are both claims the tool makes *about
itself*, and both fail silently. A key that maps to the wrong paper still
looks like a citation; a legend that describes an encoding nothing draws still
looks like a legend. The legend in particular had drifted — it announced three
node shapes for a map that draws one — and nothing caught it, because nothing
compared the legend against the drawing.
"""

import re
import sqlite3
import tempfile
from pathlib import Path

import pytest

from stage2_extraction import citations


# ---------------------------------------------------------------------------
# Citation keys
# ---------------------------------------------------------------------------

def resolved(doi, author, year, n_authors=5, second=None):
    return citations.PaperCitation(
        doi=doi, first_author=author, second_author=second,
        n_authors=n_authors, year=year, resolved=True,
    )


class TestCitationKeys:

    def test_three_or_more_authors_get_et_al(self):
        assert resolved("10.1/a", "Sanchez", 2019).base_key() == "Sanchez et al., 2019"

    def test_two_authors_are_both_named(self):
        cit = resolved("10.1/a", "Lee", 2021, n_authors=2, second="Park")
        assert cit.base_key() == "Lee & Park, 2021"

    def test_a_sole_author_stands_alone(self):
        assert resolved("10.1/a", "Okonkwo", 2020, n_authors=1).base_key() == "Okonkwo, 2020"

    def test_an_unresolved_paper_falls_back_to_its_doi(self):
        """
        Deliberately not a guess. Inventing "Unknown, n.d." would put a
        citation-shaped string next to a claim nobody can check.
        """
        assert citations.PaperCitation(doi="10.1/x").base_key() == "10.1/x"

    def test_same_author_same_year_gets_letters(self, monkeypatch):
        monkeypatch.setattr(citations, "resolve", lambda dois, **kw: {
            "10.1/a": resolved("10.1/a", "Sanchez", 2019),
            "10.1/b": resolved("10.1/b", "Sanchez", 2019, n_authors=4),
            "10.1/c": resolved("10.1/c", "Okonkwo", 2020, n_authors=1),
        })
        keys = citations.citation_keys(["10.1/a", "10.1/b", "10.1/c"])
        assert keys["10.1/a"] == "Sanchez et al., 2019a"
        assert keys["10.1/b"] == "Sanchez et al., 2019b"
        # A paper with no collision keeps a clean key.
        assert keys["10.1/c"] == "Okonkwo, 2020"

    def test_letters_are_stable_under_doi_order(self, monkeypatch):
        """
        A key that changed when a paper was added would invalidate every note
        already written against it, so the suffix follows DOI order rather
        than anything that shifts as the corpus grows.
        """
        papers = {
            "10.1/zzz": resolved("10.1/zzz", "Sanchez", 2019),
            "10.1/aaa": resolved("10.1/aaa", "Sanchez", 2019),
        }
        monkeypatch.setattr(citations, "resolve", lambda dois, **kw: papers)
        keys = citations.citation_keys(list(papers))
        assert keys["10.1/aaa"] == "Sanchez et al., 2019a"
        assert keys["10.1/zzz"] == "Sanchez et al., 2019b"

    @pytest.mark.parametrize("raw,expected", [
        ("https://doi.org/10.1234/EXAMPLE-E", "10.1234/example-e"),
        ("http://dx.doi.org/10.1/X", "10.1/x"),
        ("doi: 10.1/Y", "10.1/y"),
        ("  ", ""),
        ("nan", ""),
        (None, ""),
    ])
    def test_dois_are_normalised_before_anything_else(self, raw, expected):
        assert citations._norm(raw) == expected

    def test_key_for_normalises_its_argument(self):
        keys = {"10.1/a": "Lee, 2020"}
        assert citations.key_for("https://doi.org/10.1/A", keys) == "Lee, 2020"

    def test_key_for_falls_back_when_absent(self):
        assert citations.key_for("10.1/unknown", {}) == "10.1/unknown"


class TestCitationCache:

    def test_a_lookup_round_trips(self, tmp_path, monkeypatch):
        db = tmp_path / "cite.db"
        monkeypatch.setattr("stage2_extraction.table1_store.DB_PATH", db)

        cit = citations.PaperCitation(
            doi="10.1/a", first_author="Sanchez", second_author="Lee",
            n_authors=5, year=2019, title="A title",
            container_title="J Neuro", resolved=True,
        )
        citations._store(cit)
        back = citations._load_cached(["10.1/A"])
        assert back["10.1/a"].first_author == "Sanchez"
        assert back["10.1/a"].base_key() == "Sanchez et al., 2019"

    def test_forget_clears_one_entry(self, tmp_path, monkeypatch):
        db = tmp_path / "cite.db"
        monkeypatch.setattr("stage2_extraction.table1_store.DB_PATH", db)
        citations._store(citations.PaperCitation(doi="10.1/a"))
        citations.forget("10.1/a")
        assert citations._load_cached(["10.1/a"]) == {}

    def test_a_failed_lookup_is_cached_too(self, tmp_path, monkeypatch):
        """
        Otherwise one bad DOI re-queries Crossref on every Streamlit rerun,
        which is on every click.
        """
        db = tmp_path / "cite.db"
        monkeypatch.setattr("stage2_extraction.table1_store.DB_PATH", db)
        citations._store(
            citations.PaperCitation(doi="10.1/bad", error="Crossref has no record")
        )
        back = citations._load_cached(["10.1/bad"])
        assert back["10.1/bad"].resolved is False
        assert back["10.1/bad"].error


# ---------------------------------------------------------------------------
# The legend against the drawing
# ---------------------------------------------------------------------------

MAP_SOURCE = (Path(__file__).resolve().parent.parent / "ui" / "aop_map.py").read_text(
    encoding="utf-8"
)
LEGEND = MAP_SOURCE[MAP_SOURCE.index("ENCODINGS:") : MAP_SOURCE.index("EDGE_COLOURS =")]


def _keys_of(name: str) -> set[str]:
    body = re.search(rf"{name} = \{{(.*?)\}}", MAP_SOURCE, re.S).group(1)
    return set(re.findall(r'"(\w+)":', body))


class TestLegendMatchesTheDrawing:
    """
    The legend is the map's own account of itself, so a wrong entry is worse
    than a missing one: it tells a reader to interpret something that is not
    there. These tests read the drawing code and the legend text and insist
    they agree.
    """

    @pytest.mark.parametrize("role,phrase", [
        ("MIE", "molecular initiating"),
        ("KE", "key event"),
        ("AO", "adverse outcome"),
        ("marker", "marker"),
    ])
    def test_every_drawn_role_is_explained(self, role, phrase):
        assert role in _keys_of("ROLE_COLOURS")
        assert phrase in LEGEND.lower(), f"{role} is drawn but the legend never says so"

    @pytest.mark.parametrize("verdict,phrase", [
        ("supporting", "supported"),
        ("mixed", "mixed"),
        ("contradictory", "contradicted"),
    ])
    def test_every_edge_verdict_is_explained(self, verdict, phrase):
        assert verdict in _keys_of("EDGE_COLOURS")
        assert phrase in LEGEND.lower()

    @pytest.mark.parametrize("token", ["↑", "↓", "⚠", "dashed", "pill"])
    def test_the_marks_on_a_node_are_explained(self, token):
        assert token in LEGEND.lower()

    @pytest.mark.parametrize("shape", ["▭", "▢", "⬭"])
    def test_the_legend_does_not_claim_shapes_nothing_draws(self, shape):
        """
        The regression this file exists for. Every node is the same 200x80
        rectangle differing only in corner radius, and the legend announced a
        rectangle, a square and an ellipse.
        """
        assert shape not in LEGEND

    def test_the_grid_shows_what_was_reported_not_just_how_often(self):
        """
        A row reading "label | canonical | TRUE | 12 | 11" cannot be acted on:
        it does not say whether the event went up or down, or what was
        measured. Those columns exist in Table 1 and must reach the grid.
        """
        curate = (Path(__file__).resolve().parent.parent / "ui" / "curate.py").read_text(
            encoding="utf-8"
        )
        for column in ("Reported change", "Assay (whole claim)", "Cell type"):
            assert f'"{column}"' in curate, f"{column} is missing from the Assign grid"
        assert "_label_evidence" in curate

    def test_the_assay_column_does_not_overclaim(self):
        """
        `measured_as` is one field per relationship, not per event, so the
        assay shown against a raw label may be the measurement at the other
        end of the link. The column name and help must say so rather than
        implying this event was measured that way.
        """
        curate = (Path(__file__).resolve().parent.parent / "ui" / "curate.py").read_text(
            encoding="utf-8"
        )
        assert '"Measured as"' not in curate, "the old over-claiming column name is back"
        assert "may be the measurement at the OTHER end" in curate

    def test_all_nodes_are_the_same_rectangle(self):
        """
        If this ever stops being true, the legend has to change with it —
        which is the point of asserting it here rather than trusting a comment.
        """
        rects = re.findall(r'<rect x="\{placement\.x', MAP_SOURCE)
        assert rects, "the canvas no longer draws nodes as <rect>"
        assert "radius = {" in MAP_SOURCE, "corner radius is what distinguishes roles"
