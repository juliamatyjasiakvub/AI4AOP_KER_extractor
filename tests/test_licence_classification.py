"""
Licence classification, where a wrong answer points the wrong way.

Most misclassifications in this tool are harmless — calling CC BY "unknown"
costs nothing but a manual check. One direction is not harmless: reporting a
restricted licence as permissive tells a researcher they may process and
republish an article when they may not.

The likeliest way that happens is substring ordering. "creativecommons.org/
licenses/by" is a prefix of every other Creative Commons URL, so a naive check
labels CC BY-NC-SA as plain CC BY — stripping away both the non-commercial and
the share-alike conditions. These tests pin the order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from check_licences import _classify_licence_url  # noqa: E402


@pytest.mark.parametrize("url,expected", [
    ("https://creativecommons.org/licenses/by/4.0/", "CC BY"),
    ("https://creativecommons.org/licenses/by/3.0/", "CC BY"),
    ("http://creativecommons.org/licenses/by/2.0", "CC BY"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
])
def test_permissive_licences_are_recognised(url, expected):
    category, name = _classify_licence_url(url)
    assert name == expected
    assert category == "OPEN"


@pytest.mark.parametrize("url,expected", [
    ("https://creativecommons.org/licenses/by-nc/4.0/", "CC BY-NC"),
    ("https://creativecommons.org/licenses/by-sa/4.0/", "CC BY-SA"),
    ("https://creativecommons.org/licenses/by-nd/4.0/", "CC BY-ND"),
    ("https://creativecommons.org/licenses/by-nc-sa/4.0/", "CC BY-NC-SA"),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC BY-NC-ND"),
])
def test_conditioned_licences_are_never_reported_as_plain_cc_by(url, expected):
    """
    The failure that matters. Every URL here contains "/licenses/by", so an
    unordered prefix match reports all of them as CC BY and silently discards
    the condition that makes them restrictive.
    """
    category, name = _classify_licence_url(url)
    assert name == expected, f"{url} misread as {name}"
    assert category == "OPEN-CONDITIONS", (
        f"{expected} must not be classified as unconditionally open"
    )


def test_publisher_licences_are_not_mistaken_for_open_ones():
    """A publisher's own TDM licence has terms; it is not a CC licence."""
    assert _classify_licence_url("https://www.elsevier.com/tdm/userlicense/1.0/") is None
    assert _classify_licence_url(
        "https://www.springernature.com/gp/researchers/text-and-data-mining"
    ) is None


def test_nothing_is_recognised_from_an_empty_or_missing_url():
    assert _classify_licence_url("") is None
    assert _classify_licence_url(None) is None


def test_matching_is_case_insensitive():
    category, name = _classify_licence_url(
        "HTTPS://CreativeCommons.ORG/licenses/BY-NC/4.0/"
    )
    assert name == "CC BY-NC"
    assert category == "OPEN-CONDITIONS"


def test_an_unrelated_url_is_not_a_licence():
    assert _classify_licence_url("https://example.org/terms") is None
