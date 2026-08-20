"""
NCBI's E-utilities usage policy, enforced in code rather than in a README.

Two rules, both previously unmet. Every request must name the tool and give a
contact address so a misbehaving client can be written to instead of blocked;
`tool` was never sent at all. And the request rate must respect 3/second
anonymously or 10/second with an API key; the client slept a flat 0.11 s —
correct with a key, roughly triple the limit without one. Since the key is
optional, the default configuration was the non-compliant one.

Credentials are passed as an argument rather than read from `os.environ` at the
point of use, because Streamlit serves every browser session from one process
and environment variables are shared across all of them.
"""

from __future__ import annotations

import pytest

from stage1_search.pubmed_search import (
    NCBI_TOOL_NAME,
    NCBICredentials,
    request_delay,
)


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    """Every test starts with no NCBI environment variables set."""
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Identifying the client
# ---------------------------------------------------------------------------

def test_tool_is_always_sent_even_with_nothing_configured():
    """The one parameter NCBI asks for that costs nothing to provide."""
    assert NCBICredentials().params()["tool"] == NCBI_TOOL_NAME


def test_email_is_sent_when_supplied():
    params = NCBICredentials(email="j@vub.be").params()
    assert params["email"] == "j@vub.be"


def test_absent_fields_are_omitted_rather_than_sent_empty():
    """An empty `email=` is worse than none — it looks like a contact and isn't."""
    params = NCBICredentials().params()
    assert "email" not in params
    assert "api_key" not in params


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------

def test_anonymous_requests_stay_under_three_per_second():
    assert NCBICredentials().requests_per_second <= 3.0


def test_a_key_permits_the_higher_rate_but_stays_under_ten():
    rate = NCBICredentials(api_key="abc").requests_per_second
    assert 3.0 < rate <= 10.0


def test_a_key_is_faster_than_none():
    assert NCBICredentials(api_key="abc").delay < NCBICredentials().delay


def test_request_delay_defaults_to_the_anonymous_limit():
    """
    The failure this guards. `request_delay()` with nothing configured must
    return the cautious value, not the one that assumes a key is present.
    """
    assert 1.0 / request_delay() <= 3.0


# ---------------------------------------------------------------------------
# Where credentials come from
# ---------------------------------------------------------------------------

def test_environment_is_used_when_the_caller_supplies_nothing(monkeypatch):
    monkeypatch.setenv("NCBI_EMAIL", "env@vub.be")
    monkeypatch.setenv("NCBI_API_KEY", "env-key")
    resolved = NCBICredentials.resolve(None)
    assert resolved.email == "env@vub.be"
    assert resolved.api_key == "env-key"


def test_caller_values_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("NCBI_EMAIL", "env@vub.be")
    resolved = NCBICredentials.resolve(NCBICredentials(email="ui@vub.be"))
    assert resolved.email == "ui@vub.be"


def test_fallback_is_per_field_not_all_or_nothing(monkeypatch):
    """
    A user who types an API key into the sidebar but leaves the email blank
    should still send the address from the environment, rather than losing it
    because one field was filled in.
    """
    monkeypatch.setenv("NCBI_EMAIL", "env@vub.be")
    resolved = NCBICredentials.resolve(NCBICredentials(api_key="ui-key"))
    assert resolved.email == "env@vub.be"
    assert resolved.api_key == "ui-key"


def test_credentials_are_not_read_from_the_process_at_call_time(monkeypatch):
    """
    Explicit credentials must not be overridden by the environment. Streamlit
    runs every session in one process, so anything resolved from `os.environ`
    at the point of use belongs to whoever set it last, not to this user.
    """
    monkeypatch.setenv("NCBI_API_KEY", "someone-elses-key")
    resolved = NCBICredentials.resolve(NCBICredentials(api_key="my-key"))
    assert resolved.api_key == "my-key"


def test_credentials_are_immutable():
    """So a resolved set cannot be edited by one caller and seen by another."""
    creds = NCBICredentials(email="j@vub.be")
    with pytest.raises(Exception):
        creds.email = "someone@else.be"
