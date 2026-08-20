from __future__ import annotations

import os
import time
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from schemas import PubMedRecord

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedSearchError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "AOP-RAG-Release1/0.1"})
    return s


#: Name this client sends to NCBI. Their usage policy asks every E-utilities
#: request to carry `tool` and `email` so that a misbehaving client can be
#: identified and contacted before it is blocked. `email` was already sent when
#: the environment variable happened to be set; `tool` was not sent at all.
NCBI_TOOL_NAME = "AI4AOP_KER_extractor"

#: Seconds between requests, by whether an API key is configured.
#:
#: NCBI allows 3 requests/second anonymously and 10/second with a key. The code
#: previously slept 0.11 s unconditionally — about nine per second, which is
#: correct with a key and roughly three times the limit without one. Since the
#: key is optional, the default path was the one that broke the policy.
_DELAY_WITH_KEY = 0.11
_DELAY_WITHOUT_KEY = 0.34


@dataclass(frozen=True)
class NCBICredentials:
    """
    Who is calling NCBI, and how fast they may call.

    Passed as an argument rather than read from the process environment at the
    point of use. Streamlit serves every browser session from one process, so a
    value written into `os.environ` by one session is visible to all of them —
    the same trap that made one user's LLM API key reachable by another before
    that state was moved to thread-locals. Credentials belong to a request, not
    to the process.
    """

    email: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "NCBICredentials":
        return cls(
            email=os.getenv("NCBI_EMAIL") or None,
            api_key=os.getenv("NCBI_API_KEY") or None,
        )

    @classmethod
    def resolve(cls, given: Optional["NCBICredentials"]) -> "NCBICredentials":
        """Use what the caller supplied, falling back per-field to the environment."""
        env = cls.from_env()
        if given is None:
            return env
        return cls(
            email=given.email or env.email,
            api_key=given.api_key or env.api_key,
        )

    @property
    def delay(self) -> float:
        """Seconds to wait between calls under the limit that applies to us."""
        return _DELAY_WITH_KEY if self.api_key else _DELAY_WITHOUT_KEY

    @property
    def requests_per_second(self) -> float:
        return round(1.0 / self.delay, 1)

    def params(self) -> dict[str, str]:
        params: dict[str, str] = {"tool": NCBI_TOOL_NAME}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params


def request_delay(credentials: Optional[NCBICredentials] = None) -> float:
    """How long to wait between E-utilities calls, given the current config."""
    return NCBICredentials.resolve(credentials).delay


def _base_params(credentials: Optional[NCBICredentials] = None) -> dict[str, str]:
    return NCBICredentials.resolve(credentials).params()


def build_pubmed_query(query: str, year_start: Optional[int], year_end: Optional[int]) -> str:
    q = query.strip()
    if year_start or year_end:
        start = year_start if year_start else 1000
        end = year_end if year_end else 3000
        q = f"({q}) AND (\"{start}\"[Date - Publication] : \"{end}\"[Date - Publication])"
    return q


def search_pubmed_ids(
    query: str,
    max_records: Optional[int] = 50,
    credentials: Optional[NCBICredentials] = None,
) -> list[str]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "sort": "relevance",
        "retmax": str(max_records if max_records else 9999),
        **_base_params(credentials),
    }
    response = _session().get(ESEARCH_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def _extract_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join(text.strip() for text in node.itertext() if text and text.strip())


def _extract_year(article: ET.Element) -> Optional[int]:
    for xpath in [
        ".//PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
    ]:
        node = article.find(xpath)
        if node is not None and node.text and node.text.isdigit():
            return int(node.text)
    medline_date = article.find(".//PubDate/MedlineDate")
    if medline_date is not None and medline_date.text:
        for token in medline_date.text.split():
            if token[:4].isdigit():
                return int(token[:4])
    return None


def fetch_pubmed_records(
    pmids: list[str],
    query_used: str,
    credentials: Optional[NCBICredentials] = None,
) -> list[PubMedRecord]:
    if not pmids:
        return []

    resolved = NCBICredentials.resolve(credentials)
    records: list[PubMedRecord] = []
    batch_size = 100
    s = _session()

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            **resolved.params(),
        }
        response = s.get(EFETCH_URL, params=params, timeout=90)
        response.raise_for_status()
        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            pmid = _extract_text(article.find(".//PMID"))
            title = _extract_text(article.find(".//ArticleTitle"))
            abstract_parts = [_extract_text(node) for node in article.findall(".//Abstract/AbstractText")]
            abstract = "\n".join(part for part in abstract_parts if part)

            doi = None
            for aid in article.findall(".//ArticleId"):
                if aid.attrib.get("IdType") == "doi" and aid.text:
                    doi = aid.text.strip()
                    break

            first_author = None
            author = article.find(".//AuthorList/Author")
            if author is not None:
                last = _extract_text(author.find("LastName"))
                collective = _extract_text(author.find("CollectiveName"))
                first_author = last or collective or None

            journal = _extract_text(article.find(".//Journal/Title")) or None
            year = _extract_year(article)

            records.append(
                PubMedRecord(
                    pmid=pmid,
                    doi=doi,
                    first_author=first_author,
                    journal=journal,
                    year=year,
                    title=title,
                    abstract=abstract,
                    query_used=query_used,
                )
            )
        # Paced to whichever NCBI limit applies, rather than to the one that
        # only holds when an optional API key happens to be configured.
        time.sleep(resolved.delay)
    return records


def search_pubmed(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    max_records: Optional[int] = 50,
    credentials: Optional[NCBICredentials] = None,
) -> list[PubMedRecord]:
    resolved = NCBICredentials.resolve(credentials)
    final_query = build_pubmed_query(query, year_start, year_end)
    pmids = search_pubmed_ids(final_query, max_records=max_records, credentials=resolved)
    return fetch_pubmed_records(pmids, query_used=final_query, credentials=resolved)
