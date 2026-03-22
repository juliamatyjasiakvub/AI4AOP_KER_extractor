from __future__ import annotations

import os
import time
from typing import Iterable
from xml.etree import ElementTree as ET

import requests

from schemas import PubMedRecord

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "aop_rag_release1"
DEFAULT_TIMEOUT = 30


class PubMedSearchError(RuntimeError):
    pass


def _get_email() -> str:
    return os.getenv("NCBI_EMAIL", "example@example.com")


def _common_params() -> dict[str, str]:
    params = {
        "tool": TOOL_NAME,
        "email": _get_email(),
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def build_pubmed_query(query: str, year_start: int | None = None, year_end: int | None = None) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if year_start and year_end and year_start > year_end:
        raise ValueError("year_start cannot be greater than year_end.")

    if year_start or year_end:
        start = year_start if year_start is not None else 1000
        end = year_end if year_end is not None else 3000
        query = f"({query}) AND ({start}:{end}[pdat])"
    return query


def search_pubmed_ids(
    query: str,
    year_start: int | None = None,
    year_end: int | None = None,
    max_records: int = 50,
) -> list[str]:
    if max_records <= 0:
        return []

    term = build_pubmed_query(query, year_start, year_end)
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        **_common_params(),
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(max_records),
        "sort": "relevance",
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def _chunks(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def fetch_pubmed_records(
    query: str,
    year_start: int | None = None,
    year_end: int | None = None,
    max_records: int = 50,
    pause_seconds: float = 0.34,
) -> list[PubMedRecord]:
    pmids = search_pubmed_ids(query=query, year_start=year_start, year_end=year_end, max_records=max_records)
    if not pmids:
        return []

    records: list[PubMedRecord] = []
    for batch in _chunks(pmids, 100):
        records.extend(_fetch_details_for_pmids(batch, query_used=build_pubmed_query(query, year_start, year_end)))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return records


def _fetch_details_for_pmids(pmids: list[str], query_used: str) -> list[PubMedRecord]:
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        **_common_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise PubMedSearchError("Could not parse PubMed XML response.") from exc

    parsed: list[PubMedRecord] = []
    for article in root.findall(".//PubmedArticle"):
        record = _parse_pubmed_article(article, query_used=query_used)
        if record:
            parsed.append(record)
    return parsed


def _first_text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    found = node.find(path)
    if found is None:
        return None
    text = "".join(found.itertext()).strip()
    return text or None


def _parse_pubmed_article(article: ET.Element, query_used: str) -> PubMedRecord | None:
    medline = article.find("MedlineCitation")
    article_node = medline.find("Article") if medline is not None else None
    if medline is None or article_node is None:
        return None

    pmid = _first_text(medline, "PMID")
    title = _first_text(article_node, "ArticleTitle") or ""
    abstract_parts = ["".join(elem.itertext()).strip() for elem in article_node.findall("Abstract/AbstractText")]
    abstract = "\n".join(part for part in abstract_parts if part).strip()

    author_list = article_node.find("AuthorList")
    first_author = None
    if author_list is not None:
        first_author_node = author_list.find("Author")
        if first_author_node is not None:
            last_name = _first_text(first_author_node, "LastName")
            initials = _first_text(first_author_node, "Initials")
            collective = _first_text(first_author_node, "CollectiveName")
            if collective:
                first_author = collective
            elif last_name and initials:
                first_author = f"{last_name} {initials}"
            else:
                first_author = last_name or initials

    journal = _first_text(article_node, "Journal/Title") or _first_text(article_node, "Journal/ISOAbbreviation")

    year = None
    year_text = (
        _first_text(article_node, "Journal/JournalIssue/PubDate/Year")
        or _first_text(article_node, "ArticleDate/Year")
        or _first_text(article.find("PubmedData") if article is not None else None, "History/PubMedPubDate/Year")
    )
    if year_text:
        try:
            year = int(year_text)
        except ValueError:
            year = None

    doi = None
    for article_id in article.findall("PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            doi = (article_id.text or "").strip() or None
            break

    if not pmid or not title:
        return None

    return PubMedRecord(
        pmid=pmid,
        doi=doi,
        first_author=first_author,
        journal=journal,
        year=year,
        title=title,
        abstract=abstract,
        query_used=query_used,
    )
