from __future__ import annotations

import os
import time
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


def _base_params() -> dict[str, str]:
    params: dict[str, str] = {}
    email = os.getenv("NCBI_EMAIL")
    api_key = os.getenv("NCBI_API_KEY")
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def build_pubmed_query(query: str, year_start: Optional[int], year_end: Optional[int]) -> str:
    q = query.strip()
    if year_start or year_end:
        start = year_start if year_start else 1000
        end = year_end if year_end else 3000
        q = f"({q}) AND (\"{start}\"[Date - Publication] : \"{end}\"[Date - Publication])"
    return q


def search_pubmed_ids(query: str, max_records: Optional[int] = 50) -> list[str]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "sort": "relevance",
        "retmax": str(max_records if max_records else 9999),
        **_base_params(),
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


def fetch_pubmed_records(pmids: list[str], query_used: str) -> list[PubMedRecord]:
    if not pmids:
        return []

    records: list[PubMedRecord] = []
    batch_size = 100
    s = _session()

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            **_base_params(),
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
        time.sleep(0.11)
    return records


def search_pubmed(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    max_records: Optional[int] = 50,
) -> list[PubMedRecord]:
    final_query = build_pubmed_query(query, year_start, year_end)
    pmids = search_pubmed_ids(final_query, max_records=max_records)
    return fetch_pubmed_records(pmids, query_used=final_query)
