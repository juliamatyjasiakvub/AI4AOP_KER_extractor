from __future__ import annotations

"""
Link tissue- and organ-level Key Events to entities in physiological maps.

An AOP that says "hepatic steatosis" is more useful when that node points at a
specific anatomical entity that other resources also recognise. Rather than
hard-wiring one map provider, this module resolves anatomy through ontology
identifiers — chiefly UBERON, with CL for cell types — and then renders links
through pluggable URL templates.

That indirection matters: UBERON identifiers are the common currency of nearly
every anatomical resource, so a KE anchored to UBERON:0002107 (liver) can be
pointed at whichever map a group actually uses without re-annotating anything.

Adding a provider
-----------------
    register_provider(MapProvider(
        key="myatlas",
        label="My Institutional Atlas",
        url_template="https://atlas.example.org/region/{curie}",
        accepts=("uberon",),
    ))

`{curie}`, `{short_id}`, `{iri}` and `{label}` are substituted into the
template; `{short_id}` is the numeric part of an OBO CURIE.
"""

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd

from schemas import PhysioMapLink
from stage2_extraction import ols4_client

#: KE levels that describe anatomy and are therefore placeable on a body map.
MAPPABLE_LEVELS: tuple[str, ...] = ("Tissue", "Organ")

#: Levels that may optionally be included when the curator wants a fuller map.
EXTENDED_LEVELS: tuple[str, ...] = ("Cellular", "Tissue", "Organ", "Individual")


@dataclass(frozen=True)
class MapProvider:
    """A physiological map that can render an ontology entity."""

    key: str
    label: str
    url_template: str
    accepts: tuple[str, ...] = ("uberon",)
    description: str = ""

    def build_url(self, curie: str, iri: str, entity_label: str) -> str:
        short_id = curie.split(":")[-1] if ":" in curie else curie
        return self.url_template.format(
            curie=urllib.parse.quote(curie, safe=""),
            short_id=urllib.parse.quote(short_id, safe=""),
            iri=urllib.parse.quote(iri, safe=""),
            label=urllib.parse.quote(entity_label, safe=""),
        )


#: Providers available out of the box. All of these resolve a UBERON or CL
#: identifier, so they work for any KE that carries anatomy annotation.
BUILTIN_PROVIDERS: tuple[MapProvider, ...] = (
    MapProvider(
        key="uberon",
        label="UBERON term page",
        url_template="https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/{iri}",
        accepts=("uberon",),
        description="Canonical anatomy record with synonyms, parents and cross-references.",
    ),
    MapProvider(
        key="cl",
        label="Cell Ontology term page",
        url_template="https://www.ebi.ac.uk/ols4/ontologies/cl/classes/{iri}",
        accepts=("cl",),
        description="Canonical cell-type record.",
    ),
    MapProvider(
        key="ontobee",
        label="Ontobee",
        url_template="https://ontobee.org/ontology/UBERON?iri={iri}",
        accepts=("uberon", "cl"),
        description="Linked-data browser showing the term in its ontology context.",
    ),
    MapProvider(
        key="bgee",
        label="Bgee anatomy",
        url_template="https://www.bgee.org/search/anatomical-homology?ids={curie}",
        accepts=("uberon",),
        description="Gene expression by anatomical structure across species.",
    ),
)

_REGISTRY: dict[str, MapProvider] = {p.key: p for p in BUILTIN_PROVIDERS}


def register_provider(provider: MapProvider) -> None:
    """Add or replace a map provider."""
    _REGISTRY[provider.key] = provider


def unregister_provider(key: str) -> None:
    _REGISTRY.pop(key, None)


def list_providers() -> list[MapProvider]:
    return list(_REGISTRY.values())


def get_provider(key: str) -> Optional[MapProvider]:
    return _REGISTRY.get(key)


# ---------------------------------------------------------------------------
# Anatomy resolution
# ---------------------------------------------------------------------------

#: Anatomy words that commonly appear inside a KE label. When a KE has no
#: usable ontology annotation we fall back to searching for the anatomical
#: noun rather than the whole event phrase, because "hepatic steatosis" will
#: not match an anatomy ontology but "liver" will.
_ANATOMY_HINTS: dict[str, str] = {
    r"\bhepat\w*|\bliver\b": "liver",
    r"\brenal\b|\bkidney\b|\bnephro\w*": "kidney",
    r"\bcardiac\b|\bheart\b|\bmyocard\w*": "heart",
    r"\bpulmonary\b|\blung\b|\balveol\w*": "lung",
    r"\bneur\w*|\bbrain\b|\bcerebr\w*|\bcortex\b": "brain",
    r"\bhepatocyte\b": "hepatocyte",
    r"\bthyroid\b": "thyroid gland",
    r"\bgonad\w*|\btestis\b|\btestic\w*": "gonad",
    r"\bovar\w*": "ovary",
    r"\bintestin\w*|\bgut\b|\bcolon\b": "intestine",
    r"\bpancrea\w*": "pancreas",
    r"\bskin\b|\bdermal\b|\bepiderm\w*": "skin",
    r"\bblood\b|\bplasma\b|\bserum\b": "blood",
    r"\bvascular\b|\bendotheli\w*|\bartery\b": "blood vessel",
    r"\bbone\b|\bskelet\w*": "bone element",
    r"\bmuscle\b|\bmyocyte\b|\bskeletal muscle\b": "muscle organ",
    r"\bimmune\b|\blymph\w*|\bspleen\b": "spleen",
    r"\bplacenta\w*": "placenta",
    r"\bretina\w*|\bocular\b|\beye\b": "eye",
    r"\bgill\b": "gill",
    r"\bswim bladder\b": "swim bladder",
}

_COMPILED_HINTS = [(re.compile(p, re.I), term) for p, term in _ANATOMY_HINTS.items()]


def anatomy_hint(ke_name: str) -> Optional[str]:
    """Extract an anatomical noun from a KE label, if one is recognisable."""
    for pattern, term in _COMPILED_HINTS:
        if pattern.search(ke_name or ""):
            return term
    return None


def _is_anatomy_curie(curie: Optional[str], source: Optional[str]) -> bool:
    if source and source.lower() in ols4_client.ANATOMY_ONTOLOGIES:
        return True
    if curie:
        prefix = curie.split(":")[0].lower()
        return prefix in {"uberon", "cl", "ma", "emapa", "fma"}
    return False


def resolve_anatomy(
    ke_name: str,
    level: str,
    *,
    existing_curie: Optional[str] = None,
    existing_iri: Optional[str] = None,
    existing_label: Optional[str] = None,
    existing_source: Optional[str] = None,
    existing_score: float = 0.0,
    lookup: bool = True,
    min_score: float = 0.45,
) -> Optional[tuple[str, str, str, str, float]]:
    """
    Resolve a KE to an anatomical entity.

    Returns (curie, iri, label, ontology, confidence) or None.

    Uses the KE's existing ontology annotation when that annotation is already
    anatomical. Otherwise it retries the lookup against anatomy ontologies,
    first with the full label and then with an extracted anatomical noun.
    """
    if _is_anatomy_curie(existing_curie, existing_source) and existing_curie and existing_iri:
        return (
            existing_curie,
            existing_iri,
            existing_label or ke_name,
            (existing_source or existing_curie.split(":")[0]).lower(),
            float(existing_score or 0.75),
        )

    if not lookup:
        return None

    anatomy_ontologies = ("uberon", "cl")

    for query, penalty in ((ke_name, 0.0), (anatomy_hint(ke_name), 0.1)):
        if not query:
            continue
        result = ols4_client.search(query, ontologies=anatomy_ontologies, rows=5)
        if result.error or not result.matches:
            continue
        best = result.matches[0]
        confidence = round(max(0.0, best.score - penalty), 4)
        if confidence >= min_score:
            return best.curie, best.iri, best.label, best.ontology, confidence

    return None


# ---------------------------------------------------------------------------
# Link building
# ---------------------------------------------------------------------------

def build_links_for_kes(
    canonical_df: pd.DataFrame,
    *,
    providers: Optional[Sequence[str]] = None,
    levels: Sequence[str] = MAPPABLE_LEVELS,
    lookup: bool = True,
    min_score: float = 0.45,
    progress: Optional[callable] = None,
) -> tuple[list[PhysioMapLink], list[str]]:
    """
    Build physiological-map links for every eligible canonical KE.

    Parameters
    ----------
    canonical_df
        Output of `table1_store.load_canonical_kes()`.
    providers
        Provider keys to emit links for. Defaults to every registered provider.
    levels
        Which biological levels to attempt. Defaults to Tissue and Organ.
    lookup
        Set False to work purely from ontology annotations already stored,
        making the call offline and instant.

    Returns (links, warnings).
    """
    if canonical_df is None or canonical_df.empty:
        return [], []

    provider_keys = list(providers) if providers else list(_REGISTRY.keys())
    chosen = [_REGISTRY[k] for k in provider_keys if k in _REGISTRY]

    eligible = canonical_df[canonical_df["level"].isin(levels)]
    links: list[PhysioMapLink] = []
    warnings: list[str] = []

    total = len(eligible)
    for i, (_, row) in enumerate(eligible.iterrows(), start=1):
        ke_name = str(row.get("canonical_name") or "")
        resolved = resolve_anatomy(
            ke_name,
            str(row.get("level") or ""),
            existing_curie=_opt(row.get("ontology_curie")),
            existing_iri=_opt(row.get("ontology_iri")),
            existing_label=_opt(row.get("ontology_label")),
            existing_source=_opt(row.get("ontology_source")),
            existing_score=float(row.get("ontology_score") or 0.0),
            lookup=lookup,
            min_score=min_score,
        )

        if progress is not None:
            try:
                progress(i, total, ke_name)
            except Exception:
                pass

        if resolved is None:
            warnings.append(f"No anatomical entity found for '{ke_name}'.")
            continue

        curie, iri, label, ontology, confidence = resolved
        for provider in chosen:
            if ontology not in provider.accepts:
                continue
            links.append(
                PhysioMapLink(
                    canonical_id=int(row["canonical_id"]),
                    provider=provider.key,
                    entity_label=label,
                    entity_id=curie,
                    url=provider.build_url(curie, iri, label),
                    confidence=confidence,
                )
            )

    return links, warnings


def _opt(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def links_by_ke(links_df: pd.DataFrame) -> dict[int, list[dict]]:
    """canonical_id -> list of link dicts, for rendering in the evidence panel."""
    out: dict[int, list[dict]] = {}
    if links_df is None or links_df.empty:
        return out
    for _, row in links_df.iterrows():
        out.setdefault(int(row["canonical_id"]), []).append(
            {
                "provider": row["provider"],
                "provider_label": (
                    _REGISTRY[row["provider"]].label
                    if row["provider"] in _REGISTRY
                    else row["provider"]
                ),
                "entity_label": row["entity_label"],
                "entity_id": row["entity_id"],
                "url": row["url"],
                "confidence": float(row.get("confidence") or 0.0),
            }
        )
    return out


def coverage_summary(canonical_df: pd.DataFrame, links_df: pd.DataFrame) -> dict[str, float]:
    """How much of the map is anchored to a physiological entity."""
    if canonical_df is None or canonical_df.empty:
        return {"eligible": 0, "linked": 0, "coverage": 0.0}
    eligible = canonical_df[canonical_df["level"].isin(MAPPABLE_LEVELS)]
    linked = (
        links_df["canonical_id"].nunique()
        if links_df is not None and not links_df.empty
        else 0
    )
    n_eligible = len(eligible)
    return {
        "eligible": n_eligible,
        "linked": int(linked),
        "coverage": round(linked / n_eligible, 3) if n_eligible else 0.0,
    }


__all__ = [
    "MapProvider",
    "MAPPABLE_LEVELS",
    "EXTENDED_LEVELS",
    "BUILTIN_PROVIDERS",
    "register_provider",
    "unregister_provider",
    "list_providers",
    "get_provider",
    "anatomy_hint",
    "resolve_anatomy",
    "build_links_for_kes",
    "links_by_ke",
    "coverage_summary",
]
