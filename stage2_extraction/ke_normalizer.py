from __future__ import annotations

"""
Canonical Key Event normalization.

Twelve papers describing the same biology will produce twelve different strings:
"mitochondrial ROS accumulation", "increased mitochondrial reactive oxygen
species", "elevated mtROS", "mitochondrial oxidative stress". Left alone these
become four separate nodes and the AOP map fragments into unreadable confetti.

This module merges equivalent labels into `CanonicalKE` records **while
preserving every original string as an alias**, so no paper's terminology is
ever destroyed — a curator can always see exactly what was written and where.

Merging rules, in order of authority
------------------------------------
1. Same AOP-Wiki KE id                     -> merge (authoritative)
2. Same ontology CURIE from OLS4           -> merge (authoritative)
3. Identical normalised string             -> merge
4. Fuzzy string similarity above threshold -> merge, but only if the two
   labels agree on biological level AND on direction polarity

Rule 4's polarity guard is the important one. "Increased apoptosis" and
"decreased apoptosis" are 90 % similar as strings and biologically opposite;
merging them would silently invert the meaning of an edge. Labels whose
polarity conflicts are never merged, no matter how similar they look.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

import pandas as pd

from schemas import CanonicalKE, OntologyMatch, level_index

# ---------------------------------------------------------------------------
# Lexical normalization
# ---------------------------------------------------------------------------

#: Domain abbreviations expanded before comparison so "ROS" and "reactive
#: oxygen species" land in the same bucket.
_ABBREVIATIONS: dict[str, str] = {
    r"\bros\b": "reactive oxygen species",
    r"\brns\b": "reactive nitrogen species",
    r"\bmtros\b": "mitochondrial reactive oxygen species",
    r"\boxphos\b": "oxidative phosphorylation",
    r"\bmmp\b": "mitochondrial membrane potential",
    r"\bgsh\b": "glutathione",
    r"\bmda\b": "malondialdehyde",
    r"\bsod\b": "superoxide dismutase",
    r"\bldh\b": "lactate dehydrogenase",
    r"\bahr\b": "aryl hydrocarbon receptor",
    r"\bppar\b": "peroxisome proliferator activated receptor",
    r"\ber\s+stress\b": "endoplasmic reticulum stress",
    r"\bdna\s+dsb\b": "dna double strand break",
    r"\bdsb\b": "double strand break",
    r"\bssb\b": "single strand break",
    r"\bnf-?kb\b": "nuclear factor kappa b",
    r"\bnrf2\b": "nuclear factor erythroid 2 related factor 2",
    r"\btnf-?a(?:lpha)?\b": "tumor necrosis factor alpha",
    r"\bil-?(\d+)\b": r"interleukin \1",
    r"\batp\b": "adenosine triphosphate",
    r"\bbbb\b": "blood brain barrier",
    r"\bepithelial[- ]mesenchymal\s+transition\b": "emt",
    r"\bhpg\b": "hypothalamic pituitary gonadal",
    r"\bhpt\b": "hypothalamic pituitary thyroid",
    r"\bt3\b": "triiodothyronine",
    r"\bt4\b": "thyroxine",

    # Neurotoxicology. The table above was built around oxidative stress and
    # general tox; a neuro corpus abbreviates a different vocabulary, and an
    # unexpanded "OL" scored 0.40 against "oligodendrocytes".
    r"\bols?\b": "oligodendrocyte",
    r"\boligodendrocytes\b": "oligodendrocyte",
    r"\bopcs?\b": "oligodendrocyte progenitor cell",
    r"\bmbp\b": "myelin basic protein",
    r"\bplp\b": "proteolipid protein",
    r"\bmag\b": "myelin associated glycoprotein",
    r"\bmog\b": "myelin oligodendrocyte glycoprotein",
    r"\bcns\b": "central nervous system",
    r"\bpns\b": "peripheral nervous system",
    r"\bbdnf\b": "brain derived neurotrophic factor",
    r"\bnmda\b": "n methyl d aspartate",
    r"\bampa\b": "alpha amino 3 hydroxy 5 methyl 4 isoxazolepropionic acid",
    r"\bgaba\b": "gamma aminobutyric acid",
    r"\bach\b": "acetylcholine",
    r"\bache\b": "acetylcholinesterase",
    r"\bvgcc\b": "voltage gated calcium channel",
    r"\bvgsc\b": "voltage gated sodium channel",
    r"\bnav\s*1\.?(\d+)\b": r"voltage gated sodium channel 1\1",
    r"\bcav\s*1\.?(\d+)\b": r"voltage gated calcium channel 1\1",
    r"\bstxs?\b": "saxitoxin",
    r"\bttx\b": "tetrodotoxin",
}

#: Word-level synonyms folded to one representative.
_SYNONYMS: dict[str, str] = {
    "elevated": "increased",
    "elevation": "increased",
    "raised": "increased",
    "enhanced": "increased",
    "enhancement": "increased",
    "augmented": "increased",
    "upregulated": "increased",
    "upregulation": "increased",
    "accumulation": "increased",
    "induction": "increased",
    "induced": "increased",
    "overproduction": "increased",
    "excess": "increased",
    "excessive": "increased",
    "higher": "increased",
    "reduced": "decreased",
    "reduction": "decreased",
    "lowered": "decreased",
    "diminished": "decreased",
    "depletion": "decreased",
    "depleted": "decreased",
    "downregulated": "decreased",
    "downregulation": "decreased",
    "suppression": "decreased",
    "suppressed": "decreased",
    "loss": "decreased",
    "lower": "decreased",
    "inhibition": "decreased",
    "inhibited": "decreased",
    "impairment": "impaired",
    "dysfunction": "impaired",
    "disruption": "impaired",
    "disrupted": "impaired",
    "damage": "damaged",
    "injury": "damaged",
    "activation": "activated",
    "levels": "level",
    "cells": "cell",
    "tissues": "tissue",
    "responses": "response",
    "changes": "change",
    "mitochondria": "mitochondrial",
    "neurons": "neuronal",
    "neuron": "neuronal",
    "hepatocytes": "hepatocyte",
    "oxidant": "oxidative",
}

#: Filler words dropped before comparison. Note that direction words are NOT
#: dropped — they are load-bearing.
_STOPWORDS = {
    "of", "in", "the", "a", "an", "and", "to", "on", "at", "for", "with",
    "by", "from", "into", "level", "amount", "content", "status", "rate",
}

#: Direction polarity detection. Two labels with different polarity are never
#: fuzzy-merged.
#: Words that make a Key Event name state its own direction.
#:
#: The list is longer than the obvious "increased"/"decreased" pair because
#: real AOP names rarely use those. A neuroscience corpus writes "reduced
#: presynaptic excitability", "dispersed sodium channel clustering",
#: "shortened distal internode", "conduction failures" and "abolished
#: spiking" — every one of them directional, none of them matching a short
#: keyword list. Missing them has two costs: two labels of opposite meaning
#: can be merged, and a name that already says "reduced" gets an arrow
#: prefixed to it and reads "↓ reduced …".
_POSITIVE_MARKERS = re.compile(
    r"\b(increased|elevated|enhanced|excess\w*|activated|induction|"
    r"accumulation|overproduction|gain|hyper\w*|stimulat\w*|proliferat\w*|"
    r"upregulat\w*|prolonged|aberrant)\b", re.I
)
_NEGATIVE_MARKERS = re.compile(
    r"\b(decreased|reduced|diminish\w*|lower\w*|inhibited|impaired|damaged|"
    r"loss|lost|depletion|deficiency|deficit\w*|hypo\w*|failure|failures|"
    r"arrest|apoptosis|death|degeneration|atrophy|downregulat\w*|"
    r"abolish\w*|disrupt\w*|disorganiz\w*|disorganis\w*|dispersed|"
    r"shortened|thinner|delayed|defective|blocked)\b", re.I
)

_PUNCT_RE = re.compile(r"[^a-z0-9\s\-]")
_WS_RE = re.compile(r"\s+")

#: Modifiers that hedge an association without changing which event is meant.
#: "myelin-associated genes" and "myelin-related genes" are the same thing said
#: twice; keeping the modifier scored them at 0.68, below any threshold that
#: could safely merge them.
_EQUIVALENT_MODIFIERS = {
    "associated", "related", "linked", "dependent", "specific", "mediated",
    "positive", "containing",
}

#: Experimental context that belongs to the study, not to the Key Event.
#: An AOP Key Event is a biological state that generalises past the experiment
#: that revealed it, so "in Oln93 cells" and "in mature oligodendrocytes" are
#: describing where it was observed. That belongs in `study_design` and the
#: applicability fields — which the pipeline already extracts separately — not
#: in the identity of the event.
_CONTEXT_PHRASE_RE = re.compile(
    r"\s+in\s+(?:mature\s+|immature\s+|primary\s+|cultured\s+|isolated\s+)?"
    r"[a-z0-9\-]+(?:\s+cells?|\s+cell\s+lines?|\s+cultures?)\b",
    re.I,
)

#: Maturity and preparation adjectives, dropped for the same reason.
_CONTEXT_WORDS = {
    "mature", "immature", "primary", "cultured", "isolated", "vitro", "vivo",
    "adult", "neonatal", "juvenile",
}


#: Direction words that papers split with a hyphen or a space. Rejoining them
#: before anything else means the synonym table and the polarity markers each
#: only need the one closed-up spelling.
_DIRECTION_COMPOUND_RE = re.compile(
    r"\b(down|up|over|under|co)[\s\-]+(regulat\w*|express\w*|activat\w*|produc\w*)",
    re.I,
)


def _join_direction_compounds(text: str) -> str:
    """"Down-regulation" and "down regulation" -> "downregulation"."""
    return _DIRECTION_COMPOUND_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)


def _flatten_hyphens(text: str) -> str:
    """
    Split hyphenated compounds into separate words.

    `_PUNCT_RE` deliberately preserves hyphens, which meant "down-regulation"
    never matched the synonym entry for "downregulation" — it stayed a single
    unknown token, kept polarity 0, and so was excluded from fuzzy merging
    before similarity was ever computed. Splitting also exposes the modifier
    in "myelin-related" so it can be dropped.

    Runs after abbreviation expansion, whose patterns already tolerate the
    hyphen they need ("nf-?kb").
    """
    return text.replace("-", " ")


def normalise_label(label: str) -> str:
    """Reduce a raw KE label to a comparable canonical string."""
    text = (label or "").lower().strip()
    if not text:
        return ""

    text = _join_direction_compounds(text)

    # "increase in X" / "decrease of X" -> "increased X"
    text = re.sub(r"\bincreases?\s+(?:in|of)\s+", "increased ", text)
    text = re.sub(r"\bdecreases?\s+(?:in|of)\s+", "decreased ", text)
    text = re.sub(r"\bactivation\s+of\s+", "activated ", text)
    text = re.sub(r"\binhibition\s+of\s+", "decreased ", text)
    text = re.sub(r"\bimpairment\s+of\s+", "impaired ", text)
    text = re.sub(r"\bloss\s+of\s+", "decreased ", text)
    text = re.sub(r"\bdisruption\s+of\s+", "impaired ", text)

    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.I)

    text = _CONTEXT_PHRASE_RE.sub(" ", text)
    text = _flatten_hyphens(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    tokens: list[str] = []
    for token in text.split():
        token = _SYNONYMS.get(token, token)
        if token in _STOPWORDS or token in _CONTEXT_WORDS:
            continue
        if token in _EQUIVALENT_MODIFIERS:
            continue
        tokens.append(token)

    # Deduplicate consecutive repeats produced by synonym folding
    # ("increased increased ...").
    deduped: list[str] = []
    for token in tokens:
        if not deduped or deduped[-1] != token:
            deduped.append(token)

    return " ".join(deduped)


def polarity(label: str) -> int:
    """
    Return +1 (increase), -1 (decrease/impairment) or 0 (neutral).

    Hyphens are flattened first. "Down-regulation" otherwise failed to match
    `downregulat\\w*` and came back neutral, which quietly excluded it from
    merging with the identical "Downregulation" label — direction is the one
    signal here that must never be missed by accident.
    """
    flattened = _flatten_hyphens(_join_direction_compounds((label or "").lower()))
    pos = bool(_POSITIVE_MARKERS.search(flattened))
    neg = bool(_NEGATIVE_MARKERS.search(flattened))
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


def token_key(normalised: str) -> str:
    """
    Order-independent fingerprint of a normalised label.

    "hepatocyte apoptosis" and "apoptosis of hepatocytes" normalise to the same
    two content words in different orders. A Key Event is a thing, not a
    sentence, so word order carries no meaning here and the two are the same
    event. Sorting the tokens makes that identity explicit.
    """
    return " ".join(sorted(set(normalised.split())))


def similarity(a: str, b: str) -> float:
    """Similarity between two already-normalised labels."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # Compare both in written order and in canonical token order, and keep the
    # better of the two, so a re-ordered phrasing is not penalised.
    seq = max(
        SequenceMatcher(None, a, b, autojunk=False).ratio(),
        SequenceMatcher(None, token_key(a), token_key(b), autojunk=False).ratio(),
    )
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    containment = 1.0 if (a in b or b in a) else 0.0
    return round(0.45 * seq + 0.40 * jaccard + 0.15 * containment, 4)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

@dataclass
class _Cluster:
    """Working state for one group of equivalent labels."""

    key: str
    normalised: str
    levels: Counter = field(default_factory=Counter)
    raw_labels: Counter = field(default_factory=Counter)
    aopwiki_ids: Counter = field(default_factory=Counter)
    ontology: Optional[OntologyMatch] = None
    polarity: int = 0
    n_rows: int = 0

    def dominant_level(self) -> str:
        if not self.levels:
            return "Molecular"
        # Ties break toward the earlier (more upstream) level, which keeps a
        # KE from drifting rightwards across lanes between sessions.
        best = max(self.levels.values())
        candidates = [lvl for lvl, n in self.levels.items() if n == best]
        return min(candidates, key=level_index)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class NormalizationReport:
    """What the normalizer did, for display in the UI."""

    n_raw_labels: int
    n_canonical: int
    n_merged_by_ontology: int
    n_merged_by_aopwiki: int
    n_merged_by_string: int
    ontology_coverage: float   # fraction of canonical KEs with an ontology term

    #: Set by `normalize_table1` when OLS4 was consulted; None when annotation
    #: was switched off, which is different from having been consulted and
    #: having found nothing.
    n_ontology_lookups: Optional[int] = None
    ontology_error: Optional[str] = None

    #: Labels that appear in Table 1 under more than one cell type. These are
    #: the merges most likely to be wrong and least likely to look wrong: the
    #: strings are identical, so no clustering threshold and no curator
    #: reading a list of names will separate them. Surfaced rather than split
    #: automatically, because splitting one node into two changes every edge
    #: attached to it and that is a decision with a person's name on it.
    cell_type_conflicts: list[dict] = field(default_factory=list)

    #: One entry per raw label: which canonical event it went to, whether it
    #: is the name of that event or one of its synonyms, and the rule and
    #: evidence that put it there. This is the account of the step between
    #: Table 1 and the canonical Key Events. Before it existed the only thing
    #: shown was the pair of totals, and a curator reading "31 raw labels → 18
    #: canonical Key Events" had no way to find out which thirteen labels
    #: moved, where they went, or why — which made the number read as a loss
    #: of thirteen findings rather than as thirteen wordings for events
    #: already on the list.
    #:
    #: Keys: raw_label, canonical_name, level, mentions, is_event_name,
    #: basis, detail.
    crosswalk: list[dict] = field(default_factory=list)

    @property
    def compression_pct(self) -> float:
        if not self.n_raw_labels:
            return 0.0
        return 100.0 * (1.0 - self.n_canonical / self.n_raw_labels)

    @property
    def n_raw(self) -> int:
        """Alias for `n_raw_labels`, which the UI reads under this name."""
        return self.n_raw_labels


def collect_raw_kes(table1_df: pd.DataFrame) -> list[tuple[str, str, Optional[int]]]:
    """
    Pull every (label, level, aopwiki_ke_id) triple out of Table 1.

    Both endpoints of every row are collected, since a KE that is downstream in
    one paper is very often upstream in another.
    """
    return [(label, level, wiki_id)
            for label, level, wiki_id, _ in collect_raw_kes_with_context(table1_df)]


def collect_raw_kes_with_context(
    table1_df: pd.DataFrame,
) -> list[tuple[str, str, Optional[int], Optional[str]]]:
    """
    As `collect_raw_kes`, but carrying the cell type each label was seen in.

    Two papers can write the identical string for events that are not the same
    event: "voltage-gated sodium channel" in an oligodendrocyte and in an
    activated microglial cell are different Key Events with different, in
    places opposite, consequences. String clustering cannot tell them apart —
    they are the same string — so the cell type has to travel alongside the
    label and become part of its identity.
    """
    out: list[tuple[str, str, Optional[int], Optional[str]]] = []
    if table1_df is None or table1_df.empty:
        return out

    for _, row in table1_df.iterrows():
        for name_col, level_col, id_col, cell_col in (
            ("upstream_ke_name", "upstream_ke_level", "upstream_ke_id",
             "upstream_cell_type"),
            ("downstream_ke_name", "downstream_ke_level", "downstream_ke_id",
             "downstream_cell_type"),
        ):
            name = row.get(name_col)
            if pd.isna(name) or not str(name).strip():
                continue
            level = row.get(level_col)
            level = str(level).strip() if pd.notna(level) and str(level).strip() else "Molecular"
            wiki_id = row.get(id_col)
            wiki_id = int(wiki_id) if pd.notna(wiki_id) and str(wiki_id).strip() not in ("", "None") else None
            cell = row.get(cell_col)
            cell = str(cell).strip() if pd.notna(cell) and str(cell).strip() else None
            out.append((str(name).strip(), level, wiki_id, cell))
    return out


def build_canonical_kes(
    raw_kes: Sequence[tuple[str, str, Optional[int]]],
    ontology_matches: Optional[dict[str, OntologyMatch]] = None,
    *,
    fuzzy_threshold: float = 0.86,
    respect_polarity: bool = True,
) -> tuple[list[CanonicalKE], dict[str, int], NormalizationReport]:
    """
    Merge raw KE labels into canonical Key Events.

    Parameters
    ----------
    raw_kes
        (label, level, aopwiki_ke_id) triples, typically from `collect_raw_kes`.
    ontology_matches
        Optional map of raw label -> best OntologyMatch, from `ols4_client`.
    fuzzy_threshold
        Minimum similarity for rule 4. Raise it to merge less aggressively.
    respect_polarity
        Keep True unless you have a reason not to; see the module docstring.

    Returns
    -------
    (canonical_kes, label_to_canonical_index, report)
        `label_to_canonical_index` maps every raw label to the position of its
        canonical KE in the returned list. Canonical ids are assigned by the
        store on insert, so here the mapping is positional.
    """
    ontology_matches = ontology_matches or {}

    # --- Stage 1: bucket by normalised string ------------------------------
    buckets: dict[str, _Cluster] = {}
    label_to_bucket: dict[str, str] = {}

    for label, level, wiki_id in raw_kes:
        norm = normalise_label(label)
        if not norm:
            norm = label.strip().lower()
        cluster = buckets.get(norm)
        if cluster is None:
            cluster = _Cluster(key=norm, normalised=norm, polarity=polarity(label))
            buckets[norm] = cluster
        cluster.levels[level] += 1
        cluster.raw_labels[label] += 1
        cluster.n_rows += 1
        if wiki_id is not None:
            cluster.aopwiki_ids[wiki_id] += 1
        label_to_bucket[label] = norm

        match = ontology_matches.get(label)
        if match is not None and (cluster.ontology is None or match.score > cluster.ontology.score):
            cluster.ontology = match

    n_raw_labels = len(label_to_bucket)

    # --- Stage 2: union-find across buckets --------------------------------
    uf = _UnionFind()
    for key in buckets:
        uf.find(key)

    merged_by_ontology = 0
    merged_by_aopwiki = 0
    merged_by_string = 0

    # Why each bucket left its own group, keyed by bucket. Recorded at the
    # moment of the union and never overwritten, so the reason kept is the
    # first and strongest rule that applied — the rules run in order of
    # authority, which is the order the methods text claims they run in.
    #
    # A bucket absent from this map was not absorbed by anything: it is either
    # a singleton or the anchor other buckets were folded into.
    basis: dict[str, tuple[str, str]] = {}

    def _record(bucket_key: str, rule: str, detail: str) -> None:
        basis.setdefault(bucket_key, (rule, detail))

    def _example(bucket_key: str) -> str:
        """The most-used raw wording in a bucket, for naming it to a human."""
        cluster = buckets.get(bucket_key)
        if cluster is None or not cluster.raw_labels:
            return bucket_key
        return cluster.raw_labels.most_common(1)[0][0]

    # Rule 1 — same AOP-Wiki KE id.
    by_wiki: dict[int, list[str]] = defaultdict(list)
    for key, cluster in buckets.items():
        if cluster.aopwiki_ids:
            dominant = cluster.aopwiki_ids.most_common(1)[0][0]
            by_wiki[dominant].append(key)
    for wiki_id, keys in by_wiki.items():
        for other in keys[1:]:
            if uf.find(keys[0]) != uf.find(other):
                uf.union(keys[0], other)
                merged_by_aopwiki += 1
                _record(
                    other,
                    "aopwiki",
                    f"Both extracted with AOP-Wiki Key Event {wiki_id} — "
                    f"same id as “{_example(keys[0])}”.",
                )

    # Rule 2 — same ontology CURIE, provided the match is confident.
    by_curie: dict[str, list[str]] = defaultdict(list)
    for key, cluster in buckets.items():
        if cluster.ontology and cluster.ontology.score >= 0.75 and cluster.ontology.curie:
            by_curie[cluster.ontology.curie].append(key)
    for curie, keys in by_curie.items():
        for other in keys[1:]:
            if uf.find(keys[0]) != uf.find(other):
                # Ontology terms are direction-agnostic ("apoptotic process"
                # matches both increased and decreased apoptosis), so the
                # polarity guard still applies here.
                if respect_polarity and buckets[keys[0]].polarity * buckets[other].polarity < 0:
                    continue
                uf.union(keys[0], other)
                merged_by_ontology += 1
                _score = getattr(buckets[other].ontology, "score", 0.0) or 0.0
                _record(
                    other,
                    "ontology",
                    f"Matched to the same ontology term {curie} as "
                    f"“{_example(keys[0])}” (match score {_score:.2f}).",
                )

    # Rule 3b — identical content words in a different order.
    by_tokens: dict[str, list[str]] = defaultdict(list)
    for key in buckets:
        by_tokens[token_key(key)].append(key)
    for keys_group in by_tokens.values():
        anchor = keys_group[0]
        for other in keys_group[1:]:
            if uf.find(anchor) == uf.find(other):
                continue
            ca, cb = buckets[anchor], buckets[other]
            if respect_polarity and ca.polarity * cb.polarity < 0:
                continue
            if ca.dominant_level() != cb.dominant_level():
                continue
            uf.union(anchor, other)
            merged_by_string += 1
            _record(
                other,
                "token_order",
                f"Same content words as “{_example(anchor)}”, in a different "
                f"order, at the same biological level.",
            )

    # Rule 4 — fuzzy string similarity within the same level and polarity.
    keys = sorted(buckets)
    for i, a in enumerate(keys):
        ca = buckets[a]
        for b in keys[i + 1 :]:
            cb = buckets[b]
            if uf.find(a) == uf.find(b):
                continue
            if respect_polarity and ca.polarity * cb.polarity < 0:
                continue
            if ca.dominant_level() != cb.dominant_level():
                continue
            score = similarity(a, b)
            if score >= fuzzy_threshold:
                uf.union(a, b)
                merged_by_string += 1
                _record(
                    b,
                    "lexical",
                    f"Wording {score:.2f} similar to “{_example(a)}” — above "
                    f"the {fuzzy_threshold:.2f} threshold — at the same level "
                    f"and the same direction polarity.",
                )

    # --- Stage 3: materialise canonical KEs --------------------------------
    groups: dict[str, list[str]] = defaultdict(list)
    for key in buckets:
        groups[uf.find(key)].append(key)

    canonical_kes: list[CanonicalKE] = []
    bucket_to_index: dict[str, int] = {}

    for root, member_keys in sorted(groups.items(), key=lambda kv: -sum(buckets[k].n_rows for k in kv[1])):
        levels: Counter = Counter()
        raw_labels: Counter = Counter()
        wiki_ids: Counter = Counter()
        best_ontology: Optional[OntologyMatch] = None
        n_rows = 0

        for key in member_keys:
            cluster = buckets[key]
            levels.update(cluster.levels)
            raw_labels.update(cluster.raw_labels)
            wiki_ids.update(cluster.aopwiki_ids)
            n_rows += cluster.n_rows
            if cluster.ontology and (best_ontology is None or cluster.ontology.score > best_ontology.score):
                best_ontology = cluster.ontology

        merged = _Cluster(
            key=root,
            normalised=root,
            levels=levels,
            raw_labels=raw_labels,
            aopwiki_ids=wiki_ids,
            ontology=best_ontology,
            n_rows=n_rows,
        )

        canonical_name = _choose_canonical_name(merged)
        method = (
            "ontology" if (best_ontology and best_ontology.score >= 0.75)
            else "auto"
        )

        # Why each wording in this group is in this group. Two questions, both
        # asked by a curator looking at a merge: why is this label in the same
        # *bucket* as its neighbours (its string normalised to theirs), and why
        # did that bucket join this *group* (one of the four rules). A label
        # can have both answers, and the group answer is the one that needs
        # defending, so it leads.
        alias_basis: dict[str, list[str]] = {}
        for key in member_keys:
            cluster = buckets[key]
            bucket_labels = list(cluster.raw_labels)
            bucket_rule, bucket_detail = basis.get(key, ("", ""))

            for label in bucket_labels:
                siblings = [other for other in bucket_labels if other != label]
                if bucket_rule:
                    rule, detail = bucket_rule, bucket_detail
                    if siblings:
                        detail = (
                            f"{detail} Also identical after normalisation to "
                            f"“{siblings[0]}”."
                        )
                elif siblings:
                    rule = "normalised_string"
                    detail = (
                        f"Identical to “{siblings[0]}” once case, plurals, "
                        f"punctuation and direction wording are normalised "
                        f"(“{cluster.normalised}”)."
                    )
                elif len(member_keys) > 1:
                    rule = "own_group"
                    detail = (
                        f"The wording the event is named from; "
                        f"{len(member_keys) - 1} other wording(s) were merged "
                        f"into it."
                    )
                else:
                    rule = "own_group"
                    detail = (
                        "Nothing matched it — it stands as its own Key Event "
                        "on its own wording."
                    )
                alias_basis[label] = [rule, detail]

        index = len(canonical_kes)
        canonical_kes.append(
            CanonicalKE(
                canonical_id=None,
                canonical_name=canonical_name,
                level=merged.dominant_level(),
                aliases=[lbl for lbl, _ in raw_labels.most_common()],
                alias_basis=alias_basis,
                ontology_curie=best_ontology.curie if best_ontology else None,
                ontology_iri=best_ontology.iri if best_ontology else None,
                ontology_label=best_ontology.label if best_ontology else None,
                ontology_source=best_ontology.ontology if best_ontology else None,
                ontology_score=best_ontology.score if best_ontology else 0.0,
                aopwiki_ke_id=wiki_ids.most_common(1)[0][0] if wiki_ids else None,
                merge_method=method,
                n_source_rows=n_rows,
            )
        )
        for key in member_keys:
            bucket_to_index[key] = index

    label_to_index = {
        label: bucket_to_index[bucket]
        for label, bucket in label_to_bucket.items()
        if bucket in bucket_to_index
    }

    # How many Table 1 mentions each wording accounts for. Needed so the
    # crosswalk can say that a label folded away carried eleven claims with it
    # rather than looking like a wording nobody used.
    mentions_by_label: Counter = Counter(label for label, _, _ in raw_kes)

    crosswalk: list[dict] = []
    for ke in canonical_kes:
        for label in ke.aliases:
            rule, detail = (ke.alias_basis.get(label) or ["", ""])[:2]
            crosswalk.append(
                {
                    "raw_label": label,
                    "canonical_name": ke.canonical_name,
                    "level": ke.level,
                    "mentions": int(mentions_by_label.get(label, 0)),
                    "is_event_name": label.strip().casefold()
                    == ke.canonical_name.strip().casefold(),
                    "basis": rule,
                    "detail": detail,
                }
            )
    crosswalk.sort(key=lambda row: (row["canonical_name"], not row["is_event_name"],
                                    -row["mentions"], row["raw_label"]))

    with_ontology = sum(1 for ke in canonical_kes if ke.ontology_curie)
    report = NormalizationReport(
        n_raw_labels=n_raw_labels,
        n_canonical=len(canonical_kes),
        n_merged_by_ontology=merged_by_ontology,
        n_merged_by_aopwiki=merged_by_aopwiki,
        n_merged_by_string=merged_by_string,
        ontology_coverage=round(with_ontology / len(canonical_kes), 3) if canonical_kes else 0.0,
        crosswalk=crosswalk,
    )

    return canonical_kes, label_to_index, report


def normalize_table1(
    table1_df: pd.DataFrame,
    *,
    threshold: float = 0.86,
    ols4_enabled: bool = True,
    ols4_min_score: float = 0.45,
    respect_polarity: bool = True,
    progress: Optional[callable] = None,
) -> NormalizationReport:
    """
    Normalize Table 1 end to end and persist the result.

    `build_canonical_kes` is deliberately pure — labels in, records out, no
    network and no database — which makes it testable but leaves three steps
    for somebody else to do: collect the labels, look them up in OLS4, and
    write the result. Every caller needs all three in the same order, so they
    belong here rather than in the UI, where they were previously assumed to
    exist and were never written.

    Returns the report. The canonical KEs themselves are read back with
    `table1_store.load_canonical_kes()`, which is what the rest of the
    pipeline already does.
    """
    # Imported inside the function: `table1_store` imports `schemas` and the
    # extractor, and a module-level import here would close that loop.
    from stage2_extraction import ols4_client, table1_store

    with_context = collect_raw_kes_with_context(table1_df)
    raw_kes = [(label, level, wiki) for label, level, wiki, _ in with_context]
    if not raw_kes:
        table1_store.replace_canonical_kes([], {})
        return NormalizationReport(
            n_raw_labels=0, n_canonical=0, n_merged_by_ontology=0,
            n_merged_by_aopwiki=0, n_merged_by_string=0, ontology_coverage=0.0,
        )

    ontology_matches: dict[str, OntologyMatch] = {}
    n_lookups: Optional[int] = None
    ontology_error: Optional[str] = None

    if ols4_enabled:
        try:
            results = ols4_client.annotate_many(
                [(label, level) for label, level, _ in raw_kes],
                min_score=float(ols4_min_score),
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            # A blocked or slow OLS4 must not cost the curator their run.
            # Clustering still works on strings alone; the report says so.
            ontology_error = f"{type(exc).__name__}: {exc}"
            results = {}
        n_lookups = len(results)
        errors = [r.error for r in results.values() if getattr(r, "error", None)]
        if errors and ontology_error is None:
            ontology_error = f"{len(errors)} of {len(results)} lookups failed."
        ontology_matches = {
            name: result.best
            for name, result in results.items()
            if getattr(result, "best", None) is not None
        }

    canonical_kes, label_to_index, report = build_canonical_kes(
        raw_kes,
        ontology_matches,
        fuzzy_threshold=float(threshold),
        respect_polarity=bool(respect_polarity),
    )

    table1_store.replace_canonical_kes(canonical_kes, label_to_index)

    report.n_ontology_lookups = n_lookups
    report.ontology_error = ontology_error
    report.cell_type_conflicts = find_cell_type_conflicts(with_context)
    return report


def find_context_conflicts(table1_df: pd.DataFrame) -> list[dict]:
    """
    Key Events whose evidence comes from incompatible study models.

    A finding from a spinal-cord-injury remyelination experiment and one from
    normal postnatal development are different biology, and the tool has no
    way to notice: both papers write "oligodendrocyte differentiation", so
    both rows land on one node and the injury result is chained onto the
    developmental pathway. That is where the motor-evoked-potential edge came
    from — a real result from a real paper, attached to a pathway it says
    nothing about.

    Reported rather than split, because unlike cell lineage the right answer
    is often to keep one node and note the difference.
    """
    if table1_df is None or table1_df.empty:
        return []
    if "study_context" not in table1_df.columns:
        return []

    by_event: dict[str, Counter] = defaultdict(Counter)
    for _, row in table1_df.iterrows():
        context = row.get("study_context")
        if pd.isna(context) or not str(context).strip():
            continue
        for column in ("upstream_ke_name", "downstream_ke_name"):
            name = row.get(column)
            if pd.notna(name) and str(name).strip():
                by_event[str(name).strip()][str(context).strip()] += 1

    return sorted(
        (
            {
                "label": label,
                "contexts": [c for c, _ in contexts.most_common()],
                "n_rows": sum(contexts.values()),
            }
            for label, contexts in by_event.items()
            if len(contexts) > 1
        ),
        key=lambda c: -c["n_rows"],
    )


def propose_specific_names(table1_df: pd.DataFrame) -> list[dict]:
    """
    Suggest a precise name for each Key Event from what the rows already say.

    A targeted run names every event after the question, because that is what
    makes separate papers join up — and the cost is a map labelled
    "Voltage-gated sodium channel activity" when the evidence is specifically
    about SCN2A/Nav1.2 in immature oligodendroglia. The direction, the
    isoform and the cell lineage are all on the rows by now; this assembles
    them into the name the map should carry.

    Suggestions only. Renaming is a curator's decision and stays one.
    """
    from stage2_extraction import cell_lineage

    if table1_df is None or table1_df.empty:
        return []

    facts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"target": Counter(), "lineage": Counter(), "change": Counter()}
    )
    for _, row in table1_df.iterrows():
        for side in ("upstream", "downstream"):
            name = row.get(f"{side}_ke_name")
            if pd.isna(name) or not str(name).strip():
                continue
            entry = facts[str(name).strip()]
            for key, column in (
                ("target", f"{side}_target"),
                ("change", f"{side}_change"),
            ):
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    entry[key][str(value).strip()] += 1
            cell = row.get(f"{side}_cell_type")
            if pd.notna(cell) and str(cell).strip():
                name_of = cell_lineage.lineage(str(cell))
                if name_of != cell_lineage.UNSPECIFIED:
                    entry["lineage"][name_of] += 1

    proposals: list[dict] = []
    for label, entry in facts.items():
        target = entry["target"].most_common(1)[0][0] if entry["target"] else None
        lineage_name = entry["lineage"].most_common(1)[0][0] if entry["lineage"] else None
        change = entry["change"].most_common(1)[0][0] if entry["change"] else None

        # Only add what the name does not already carry, and only when the
        # rows agree. A split vote is the signal to leave the name general.
        stated = polarity(label)
        direction_word = ""
        if change and not stated and len(entry["change"]) == 1:
            direction_word = {"increased": "Increased", "decreased": "Decreased",
                              "lost": "Loss of", "abolished": "Loss of"}.get(
                change.lower(), "")

        parts = [p for p in (direction_word, target or label) if p]
        suggested = " ".join(parts)
        if target and target.lower() not in label.lower():
            suggested = f"{suggested} ({label})" if direction_word else f"{target} ({label})"
        if lineage_name and len(entry["lineage"]) == 1:
            suggested = f"{suggested} {cell_lineage.suffix_for(lineage_name)}"

        suggested = " ".join(suggested.split())
        if suggested and suggested.lower() != label.lower():
            proposals.append(
                {
                    "label": label,
                    "suggested": suggested,
                    "target": target,
                    "lineage": lineage_name,
                    "direction": direction_word or None,
                    "n_rows": sum(entry["change"].values()) or 0,
                }
            )
    return sorted(proposals, key=lambda p: p["label"])


def find_cell_type_conflicts(
    raw_kes_with_context: Sequence[tuple[str, str, Optional[int], Optional[str]]],
) -> list[dict]:
    """
    Labels written identically for events observed in different cell types.

    This is the failure that string clustering cannot see and a curator
    reading a list of canonical names cannot see either, because there is
    nothing to see: one string, one node, two biologies. It has to be found by
    looking at what the rows say about where each event was observed.
    """
    from stage2_extraction import cell_lineage

    seen: dict[str, Counter] = defaultdict(Counter)
    for label, _level, _wiki, cell in raw_kes_with_context:
        if cell:
            seen[label][cell] += 1

    conflicts = []
    for label, cells in seen.items():
        # Compared by lineage, not by string. Nine spellings of
        # "oligodendrocyte precursor cell" are one Key Event; an
        # oligodendrocyte and a nerve terminal are two, and only the second
        # is worth interrupting anyone about.
        lineages = cell_lineage.distinct_lineages(cells)
        if len(lineages) < 2:
            continue
        by_lineage: dict[str, list[str]] = defaultdict(list)
        for cell, _n in cells.most_common():
            by_lineage[cell_lineage.lineage(cell)].append(cell)
        conflicts.append(
            {
                "label": label,
                "lineages": lineages,
                "cell_types": [c for c, _ in cells.most_common()],
                "examples": {
                    name: by_lineage[name][:3] for name in lineages
                },
                "n_rows": sum(cells.values()),
            }
        )
    return sorted(conflicts, key=lambda c: -c["n_rows"])


def _choose_canonical_name(cluster: _Cluster) -> str:
    """
    Pick the display name for a merged group.

    Preference order: the most frequently written raw label, unless a confident
    ontology label exists and the raw labels disagree with each other — in
    which case the ontology label is the more neutral choice. Direction is
    re-attached to the ontology label when the raw labels carry one, because
    ontologies name processes ("apoptotic process") without direction.
    """
    if not cluster.raw_labels:
        return cluster.normalised or "unnamed key event"

    ranked = cluster.raw_labels.most_common()
    most_common_label, most_common_n = ranked[0]

    distinct = len(ranked)
    total = sum(cluster.raw_labels.values())
    dominant = most_common_n / total if total else 0.0

    ontology = cluster.ontology
    if ontology and ontology.score >= 0.8 and distinct > 2 and dominant < 0.6:
        name = ontology.label
        pol = cluster.polarity or _majority_polarity(cluster.raw_labels)
        if pol > 0 and not _POSITIVE_MARKERS.search(name):
            name = f"Increased {name[0].lower()}{name[1:]}"
        elif pol < 0 and not _NEGATIVE_MARKERS.search(name):
            name = f"Decreased {name[0].lower()}{name[1:]}"
        return name

    return most_common_label


def _majority_polarity(raw_labels: Counter) -> int:
    score = 0
    for label, n in raw_labels.items():
        score += polarity(label) * n
    return (score > 0) - (score < 0)


def apply_canonical_names(
    table1_df: pd.DataFrame,
    canonical_kes: Sequence[CanonicalKE],
    label_to_index: dict[str, int],
) -> pd.DataFrame:
    """
    Return a copy of Table 1 with canonical KE columns added.

    Adds: upstream_ke_canonical, downstream_ke_canonical, and their level and
    index columns. The original *_ke_name columns are left untouched so the raw
    view keeps showing exactly what each paper said.
    """
    df = table1_df.copy()
    if df.empty:
        for col in (
            "upstream_ke_canonical", "downstream_ke_canonical",
            "upstream_ke_canonical_index", "downstream_ke_canonical_index",
            "upstream_ke_canonical_level", "downstream_ke_canonical_level",
        ):
            df[col] = pd.Series(dtype="object")
        return df

    def resolve(name, fallback_level):
        if pd.isna(name):
            return None, None, fallback_level
        idx = label_to_index.get(str(name).strip())
        if idx is None or idx >= len(canonical_kes):
            return str(name).strip(), None, fallback_level
        ke = canonical_kes[idx]
        return ke.canonical_name, idx, ke.level

    up_names, up_idx, up_levels = [], [], []
    down_names, down_idx, down_levels = [], [], []

    for _, row in df.iterrows():
        n, i, lvl = resolve(row.get("upstream_ke_name"), row.get("upstream_ke_level"))
        up_names.append(n); up_idx.append(i); up_levels.append(lvl)
        n, i, lvl = resolve(row.get("downstream_ke_name"), row.get("downstream_ke_level"))
        down_names.append(n); down_idx.append(i); down_levels.append(lvl)

    df["upstream_ke_canonical"] = up_names
    df["upstream_ke_canonical_index"] = up_idx
    df["upstream_ke_canonical_level"] = up_levels
    df["downstream_ke_canonical"] = down_names
    df["downstream_ke_canonical_index"] = down_idx
    df["downstream_ke_canonical_level"] = down_levels
    return df


# ---------------------------------------------------------------------------
# Merge suggestions
# ---------------------------------------------------------------------------

#: Similarity below the merge threshold but high enough to be worth a human
#: glance. Pairs in this band are the ones no string metric can settle, so they
#: are suggested rather than merged.
#:
#: Calibrated against real pairs rather than chosen by intuition, because
#: `similarity` is a weighted blend (0.45 sequence + 0.40 Jaccard + 0.15
#: containment) that compresses scores — one differing token in a four-token
#: label costs about 0.16 on the Jaccard term alone. Pairs a curator would
#: merge score lower than the raw ratio suggests:
#:
#:     axon degeneration      / axonal degeneration            0.56
#:     impaired mitochondrial function / ... respiration       0.57
#:     decreased myelin gene / protein expression              0.64
#:
#: while clearly distinct events sit around 0.40. A floor of 0.55 catches the
#: former without dragging in the latter. It is deliberately generous: a false
#: suggestion costs one click to dismiss, a missed duplicate silently splits an
#: event across the map.
REVIEW_FLOOR = 0.55

#: Share of the shorter label's content words that must appear in the longer
#: one before the pair is suggested regardless of blended similarity.
#:
#: Calibrated on the pairs this missed. The two heminode labels share
#: dispersed / sodium / channel / clustering / heminode — five of the shorter
#: label's seven words, 0.71 — and differ only in that one of them also
#: mentions the shortened internode. "Decreased myelin thickness" and
#: "decreased myelin-associated gene expression" share decreased / myelin,
#: two of three, 0.67, and are genuinely different events. The line goes
#: between them. Erring low costs a click to dismiss; erring high splits one
#: event across the map, which nobody notices.
CONTAINMENT_FLOOR = 0.70


def _bare_tokens(normalised: str) -> set[str]:
    """
    Content words, plural-insensitive.

    "heminode" and "heminodes" are the same word and were being counted as
    two, which alone dropped the heminode pair below any workable threshold.
    """
    out: set[str] = set()
    for token in normalised.split():
        if token in _STOPWORDS:
            continue
        out.add(token[:-1] if len(token) > 4 and token.endswith("s") else token)
    return out


def suggest_merges(
    canonical_df: pd.DataFrame,
    *,
    floor: float = REVIEW_FLOOR,
    ceiling: float = 0.86,
    same_level_only: bool = True,
    respect_polarity: bool = True,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Near-miss pairs of canonical Key Events, for a curator to accept or reject.

    Runs after normalization, over the canonical names the user actually sees,
    so a suggestion always refers to something on screen. Pairs whose polarity
    is opposed are never suggested — "increased" and "decreased apoptosis" look
    almost identical to any string metric and must not be offered as the same
    event.

    Returns columns: source_id, source_name, target_id, target_name, level,
    similarity — sorted most similar first.
    """
    if canonical_df is None or canonical_df.empty:
        return pd.DataFrame()

    rows = [
        {
            "id": int(r["canonical_id"]),
            "name": str(r["canonical_name"]),
            "level": str(r.get("level") or ""),
            "norm": normalise_label(str(r["canonical_name"])),
            "pol": polarity(str(r["canonical_name"])),
            "n_rows": int(r.get("n_source_rows") or 0),
        }
        for _, r in canonical_df.iterrows()
    ]

    suggestions: list[dict] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if same_level_only and a["level"] != b["level"]:
                continue
            if respect_polarity and a["pol"] * b["pol"] < 0:
                continue
            score = similarity(a["norm"], b["norm"])

            # A blended similarity punishes a label for saying MORE. These two
            # are the same event:
            #
            #   dispersed voltage-gated sodium channel clustering at heminodes
            #   shortened distal internode and dispersed sodium channel
            #       clustering at heminode
            #
            # and they score 0.47 — below the suggestion floor — purely
            # because the second describes a second finding in the same
            # sentence. Descriptive AOP names do that constantly. So a pair
            # also qualifies when one label's content words are almost wholly
            # contained in the other's, whatever the blended score says.
            tokens_a = _bare_tokens(a["norm"])
            tokens_b = _bare_tokens(b["norm"])
            contained = 0.0
            if tokens_a and tokens_b:
                overlap = len(tokens_a & tokens_b)
                contained = overlap / min(len(tokens_a), len(tokens_b))

            if not (floor <= score < ceiling) and contained < CONTAINMENT_FLOOR:
                continue
            if score >= ceiling:
                continue
            # Merging the smaller cluster into the larger keeps the better
            # supported name as the survivor by default.
            source, target = (a, b) if a["n_rows"] <= b["n_rows"] else (b, a)
            suggestions.append(
                {
                    "source_id": source["id"],
                    "source_name": source["name"],
                    "target_id": target["id"],
                    "target_name": target["name"],
                    "level": a["level"],
                    "similarity": round(score, 3),
                }
            )

    if not suggestions:
        return pd.DataFrame()

    out = pd.DataFrame(suggestions).sort_values("similarity", ascending=False)
    return out.head(limit).reset_index(drop=True)


__all__ = [
    "normalise_label",
    "polarity",
    "similarity",
    "token_key",
    "collect_raw_kes",
    "collect_raw_kes_with_context",
    "find_cell_type_conflicts",
    "find_context_conflicts",
    "propose_specific_names",
    "build_canonical_kes",
    "normalize_table1",
    "apply_canonical_names",
    "suggest_merges",
    "REVIEW_FLOOR",
    "NormalizationReport",
]
