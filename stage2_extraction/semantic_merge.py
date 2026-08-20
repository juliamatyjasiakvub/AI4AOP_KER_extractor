from __future__ import annotations

"""
Semantic classification of proposed Key Event merges.

The problem this replaces
-------------------------
`ke_normalizer.suggest_merges` ranks candidate pairs by string similarity. That
is a reasonable way to *find* candidates and a dangerous way to *judge* them,
because the strings that matter most in an AOP differ by one word:

    restored nodal protein organization   vs  disrupted nodal protein organization
    no change in OPC proliferation        vs  increased OPC proliferation
    NaV1.2                                vs  voltage-gated sodium channel

A lexical metric scores each of those pairs high. The first two are direct
contradictions; the third is a subtype and its class, whose evidence must never
be pooled. Merging any of them corrupts the map in a way that is invisible
afterwards — the aliases are gone and the resulting Key Event reads as though
the papers agreed.

What this module does instead
-----------------------------
Every proposed relationship is classified into exactly one of:

    equivalent                  the same Key Event, said twice
    broader_than                A subsumes B
    narrower_than               A is a special case of B
    related_but_distinct        genuinely different events that co-occur
    contradictory_or_incompatible   opposed direction, or opposed state
    uncertain                   not enough signal to say

Only `equivalent` may be merged. Everything else is offered to the curator as
a different action — map to a broader concept, record a biological relation,
or keep separate — and the classification is stored alongside the decision.

Eight checks, run in order, each returning pass / fail / unknown:

    1. object type compatibility
    2. entity vs. biological event
    3. biological process and participants
    4. direction and state
    5. biological level
    6. ontology identifiers and hierarchy
    7. semantic entailment or contradiction
    8. explanation for the curator

Checks 1, 2 and 4 are vetoes: a failure there fixes the relationship as
contradictory or distinct no matter how similar the strings are. The remaining
checks can only ever *downgrade* a pair from equivalent, never upgrade one.

On language models
------------------
An LLM or an embedding model may rank which candidates a curator sees first —
`rank_candidates` takes an optional scorer for exactly that. Neither may set
`mergeable`, and nothing in this module merges anything: it returns a judgement
and a rationale, and the curator acts on it.
"""

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence

from stage2_extraction.ke_normalizer import (
    normalise_label,
    similarity,
    token_key,
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Relationship(str, Enum):
    """How two candidate records stand to one another."""

    EQUIVALENT = "equivalent"
    BROADER_THAN = "broader_than"
    NARROWER_THAN = "narrower_than"
    RELATED_DISTINCT = "related_but_distinct"
    CONTRADICTORY = "contradictory_or_incompatible"
    UNCERTAIN = "uncertain"

    @property
    def label(self) -> str:
        return {
            "equivalent": "Equivalent",
            "broader_than": "Broader than",
            "narrower_than": "Narrower than",
            "related_but_distinct": "Related but distinct",
            "contradictory_or_incompatible": "Contradictory or incompatible",
            "uncertain": "Uncertain",
        }[self.value]


#: The one relationship that permits a merge. Kept as a set of one so the
#: guard reads as a membership test rather than an equality that could be
#: loosened later without anyone noticing.
MERGEABLE = frozenset({Relationship.EQUIVALENT})


class ObjectType(str, Enum):
    """What kind of thing a label names."""

    EVENT = "event"              # a change or process: "increased apoptosis"
    ENTITY = "entity"            # a thing: "NaV1.2", "myelin basic protein"
    OBSERVATION = "observation"  # a study result: "no change in proliferation"
    UNKNOWN = "unknown"


#: Words that mark a *result reported by a study* rather than a Key Event.
#:
#: "No change in OPC proliferation" is a finding about an experiment. It has no
#: place as a node on a map of what happens, because nothing happened — and
#: treating it as one produces a Key Event that contradicts its own neighbours.
_NO_CHANGE_RE = re.compile(
    r"\b(?:"
    r"no\s+(?:significant\s+)?(?:change|changes|difference|differences|effect|"
    r"effects|alteration|alterations|impact)"
    r"|unchanged|unaltered|unaffected|not\s+(?:significantly\s+)?(?:changed|"
    r"altered|affected|different)"
    r"|without\s+(?:significant\s+)?(?:change|alteration|effect)"
    r"|comparable\s+to\s+control"
    r"|similar\s+to\s+control"
    r")\b",
    re.I,
)

#: Direction markers. Broader than `ke_normalizer`'s pair because this module
#: has to separate states that a purely quantitative reading treats alike:
#: "restored" and "disrupted" are both changes, and they are opposite ones.
_INCREASE_RE = re.compile(
    r"\b(?:increase[sd]?|increasing|elevated|elevation|enhanced|enhancement|"
    r"induced|induction|activated|activation|accumulation|accumulated|"
    r"upregulat\w*|over[\s\-]?express\w*|overproduction|hyper\w+|gain|"
    r"stimulat\w*|proliferat\w*|potentiat\w*|augment\w*|excess\w*|"
    r"prolonged|greater|higher)\b",
    re.I,
)
_DECREASE_RE = re.compile(
    r"\b(?:decrease[sd]?|decreasing|reduced|reduction|diminish\w*|lowered|"
    r"inhibit\w*|suppress\w*|attenuat\w*|downregulat\w*|under[\s\-]?express\w*|"
    r"depletion|depleted|deficien\w*|deficit\w*|loss|lost|hypo\w+|"
    r"blocked|blockade|abolish\w*|shorter|lower|fewer|slowed|delayed)\b",
    re.I,
)

#: Structural damage, distinct from a quantitative decrease. "Disrupted nodal
#: protein organization" is not less organization on a dial, it is organization
#: broken — and its opposite is restoration, not increase.
_DISRUPTION_RE = re.compile(
    r"\b(?:disrupt\w*|disorganiz\w*|disorganis\w*|disorder\w*|damage[sd]?|"
    r"damaging|degenerat\w*|degradat\w*|degraded|breakdown|fragment\w*|"
    r"lesion\w*|injur\w*|impair\w*|abnormal\w*|aberrant|malform\w*|"
    r"demyelinat\w*|denervat\w*|dysfunction\w*|dysregulat\w*|perturb\w*|"
    r"compromis\w*|deteriorat\w*|disintegrat\w*)\b",
    re.I,
)
_RESTORATION_RE = re.compile(
    r"\b(?:restor\w*|recover\w*|rescu\w*|normaliz\w*|normalis\w*|"
    r"re[\s\-]?establish\w*|reorganiz\w*|reorganis\w*|remyelinat\w*|"
    r"reinnervat\w*|repair\w*|preserv\w*|maintain\w*|protect\w*|"
    r"amelior\w*|reversal|reversed|improv\w*|regenerat\w*)\b",
    re.I,
)

#: Change words that carry no direction.
#:
#: "Altered Nav channel clustering" and "shortened internodes" both say
#: something happened; neither says which way on a signed axis, and neither
#: matched any of the four polarity patterns above. `object_type` was
#: therefore calling them ENTITY — the same verdict it gives "voltage-gated
#: sodium channels" — which is wrong in a way that matters, because the
#: extraction guard uses that verdict to decide whether the model returned a
#: bare noun phrase.
#:
#: Kept separate from `_DISRUPTION_RE` deliberately. Feeding these into the
#: polarity machinery would give "altered X" an integrity polarity it does not
#: have, and the polarity guard would then refuse merges on the strength of a
#: direction nobody stated. This pattern answers only "is this a change at
#: all", which is the one question `object_type` asks.
_UNSIGNED_CHANGE_RE = re.compile(
    r"\b(?:alter\w*|chang\w*|modif\w*|shift\w*|redistribut\w*|"
    r"mislocali[sz]\w*|dispers\w*|shorten\w*|lengthen\w*|thicken\w*|"
    r"thinn\w*|widen\w*|narrow\w*|remodel\w*|reposition\w*)\b",
    re.I,
)

#: Words that make a label a process or a change — the test for "event".
_PROCESS_RE = re.compile(
    r"\b(?:\w*ation|\w*ition|\w*ysis|\w*osis|\w*esis|\w*olysis|\w*aemia|\w*emia|"
    r"release|uptake|influx|efflux|transport|signalling|signaling|binding|"
    r"expression|response|flux|turnover|death|survival|migration|"
    r"differentiation|maturation|myelination|conduction|excitability|"
    r"permeability|viability|integrity|organization|organisation|function|"
    r"level|levels|activity|number|density|count|rate|failure|arrest|"
    r"leakage|swelling|shrinkage|stress|potential|"
    # Structural processes an AOP records that no suffix rule catches.
    r"clustering|spacing|targeting|trafficking|sprouting|pruning|"
    r"branching|spiking|firing)\b",
    re.I,
)

#: Ion-channel and receptor subtype patterns, mapped to the class they belong
#: to. Deliberately small and explicit: the point is to get the well-known
#: subtype-vs-class traps right, not to pretend at general coverage. When a
#: pattern does not fire the classifier falls back to `uncertain`, which is the
#: correct answer for "I do not know whether these are the same".
_SUBTYPE_CLASSES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bna[\s_]?v\s?1\.\d+\b|\bscn\d+a\b", re.I),
     "voltage gated sodium channel"),
    (re.compile(r"\bk[\s_]?v\s?\d+\.\d+\b|\bkcn[a-z]\d+\b", re.I),
     "voltage gated potassium channel"),
    (re.compile(r"\bca[\s_]?v\s?\d+\.\d+\b|\bcacna\d[a-z]\b", re.I),
     "voltage gated calcium channel"),
    (re.compile(r"\bglun\d[a-z]?\b|\bnr2[a-d]\b|\bgrin\d[a-z]?\b", re.I),
     "nmda receptor"),
    (re.compile(r"\bglua\d\b|\bgria\d\b", re.I), "ampa receptor"),
    (re.compile(r"\bgabra\d\b|\bgabrb\d\b", re.I), "gaba a receptor"),
    (re.compile(r"\bnachr\s?alpha\s?\d\b|\bchrna\d\b", re.I),
     "nicotinic acetylcholine receptor"),
)

#: Head words naming a *class* of things. A label containing one of these and
#: nothing subtype-specific is the general case.
_CLASS_HEAD_RE = re.compile(
    r"\b(?:channels?|receptors?|transporters?|enzymes?|kinases?|proteins?|"
    r"cells?|genes?|family|families|class|classes|subtypes?|isoforms?|"
    r"pathways?|systems?)\b",
    re.I,
)

#: Qualifiers that narrow a label without changing what process it names.
#: "Mitochondrial ROS accumulation" is ROS accumulation, in one compartment.
_NARROWING_QUALIFIERS = frozenset({
    "mitochondrial", "cytosolic", "nuclear", "membrane", "synaptic",
    "axonal", "dendritic", "presynaptic", "postsynaptic", "extracellular",
    "intracellular", "peripheral", "central", "cortical", "hippocampal",
    "cerebellar", "striatal", "hepatic", "renal", "cardiac", "pulmonary",
    "neuronal", "glial", "astrocytic", "microglial", "oligodendrocyte",
    "oligodendroglial", "endothelial", "epithelial", "nodal", "paranodal",
    "juxtaparanodal", "myelin", "mature", "immature",
})

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.\-]*")


# ---------------------------------------------------------------------------
# Records and results
# ---------------------------------------------------------------------------

@dataclass
class KERecord:
    """
    One thing being compared — a canonical Key Event or a raw extracted label.

    Everything except `label` is optional so a bare pair of strings can still
    be classified; the checks that need the missing field return `unknown`
    rather than guessing, and an unknown never licenses a merge.
    """

    key: str
    label: str
    level: Optional[str] = None
    ontology_curie: Optional[str] = None
    ontology_iri: Optional[str] = None
    ontology_label: Optional[str] = None
    ontology_source: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    n_claims: int = 0
    source_papers: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: Any) -> "KERecord":
        """Build from a `ke_canonical` row (dict-like or pandas Series)."""
        def g(name: str, default: Any = None) -> Any:
            try:
                value = row[name]
            except (KeyError, IndexError, TypeError):
                return default
            if value is None:
                return default
            # pandas NaN
            if isinstance(value, float) and value != value:
                return default
            return value

        return cls(
            key=str(g("canonical_id", "")),
            label=str(g("canonical_name", "")),
            level=(str(g("level")) if g("level") else None),
            ontology_curie=(str(g("ontology_curie")) if g("ontology_curie") else None),
            ontology_iri=(str(g("ontology_iri")) if g("ontology_iri") else None),
            ontology_label=(str(g("ontology_label")) if g("ontology_label") else None),
            ontology_source=(str(g("ontology_source")) if g("ontology_source") else None),
            n_claims=int(g("n_source_rows", 0) or 0),
        )


@dataclass
class Check:
    """One of the eight ordered checks."""

    name: str
    outcome: str            # "pass" | "fail" | "unknown"
    detail: str

    @property
    def icon(self) -> str:
        return {"pass": "✓", "fail": "✗", "unknown": "?"}.get(self.outcome, "?")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Classification:
    """The verdict on one candidate pair, with its reasoning."""

    source: str
    target: str
    relationship: Relationship
    checks: list[Check] = field(default_factory=list)
    explanation: str = ""
    similarity: float = 0.0
    #: Ranking hint only — never a reason to merge. Set by an optional external
    #: scorer so the curator sees the most likely duplicates first.
    rank_score: Optional[float] = None

    @property
    def mergeable(self) -> bool:
        """Whether a *curator* is allowed to merge this pair at all."""
        return self.relationship in MERGEABLE

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if c.outcome == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship.value,
            "relationship_label": self.relationship.label,
            "mergeable": self.mergeable,
            "similarity": self.similarity,
            "rank_score": self.rank_score,
            "explanation": self.explanation,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class _State:
    """The direction-and-state reading of one label."""

    increase: bool = False
    decrease: bool = False
    disruption: bool = False
    restoration: bool = False
    no_change: bool = False

    @property
    def axis(self) -> Optional[str]:
        """Which axis the label moves along, if any."""
        if self.no_change:
            return "none"
        if self.disruption or self.restoration:
            return "integrity"
        if self.increase or self.decrease:
            return "quantity"
        return None

    @property
    def sign(self) -> int:
        """+1 toward more/intact, -1 toward less/broken, 0 for neither."""
        if self.no_change:
            return 0
        if self.increase and not self.decrease:
            return 1
        if self.decrease and not self.increase:
            return -1
        if self.restoration and not self.disruption:
            return 1
        if self.disruption and not self.restoration:
            return -1
        return 0

    def describe(self) -> str:
        parts = []
        if self.no_change:
            parts.append("no change")
        if self.increase:
            parts.append("increase")
        if self.decrease:
            parts.append("decrease")
        if self.disruption:
            parts.append("disruption")
        if self.restoration:
            parts.append("restoration")
        return ", ".join(parts) if parts else "no direction stated"


def read_state(label: str) -> _State:
    """Read direction and state markers off a raw label."""
    text = (label or "")
    no_change = bool(_NO_CHANGE_RE.search(text))
    if no_change:
        # "No change in increased X" is not a thing; once the no-change frame
        # is present it governs the whole label and the other markers are
        # describing what did *not* happen.
        return _State(no_change=True)
    return _State(
        increase=bool(_INCREASE_RE.search(text)),
        decrease=bool(_DECREASE_RE.search(text)),
        disruption=bool(_DISRUPTION_RE.search(text)),
        restoration=bool(_RESTORATION_RE.search(text)),
    )


def is_key_event(label: str) -> tuple[bool, str]:
    """
    Whether a label can be a Key Event at all.

    A Key Event is something that happens. "No change in OPC proliferation" is
    a study observation about a measurement that stayed put — real, worth
    keeping as evidence, and not a node on the map.
    """
    if not (label or "").strip():
        return False, "Empty label."
    if _NO_CHANGE_RE.search(label):
        return (
            False,
            "States that a measurement did not change. That is a study "
            "observation about the evidence, not a Key Event — record it "
            "against the KER it was measured for.",
        )
    return True, ""


def object_type(label: str) -> ObjectType:
    """Classify a label as an event, an entity, or a study observation."""
    text = (label or "").strip()
    if not text:
        return ObjectType.UNKNOWN
    if _NO_CHANGE_RE.search(text):
        return ObjectType.OBSERVATION

    state = read_state(text)
    if state.axis in {"quantity", "integrity"}:
        return ObjectType.EVENT
    # A change with no direction is still a change. Checked before the process
    # pattern because "shortened internodes" contains no process word at all.
    if _UNSIGNED_CHANGE_RE.search(text):
        return ObjectType.EVENT
    if _PROCESS_RE.search(text):
        return ObjectType.EVENT
    return ObjectType.ENTITY


def _tokens(label: str) -> list[str]:
    return _TOKEN_RE.findall((label or "").lower())


def participants(label: str) -> set[str]:
    """
    The nouns a label is about, with direction and process words removed.

    "Increased mitochondrial ROS" and "decreased mitochondrial ROS" have the
    same participants and opposite directions — separating the two is the whole
    point, because a metric that mixes them cannot tell agreement from
    contradiction.
    """
    normalised = normalise_label(label)
    out: set[str] = set()
    for token in normalised.split():
        if _INCREASE_RE.fullmatch(token) or _DECREASE_RE.fullmatch(token):
            continue
        if _DISRUPTION_RE.fullmatch(token) or _RESTORATION_RE.fullmatch(token):
            continue
        if len(token) <= 1:
            continue
        out.add(token)
    return out


def _subtype_class(label: str) -> Optional[str]:
    """The class a subtype belongs to, if the label names a known subtype."""
    for pattern, parent in _SUBTYPE_CLASSES:
        if pattern.search(label or ""):
            return parent
    return None


def _names_a_class(label: str) -> bool:
    """Whether a label names a general class rather than a specific member."""
    return bool(_CLASS_HEAD_RE.search(label or "")) and _subtype_class(label) is None


# ---------------------------------------------------------------------------
# The eight checks
# ---------------------------------------------------------------------------

def _check_object_type(a: KERecord, b: KERecord) -> tuple[Check, Optional[Relationship]]:
    """1. Compatible object types."""
    ta, tb = object_type(a.label), object_type(b.label)

    if ObjectType.OBSERVATION in (ta, tb):
        which = a.label if ta is ObjectType.OBSERVATION else b.label
        return (
            Check(
                "Object type",
                "fail",
                f"“{which}” reports that a measurement did not change. A study "
                f"observation and a Key Event are different kinds of thing and "
                f"cannot be the same record.",
            ),
            Relationship.CONTRADICTORY,
        )

    if ObjectType.UNKNOWN in (ta, tb):
        return Check("Object type", "unknown", "One label could not be typed."), None

    if ta is not tb:
        return (
            Check(
                "Object type",
                "fail",
                f"“{a.label}” is {_article(ta.value)} and “{b.label}” is "
                f"{_article(tb.value)}. Different object types are never the "
                f"same record.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    return Check("Object type", "pass", f"Both are of type “{ta.value}”."), None


def _article(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _check_entity_vs_event(a: KERecord, b: KERecord) -> tuple[Check, Optional[Relationship]]:
    """2. Entities are kept separate from biological events."""
    ta, tb = object_type(a.label), object_type(b.label)

    if {ta, tb} == {ObjectType.ENTITY, ObjectType.EVENT}:
        entity = a.label if ta is ObjectType.ENTITY else b.label
        event = b.label if ta is ObjectType.ENTITY else a.label
        return (
            Check(
                "Entity vs. event",
                "fail",
                f"“{entity}” names a thing; “{event}” names something that "
                f"happens to it. Merging them would turn a participant into "
                f"an event.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    if ta is ObjectType.ENTITY and tb is ObjectType.ENTITY:
        return (
            Check(
                "Entity vs. event",
                "pass",
                "Both name entities rather than events. An entity can be a "
                "Key Event's participant but is not itself a Key Event; check "
                "that this record belongs on the map at all.",
            ),
            None,
        )

    return Check("Entity vs. event", "pass", "Both name biological events."), None


def _check_process_and_participants(
    a: KERecord, b: KERecord
) -> tuple[Check, Optional[Relationship]]:
    """3. The biological process and the participants."""
    pa, pb = participants(a.label), participants(b.label)
    if not pa or not pb:
        return (
            Check("Process and participants", "unknown",
                  "One label has no content words left after normalisation."),
            None,
        )

    shared = pa & pb
    union = pa | pb
    overlap = len(shared) / len(union) if union else 0.0

    if not shared:
        return (
            Check(
                "Process and participants",
                "fail",
                f"No participant in common: {{{', '.join(sorted(pa))}}} vs. "
                f"{{{', '.join(sorted(pb))}}}.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    if pa == pb:
        return (
            Check("Process and participants", "pass",
                  f"Identical participants: {{{', '.join(sorted(pa))}}}."),
            None,
        )

    if overlap >= 0.5:
        extra = (pa - pb) | (pb - pa)
        return (
            Check(
                "Process and participants",
                "pass",
                f"Shares {{{', '.join(sorted(shared))}}}; differs by "
                f"{{{', '.join(sorted(extra))}}}.",
            ),
            None,
        )

    return (
        Check(
            "Process and participants",
            "fail",
            f"Only {{{', '.join(sorted(shared))}}} in common out of "
            f"{len(union)} participants — too little to be the same event.",
        ),
        Relationship.RELATED_DISTINCT,
    )


def _check_direction_and_state(
    a: KERecord, b: KERecord
) -> tuple[Check, Optional[Relationship]]:
    """
    4. Direction and state.

    The veto that matters most. Two labels that differ only in direction score
    near-identically on any string metric, and merging them fabricates
    agreement out of a contradiction.
    """
    sa, sb = read_state(a.label), read_state(b.label)

    if sa.no_change != sb.no_change:
        stated = a.label if sa.no_change else b.label
        moved = b.label if sa.no_change else a.label
        return (
            Check(
                "Direction and state",
                "fail",
                f"“{stated}” says nothing changed; “{moved}” says something "
                f"did. These are opposite claims about the same measurement.",
            ),
            Relationship.CONTRADICTORY,
        )

    if sa.sign and sb.sign and sa.sign != sb.sign:
        return (
            Check(
                "Direction and state",
                "fail",
                f"Opposite direction: “{a.label}” reads as {sa.describe()}, "
                f"“{b.label}” reads as {sb.describe()}.",
            ),
            Relationship.CONTRADICTORY,
        )

    if sa.axis and sb.axis and sa.axis != sb.axis:
        return (
            Check(
                "Direction and state",
                "fail",
                f"Different kind of change: “{a.label}” is a change in "
                f"{sa.axis}, “{b.label}” is a change in {sb.axis}. A quantity "
                f"moving and a structure breaking are not the same event.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    if not sa.axis and not sb.axis:
        return (
            Check("Direction and state", "unknown",
                  "Neither label states a direction."),
            None,
        )

    if bool(sa.axis) != bool(sb.axis):
        return (
            Check(
                "Direction and state",
                "unknown",
                f"Only one label states a direction "
                f"({(sa if sa.axis else sb).describe()}); the other is silent.",
            ),
            None,
        )

    return (
        Check("Direction and state", "pass",
              f"Same direction and state ({sa.describe()})."),
        None,
    )


def _check_biological_level(
    a: KERecord, b: KERecord
) -> tuple[Check, Optional[Relationship]]:
    """5. Biological level."""
    if not a.level or not b.level:
        return Check("Biological level", "unknown", "Level missing on one record."), None
    if a.level == b.level:
        return Check("Biological level", "pass", f"Both at the {a.level} level."), None
    return (
        Check(
            "Biological level",
            "fail",
            f"Different levels of biological organisation: {a.level} vs. "
            f"{b.level}. The same event does not sit at two levels; one is "
            f"more likely upstream of the other.",
        ),
        Relationship.RELATED_DISTINCT,
    )


def _check_ontology(
    a: KERecord,
    b: KERecord,
    ancestors_of: Optional[Callable[[str, Optional[str]], set[str]]] = None,
) -> tuple[Check, Optional[Relationship]]:
    """
    6. Exact ontology identifiers and hierarchy.

    An identical CURIE is the strongest evidence of equivalence available.
    An ancestor relation is the strongest evidence *against* it: a subtype and
    its parent class are related precisely because they are not the same, and
    pooling their evidence attributes findings about one channel to every
    channel of that kind.
    """
    ca = (a.ontology_curie or "").strip().upper()
    cb = (b.ontology_curie or "").strip().upper()

    if ca and cb and ca == cb:
        return (
            Check("Ontology identity", "pass",
                  f"Both annotated to the same term, {ca}"
                  + (f" ({a.ontology_label})." if a.ontology_label else ".")),
            None,
        )

    if ca and cb and ancestors_of is not None:
        try:
            anc_a = {c.upper() for c in ancestors_of(ca, a.ontology_source)}
            anc_b = {c.upper() for c in ancestors_of(cb, b.ontology_source)}
        except Exception:
            anc_a = anc_b = set()
        if cb in anc_a:
            return (
                Check("Ontology identity", "fail",
                      f"{cb} is an ancestor of {ca}: “{a.label}” is a kind of "
                      f"“{b.label}”, not the same term."),
                Relationship.NARROWER_THAN,
            )
        if ca in anc_b:
            return (
                Check("Ontology identity", "fail",
                      f"{ca} is an ancestor of {cb}: “{b.label}” is a kind of "
                      f"“{a.label}”, not the same term."),
                Relationship.BROADER_THAN,
            )

    if ca and cb and ca != cb:
        return (
            Check(
                "Ontology identity",
                "fail",
                f"Annotated to different terms, {ca} and {cb}. Two ontology "
                f"terms are two concepts unless the hierarchy says otherwise.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    return (
        Check("Ontology identity", "unknown",
              "At least one record has no ontology annotation."),
        None,
    )


def _check_entailment(a: KERecord, b: KERecord) -> tuple[Check, Optional[Relationship]]:
    """
    7. Semantic entailment or contradiction.

    Subsumption without an ontology to consult: a known subtype against its
    class, or a label that is another label plus qualifying words.
    """
    class_a, class_b = _subtype_class(a.label), _subtype_class(b.label)
    norm_a, norm_b = normalise_label(a.label), normalise_label(b.label)

    # A named subtype against the class it belongs to.
    if class_a and not class_b and _mentions(norm_b, class_a):
        return (
            Check(
                "Semantic entailment",
                "fail",
                f"“{a.label}” is a specific {class_a}; “{b.label}” names the "
                f"class. Keep the subtype and attach the class as a broader "
                f"concept — evidence about one subtype is not evidence about "
                f"all of them.",
            ),
            Relationship.NARROWER_THAN,
        )
    if class_b and not class_a and _mentions(norm_a, class_b):
        return (
            Check(
                "Semantic entailment",
                "fail",
                f"“{b.label}” is a specific {class_b}; “{a.label}” names the "
                f"class. Keep the subtype and attach the class as a broader "
                f"concept.",
            ),
            Relationship.BROADER_THAN,
        )
    if class_a and class_b and class_a == class_b and norm_a != norm_b:
        return (
            Check(
                "Semantic entailment",
                "fail",
                f"Two different subtypes of the same {class_a}. Sibling "
                f"subtypes are distinct entities.",
            ),
            Relationship.RELATED_DISTINCT,
        )

    # One label is the other plus narrowing qualifiers.
    ta, tb = set(norm_a.split()), set(norm_b.split())
    if ta and tb and ta != tb:
        if tb < ta and (ta - tb) <= _NARROWING_QUALIFIERS:
            return (
                Check(
                    "Semantic entailment",
                    "fail",
                    f"“{a.label}” is “{b.label}” restricted to "
                    f"{', '.join(sorted(ta - tb))}. That is a narrower event, "
                    f"not the same one.",
                ),
                Relationship.NARROWER_THAN,
            )
        if ta < tb and (tb - ta) <= _NARROWING_QUALIFIERS:
            return (
                Check(
                    "Semantic entailment",
                    "fail",
                    f"“{b.label}” is “{a.label}” restricted to "
                    f"{', '.join(sorted(tb - ta))}.",
                ),
                Relationship.BROADER_THAN,
            )

    # A bare class head against a specific member of that class.
    if _names_a_class(a.label) != _names_a_class(b.label):
        general = a.label if _names_a_class(a.label) else b.label
        specific = b.label if _names_a_class(a.label) else a.label
        if participants(general) & participants(specific):
            rel = (Relationship.BROADER_THAN if general == a.label
                   else Relationship.NARROWER_THAN)
            return (
                Check(
                    "Semantic entailment",
                    "fail",
                    f"“{general}” names a class and “{specific}” names a "
                    f"member of it.",
                ),
                rel,
            )

    if norm_a == norm_b:
        return (
            Check("Semantic entailment", "pass",
                  "Both reduce to the same normalised form, with no "
                  "subsumption or contradiction detected."),
            None,
        )

    return (
        Check("Semantic entailment", "pass",
              "No subsumption or contradiction detected."),
        None,
    )


def _mentions(normalised: str, phrase: str) -> bool:
    """Whether a normalised label contains most of a class phrase."""
    want = {t for t in phrase.split() if len(t) > 2}
    have = set(normalised.split())
    if not want:
        return False
    return len(want & have) / len(want) >= 0.6


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: How firmly each relationship is held. A later check may replace an earlier
#: verdict only if it is at least as firm, so "contradictory" from the
#: direction check is never softened to "related but distinct" by the level
#: check further down.
_PRECEDENCE = {
    Relationship.CONTRADICTORY: 5,
    Relationship.NARROWER_THAN: 4,
    Relationship.BROADER_THAN: 4,
    Relationship.RELATED_DISTINCT: 3,
    Relationship.UNCERTAIN: 1,
    Relationship.EQUIVALENT: 0,
}

def severity(relationship: Relationship) -> int:
    """
    How strongly a relationship objects to a merge.

    Used to reduce a group of records to a single verdict: a group is only
    equivalent if every member is equivalent to every other, so the worst
    pairwise verdict governs the whole selection.
    """
    return _PRECEDENCE[relationship]


def worst(classifications: Sequence[Classification]) -> Optional[Classification]:
    """The most objecting classification in a set, or None if empty."""
    if not classifications:
        return None
    return max(classifications, key=lambda c: severity(c.relationship))


#: Similarity below which two labels are not offered as equivalent even when
#: every check passes. Matches `ke_normalizer.REVIEW_FLOOR` so a pair that was
#: never suggested cannot be confirmed by a different route.
EQUIVALENCE_FLOOR = 0.55


def classify(
    a: KERecord,
    b: KERecord,
    *,
    ancestors_of: Optional[Callable[[str, Optional[str]], set[str]]] = None,
) -> Classification:
    """
    Classify one candidate pair.

    Runs all eight checks even after a veto fires, because the curator is owed
    the full picture: knowing that two records share their participants *and*
    contradict each other on direction is more useful than being told only
    that they were rejected.

    `ancestors_of(curie, ontology) -> set[curie]` is optional; without it the
    hierarchy check reports `unknown` and the pair falls back on the lexical
    and morphological evidence.
    """
    checks: list[Check] = []
    verdict: Optional[Relationship] = None

    def apply(result: tuple[Check, Optional[Relationship]]) -> None:
        nonlocal verdict
        check, proposed = result
        checks.append(check)
        if proposed is None:
            return
        if verdict is None or _PRECEDENCE[proposed] > _PRECEDENCE[verdict]:
            verdict = proposed

    apply(_check_object_type(a, b))
    apply(_check_entity_vs_event(a, b))
    apply(_check_process_and_participants(a, b))
    apply(_check_direction_and_state(a, b))
    apply(_check_biological_level(a, b))
    apply(_check_ontology(a, b, ancestors_of=ancestors_of))
    apply(_check_entailment(a, b))

    score = similarity(normalise_label(a.label), normalise_label(b.label))

    if verdict is None:
        # Nothing objected. Equivalence still has to be positively earned:
        # either an identical ontology term, an identical normalised form, or
        # a high enough lexical score with no unknowns on the vetoing checks.
        verdict = _positive_verdict(a, b, checks, score)

    explanation = _explain(a, b, verdict, checks, score)
    return Classification(
        source=a.key or a.label,
        target=b.key or b.label,
        relationship=verdict,
        checks=checks,
        explanation=explanation,
        similarity=round(score, 4),
    )


def _positive_verdict(
    a: KERecord, b: KERecord, checks: list[Check], score: float
) -> Relationship:
    """Decide between equivalent and uncertain when no check objected."""
    same_curie = any(
        c.name == "Ontology identity" and c.outcome == "pass" for c in checks
    )
    if same_curie:
        return Relationship.EQUIVALENT

    if normalise_label(a.label) == normalise_label(b.label):
        return Relationship.EQUIVALENT

    if token_key(normalise_label(a.label)) == token_key(normalise_label(b.label)):
        return Relationship.EQUIVALENT

    # A pair that reaches here differs in wording only. Require both a decent
    # lexical score and that the direction check actually confirmed agreement
    # rather than shrugging — "unknown" on direction plus a good string score
    # is exactly the situation that produces confident nonsense.
    direction_confirmed = any(
        c.name == "Direction and state" and c.outcome == "pass" for c in checks
    )
    if score >= 0.80 and direction_confirmed:
        return Relationship.EQUIVALENT

    return Relationship.UNCERTAIN


def _explain(
    a: KERecord,
    b: KERecord,
    verdict: Relationship,
    checks: list[Check],
    score: float,
) -> str:
    """One paragraph a curator can act on, then the check that settled it."""
    failed = [c for c in checks if c.outcome == "fail"]
    lead = {
        Relationship.EQUIVALENT:
            f"“{a.label}” and “{b.label}” describe the same Key Event and may "
            f"be merged.",
        Relationship.BROADER_THAN:
            f"“{a.label}” is broader than “{b.label}”. Keep both and attach "
            f"the broader term as an ontology parent rather than merging.",
        Relationship.NARROWER_THAN:
            f"“{a.label}” is narrower than “{b.label}”. Keep the specific "
            f"record and attach the broader term as an ontology parent.",
        Relationship.RELATED_DISTINCT:
            f"“{a.label}” and “{b.label}” are related but distinct events. "
            f"Keep them separate; a biological relationship can be recorded "
            f"between them.",
        Relationship.CONTRADICTORY:
            f"“{a.label}” and “{b.label}” are incompatible. They must not be "
            f"merged; the disagreement itself is evidence and belongs in the "
            f"KER's uncertainties.",
        Relationship.UNCERTAIN:
            f"There is not enough signal to say how “{a.label}” and "
            f"“{b.label}” relate. Leave them unresolved rather than guessing.",
    }[verdict]

    reason = failed[0].detail if failed else (
        f"All checks passed at a lexical similarity of {score:.2f}."
    )
    return f"{lead} {reason}"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def classify_pairs(
    records: Sequence[KERecord],
    pairs: Iterable[tuple[str, str]],
    *,
    ancestors_of: Optional[Callable[[str, Optional[str]], set[str]]] = None,
) -> list[Classification]:
    """Classify an explicit list of (key, key) pairs."""
    by_key = {r.key: r for r in records}
    out: list[Classification] = []
    for left, right in pairs:
        a, b = by_key.get(left), by_key.get(right)
        if a is None or b is None:
            continue
        out.append(classify(a, b, ancestors_of=ancestors_of))
    return out


def classify_all(
    records: Sequence[KERecord],
    *,
    floor: float = EQUIVALENCE_FLOOR,
    ancestors_of: Optional[Callable[[str, Optional[str]], set[str]]] = None,
    limit: int = 400,
) -> list[Classification]:
    """
    Classify every pair whose labels are similar enough to be worth a look.

    The lexical floor is a *recall* device — it decides which pairs a human
    ever sees. It has no say in the verdict, which is what keeps a high string
    score from being mistaken for agreement.
    """
    scored: list[tuple[float, KERecord, KERecord]] = []
    normalised = {r.key: normalise_label(r.label) for r in records}

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            score = similarity(normalised[a.key], normalised[b.key])
            if score < floor:
                continue
            scored.append((score, a, b))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        classify(a, b, ancestors_of=ancestors_of)
        for _, a, b in scored[:limit]
    ]


def rank_candidates(
    classifications: Sequence[Classification],
    scorer: Optional[Callable[[Classification], float]] = None,
) -> list[Classification]:
    """
    Order candidates for review.

    `scorer` may be an embedding distance or a language model — anything that
    helps a curator see the likeliest duplicates first. It writes to
    `rank_score` and nothing else. `mergeable` is derived from `relationship`,
    which only the checks can set, so no scorer can promote a pair into being
    mergeable however confident it is.
    """
    ranked = list(classifications)
    if scorer is not None:
        for c in ranked:
            try:
                c.rank_score = float(scorer(c))
            except Exception:
                c.rank_score = None

    def sort_key(c: Classification) -> tuple[int, float, float]:
        # Mergeable pairs first — those are the ones with an action attached.
        return (
            0 if c.mergeable else 1,
            -(c.rank_score if c.rank_score is not None else 0.0),
            -c.similarity,
        )

    ranked.sort(key=sort_key)
    return ranked


def summarise(classifications: Sequence[Classification]) -> dict[str, int]:
    """Count classifications by relationship, for a status line."""
    counts = {r.value: 0 for r in Relationship}
    for c in classifications:
        counts[c.relationship.value] += 1
    return counts


__all__ = [
    "Relationship",
    "ObjectType",
    "MERGEABLE",
    "EQUIVALENCE_FLOOR",
    "KERecord",
    "Check",
    "Classification",
    "classify",
    "classify_pairs",
    "classify_all",
    "rank_candidates",
    "summarise",
    "severity",
    "worst",
    "is_key_event",
    "object_type",
    "participants",
    "read_state",
]
