# Stage 2 — Key Event Relationship extraction and synthesis

*Draft methods text, ~500 words.*

Stage 2 converts the full-text papers retained at screening into a structured
Adverse Outcome Pathway. It proceeds in five steps, and the order is enforced:
no step reads records that the preceding step has not signed off.

**Extraction.** Each PDF is parsed into a page- and section-aware document and
submitted to a large language model in a sequence of narrow calls — pathway
reconstruction, pair listing, classification, empirical evidence,
applicability, quantitative relationships and study metadata — each with its
own output-token ceiling. By default the whole paper is sent; optional
relevance scoring, intended for models whose context window cannot hold a full
text, selects passages against a character budget. Every field the model
returns must be accompanied by verbatim quotations, which are then searched for
in the source text character by character and marked verified or unlocated.
Unlocated quotations are retained and flagged rather than discarded, since they
indicate paraphrase or fabrication and are the principal signal of extraction
failure. The product is Table 1: one row per relationship per paper, expressed
in that paper's own terminology, with nothing merged or interpreted.

**Normalisation and curation.** Because *n* papers describing the same biology
produce *n* different strings, raw labels are clustered into canonical Key
Events. Merging is authorised, in order of precedence, by shared AOP-Wiki Key
Event identifier, shared ontology CURIE from OLS4, identical normalised string,
identical content words in a different order, or lexical similarity above a
threshold — the last two permitted only where the two labels agree on
biological level and on direction polarity, so that "increased apoptosis" and
"decreased apoptosis" cannot merge. The authorising rule and its evidence are
recorded per label and reported as a crosswalk, one row per wording, so that
the step from Table 1 to the canonical events is inspectable rather than
summarised as two totals: a corpus of *n* claims names 2*n* Key Event mentions
in some smaller number of wordings, and the crosswalk states which wording
became which event and on what authority. Curation is then performed one Table
1 row at a time rather than one label at a time, because the unit of the
decision is the claim: a single wording such as "voltage-gated sodium channels"
may be blocked in one experiment and activated in another, and each row is
therefore pointed at its own Key Event while the wording is retained as an
alias of both. Every original wording is preserved as an alias. Candidate pairs
are additionally classified as
equivalent, broader, narrower, related but distinct, contradictory or
uncertain; only pairs classified equivalent may be merged, and curator
overrides are recorded as such. Each merge stores complete before-and-after
state, making every merge individually reversible.

**Approval.** Records advance through raw, normalisation-proposed, curated,
approved and synthesised states. Approval is the gate that opens synthesis and
the map. Editing an approved record retracts its approval and marks everything
derived from it stale. The curator also declares which Key Events are the
molecular initiating event and, where one exists, the adverse outcome; no
endpoint is labelled an adverse outcome by the tool, since that designation is
a regulatory claim rather than an observation.

**Synthesis.** Approved relationships are consolidated into Table 2 and scored
on a transparent heuristic combining supporting papers (saturating at five and
discounted by the contradicting fraction), essentiality evidence, quantitative
data, quotation verification rate, extraction confidence and AOP-Wiki presence.
The result is multiplied by a factor reflecting experimental design — rescue,
perturbation, common-stressor, correlational or reverse-only — halved where
papers disagree on direction, and banded as high, moderate or low. A single
model call per relationship then writes a consolidated narrative in AOP
Handbook structure, from the extracted fields alone, in which uncertainties and
contradicting findings are a required output.

**Pathway assembly.** Only approved Key Events and relationships are drawn.
Causal ordering is computed from the graph and cannot be manually overridden;
graphs may be frozen as snapshots so that a cited figure does not change.

Every run records provider, model as requested and as reported, endpoint,
sampling parameters and seed, prompt fingerprint, token budgets and chunking
settings, so that two runs are only compared when those conditions match.
