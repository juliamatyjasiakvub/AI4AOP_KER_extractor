# Legal notices and data handling

*These notices describe how the AI4AOP KER extractor handles the material you
give it, and what you are responsible for when you use it. They are provided
for transparency and are **not legal advice**. If you are deploying this tool
at an institution, have your legal and research-integrity offices review both
this document and the deployment before your colleagues use it.*

---

## 1. The papers you upload

You are responsible for having lawful access to every paper you upload, and for
complying with the publisher and institutional terms that cover it.

This tool does not obtain papers on your behalf, does not circumvent access
controls, and does not attempt to retrieve full text from any source. Stage 1
queries PubMed for titles and abstracts only; full text reaches the tool only
because you upload it.

## 2. Where your papers go — the point that matters most

This is the part users most often miss, so it is stated plainly.

**With Ollama (local) selected**, paper text is processed by a model running on
the same machine as the app. Nothing is transmitted to a third party.

**With Anthropic or OpenAI selected**, the passages of each paper selected for
extraction are transmitted over the network to that provider's API. On the
hosted Streamlit deployment, the uploaded PDF also passes through the hosting
provider's servers before it reaches the app.

Publisher subscription licences commonly permit personal, non-commercial
research use while restricting **systematic downloading** and **disclosure of
licensed content to third parties**. Transmitting the full text of a licensed
article to a commercial LLM API is plausibly a disclosure to a third party and
may fall outside your licence, regardless of whether the provider trains on it.
Text and data mining rights vary further by jurisdiction and by agreement.

Practical guidance:

- For **open-access** papers, check the specific licence (CC-BY, CC-BY-NC and
  CC-BY-ND differ in what they permit downstream).
- For **subscription or pay-per-view** papers, either check your institutional
  agreement first, or run Stage 2 with the local Ollama provider, which sends
  nothing outward.
- Review the data-usage and retention terms of whichever API provider you
  select. Those terms are theirs, not ours, and they change.

The app shows a warning to this effect at the point of upload whenever a cloud
provider is selected — not only here, where nobody would read it.

## 3. What the tool stores

Written to the local `aop_rag.db` SQLite file:

- extracted Key Events and Key Event Relationships
- verbatim quotations with their page and section locations
- canonical Key Events, aliases and ontology annotations
- curation decisions and map layouts
- run manifests: model, settings, prompt fingerprint and robustness counters

Uploaded PDFs are read into memory for processing and are not retained as files
afterwards. Extracted quotations from those PDFs **are** retained in the
database, because they are the evidence the tool exists to preserve.

API keys entered in the sidebar are held for the session only and are not
written to the database.

Nothing is transmitted, published or redistributed by the tool other than the
model calls described in section 2 and the optional lookups in section 5.

## 4. Quotations and exports

Extracted evidence includes short verbatim quotations from the source papers,
retained so that every claim can be checked against its source. Short
quotations for the purpose of criticism, review or citation are treated
differently from bulk reproduction in most jurisdictions, but the boundaries
differ and the responsibility for staying inside them is the user's.

Exports (CSV, JSON, the QC report) carry these quotations with them and are
stamped with a disclaimer line. How you reuse, circulate or publish an export is
your responsibility, and republishing extracted quotations at scale is a
different act from consulting them privately.

## 5. Third-party services and data

- **PubMed / NCBI E-utilities** — used in Stage 1 for search and abstracts.
  Subject to NCBI's usage policies and rate limits.
- **AOP-Wiki** — an XML dump is bundled and can be updated from within the app.
  Subject to the AOP-Wiki terms of use; check the current licence and
  attribution requirement before publishing anything derived from it.
- **EBI Ontology Lookup Service (OLS4)** — optional Key Event enrichment
  against GO, UBERON, CL, HP, MP, ChEBI and PATO. Subject to EMBL-EBI's terms;
  each source ontology also carries its own licence.
- **LLM providers** — Anthropic, OpenAI, or a local Ollama instance, as
  selected in the sidebar.

Cite and attribute these sources in any publication derived from this tool's
output.

## 6. Scientific limitations — no warranty

This tool produces **machine-extracted candidate** Key Events and Key Event
Relationships.

Quote verification confirms that a sentence exists in the source document. It
does **not** confirm that the relationship inferred from that sentence is
correct, that the direction of effect is right, or that the extraction is
complete. Quotations flagged as unverified could not be located in the source
and may be paraphrases or fabrications.

Output is not suitable for regulatory submission, publication or decision-making
without expert review. The software is provided without warranty of any kind;
see `LICENSE`.

## 7. Reproducibility

Identical inputs do not guarantee identical output. Sampling is stochastic,
hosted model aliases are re-pointed to new weights without notice, and
server-side batching makes results non-deterministic even at temperature 0.

Each run therefore records a manifest — provider, model requested and model
reported, temperature, seed, prompt fingerprint, chunk budget, code version,
AOP-Wiki dump version — together with counters for everything the pipeline
recovered from silently. The QC report renders both. Treat any single run as
one sample of the model's behaviour, not as a reading of the paper.

## 8. Personal data

The tool is designed for scientific literature and is not intended for
documents containing personal or patient data. Do not upload clinical records,
identifiable participant data or other personal data; nothing here is
configured or assessed for that purpose.

---

*Last reviewed: see git history for this file. Report problems with these
notices through the project's issue tracker.*
