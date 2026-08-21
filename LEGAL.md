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

**With Anthropic or OpenAI selected**, the full text of each paper is
transmitted over the network to that provider's API. Chunk scoring — which
would send only the passages it selects — is **off by default**, so unless you
switch it on in the sidebar, what leaves this machine is the whole article and
not an excerpt. On the hosted Streamlit deployment, the uploaded PDF also
passes through the hosting provider's servers before it reaches the app.

**The provider caches it.** Extraction asks about thirty questions of each
paper, and re-sending the article thirty times would be both slow and
expensive, so it is sent once under a caching marker and the later calls read
it from the provider's cache. The practical consequence is that the article is
*stored* on the provider's infrastructure for the lifetime of that cache, not
merely processed in transit. Several publisher agreements treat copies **held**
by a third party differently from disclosure to one, and Article 3(2) of the EU
Copyright in the Digital Single Market Directive requires copies retained for
text and data mining to be stored "with an appropriate level of security" — a
condition you can only speak to if you know what your provider keeps, and for
how long. Provider terms that guarantee zero retention, or an institutional
agreement under which the provider processes on your behalf, are what change
the answer here.

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
provider is selected — not only here, where nobody would read it. It will not
run a hosted extraction at all until you confirm a lawful basis and say what it
is, and it records that statement with the run. It does not verify the claim,
and it cannot.

**Stage 1 is not covered by that gate.** Screening sends each record's title and
abstract to whichever provider is selected for Stage 1, with no acknowledgement
step. Abstracts are publisher copyright too, and NCBI's terms constrain
redistribution of E-utilities output. If that matters for your corpus, screen
with the local provider.

## 3. What the tool stores

Written to the local `aop_rag.db` SQLite file:

- **the extracted text of each paper, chunk by chunk** (`paper_chunk`), with its
  page and section offsets
- extracted Key Events and Key Event Relationships
- verbatim quotations with their page and section locations
- canonical Key Events, aliases and ontology annotations
- curation decisions and map layouts
- run manifests: model, settings, endpoint host, prompt fingerprint, the
  lawful-basis statement for any hosted run, and robustness counters

Uploaded PDFs are read into memory and the **file** is not retained afterwards.
Its **contents** are: the text of every paper you process is kept so that a
quotation can be located and its page reported, and so that a question about
what a paper says downstream of some event does not require re-uploading it. A
modest corpus leaves hundreds of kilobytes of publisher text in that file.

It is easy to read "the PDF is not kept" as "the paper is not kept". It is
kept. **Treat `aop_rag.db` as a copy of your corpus**: keep it local, do not
commit it to a repository, and do not attach it to a manuscript or pass it to
anyone not licensed for those articles. `python tools/check_publishable.py`
fails a release if a database, a PDF or a credentials file is in the working
tree, the git index or anywhere in the git history.

**How long it is kept.** With the `AOP_RAG_DB` environment variable set — the
single-user desktop configuration — the database persists until you delete it;
there is no expiry. Without it, each browser session gets its own database in a
temporary directory, swept twelve hours after the session ends and removed when
the process stops. The file is not encrypted, and the app has no login, so on a
shared machine anyone who can read the file can read your corpus.

**Removing things.** *Clear all extraction data* and *Reset everything* empty
the database and reclaim the file space. There is currently no way to remove a
single paper — if you upload something you should not have, clear the whole
database and re-run the papers you meant to keep.

API keys entered in the sidebar are held for the session only and are not
written to the database, to a log, or to any export. One caveat: if you point
the API base URL at a proxy and put credentials in the URL itself, the host
portion of that URL is recorded in the run manifest and shown in the QC report.
Keep credentials out of the base URL.

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

Some publisher text-and-data-mining terms cap the length of a snippet that may
appear in published output — Elsevier's, for instance, at around 200
characters. The tool applies no such cap, and an export covering a mixed corpus
is governed by the strictest terms among the papers in it. Check the quotations
in an export before it becomes supplementary material.

## 5. Third-party services and data

- **PubMed / NCBI E-utilities** — used in Stage 1 for search and abstracts.
  Subject to NCBI's usage policies and rate limits. Requests carry the tool
  name and, if you supply one, a contact address. An NCBI API key is passed as
  a URL parameter, which is how E-utilities works — if you run the tool behind
  an institutional proxy, assume the key appears in that proxy's logs.
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

This matters more than it would in a tool that forgets its inputs. As section 3
explains, the text of everything you process is retained in `aop_rag.db`, and
there is no way to remove a single document — so an accidental upload is undone
by clearing the whole database, not by deleting one row. If a cloud provider was
selected, the text has also already been sent.

---

*Last reviewed: see git history for this file. Report problems with these
notices through the project's issue tracker.*
