# AI4AOP KER Extractor

AI4AOP KER Extractor is a Streamlit application that helps an Adverse Outcome
Pathway (AOP) developer find literature, extract candidate Key Events (KEs) and
Key Event Relationships (KERs), curate the terminology, evaluate the evidence,
and build an approved AOP figure.

The application is evidence-first and curator-controlled. It does not treat an
LLM extraction, similarity score, or calculated confidence score as an expert
decision. Raw evidence remains traceable to its source, and nothing is
synthesized or drawn in the final AOP until a user approves it.

Feedback form: <https://forms.cloud.microsoft/e/Wx7vF2nvn2>

## What happens in the app

The main workflow is deliberately linear:

```text
Search and screen
      ↓
1. Extract evidence
      ↓
2. Normalize and curate
      ↓
3. Approve
      ↓
4. Synthesize evidence
      ↓
5. Final AOP
```

Each step depends on the previous one. If a curated record changes after
approval, the application retracts the effective approval, marks dependent
syntheses as stale, and requires review again.

### Search and screen

The optional Stage 1 workflow searches PubMed and uses the selected language
model to screen titles and abstracts against user-supplied inclusion and
exclusion criteria. Results can be exported as CSV.

Stage 1 identifies potentially relevant papers; it does not download or process
full text.

### 1. Extract evidence

The user uploads full-text PDFs and chooses one of two extraction modes:

- **Open discovery** extracts the candidate relationships found throughout each
  paper.
- **Targeted KER** asks how two user-defined KEs are connected and extracts the
  supported chain, including intermediate events. This is usually more focused
  and requires less normalization later.

For every paper, the application:

1. Detects the DOI and allows the user to correct it.
2. Reads the PDF by page and section.
3. Sends either the full extracted text or selected chunks to the configured
   model.
4. Extracts candidate upstream and downstream events, experimental context,
   applicability, direction, and supporting or contradicting findings.
5. Requests verbatim supporting quotations.
6. Searches for each quotation in the source and records its page and section.
7. Stores the raw result in the paper's original terminology.

The extraction page displays only source-level material. It does not merge KEs,
calculate a final weight of evidence, or build the AOP. The provenance drawer
under each extracted claim shows the associated quotations and whether they
were located verbatim.

Quote verification confirms that text was found in the PDF. It does not prove
that the model interpreted the text correctly.

### Entering a claim by hand

Extraction misses things. Any relationship can be added, corrected or removed
by the curator, from the extraction page or from **Normalize and curate**.

A hand-entered claim is stored as an ordinary Table 1 row, so it is normalized,
approved and synthesized exactly like an extracted one. What differs is its
label. Every row records its origin, and three things read it:

- the QC report excludes curator rows from the quotation-verification rate and
  the confidence distribution, and reports them on their own line — a
  hand-typed row contains no model output to verify;
- the provenance drawer states who entered the claim and the rationale they
  gave;
- the final map draws a relationship no paper states in violet, with an
  `asserted` badge, and carries that through to the CSV and JSON exports.

Where the source paper is already in the corpus, a pasted quotation is checked
against its stored text and verified the same way an extracted quotation is. A
hand-entered claim with a located quotation is as traceable as a model-extracted
one; a claim with no source at all is an assertion, and is drawn as one.

Editing an extracted row marks it as curator-edited and archives the previous
version; deleting one archives it as well. Nothing is overwritten silently, and
either change retracts the dependent approval and marks dependent syntheses
stale.

A Key Event with no relationship yet — typically an adverse outcome the corpus
has not reached — can be added directly in **Normalize and curate**. Unlike
derived Key Events, it survives re-running normalization.

### 2. Normalize and curate

This is the only workspace where KEs are grouped, renamed, mapped, merged, or
rejected.

The application proposes candidate canonical groups, but the curator makes the
decision. Raw wording, source papers, direction, biological level, and evidence
remain available while reviewing each proposal.

Supported decisions include:

- merge scientifically equivalent records;
- keep records separate;
- map a specific record to a broader ontology concept;
- record a biological relationship between distinct records;
- reject a record that is not a KE;
- leave an uncertain record unresolved; and
- split or undo an earlier grouping.

Equivalence and ontology hierarchy are intentionally different operations. For
example, `NaV1.2` may be mapped to the broader class `voltage-gated sodium
channel`, but evidence about that subtype is not automatically pooled with
evidence about the entire class.

Merge recommendations are classified as one of:

- equivalent;
- broader than;
- narrower than;
- related but distinct;
- contradictory or incompatible; or
- uncertain.

Only **equivalent** records can be merged. String similarity may help find
candidates, but it cannot authorize a merge. This prevents errors such as
combining "restored nodal protein organization" with "disrupted nodal protein
organization." A finding such as "no change in OPC proliferation" is retained
as an observation rather than converted into a pathway event.

Every decision records its semantic classification, explanation, curator,
rationale, and before/after state. The **Canonical groups** view shows the
resulting KE, its aliases, source publications, ontology mapping, and available
undo or split actions.

### 3. Approve

Approval is the gate between curation and scientific synthesis.

The curator first approves the canonical KEs. A KER can then be approved only
when both its upstream and downstream KEs are approved. The workflow states are:

```text
raw → normalization proposed → curated → approved → synthesized
```

The approval page records who approved each object and when. If an approved KE
or KER changes, dependent evidence synthesis becomes stale and must be
regenerated and reviewed.

### 4. Synthesize evidence

The application provides one evidence page for each approved canonical KER.
The page contains:

1. KER identity and applicability;
2. one evidence block per independent study;
3. biological plausibility;
4. empirical support, including temporal and dose-response concordance;
5. uncertainties, inconsistencies, null findings, and contradicting evidence;
6. quantitative understanding; and
7. the AOP developer's assessment and rationale.

The language model can draft the cross-study synthesis from the approved source
records. The calculated confidence score is displayed only as decision support.
The developer must independently select and justify the final High, Moderate,
or Low assessment.

Superseded syntheses are archived rather than silently overwritten.

### 5. Final AOP

The final map contains only approved KEs and approved KERs. Provisional records
are withheld.

The graph is arranged from left to right by causal order:

```text
MIE → early KEs → intermediate KEs → late KEs → AO
```

Horizontal placement is recalculated from the graph and cannot be overridden by
a saved layout. The user may adjust and save vertical positions to reduce
overlap. The legend explains node roles, biological-level badges, approval
status, evidence colours, adjacency, and arrow direction.

**Every event is drawn as a Key Event unless the curator declares otherwise.**
Neither endpoint is inferred. A node with nothing upstream of it says only that
no paper in this corpus reported an earlier step — the ordinary state of a
corpus assembled around the middle of a pathway — and naming it the molecular
initiating event would turn a gap in the literature into a claim, in the
leftmost column of the figure. The MIE and the AO are assigned and approved in
**Approve**, under *Pathway endpoints*, which also suggests where each might go
and says why the suggestion is only a suggestion.

The curator can also freeze a named snapshot and export nodes, edges, and the
graph as CSV or JSON.

#### Direction conflicts

A node marked **±** has claims that disagree: some recorded an increase and
others a decrease. A node marked **⚠** is named for one direction while its
claims report the other. Neither is a defect in the map; both are findings
about the corpus.

The **Direction conflicts** panel below the map lists every flagged Key Event
with the individual claims behind it — which paper, which recorded change,
which assay, which model — and offers the three things that actually resolve
one:

- **a claim is wrong** — correct that row in place, which removes the conflict
  at its source;
- **these are two events** — the name is too general to separate them, and the
  wordings are split in **Normalize and curate**;
- **the split is real** — record which direction the AOP asserts, and why.

Ruling that a conflict is genuine keeps the **±** on the figure. It is an
answer rather than a dismissal, and the map's job is then to keep showing it.
Any ruling records the curator and the rationale, and retracts approvals that
depended on the previous arrow.

## Raw evidence versus derived output

| Content | Origin | Requires approval? |
|---|---|---:|
| Extracted claim | Language-model reading of one paper | No; it remains raw |
| Curator-entered claim | A person, with a recorded rationale | No; it remains raw, and is labelled as entered |
| Quotation and page | Source PDF plus verification lookup | No; review recommended |
| Canonical KE | Curator decision over raw labels | Yes |
| Canonical KER | Approved upstream/downstream pair | Yes |
| Evidence synthesis | Model-assisted summary of approved study records | Yes; developer assessment required |
| Final AOP | Approved graph snapshot | Yes |

## Providers and data handling

The app supports separate provider settings for Stage 1 and Stage 2:

- **Ollama** for local processing;
- **Anthropic Claude**; and
- **OpenAI GPT**.

**Local processing is the default.** Ollama is preselected, and with it the
full text of your papers never leaves the machine.

Anthropic and OpenAI are hosted services: selecting one sends the full text of
every uploaded paper to that company. The application will not run a hosted
extraction until you confirm a lawful basis and state what it is, and it
records that statement with the run. **It does not verify your basis, and it
cannot.** Whether you may transmit a paper to a third party depends on your
subscription agreements and institutional policy, which no software can read —
a subscription grants reading, not automatically the right to send the text
elsewhere, and free-to-read is not a reuse licence. If you are unsure, use
Ollama.

The local path is the default and is the one this tool is designed around, but
the results reported for the AI4AOP Challenge were produced with a hosted
model. Extraction quality on small local models is noticeably lower —
quotation matching in particular — and the local path has not been benchmarked
at corpus scale in this release.

**`aop_rag.db` contains the papers.** The PDF file itself is processed in
memory and not kept, which is easy to mistake for "the paper is not kept" — it
is. To locate a quotation and report its page, the application stores each
paper's extracted text chunk by chunk in the `paper_chunk` table, alongside the
verbatim quotations in `evidence_spans`. A thirteen-paper corpus leaves roughly
750 KB of publisher text in that file. Treat the database as a copy of your
corpus: keep it local, do not commit it, and do not attach it to a manuscript
or share it with anyone not licensed for those articles.

Run `python tools/check_publishable.py` before pushing or releasing. It fails
if a database, a PDF or a credentials file is tracked, sitting untracked in the
working tree, or present anywhere in the git history — the last of which is not
fixed by deleting the file, because every clone still receives it.

Do not upload personal or patient data. Before uploading subscription content,
read [LEGAL.md](LEGAL.md) and check the relevant publisher and institutional
licences.

## Installation

### Requirements

- Python 3.10 or later;
- Git; and
- either a local Ollama installation or an API key for a supported cloud
  provider.

### Windows with Git Bash

```bash
git clone https://github.com/juliamatyjasiakvub/AI4AOP_KER_extractor.git
cd AI4AOP_KER_extractor
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application normally opens at <http://localhost:8501>. Stop it with
`Ctrl+C`. Run `deactivate` to leave the virtual environment.

### macOS or Linux

```bash
git clone https://github.com/juliamatyjasiakvub/AI4AOP_KER_extractor.git
cd AI4AOP_KER_extractor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Configuration

Keys can be entered in the Streamlit sidebar for the current session or
provided as environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI access for Stage 1 or Stage 2 |
| `ANTHROPIC_API_KEY` | Anthropic access for Stage 1 or Stage 2 |
| `OLLAMA_URL` | Ollama endpoint; defaults to `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model used by the legacy Stage 1 helper |
| `NCBI_EMAIL` | Contact email supplied to NCBI E-utilities. Set it — NCBI's usage policy asks that requests identify their sender |
| `NCBI_API_KEY` | Optional NCBI API key. Without one the tool paces itself to NCBI's anonymous limit of 3 requests/second; with one it uses the 10/second allowance, so a large search finishes about three times faster |

Example for Git Bash:

```bash
export OPENAI_API_KEY="your-key"
python -m streamlit run app.py
```

API keys entered in the sidebar are not written to the database.

## Ontology and AOP-Wiki enrichment

OLS4 enrichment is optional. It can attach identifiers and hierarchy from
ontologies including GO, UBERON, CL, HP, MP, ChEBI, and PATO. If OLS4 is
unavailable, extraction and curation continue without it.

The AOP-Wiki XML dump is **not bundled**. It is AOP-Wiki's curated database
rather than part of this tool, so the repository ships the code and the
application downloads the dump on request — sidebar → **AOP-Wiki dump** →
*Download the latest dump*. Without it, extraction and curation run normally
but Key Events are not matched to AOP-Wiki identifiers, exactly as when OLS4 is
unreachable. Attribution and licence terms for that data are AOP-Wiki's.

## Reproducibility and quality control

Every extraction run records a manifest containing the provider, requested and
reported model, prompt fingerprint, token settings, optional sampling seed,
chunking settings, code version, schema version, and AOP-Wiki dump version.

The QC report includes quotation verification rates, model-response repairs,
failed steps, confidence distributions, contradicted rows, and unverified
quotations. It can be exported as Markdown, JSON, or CSV.

Language-model output is not deterministic. A fixed seed may reduce variation
for providers that support it, but does not make an extraction fully
reproducible.

## Persistence and reset behavior

Application state is stored in the local SQLite file `aop_rag.db`, including:

- raw extractions and evidence spans;
- superseded and deleted versions of edited rows;
- run manifests;
- canonical KEs and aliases;
- semantic merge decisions and ontology mappings;
- workflow state and approval history;
- KER syntheses and prior versions;
- MIE/KE/AO assignments;
- vertical layout adjustments; and
- frozen AOP snapshots.

The sidebar provides separate controls to clear extraction data or reset all
stored application state. **Reset everything cannot be undone.** Database
schema upgrades may create a timestamped backup before migration.

Do not commit personal working databases, database backups, uploaded content,
API keys, or Python cache files to Git. `.gitignore` covers all of these, but
it has no effect on a file that is already tracked and none at all on the git
history — so `python tools/check_publishable.py` is what actually enforces it.

## Tests

Run the automated suite from the repository root:

```bash
python -m pytest -q
```

The tests include safeguards against incompatible merges, workflow-gate
bypasses, stale approvals, and invalid causal layouts.

## Scientific limitations

This tool produces candidate scientific records and model-assisted summaries.
It may miss evidence, misread direction, extract an incorrect quotation, or
produce an incomplete pathway. It does not replace expert review or the AOP
Developers' Handbook.

Do not use the output for regulatory submission, publication, or scientific
decision-making without reviewing the source evidence, curation history,
applicability, inconsistencies, and developer assessment.

## Licence

See [LICENSE](LICENSE). Third-party services and datasets retain their own
terms and attribution requirements.

Example citation: Matyjasiak J and Camargo E. AI4AOP KER Extractor [computer software]. GitHub; 2026 [cited 2026 Aug 21]. Available from: https://github.com/juliamatyjasiakvub/AI4AOP_KER_extractor/
