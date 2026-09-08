# Hospital RAG Assistant

A patient information assistant for a hospital. It answers practical questions
from the hospital's own documents, shows the source of every claim, refuses
anything clinical, and books outpatient appointments.

Built with **Django 5.2**, **PostgreSQL 17 + pgvector**, **HTMX**, and
**Claude**. Server-rendered throughout — no JavaScript framework, no separate
API, one deployable.

```
Question ─▶ Router ─┬─ emergency        ─▶ call 999 (never reaches a model)
                    ├─ clinical advice  ─▶ refused, redirected to a clinician
                    ├─ appointment      ─▶ booking tools ─▶ writes to the database
                    └─ information      ─▶ hybrid search ─▶ grounded, cited answer
                                                        └─ nothing relevant? say so
```

---

## Why this project exists

Most RAG tutorials build a chatbot over some PDFs and stop at "it answers
questions". That skips the three things that decide whether a system like this
can be deployed:

1. **Knowing when not to answer.** Hospital documents sit next to questions
   nobody should answer from a leaflet. "How long must I fast before a
   colonoscopy?" is published guidance. "Should I stop my warfarin first?"
   depends on a medical record. The second must be refused, every time, and the
   line between them is not obvious to a model that is trying to be helpful.

2. **The boundary between retrieval and action.** Answering a question is
   search. Booking an appointment is a database write with consequences.
   Systems that blur the two produce "AI booking agents" that are really search
   engines with optimism.

3. **Being able to prove it works.** `manage.py evaluate` scores retrieval and
   routing against a labelled question set. Without that, "I improved the
   chunking" is an opinion.

---

## What it does

| | |
|---|---|
| **Answers from documents** | Markdown, PDF, and Word files are chunked, embedded, and indexed. Every answer cites the passage it came from, with the document's review date. |
| **Hybrid search** | Vector search for meaning, Postgres full-text for exact terms, merged with Reciprocal Rank Fusion. Both live in the same database. |
| **Says "I don't know"** | A relevance gate rejects questions the corpus does not cover, instead of building an answer out of six irrelevant passages. |
| **Refuses clinical questions** | A router classifies every message before anything is retrieved. Emergencies get a fixed reply that never touches a model. |
| **Books appointments** | Two tools — search the appointment book, take a slot — with row-level locking so the same slot cannot be booked twice. |
| **Streams** | Answers appear token by token over Server-Sent Events, with progress while search runs. |
| **Is auditable** | Every conversation, route decision, and cited passage is stored and readable in Django admin. |

**One API key.** Embeddings run locally on ONNX, so the only key you need is
Claude's.

---

## Running it

Requires Python 3.10+, Docker, and a Claude API key from
[console.anthropic.com](https://console.anthropic.com/settings/keys).

```bash
git clone https://github.com/aalhommada/hospital-rag-assistant.git
cd hospital-rag-assistant

make install            # virtualenv + dependencies
cp .env.example .env    # then put your ANTHROPIC_API_KEY in it

make db-up              # Postgres 17 + pgvector, on port 5433
make seed               # migrate, ingest the documents, create appointment slots
make run                # http://127.0.0.1:8000
```

First run downloads the embedding model (~130 MB) once.

```bash
make test               # 70 tests, no API key needed
make evaluate           # score retrieval (no API key needed)
make evaluate-routing   # also score routing (uses the API)
make lint
```

An admin account for browsing documents and transcripts:

```bash
.venv/bin/python manage.py createsuperuser   # then visit /admin/
```

### Try these

| Question | What should happen |
|---|---|
| `Can I eat before an MRI scan?` | Answered, citing the MRI leaflet |
| `What number do I call for radiology?` | Exact number, found by the keyword arm |
| `And for the children's ward?` (after asking about visiting hours) | Follow-up resolved into a standalone search |
| `Should I stop my warfarin before my colonoscopy?` | Refused, redirected to a clinician |
| `I have crushing chest pain` | Immediate 999 redirect, no model call |
| `What is the capital of France?` | "I could not find that in the hospital's documents" |
| `Book me a cardiology appointment next week` | Offers real slots, asks for name and number, books one |

---

## How it works

### Ingestion

```
sample_data/*.md,pdf,docx
      │
      ▼
  loaders.py     read the file, keep headings and paragraphs
      │
      ▼
  chunker.py     split on headings, never split a paragraph
      │
      ▼
 embeddings.py   384 numbers per chunk (local ONNX model)
      │
      ▼
  pipeline.py    write chunks + build the keyword index, in one transaction
```

Ingestion is idempotent — every document stores the SHA-256 of its source file,
so `make ingest` skips anything unchanged. When a file does change, its old
chunks are replaced atomically; there is no moment where half the old leaflet
and half the new one are both retrievable.

**Chunking follows the document, not a character count.** A new heading always
starts a new chunk, and a paragraph is never split. This matters more than it
sounds: a chunk is the smallest thing search can return, so if "you must not eat
for six hours" is split from "before your MRI scan", no retriever and no model
can put it back together. Each chunk also carries its heading path — `Visiting
hours > Intensive care unit` — which goes into both the embedding and the prompt.

### Retrieval

Two searches run over the same table and are merged.

| | Good at | Bad at |
|---|---|---|
| **Vector search** | Paraphrase. Finds "you must not eat for six hours" from "can I have breakfast first?" | Exact tokens: phone numbers, ward names |
| **Keyword search** | Exact tokens, rare strings, identifiers | Anything phrased differently from the document |

They fail in opposite places, which is the whole argument for running both.

Merging uses **Reciprocal Rank Fusion**: `score = Σ 1 / (60 + rank)`. It throws
the scores away and keeps only the ranks, because a cosine distance and a
`ts_rank` are different units that cannot be compared directly.

**Ordering and relevance are separate jobs.** RRF is excellent at the first and
useless at the second: its top result always scores `1/61` whether the question
was "how much is parking?" or "what is the capital of France?". So a chunk earns
its place in the prompt by clearing one of two independent bars:

- **Semantic**: cosine similarity ≥ `RETRIEVAL_MIN_SIMILARITY` (0.65). Measured
  on this corpus, answerable questions score 0.68–0.91 and unanswerable ones
  0.41–0.56, so the floor sits in the gap.
- **Lexical**: the keyword arm matched it at all. Needed because a bare
  identifier carries almost no semantic content — searching `020 7946 0400`
  scores 0.614 and would otherwise be discarded with the right answer in hand.
  Safe because `websearch_to_tsquery` requires *every* term to appear, which is
  why off-topic questions return no keyword hits whatsoever.

When nothing clears either bar, retrieval returns nothing and the assistant says
so. That single decision is most of what separates this from a demo.

### Routing and safety

One cheap call classifies each message and rewrites it into a standalone search
query, using the conversation so far — so "and for the children's ward?" becomes
"visiting hours for the children's ward" before anything is searched.

| Intent | Handling |
|---|---|
| `emergency` | Fixed reply. Never reaches a model — it must be identical every time and instant. |
| `clinical_advice` | Fixed refusal naming who *can* answer. Retrieval never runs. |
| `appointment` | Booking tools. |
| `information` | Retrieve, ground, cite. |
| `other` | Greeting. |

Ambiguity resolves toward refusal: a message the router cannot classify is
treated as clinical. Refusing something harmless is cheap; answering something
clinical is not.

The answering prompt adds a second layer — answer only from the numbered
passages, cite every claim, copy times and numbers exactly, never interpret a
symptom. And because a request can come back with `stop_reason == "refusal"`
and no content at all, that case is checked for explicitly rather than
discovered as an empty answer.

### The two halves

The booking tools are deliberately ordinary Python. Nothing about them is AI:

- **Reads are wide, writes are narrow.** Searching slots takes fuzzy input.
  Booking takes an exact `slot_id` the model can only have got from a search.
- **The write is guarded in the database.** `select_for_update` means two people
  racing for the last slot cannot both get it, however the conversation went.
- **Failed tools return explanations, not exceptions.** "That slot has just been
  taken, here are three others" is something a model can recover from.

---

## Configuration

Everything lives in `.env` — see `.env.example` for the annotated list.

| Setting | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | The only key required |
| `CLAUDE_ANSWER_MODEL` | `claude-sonnet-5` | Writes the grounded answer. Move up to `claude-opus-5` if you see paraphrased numbers or missing citations |
| `CLAUDE_ROUTER_MODEL` | `claude-sonnet-5` | Classifies and rewrites. Runs on every message, so it decides your bill — and it is safety-critical. Change it with `make evaluate-routing`, not on a hunch |
| `CLAUDE_ENABLE_FALLBACKS` | `true` | Server-side refusal fallbacks (see below) |
| `EMBEDDING_PROVIDER` | `local` | `local`, `voyage`, or `openai` |
| `RETRIEVAL_TOP_K` | `6` | Passages placed in the prompt |
| `RETRIEVAL_CANDIDATES` | `30` | Results per search arm before fusion |
| `RETRIEVAL_MIN_SIMILARITY` | `0.65` | The relevance floor |
| `CHUNK_TARGET_WORDS` | `180` | Chunk size budget |

### Refusal fallbacks

A hospital corpus sits close to topics that safety classifiers watch —
medicines, doses, procedures. A request can return `stop_reason: "refusal"` with
no content. With `CLAUDE_ENABLE_FALLBACKS=true` (the default) the API retries
the same request on a fallback model inside the same call, so a borderline
question about fasting before an operation still gets answered from the
hospital's own leaflet. Set it to `false` to see refusals directly.

### Model choice and cost

Two calls happen per question: a router call on every message, and a generation
call on the ones that get answered. Both models are set independently.

Every call logs its own token count and estimated cost at `INFO` level:

```
INFO assistant.llm: usage router model=claude-sonnet-5 in=812 cached=0 out=141 est=$0.00303
INFO assistant.llm: usage answer model=claude-sonnet-5 in=2794 cached=0 out=387 est=$0.00946
```

Watch the output number. With adaptive thinking on, reasoning tokens are billed
as output even though the patient never sees them — invisible in the transcript
and very visible on the bill.

Rough per-question totals, both calls included:

| | Per question | 1,000 questions | 100,000 questions |
|---|---|---|---|
| `claude-opus-5` | ~$0.029 | ~$29 | ~$2,900 |
| `claude-sonnet-5` *(default)* | ~$0.012 | ~$12 | ~$1,160 |
| `claude-haiku-4-5` router + Sonnet answer | ~$0.009 | ~$9 | ~$870 |

For trying the project out, this is pennies either way. It only starts to matter
at deployment volume — and by then you should be choosing from `make
evaluate-routing` output rather than from this table.

**Do not downgrade the router without measuring it.** It is the component that
separates "how long must I fast?" from "should I stop my warfarin?". The
evaluation reports refusal mistakes separately and in red, precisely because
that is the one number that must not regress:

```
  Routed correctly                             44/47  (94%)
  1 question(s) that should have been refused were not. Fix this before anything else.
```

Prompt caching is not used here. Both system prompts are under a thousand
tokens, below the minimum cacheable prefix, so there would be nothing to cache.
That changes if you grow the prompts substantially.

### Changing the embedding model

The vector width is compiled into the database column, so it cannot be changed
by editing `.env` alone. To move to a model with different dimensions:

1. Change `EMBEDDING_DIMENSIONS` in `knowledge/models.py`.
2. `python manage.py makemigrations knowledge && python manage.py migrate`
3. `make reset` — re-embeds every chunk.
4. `make evaluate` — the similarity floor is model-specific and will need
   re-tuning. The command prints both ranges and tells you where the floor
   should sit.

`get_embedding_provider()` raises at startup if a provider's width does not
match the column, rather than letting wrongly sized vectors reach Postgres.

---

## Layout

```
config/          Django settings, URLs
knowledge/       documents, chunks, ingestion, retrieval
  ingest/        loaders.py · chunker.py · pipeline.py
  embeddings.py  swappable providers, local by default
  retrieval.py   hybrid search + RRF + the relevance gate
appointments/    departments, slots, bookings, and the two tools
assistant/       router · prompts · answering · rendering · views
evaluation/      questions.yaml — the labelled question set
sample_data/     12 hospital documents (10 markdown, 1 PDF, 1 Word)
tests/           70 tests
```

Worth reading first: `knowledge/retrieval.py` for the search design,
`assistant/answering.py` for the orchestration, and `assistant/prompts.py` for
every instruction the model receives.

---

## Known limits

Stated plainly, because a system like this is defined as much by what it does
not do.

- **Scanned documents are not supported.** The PDF loader reflows extracted
  text and rebuilds headings from line widths, which works on plainly formatted
  leaflets. A scan contains no text to extract, and a two-column layout will
  interleave. Both need a layout-aware parser or a vision model.
- **No reranking.** Retrieve-wide-then-rerank with a cross-encoder is the next
  quality step and is not here. On a corpus this size the similarity gate is
  doing the work reranking would.
- **No authentication or multi-tenancy.** Conversations are keyed to a browser
  session. A real deployment needs patient identity before it can say anything
  personal — and this assistant deliberately never does.
- **English only.** The full-text index uses the `english` configuration, and
  the embedding model is English. Other languages need a different config and a
  multilingual model.
- **Sample data is fictional.** Riverside General Hospital does not exist. Every
  telephone number, price, and opening time is invented.
- **This is a teaching project.** It is not clinically governed, and it must not
  be deployed to real patients without review by the people accountable for the
  information it repeats.

---

## Licence

MIT.
