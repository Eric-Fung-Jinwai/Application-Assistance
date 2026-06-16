# Evidence-Aware Career Application Assistant

A Python application for parsing resumes, analyzing job descriptions, scoring fit, generating evidence-grounded resume tailoring suggestions, reviewing those suggestions with a human in the loop, and exporting ATS-friendly HTML/PDF resumes.

The project is built around one safety rule: resume rewrites should improve wording and targeting without inventing skills, scope, metrics, or responsibilities. Tailored bullets are scored by an integrity judge before review, and human decisions are stored in an immutable resume version tree.

## What It Does

- Parses TXT, PDF, and DOCX resumes into structured resume data.
- Extracts job-description requirements: required/preferred skills, tools, responsibilities, domain, seniority, and years of experience.
- Embeds resume bullets and JD requirement chunks into ChromaDB.
- Scores resume/JD fit across skill, experience, seniority, location, and compensation dimensions.
- Generates tailoring suggestions for resume bullets using allowed operations only.
- Scores each rewrite for integrity using LLM grounding plus embedding similarity.
- Supports accept, reject, customize, refine, and accept-all review flows.
- Tracks immutable resume versions with stable bullet `lineage_id`s.
- Exports ATS-friendly HTML and PDF using selectable text, not rasterized images.
- Tailors one uploaded resume against many job descriptions in a session: each new JD starts from your original resume, so you never re-upload.
- Provides a FastAPI backend and Streamlit MVP frontend.

## Architecture

Core packages:

- `career_assistant/ingest`: file parsing and structured resume extraction.
- `career_assistant/jd`: job-description analysis, chunking, embeddings, coverage.
- `career_assistant/kb`: resume bullet persistence and vector indexing.
- `career_assistant/fit`: fit scoring.
- `career_assistant/tailor`: rewrite generation and integrity scoring.
- `career_assistant/review`: human review actions and version application.
- `career_assistant/versioning`: immutable resume version tree.
- `career_assistant/export`: render-ready document model, HTML, and PDF export.
- `career_assistant/api`: FastAPI app, schemas, routes, dependencies, error mapping.
- `career_assistant/ui`: Streamlit app and API client.
- `career_assistant/eval`: smoke tests, budget estimator, regression harness, and the benchmark labeling helper.
- `career_assistant/storage`: SQLite models/repo, lightweight migrations, Chroma helpers.
- `career_assistant/llm`: swappable LLM and embedding client interfaces.

Storage:

- SQLite is the source of truth for resumes, versions, bullets, JDs, fit scores, suggestions, and edit history.
- ChromaDB stores vectors keyed by concrete bullet/chunk IDs.
- Bullet `id` identifies one exact bullet row/snapshot.
- Bullet `lineage_id` identifies the same logical bullet across versions.

## Requirements

- Python 3.12+
- OpenAI API key for live LLM calls
- Local or OpenAI embeddings
- Chromium installed through Playwright for PDF export

Python dependencies are declared in `pyproject.toml`.

## Setup

Create and activate an environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install Chromium if you need PDF export:

```bash
playwright install chromium
```

Create `.env` if you want to override defaults:

```bash
OPENAI_API_KEY=...
LLM_MODEL=gpt-5.4-nano
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SQLITE_PATH=./data/career_assistant.db
CHROMA_PATH=./data/chroma
```

The default embedding provider is local Hugging Face `sentence-transformers/all-MiniLM-L6-v2`. The embedding factory is cached per process, so the model loads once after backend startup rather than once per request.

## Run

Start the FastAPI backend:

```bash
uvicorn career_assistant.api.app:app --reload
```

Start the Streamlit UI in another terminal:

```bash
streamlit run career_assistant/ui/app.py
```

The UI talks to the API over HTTP. Configure the API URL with:

```bash
export CAREER_ASSISTANT_API_URL=http://localhost:8000
```

Default:

```text
http://localhost:8000
```

### Tailoring for multiple jobs

The app is a single-user personal tool. Upload your resume once, then work one JD at a time: paste a job description, review and apply tailored bullets, and export. To target the next job, click **"Tailor for another job — keep my resume"** (after export) or **"Analyze a different JD"** — both clear the JD and rewind to your original resume, so each application is tailored from the clean resume rather than the previous job's edits. The resume persists for the session; re-select or re-upload it after a full restart.

## API

Main endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/parse_resume` | Upload resume file, parse, create original version, index bullets |
| `POST` | `/analyze_jd` | Analyze JD text, persist parsed JD, index requirement chunks |
| `POST` | `/fit_score` | Score one resume version against one JD |
| `POST` | `/recommendation` | Build an apply recommendation from fit and coverage |
| `POST` | `/tailor` | Generate persisted tailoring suggestions |
| `POST` | `/review` | Accept, reject, customize, or refine one suggestion |
| `POST` | `/accept_all` | Apply multiple pending suggestions into one accepted version |
| `GET` | `/budget` | Estimate cost/latency budget |
| `POST` | `/generate_html` | Render a resume version to HTML |
| `POST` | `/generate_pdf` | Render a resume version to PDF |

Example:

```bash
curl -X POST http://localhost:8000/analyze_jd \
  -H "Content-Type: application/json" \
  -d '{"text": "We need a senior Python engineer with Docker experience."}'
```

## Review Flow

Tailoring suggestions are initially `pending`.

Review actions:

- `accept`: creates an `accepted` child version using the suggested text.
- `reject`: returns the latest accepted ancestor on the current branch, or original when none exists.
- `customize`: re-scores human-edited text, then creates a `customized` child version.
- `refine`: uses chat-style instruction to update the pending suggestion and re-score it; no version is created yet.
- `accept_all`: applies several pending suggestions into one accepted child version.

When reviewing several suggestions, pass the current `head_version_id` as `base_version_id` so accepts/customizations chain onto one branch.

## Export

Templates:

- `ats`
- `technical`
- `pm`
- `modern`

HTML uses Jinja2 autoescaping. PDF export uses Playwright/Chromium and preserves selectable text for ATS compatibility.

If Chromium is missing, PDF export returns a 503-style runtime error with the setup hint:

```bash
playwright install chromium
```

## Evaluation And Tests

Run targeted non-network suites:

```bash
python -m pytest tests/test_api_phase14.py tests/test_ui_phase13.py tests/test_export_phase12.py tests/test_review_phase10.py tests/test_tailor_phase8.py -q
```

Run the regression harness:

```bash
python -m career_assistant.eval.harness
```

The harness runs two offline gates over the labeled datasets in `career_assistant/eval/datasets/`: fit-band accuracy (must stay at or above 80%) and integrity precision/recall (every planted fabrication must be caught). Both use cached signals, so the default run needs no network. Optional flags:

```bash
python -m career_assistant.eval.harness --sweep      # grid-search integrity weights/cutoffs (prints recommendation only)
python -m career_assistant.eval.harness --live       # run the real LLM integrity judge end-to-end (needs API access)
python -m career_assistant.eval.harness --benchmark  # also score your local fit_pairs_benchmark.json (see below)
```

`--live` exercises the actual judge prompt, parsing, and embedding pipeline so a prompt regression that stops catching a fabrication fails the build; it calls the configured LLM provider. `--benchmark` folds your hand-labeled benchmark into the fit gate on its own line.

### Building your own fit benchmark

In case you want to personalize it. The larger fit benchmark (`career_assistant/eval/datasets/fit_pairs_benchmark.json`) is **not committed** — it contains your real resume data, so it is gitignored and each developer creates their own. A focused single-role set of roughly 20 pairs spanning strong/moderate/weak fit is a real regression gate, since the harness checks a percentage rather than a fixed count.

Each pair is a real resume, a real JD, and *your* honest fit band — the band label is your judgment, never machine-generated. The labeling helper runs the real parsers once on raw text and appends a pre-parsed, labeled entry, so you never hand-write the JSON:

```bash
# 1. Put your resume text in one file and a JD in another (e.g. under a gitignored scratch/ dir).
# 2. Add one labeled pair (parses the resume + JD; needs API access):
python -m career_assistant.eval.label add \
    --name myrole_strong_01 --role "AI-Eng" --band strong \
    --resume scratch/resume.txt --jd scratch/jd.txt

# 3. Overwrite the JD file with the next posting and repeat with a new --name and band.

python -m career_assistant.eval.label stats     # progress and band spread
python -m career_assistant.eval.label validate  # schema-check the file
```

`--role` is free-form metadata; `--band` must be `strong`, `moderate`, or `weak`. Once you have pairs, score them with `python -m career_assistant.eval.harness --benchmark`. The file lives only on your machine, so back it up separately if you want to keep it.

Run the budget estimator:

```bash
python -m career_assistant.eval.budget
```

Run lint:

```bash
python -m ruff check .
```

Known current lint note: full `ruff check .` reports `UP042` for three older `domain.py` enums that still inherit from `str, Enum`. Phase 14 already uses `StrEnum` for new API enums.

The full `pytest -q` suite includes `tests/test_smoke_phase0.py`, which makes a live OpenAI request. In restricted/offline environments, that test fails with an API connection error. Use targeted suites for offline development.

## Migrations

`repo.create_db(engine)` runs `Base.metadata.create_all(engine)` and then `storage/migrations.py::run_migrations(engine)`.

The lightweight migration layer handles pre-Alembic schema drift, including:

- adding/backfilling `bullets.lineage_id`
- adding `tailoring_suggestions.integrity_flags_json`
- reconstructing legacy lineage by parent/child `(section, order_index)` where safe
- creating lineage indexes
- failing loudly on ambiguous duplicate legacy positions

Full Alembic-style migrations are deferred.

## Scope And Limitations

This is a single-user personal tool. Resume PII is sent to the configured LLM provider for parsing and tailoring — appropriate for personal use, not a shared deployment.

- Preferences/questionnaire data in the UI is captured locally only; it is not fed into scoring (Phase 2, deferred).
- Fit/recommendation currently score from `resume.parsed_json`, not from edited bullet text.
- Location and compensation are not-yet-evaluated seams: the JD analyzer does not extract them, so the recommendation flags them as not evaluated rather than guessing.
- The review panel shows suggestion text, integrity score, and flags; it does not yet fetch a full original-vs-suggested diff payload.
- Streamlit styling uses best-effort CSS against Streamlit DOM structure; core layout and colors should hold, but internal selectors can drift across Streamlit versions.
- Export regrouping maps version bullet text back to roles/projects positionally. It safely falls back to ungrouped bullets if counts drift.

## Development Notes

- Do not call OpenAI directly outside `career_assistant/llm`.
- SQLite owns text; Chroma owns vectors.
- Chroma resume vector IDs should use concrete `BulletRow.id`, not `lineage_id`.
- Version edits should target `lineage_id`, so stale suggestions can apply to the current head.
- Dynamic text in Streamlit `unsafe_allow_html=True` blocks must be escaped with `theme.esc`.
- Dynamic text rendered through normal Streamlit Markdown should use `theme.esc_md` when formatting must be preserved literally.
