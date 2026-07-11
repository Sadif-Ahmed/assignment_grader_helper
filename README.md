# 📝 Automated Assignment Grader

A local Python command-line tool that batch-grades handwritten student PDF submissions against a master solution using a **two-stage multimodal LLM pipeline**. Powered by **NVIDIA NIM** (OpenAI-compatible endpoints), defaulting to `qwen/qwen3.5-397b-a17b` for every task (overridable with `--model-id`).

---

## How It Works

```
┌─────────────────┐
│  Question PDF   │───▶ Stage 1: Build Blueprint ──▶ Question Blueprint (JSON)
│  (mandatory)    │                                     │
└─────────────────┘                                     │
                                                        ▼
┌─────────────────┐    ┌──────────────────────┐      ┌─────────────────────┐
│  Master Solution│───▶│  Stage 1: Mapping    │─────▶│  Master Solution    │
│  PDF            │    │  (transcription)     │      │  Blueprint (JSON)   │
└─────────────────┘    └──────────────────────┘      └────────┬────────────┘
                                                              │
┌─────────────────┐    ┌──────────────────────┐               │
│  Student PDF    │───▶│  Stage 1: Mapping    │               │
│  (batch process)│    │  (transcription)     │               │
└─────────────────┘    └──────────┬───────────┘               │
                                  │                           │
                                  ▼                           │
                       ┌──────────────────────┐               │
                       │  Stage 2: Evaluation │◀──────────────┘
                       │  (grading & feedback)│
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Evaluation Report   │
                       │  (structured JSON)   │
                       └──────────────────────┘
```

**Workflow Efficiency:** To optimize API usage, the pipeline extracts the structure from the **Question PDF** once, and maps the **Master Solution** to that structure once. Then, for each student submission, it maps their answers to the pre-computed question blueprint and passes it to Stage 2 for grading. Submissions run concurrently (a thread pool, bounded by the API's rate limit), not one at a time.

**Stage 1 – Structural Mapping:** Extracts and transcribes all handwritten content from a student PDF without evaluating correctness, slotting answers directly into the question blueprint. Two completeness checks guard this stage: OCR extraction rejects blank/failed page transcriptions, and the mapping step rejects silently truncated output (a model returning fewer questions than it declared) — both retry automatically instead of passing bad data downstream.

**Stage 2 – Evaluation & Grading:** Compares the student's mapped answers against the master solution blueprint using your chosen criteria, producing a structured JSON report. A completeness check rejects grading output that doesn't cover every top-level question (a model that stops partway through), retrying automatically.

---

## Prerequisites

- **Python 3.11+** — [Download here](https://www.python.org/downloads/)

### How to Acquire an NVIDIA API Key
To use the high-performance vision and reasoning models for free, you need an NVIDIA NIM API Key:
1. Go to [NVIDIA Build](https://build.nvidia.com/) and create a free account or log in.
2. Navigate to any model (e.g., search for *Llama-3.2-90B-Vision-Instruct*).
3. Click the **"Get API Key"** button (often located in the top corner or under the model playground).
4. Generate a new key. It will start with `nvapi-`. Copy this key.
5. Save it in a file named `api_key.txt` in the project root (excluded from git) — the CLI reads it automatically. Alternatively, pass it per-run with `--api-key`.

---

## Setup

### 1. Clone or Download

```bash
git clone <repo-url>
cd assignment_checker
```

Or simply download and extract the project folder.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `openai` | API communication with NVIDIA NIM |
| `tenacity` | Exponential backoff and retry handling |
| `PyMuPDF` | Converts PDFs to images for multimodal LLM processing |
| `Pillow` | Image resizing to fit NIM's resolution limits |
| `pydantic` | Dynamic JSON schema generation |

> JSON file tracker is used automatically — no separate database install needed.

---

## Prepare Your Files

Organize your student submissions in a single folder:

```
submissions/
├── 1042.pdf
├── 1043.pdf
├── 1044.pdf
└── ...
```

- Each PDF filename becomes the **student ID** — the tool extracts the longest run of digits in the filename (e.g., `1042.pdf`, `1042 - Jane Doe.pdf`, and `CSE340_1042_A1.pdf` all become student `1042`).
- Filenames with spaces are automatically renamed (spaces → underscores) the first time you run a batch against a folder.
- Two files that resolve to the same student ID (an accidental double-upload) are both graded, but the summary CSV keeps only the more complete one (see [Batch Summary CSV](#batch-summary-csv) below) — which file is treated as "primary" vs `[duplicate]` is decided by sorted filename order, so it's stable across reruns.
- Have your **question PDF** and **master solution PDF** ready as separate files (paths are passed on the command line, they don't need to live inside the submissions folder).

---

## Run the Tool

```bash
python cli.py --question-pdf Q.pdf --master-pdf S.pdf --target-dir submissions/
```

| Flag | Required | Meaning |
|---|---|---|
| `--question-pdf` | yes | Assignment question paper PDF |
| `--master-pdf` | yes | Master solution / answer key PDF |
| `--target-dir` | yes | Folder containing student submission PDFs |
| `--api-key` | no | Overrides `api_key.txt` |
| `--model-id` | no | Single model override; default is `qwen/qwen3.5-397b-a17b` for every task |
| `--criteria` | no | Evaluation criterion text; repeatable. Default is the 3 standard criteria below |
| `--workers` | no | Concurrent submissions in flight (default 6) |
| `--limit` | no | Only process the first N pending submissions — handy for a quick test batch before running the whole class |

Example with custom criteria and a smaller worker pool:

```bash
python cli.py --question-pdf Q.pdf --master-pdf S.pdf --target-dir submissions/ \
    --criteria "Answered all questions?" --criteria "Used proper commenting?" \
    --workers 3
```

The default criteria (used when `--criteria` isn't passed) are:
- Answered all questions?
- Answers are correct?
- Overall remarks (A/B/C)?

### What happens when you run it

1. A pre-flight check confirms the API key/model work before touching anything, so a bad key fails in seconds instead of after grinding through retries for every student.
2. Scans the target directory for PDFs, sanitizes filenames, registers them in `reports/tracker.json`.
3. Builds (or loads a cached) question blueprint and master solution blueprint.
4. Processes each pending submission through Stage 1 → Stage 2, several at a time.
5. Writes results to a `reports/` subfolder as it goes, with live console progress and an ETA.

Interrupt anytime with Ctrl+C — already-graded submissions stay recorded in `tracker.json`; rerunning the same command resumes where it left off.

**Step 3 is all-or-nothing on purpose.** The question and master blueprints are shared across every student in the batch and cached to `_question_blueprint.json` / `_master_blueprint.json` — if building either one fails after retries, the whole run stops immediately with a clear error instead of continuing with an empty/garbage blueprint that would silently corrupt grading for every student. Just rerun the same command; nothing is cached until it succeeds.

### Review Results

After processing, find results in:

```
submissions/
└── reports/
    ├── tracker.json                   # Processing state tracking
    ├── _question_blueprint.json       # Run-once question hierarchy
    ├── _master_blueprint.json         # Run-once master solution answers
    ├── 1042_map.json                  # Stage 1: structural mapping
    ├── 1042_evaluation.json           # Stage 2: grading report
    ├── 1043_map.json
    ├── 1043_evaluation.json
    ├── ...
    └── report/
        └── summary.csv                # Batch summary — one row per student
```

### Batch Summary CSV

Alongside the per-student reports, the tool writes `reports/report/summary.csv` — one row per student, sorted by student ID:

| Column | Content |
|---|---|
| Student ID | Sorted numerically |
| Full Comment | Every question's full feedback text, verbatim |
| Total Sub-Questions | Count of graded sub-questions for that student |
| Complete / Missing / Partial | Sub-question status tally |
| Correct / Incorrect | Sub-question correctness tally |
| Answered All Questions | From the default criteria (or your `--criteria`) |
| Answers Correct | From the default criteria (or your `--criteria`) |
| Overall Remarks (A/B/C) | From the default criteria (or your `--criteria`) |
| Review Flag | Blank if the row looks structurally sound; a reason string if it needs a human to check it |

It's regenerated automatically at the end of every run. Duplicate submissions (two files that resolved to the same student ID) are collapsed to a single row using whichever attempt covers more top-level questions — a truncated grading response can score deceptively high on the few questions it reached, so completeness is checked before correctness score is used as a tiebreaker.

**Review Flag** — even with the retry guards in the pipeline, a model can occasionally exhaust every retry and still leave behind malformed data (e.g. it self-reports covering all questions but the structure shows otherwise) or fail every attempt outright. Rather than trust the row at face value, every student is checked against the batch's own data:
- **Incomplete coverage**: the student's graded questions cover fewer top-level questions than most of the class did (the "normal" count is inferred from the batch itself, not hardcoded) — flagged with which question numbers are missing.
- **Duplicate entries**: more raw feedback entries than distinct questions covered (a parent-question entry and its own sub-parts both present) — a sign the grading response was internally inconsistent.
- **Never successfully graded**: every attempt for that student failed (per `tracker.json`) and there's no evaluation file at all — they'd otherwise be silently absent from the summary; instead they get their own flagged row.

Flagged rows still need a human to look at the source PDF and either fix the underlying issue (bad OCR, a model having a bad day — see Troubleshooting) or grade manually; the flag exists so they can't slip through unnoticed.

---

## Checkpointing & Resume

The tool tracks each submission's status in `reports/tracker.json`:

| Status | Meaning |
|---|---|
| `PENDING` | Not yet processed |
| `MAPPED` | Stage 1 complete, Stage 2 pending |
| `EVALUATED` | Both stages complete |
| `FAILED` | An error occurred |

**Interrupted or a submission failed?** Just rerun the same command:
- `EVALUATED` files are skipped.
- `MAPPED` files resume from Stage 2.
- `FAILED` files are retried from scratch.

---

## Rate Limit Handling & Retries

- Detects rate limits/transient errors and retries with exponential backoff (up to 3 attempts per call).
- Every task (question extraction, student parsing, restructuring, evaluation) is pinned to a single model (`qwen/qwen3.5-397b-a17b` by default, or `--model-id`) rather than falling back across models — a weaker fallback model silently producing worse structured output was worse than failing loudly and retrying the same model. Use `--model-id` to override on a rerun if a specific model is having a bad day.
- Three completeness checks catch valid-but-wrong JSON that would otherwise silently corrupt downstream data, and retry it the same way as a hard API failure:
  - OCR extraction rejects a page that comes back blank.
  - Stage 1 structuring/mapping rejects output that declares N questions but only delivers some of them.
  - Stage 2 evaluation rejects output that doesn't cover every top-level question.
- Submissions process concurrently (`--workers`, default 6); the shared rate limiter is what actually caps throughput, not the worker count.
- If a submission still fails after retries, it's marked `FAILED` in `tracker.json` and picked up again automatically on the next run of the same command — no data is corrupted by a failed attempt, since results are only written on success.

## Multimodal Limitations Bypassed

NVIDIA NIM's open-weights vision endpoints (like `Llama-3.2-90B-Vision`) enforce a strict **"1 image per prompt"** limit. The pipeline works within this by sending one page per API call (each PDF page rendered and resized to fit the model's resolution limit via Pillow) rather than trying to pack a whole document into a single request.

---

## Alternative: Grading with Claude Code Instead of the NVIDIA Pipeline

If you're running this inside [Claude Code](https://claude.com/product/claude-code) and don't have (or don't want to use) an NVIDIA API key, there's a second path: `.claude/skills/grade-assignments/SKILL.md`. It has Claude Code itself do the reading, mapping, and grading — no `cli.py`, no NVIDIA API call — while writing to the exact same `reports/tracker.json` / `_question_blueprint.json` / `_master_blueprint.json` / `<id>_evaluation.json` / `summary.csv` schema described above, via `report_export.py` unmodified. Either path produces output the other can read.

**How it works:** one subagent per student PDF (Claude Code's vision-capable `Read` tool reads the PDF directly — no OCR/rendering step), run a handful in parallel, results written back to the same `reports/` folder. See the skill file for the full runbook, including duplicate-upload handling and identity-mismatch checks.

**To use it:** in Claude Code, invoke `/grade-assignments` (or just ask it to grade a folder of submissions) and point it at your question PDF, master solution PDF, and submissions folder.

**Still needed even on this path:**
- Python 3.11+ and `pip install -r requirements.txt` — the skill imports helper functions (`discover_student_pdfs`, `find_duplicate_uploads`) from `cli.py`, which pulls in the full dependency list at import time regardless of which path grades the batch.
- No `api_key.txt` needed — this path never calls the NVIDIA API.

**Tradeoff vs. the NVIDIA pipeline:** cost and concurrency are bounded differently (usage on your Claude plan vs. NVIDIA NIM's rate limits/free tier), and large classes mean one subagent invocation per student — real time and cost at scale. Prefer `cli.py` if you have an NVIDIA key and are grading dozens+ of students regularly; prefer the Claude Code skill if you'd rather not manage a key, or want Claude doing the grading directly.

---

## Project Structure

```
assignment_checker/
├── cli.py                # CLI entry point: batch orchestration, tracker, filename handling
├── nvidia_client.py      # NVIDIA API integration & PDF-to-image conversion
├── stage1.py             # Stage 1: structural mapping pipeline
├── stage2.py             # Stage 2: evaluation & grading pipeline
├── model_pools.py        # Per-task LLM model pool configuration
├── report_export.py      # Batch summary CSV export
├── check_llm.py          # CLI utility to test NVIDIA connection
├── checker_stage1.py     # CLI utility: Stage 1 only, single submission
├── checker_stage1_2.py   # CLI utility: Stage 2 only, from pre-computed JSON
├── test_stage1_validation.py  # Self-check: Stage 1 completeness/OCR guards
├── test_stage2_validation.py  # Self-check: Stage 2 completeness guard
├── test_report_export.py      # Self-check: duplicate-submission dedup logic
├── test_sanitize.py           # Self-check: unicode sanitization
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .claude/
    └── skills/
        └── grade-assignments/
            └── SKILL.md   # Alternative grading path: Claude Code as the grader, no NVIDIA API
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API key rejected | Verify your key starts with `nvapi-` and was obtained from [build.nvidia.com](https://build.nvidia.com) |
| "No student PDF files found" | Check that `--target-dir` contains `.pdf` files |
| 429 / "temporarily rate-limited upstream" | NIM models can occasionally get overloaded. Try a different model or wait a few minutes. |
| A specific student keeps ending up `FAILED` / flagged in `Review Flag` after several reruns | Some submissions are just hard for the model — long, unusual formatting, or a symbol not yet in the sanitizer's map. Rerunning the same command retries it (each retry is independent, so it can succeed on a later attempt); if it consistently fails the same way, try `--model-id` with a different model for just that one PDF, or grade it manually. This isn't a sign of a broken pipeline — it's what the Review Flag column exists to catch instead of letting it slip through silently. |
| Building the question/master blueprint fails and the whole run stops | Expected — see "Step 3 is all-or-nothing" above. Just rerun the same command. |

---

## License

This project is provided as-is for educational use.
