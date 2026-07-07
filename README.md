# 📝 Automated Assignment Grader

A local Python command-line tool that batch-grades handwritten student PDF submissions against a master solution using a **two-stage multimodal LLM pipeline**. Powered by **NVIDIA NIM** (OpenAI-compatible endpoints), allowing you to use high-performance vision models like `meta/llama-3.2-90b-vision-instruct`.

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

**Stage 1 – Structural Mapping:** Extracts and transcribes all handwritten content from a student PDF without evaluating correctness, slotting answers directly into the question blueprint. A completeness check catches silently truncated output (a model returning fewer questions than it declared) and automatically retries/falls back to another model.

**Stage 2 – Evaluation & Grading:** Compares the student's mapped answers against the master solution blueprint using your chosen criteria, producing a structured JSON report.

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
- Two files that resolve to the same student ID (an accidental double-upload) are both graded, but the summary CSV keeps only the higher-scoring one.
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
| `--model-id` | no | Single model override; default is the per-task NVIDIA model pools with automatic fallback |
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
    └── summary.csv                    # Batch summary — one row per student
```

### Batch Summary CSV

Alongside the per-student reports, the tool writes `reports/summary.csv` — one row per student, sorted by student ID, with three columns:

| Column | Content |
|---|---|
| Student ID | Sorted numerically |
| Summarised Comment | Flags per student: Partial/Missing sub-questions, "Lack of explanation" (keyword scan on feedback) |
| Overall Quality | The grade/remarks criterion for that run, or a correct-count ratio if no such criterion is active |

It's regenerated automatically at the end of every run. Duplicate submissions (two files that resolved to the same student ID) are collapsed to a single row using whichever attempt scored higher.

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

## Rate Limit Handling & Fallbacks

- Detects rate limits/transient errors and retries with exponential backoff.
- Each task (question extraction, student parsing, restructuring, evaluation) has a pool of 3 fallback models — if one exhausts its retries, the pipeline automatically moves to the next model in the pool without interrupting the batch.
- A completeness check on Stage 1's structuring/mapping output catches a model that returns valid-but-truncated JSON (declares N questions but only delivers some of them) and retries it the same way as a hard API failure.
- Submissions process concurrently (`--workers`, default 6); the shared rate limiter is what actually caps throughput, not the worker count.

## Multimodal Limitations Bypassed

NVIDIA NIM's open-weights vision endpoints (like `Llama-3.2-90B-Vision`) enforce a strict **"1 image per prompt"** limit. The pipeline works within this by sending one page per API call (each PDF page rendered and resized to fit the model's resolution limit via Pillow) rather than trying to pack a whole document into a single request.

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
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API key rejected | Verify your key starts with `nvapi-` and was obtained from [build.nvidia.com](https://build.nvidia.com) |
| "No student PDF files found" | Check that `--target-dir` contains `.pdf` files |
| 429 / "temporarily rate-limited upstream" | NIM models can occasionally get overloaded. Try a different model or wait a few minutes. |

---

## License

This project is provided as-is for educational use.
