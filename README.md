# 📝 Automated Assignment Grader

A local Python application that batch-grades handwritten student PDF submissions against a master solution using a **two-stage multimodal LLM pipeline**. Built with Streamlit for the UI and powered by **NVIDIA NIM** (OpenAI-compatible endpoints), allowing you to use high-performance vision models like `meta/llama-3.2-90b-vision-instruct`.

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

**Workflow Efficiency:** To optimize API usage, the application extracts the structure from the **Question PDF** once, and maps the **Master Solution** to that structure once. Then, for each student submission, it maps their answers to the pre-computed question blueprint and passes it to Stage 2 for grading.

**Stage 1 – Structural Mapping:** Extracts and transcribes all handwritten content from a student PDF without evaluating correctness, slotting answers directly into the question blueprint.

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
5. You will paste this key directly into the application's sidebar. *(Optional: Save it locally in an `api_key.txt` file for safekeeping—it is excluded from git).*

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
| `streamlit` | Web UI framework |
| `openai` | API communication with NVIDIA NIM |
| `tenacity` | Exponential backoff and retry handling |
| `PyMuPDF` | Converts PDFs to images for multimodal LLM processing |
| `Pillow` | Stitches PDF pages into a single vertical image to bypass NIM limits |
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

- Each PDF filename becomes the **student ID** (e.g., `1042.pdf` → student `1042`).
- Have your **master solution PDF** ready separately (it will be uploaded through the UI).

---

## Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at **http://localhost:8501**.

---

## Usage Guide

### Step 1: Configure (Sidebar)

1. **API Key** — Paste your NVIDIA API key (`nvapi-...`) into the password field.
2. **Model** — Select a model from the Global Pool fallback list or a specific multimodal model.
3. **Question PDF (Mandatory)** — Upload the original assignment question paper to generate the structural blueprint.
4. **Master Solution** — Upload your answer-key PDF.
5. **Target Directory** — Either type/paste the path to your submissions folder, or use the **📂 Browse Folders** expander to navigate to it.

### Step 2: Set Evaluation Criteria

The sidebar provides four default criteria (all enabled by default):

- ☑️ Answered all questions?
- ☑️ Answers are correct?
- ☑️ Used hallucinated instructions (mov, li, lu, ble)?
- ☑️ Overall remarks (A/B/C)?

You can:
- **Uncheck** any default criterion to exclude it.
- **Add custom criteria** using the text input + ➕ button at the bottom.
- **Remove custom criteria** using the 🗑️ button next to each.

> Any change to the criteria immediately rebuilds the evaluation prompt and the Pydantic JSON schema. You can preview the active schema using the **📐 Active Evaluation Schema** expander in the main area.

### Step 3: Process

Click **🚀 Start Batch Processing** to begin. The app will:

1. Scan the target directory for PDFs.
2. Register them in a local `tracker.json` file.
3. Process each submission through Stage 1 → Stage 2.
4. Save results to a `reports/` subfolder.
5. Show real-time progress and a live log.

### Step 4: Review Results

After processing, find results in:

```
submissions/
├── tracker.json                      # Processing state tracking
└── reports/
    ├── _question_blueprint.json      # Run-once question hierarchy
    ├── _master_blueprint.json        # Run-once master solution answers
    ├── 1042_map.json                 # Stage 1: structural mapping
    ├── 1042_evaluation.json          # Stage 2: grading report
    ├── 1043_map.json
    ├── 1043_evaluation.json
    └── ...
```

---

## Checkpointing & Resume

The application tracks each submission's status in `tracker.json`:

| Status | Meaning |
|---|---|
| `PENDING` | Not yet processed |
| `MAPPED` | Stage 1 complete, Stage 2 pending |
| `EVALUATED` | Both stages complete |
| `FAILED` | An error occurred |

**Interrupted?** Just click **Start Batch Processing** again:
- `EVALUATED` files are skipped.
- `MAPPED` files resume from Stage 2.
- `FAILED` files are retried from scratch.

---

## Rate Limit Handling & Fallbacks

The app automatically handles API rate limits (HTTP 429) via the `tenacity` library:
- Detects rate limits and waits with **exponential backoff** (up to 60 seconds).
- If using a specific model, it retries up to **10 times** per API call.
- **Auto-Fallback Mode:** If you select "Auto-Fallback NIM Models", the pipeline will try the first model 10 times. If it still fails (or is offline/404), it gracefully catches the error and **moves to the next model in the list** without interrupting your batch job.

## Multimodal Limitations Bypassed
NVIDIA NIM's open-weights vision endpoints (like `Llama-3.2-90B-Vision`) enforce a strict **"1 image per prompt"** limit. To bypass this, the pipeline uses the `Pillow` library to invisibly stitch all pages of the student submission and master solution vertically into a single seamless image strip before sending it to the API.

---

## Project Structure

```
assignment_checker/
├── app.py               # Main Streamlit UI and pipeline orchestration
├── nvidia_client.py     # NVIDIA API integration & PDF-to-image conversion
├── check_llm.py         # CLI utility to test NVIDIA connection
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API key rejected | Verify your key starts with `nvapi-` and was obtained from [build.nvidia.com](https://build.nvidia.com) |
| "No student PDF files found" | Check that the target directory contains `.pdf` files |
| 429 / "temporarily rate-limited upstream" | NIM models can occasionally get overloaded. Try a different model or wait a few minutes. |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## License

This project is provided as-is for educational use.
