---
name: grade-assignments
description: Batch-grade a folder of student PDF submissions against a master solution, with Claude Code itself (via subagents) reading each PDF directly and doing the mapping/grading. Reuses this repo's tracker.json/reports/ file schema and report_export.py CSV export unmodified. Use when asked to grade/check/mark a batch of assignment PDFs, rerun a failed grading batch, or investigate flagged/stuck students in an existing reports/ folder.
---

Runbook for grading a batch of student PDFs with Claude Code itself as the grader — a separate path from this repo's other automated pipeline (`cli.py`, `stage1.py`, `stage2.py`, and its API-backed helper modules), which is untouched and still exists if ever needed, but this skill does not call it for grading. It **does** reuse `report_export.py` and the exact `tracker.json`/`reports/` file schema, so output is fully compatible with the rest of this repo (the summary CSV, the Review Flag logic, everything in README's "Batch Summary CSV" section) regardless of which path produced it.

## Why subagents, not one giant read-everything-yourself pass

A class can be dozens of students, each a multi-page PDF. Reading every student's PDF directly into the main conversation's context does not scale — it will exhaust context long before the batch finishes. Instead: read the question + master solution PDFs **once** in the main thread (small, reused by everyone), then spawn **one subagent per student** to do that student's reading + comparison + grading (the subagent reads the PDF itself via the Read tool — no separate OCR step) in an isolated context, writing only a small JSON file back to disk. The main thread orchestrates and stays light.

## File schema (must match exactly — report_export.py depends on this)

```
<target_dir>/
└── reports/
    ├── tracker.json                # {"<student_id>": {"file_path": str, "status": "PENDING"|"EVALUATED"|"FAILED"|"MISMATCH"|"DUPLICATE_CONTENT", "last_updated": isoformat}}
    ├── _question_blueprint.json    # {"total_questions": N, "questions": [{"question_id": "1", "raw_content": "...", "sub_questions": [{"id": "1a", "content": "..."}]}]}
    ├── _master_blueprint.json      # same shape as question_blueprint, but raw_content/content hold the MASTER's answers, not the question text
    ├── <id>_evaluation.json        # see below
    └── report/
        └── summary.csv             # written by report_export.export_summary_csv — don't hand-build this
```

`<id>_evaluation.json` — one per student, this is what `report_export.py` reads:

```json
{
  "answered_all_questions": true,
  "answers_are_correct": false,
  "overall_remarks_a_b_c": "B",
  "question_wise_comments": [
    {"question_id": "1a", "comment": "", "status": "Complete", "is_correct": true},
    {"question_id": "7b", "comment": "Misread Mul as 2, should be 1.5.", "status": "Complete", "is_correct": false}
  ]
}
```

- The three top-level fields come from the default criteria (Answered all questions? / Answers are correct? / Overall remarks A/B/C?) — if the user gives custom criteria instead, field names follow the same pattern `stage2._sanitize_field_name` uses: lowercase, spaces→underscores, strip trailing `?`.
- `question_wise_comments`: **one entry per sub-question** when the question has explicit sub-parts (`1a`, `1b`, ...), else one entry per top-level question (`1`, `2`, ...). `status` is exactly one of `Complete` / `Partial` / `Missing`.
- `comment` is **`""` (empty) for anything `Complete` + `is_correct: true`** — don't praise or restate correct work. Only write a comment when `status != Complete` or `is_correct == false`, and keep it to one short sentence naming the specific mistake (e.g. "Misread Mul as 2, should be 1.5.", "5c not answered."). `report_export.py`'s `full_comment` just joins whatever's in this field (`c.get('comment', '')`), so an empty string for correct entries is fully compatible — no schema change needed downstream.
- Allow up to 5% numerical deviation before marking a final numeric answer wrong — rounding/intermediate-precision differences are not errors. Only mark wrong if the deviation exceeds 5% or the method itself is incorrect.
- `question_wise_comments` **must cover every top-level question** in the blueprint — no gaps, no duplicate parent+sub-part entries for the same question. This is exactly the failure mode `stage2._assert_full_coverage` and `report_export._review_flag` exist to catch in the repo's other pipeline; since you're grading directly, self-check this before writing the file instead of relying on an external guard.

## Steps

1. **Confirm inputs**: question PDF, master solution PDF, folder of student PDFs. Ask if not given — unless the user wants this run unattended, see "Running unattended" below.
   - **Show the existing criteria and ask for extras** — same prompt shape as `cli.py`'s `_prompt_criteria`: show the default three ("Answered all questions?" / "Answers are correct?" / "Overall remarks (A/B/C)?"), ask whether to use them, then ask if the user wants any custom criteria added on top (e.g. "Used proper commenting?"). Whatever set is agreed becomes *every* student's evaluation schema — decide it once here, before any blueprint or subagent work, not per-student. Custom criteria follow the same field-naming rule as the schema section above (`stage2._sanitize_field_name`: lowercase, spaces→underscores, strip trailing `?`) and must be threaded through to every subagent prompt in step 5.
2. **Set up `reports/`**: if `reports/` doesn't exist yet under the target dir, that just means this is a fresh run for this target dir — not an error, not something to investigate. `mkdir`, initialize `tracker.json` (`{}` since new). Discover student PDFs (`.pdf` files in the target dir, excluding the master), derive each student ID from the longest run of digits in the filename — same heuristic as `cli.py`'s `discover_student_pdfs`. Register each in `tracker.json` as `PENDING` if not already tracked (never downgrade an existing `EVALUATED` entry — this makes reruns resumable for free).
   - **Before dispatching any grading, check for duplicate-upload cross-contamination** — reuse `cli.py`'s check rather than hand-rolling a new one:
     ```python
     from cli import discover_student_pdfs, find_duplicate_uploads
     student_pdfs = discover_student_pdfs(target_dir, master_filename=None)
     find_duplicate_uploads(student_pdfs)  # -> [(id_a, id_b), ...] for every pair of byte-identical PDFs, even if they share the same nominal ID
     ```
     A byte-identical match is real but **ambiguous on its own** — it could be cross-student contamination (two filenames, regardless of whether they share a digit run, both actually contain a third student's work) or it could be a harmless same-student double-upload (one person submitting their own PDF twice under two slightly different filenames). **Same-ID and cross-ID collisions look structurally identical** from the outside — there is no reliable way to tell them apart from filenames or hashes alone. So: for every `(id_a, id_b)` pair, set **both** to `DUPLICATE_CONTENT` (not `MISMATCH` yet) — this excludes them from grading but doesn't accuse anyone until identity is actually checked.
3. **Build the question blueprint yourself** (main thread, one Read of the question PDF): produce the `_question_blueprint.json` structure above and write it.
4. **Build the master blueprint yourself** (main thread, one Read of the master solution PDF): map the master's answers into the same hierarchy, write `_master_blueprint.json`.
5. **For each `PENDING`/`FAILED`/`DUPLICATE_CONTENT` student**, spawn a subagent (`general-purpose`) — batch several in parallel in one message (e.g. 5-8 at a time, mirroring `cli.py`'s default `--workers 6`) rather than one at a time. Give each subagent a **self-contained** prompt: the student's PDF path, the question blueprint JSON, the master blueprint JSON (inline as text — don't make the subagent re-read those PDFs), the exact target schema from above, the grading conventions (5% tolerance, comment only on incomplete/incorrect sub-questions and keep it to one brief sentence, full coverage required), and the exact output path to write `<id>_evaluation.json` to. Tell it to report back pass/fail, not just "done" — if it genuinely cannot produce complete, schema-valid output (e.g. illegible pages), it should say so rather than fabricate a plausible-looking but incomplete result.
   - **Tell the subagent to read the PDF directly with the Read tool** (`Read(file_path=<pdf>, pages="1-20")`) — Claude Code's Read tool natively handles PDFs, including scanned/handwritten pages, in one call per ≤20 pages. Explicitly instruct it NOT to render pages to PNG one-by-one via PyMuPDF/pdftoppm/Bash first — that's an old OCR workaround needed by API-based pipelines that only accept one image at a time, it's unnecessary here, and it burns 10+ tool calls and minutes per student for no benefit. A whole assignment PDF (typically well under 20 pages) should usually be one Read call.
   - **Tell the subagent to check submission identity**: if the PDF has a name/ID/cover page, does it match the student ID this file was registered under? A vision-capable grader reading the whole document naturally notices this even without being asked, but say it explicitly so it doesn't get skipped. If it's a genuine mismatch (not just messy handwriting on the ID field), it must NOT write an evaluation file or grade the content as if it belonged to the tracked student — report the mismatch back instead (see below for what the main thread does with that).
   - **For a `DUPLICATE_CONTENT` student specifically**, tell the subagent the *other* ID it was matched with (from step 2's pairs) and ask it to explicitly resolve the ambiguity: does the identity written in the PDF match this student's own name/ID, or does it match the other one? That's the actual verification step 2 deferred — a hash match alone can't do it, but a subagent that reads the content can.
6. **Main thread updates `tracker.json`** as each subagent's result comes back:
   - `EVALUATED` on success (this also resolves a `DUPLICATE_CONTENT` case in the "actually fine, same student's own resubmission" direction — grading it successfully IS the resolution).
   - `FAILED` on a genuine grading failure (retryable — a future rerun will try again).
   - `MISMATCH` when a subagent confirms the identity written in the PDF doesn't match the tracked student — whether it started as `DUPLICATE_CONTENT` (step 2 found the pair, subagent confirmed it's a real mismatch) or the subagent discovered it fresh mid-grading (bytes weren't an exact duplicate of another tracked file, but the content still doesn't match). `MISMATCH` is NOT retryable by rerunning — the source PDF itself is wrong, so `get_processable`-style logic should skip it until a human fixes the file; if you build your own picking-up-PENDING logic for this skill, exclude `MISMATCH` (and unresolved `DUPLICATE_CONTENT`) from what gets automatically retried, unlike `FAILED`.
   - Always with a fresh `last_updated`. Don't let subagents write `tracker.json` directly (avoids concurrent-write races — same reason `cli.py` keeps all file/tracker writes on the main thread and workers only compute).
7. **Regenerate the summary CSV**:
   ```python
   from report_export import export_summary_csv
   export_summary_csv("<target_dir>/reports")
   ```
8. **Triage before reporting done.** Read the CSV's `Review Flag` column. Report flagged/failed students explicitly with counts, not just "batch complete." A `MISMATCH` row reads `Submitted Other's Assignment` — call these out separately from ordinary grading failures, since the fix is "get the right PDF from this student," not "rerun the grader." Any row still reading the `DUPLICATE_CONTENT` flag text means a subagent never actually resolved it (e.g. it hit the "genuinely cannot read the PDF" case) — that also needs a follow-up pass, not a shrug.

## Running unattended (no one at the keyboard)

Only relevant when the user explicitly asks for a hands-off run (e.g. "grade this while I'm away", "run the full batch overnight"). Two separate things have to hold, and only one of them lives in this file:

1. **Tool permission prompts are a session setting, not something this file controls.** Bash/Write/Agent approvals come from Claude Code's permission mode for the session — this skill can't waive them. Tell the user up front to set an auto-accept/bypass-permissions mode (or pre-approve this project directory) before walking away; otherwise the run just stalls at the first prompt no matter what the skill says.
2. **Everything this file *can* control: decide and log instead of asking.**
   - Skip step 1's criteria prompt — use the default three criteria unless custom ones were already specified when the run was kicked off. Note the choice in the final summary rather than blocking on it.
   - Missing/ambiguous inputs: don't ask — best-effort discover the standard layout (a question+solution PDF pair, a folder of student PDFs) and proceed; if genuinely not discoverable, skip that piece and say so in the final report instead of stalling.
   - `MISMATCH` / unresolved `DUPLICATE_CONTENT` / `FAILED` students never block the batch — they already have a defined tracker status and CSV flag (steps 6-8); keep grading everyone else and surface them only in the final triage report.
   - Treat setup (steps 2-4) as idempotent: if `reports/` or its tracker/blueprint files are missing when a run starts (a fresh run, or files genuinely gone mid-run), just (re)build them rather than stopping to ask what happened.
   - Keep the concurrency cap from step 5 regardless of run length; a killed/errored subagent is just a `FAILED` entry that gets retried on the next pass, not something to wait on the user for.

Net effect: once permission mode is sorted, the only difference from a normal run is that ambiguity gets resolved with a logged default instead of a question.

## Retrying flagged/stuck students

Same playbook as the repo's other pipeline:
1. Rerun just the `PENDING`/`FAILED`/`DUPLICATE_CONTENT` ones (steps 5-7) — most issues resolve on a second look, especially since it's a different subagent instance with a clean context. Don't rerun `MISMATCH` students automatically; they need a corrected source file first.
2. If one student fails 3+ independent rounds with the same kind of problem (illegible handwriting, corrupted PDF, genuinely ambiguous answer), stop — that's a real "needs a human" case, not something to keep retrying. Report it plainly.

## When NOT to use this path

Large classes (dozens+ of students) mean dozens of subagent invocations — real cost and time, same tradeoff as the repo's other pipeline's API calls but billed differently (usage on this plan vs. that pipeline's provider). If the user already has that pipeline's API key set up and cares about cost at scale, `cli.py`'s pipeline (see README) may be the better default; this skill is for when they specifically want Claude doing the grading, or don't have/want that key.
