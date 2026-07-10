---
name: grade-assignments
description: Batch-grade a folder of student PDF submissions against a master solution, with Claude Code itself (via subagents) doing the OCR/mapping/grading — no NVIDIA API, no model_pools.py. Reuses this repo's tracker.json/reports/ file schema and report_export.py CSV export unmodified. Use when asked to grade/check/mark a batch of assignment PDFs, rerun a failed grading batch, or investigate flagged/stuck students in an existing reports/ folder.
---

Runbook for grading a batch of student PDFs with Claude Code as the grader — not `cli.py`'s NVIDIA-backed pipeline. `cli.py`, `stage1.py`, `stage2.py`, `nvidia_client.py`, `model_pools.py` are untouched and still exist as the automated NVIDIA-API path if ever needed, but this skill does not call them for grading. It **does** reuse `report_export.py` and the exact `tracker.json`/`reports/` file schema, so output is fully compatible with the rest of this repo (the summary CSV, the Review Flag logic, everything in README's "Batch Summary CSV" section) regardless of which path produced it.

## Why subagents, not one giant read-everything-yourself pass

A class can be dozens of students, each a multi-page PDF of images. Reading every student's PDF directly into the main conversation's context does not scale — it will exhaust context long before the batch finishes. Instead: read the question + master solution PDFs **once** in the main thread (small, reused by everyone), then spawn **one subagent per student** to do that student's OCR + comparison + grading in an isolated context, writing only a small JSON file back to disk. The main thread orchestrates and stays light.

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
    {"question_id": "1a", "comment": "You correctly...", "status": "Complete", "is_correct": true}
  ]
}
```

- The three top-level fields come from the default criteria (Answered all questions? / Answers are correct? / Overall remarks A/B/C?) — if the user gives custom criteria instead, field names follow the same pattern `stage2._sanitize_field_name` uses: lowercase, spaces→underscores, strip trailing `?`.
- `question_wise_comments`: **one entry per sub-question** when the question has explicit sub-parts (`1a`, `1b`, ...), else one entry per top-level question (`1`, `2`, ...). `status` is exactly one of `Complete` / `Partial` / `Missing`. `comment` addresses the student directly in second person ("You correctly...", "You missed...").
- Allow up to 5% numerical deviation before marking a final numeric answer wrong — rounding/intermediate-precision differences are not errors. Only mark wrong if the deviation exceeds 5% or the method itself is incorrect.
- `question_wise_comments` **must cover every top-level question** in the blueprint — no gaps, no duplicate parent+sub-part entries for the same question. This is exactly the failure mode `stage2._assert_full_coverage` and `report_export._review_flag` exist to catch in the NVIDIA path; since you're grading directly, self-check this before writing the file instead of relying on an external guard.

## Steps

1. **Confirm inputs**: question PDF, master solution PDF, folder of student PDFs. Ask if not given.
2. **Set up `reports/`**: `mkdir`, initialize `tracker.json` (`{}`  if new). Discover student PDFs (`.pdf` files in the target dir, excluding the master), derive each student ID from the longest run of digits in the filename — same heuristic as `cli.py`'s `discover_student_pdfs`. Register each in `tracker.json` as `PENDING` if not already tracked (never downgrade an existing `EVALUATED` entry — this makes reruns resumable for free).
   - **Before dispatching any grading, check for duplicate-upload cross-contamination** — reuse `cli.py`'s check rather than hand-rolling a new one:
     ```python
     from cli import discover_student_pdfs, find_duplicate_uploads
     student_pdfs = discover_student_pdfs(target_dir, master_filename=None)
     find_duplicate_uploads(student_pdfs)  # -> [(id_a, id_b), ...] for every pair of byte-identical PDFs, even if they share the same nominal ID
     ```
     A byte-identical match is real but **ambiguous on its own** — it could be cross-student contamination (two slots both ended up with a third student's file — this is exactly what happened live: `24141084_RONY_MIAH.pdf` and `24141084_Saiful_Rony.pdf` were byte-identical to each other, different apparent names, both actually student `23201214`'s work) or it could be a harmless same-student double-upload (one person submitting their own PDF twice — also seen live, same batch: `23201214`'s own base/`[duplicate]` files were byte-identical too, just two spellings of the same name). **Same-ID and cross-ID collisions look structurally identical** (both are just "two filenames, same digit run or not, same bytes") — there is no reliable way to tell them apart from filenames or hashes alone. So: for every `(id_a, id_b)` pair, set **both** to `DUPLICATE_CONTENT` (not `MISMATCH` yet) — this excludes them from grading but doesn't accuse anyone until identity is actually checked.
3. **Build the question blueprint yourself** (main thread, one Read of the question PDF): produce the `_question_blueprint.json` structure above and write it.
4. **Build the master blueprint yourself** (main thread, one Read of the master solution PDF): map the master's answers into the same hierarchy, write `_master_blueprint.json`.
5. **For each `PENDING`/`FAILED`/`DUPLICATE_CONTENT` student**, spawn a subagent (`general-purpose`) — batch several in parallel in one message (e.g. 5-8 at a time, mirroring `cli.py`'s default `--workers 6`) rather than one at a time. Give each subagent a **self-contained** prompt: the student's PDF path, the question blueprint JSON, the master blueprint JSON (inline as text — don't make the subagent re-read those PDFs), the exact target schema from above, the grading conventions (5% tolerance, second-person feedback, full coverage required), and the exact output path to write `<id>_evaluation.json` to. Tell it to report back pass/fail, not just "done" — if it genuinely cannot produce complete, schema-valid output (e.g. illegible pages), it should say so rather than fabricate a plausible-looking but incomplete result.
   - **Tell the subagent to read the PDF directly with the Read tool** (`Read(file_path=<pdf>, pages="1-20")`) — Claude Code's Read tool natively handles PDFs, including scanned/handwritten pages, in one call per ≤20 pages. Explicitly instruct it NOT to render pages to PNG one-by-one via PyMuPDF/pdftoppm/Bash first — that's the old NVIDIA-pipeline OCR workaround (needed there because the raw NVIDIA API only accepts one image at a time), it's unnecessary here, and it burns 10+ tool calls and minutes per student for no benefit. A whole assignment PDF (typically well under 20 pages) should usually be one Read call.
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

## Retrying flagged/stuck students

Same playbook as the NVIDIA path:
1. Rerun just the `PENDING`/`FAILED`/`DUPLICATE_CONTENT` ones (steps 5-7) — most issues resolve on a second look, especially since it's a different subagent instance with a clean context. Don't rerun `MISMATCH` students automatically; they need a corrected source file first.
2. If one student fails 3+ independent rounds with the same kind of problem (illegible handwriting, corrupted PDF, genuinely ambiguous answer), stop — that's a real "needs a human" case, not something to keep retrying. Report it plainly.

## When NOT to use this path

Large classes (dozens+ of students) mean dozens of subagent invocations — real cost and time, same tradeoff as the NVIDIA pipeline's API calls but billed differently (usage on this plan vs. NVIDIA's free tier). If the user has an NVIDIA API key and cares about cost at scale, `cli.py`'s pipeline (see README) may be the better default; this skill is for when they specifically want Claude doing the grading, or don't have/want an NVIDIA key.
