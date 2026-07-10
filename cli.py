#!/usr/bin/env python3
"""
Command-line batch runner for the assignment grading pipeline.

Batch-grades student PDFs against a master solution: tracker.json
checkpointing (PENDING/MAPPED/EVALUATED/FAILED, resumable), filename
sanitization/dedup, a single pinned NVIDIA model per task (overridable via
--model-id), ThreadPoolExecutor concurrency, and a summary CSV — all
self-contained here (no UI framework).

Usage:
    python cli.py --question-pdf Q.pdf --master-pdf S.pdf --target-dir submissions/
    python cli.py --question-pdf Q.pdf --master-pdf S.pdf --target-dir submissions/ --workers 10
    python cli.py --question-pdf Q.pdf --master-pdf S.pdf --target-dir submissions/ --criteria "Used proper commenting?"

API key is read from api_key.txt next to this script by default; --api-key overrides it.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from model_pools import get_pool
from nvidia_client import call_nvidia_structured
from report_export import export_summary_csv
from stage1 import build_question_blueprint, run_student_mapping
from stage2 import run_stage2

# Configure file-based error logging
logging.basicConfig(
    filename="error.log",
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")

DB_NAME = "tracker.json"
REPORTS_DIR = "reports"

DEFAULT_CRITERIA: list[str] = [
    "Answered all questions?",
    "Answers are correct?",
    "Overall remarks (A/B/C)?",
]

# Batch pipeline concurrency: the NVIDIA rate limiter (nvidia_client's
# threading.Lock-guarded 40 RPM gate) is the real ceiling — this just stops
# a single student's request-then-wait latency from idling every other
# student in the batch. See _process_student.
BATCH_MAX_WORKERS = 6


# ═══════════════════════════════════════════════════════════════════════════════
# State Tracking Layer (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(target_dir: str) -> str:
    """Initialize the JSON tracker file (inside reports/, alongside the rest
    of the pipeline's output — not in the submissions folder) and return its
    path."""
    reports_dir = os.path.join(target_dir, REPORTS_DIR)
    os.makedirs(reports_dir, exist_ok=True)
    db_path = os.path.join(reports_dir, DB_NAME)
    if not os.path.exists(db_path):
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return db_path


def _load_db(db_path: str) -> dict:
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_db(db_path: str, data: dict) -> None:
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def upsert_submission(db_path: str, student_id: str,
                       file_path: str, status: str = "PENDING") -> None:
    """Insert a new submission or leave it alone if already tracked."""
    now = datetime.now().isoformat()
    data = _load_db(db_path)

    if student_id not in data:
        data[student_id] = {
            "file_path": file_path,
            "status": status,
            "last_updated": now
        }
    else:
        # Update path and timestamp, preserve EVALUATED status
        data[student_id]["file_path"] = file_path
        data[student_id]["last_updated"] = now
        if data[student_id]["status"] != "EVALUATED":
            data[student_id]["status"] = status

    _save_db(db_path, data)


def update_status(db_path: str, student_id: str, status: str) -> None:
    """Update the processing status of a submission."""
    data = _load_db(db_path)
    if student_id in data:
        data[student_id]["status"] = status
        data[student_id]["last_updated"] = datetime.now().isoformat()
        _save_db(db_path, data)


def get_processable(db_path: str) -> list[tuple[str, str, str]]:
    """Return submissions that still need work (PENDING / FAILED / MAPPED)."""
    data = _load_db(db_path)
    processable = []
    for sid, info in data.items():
        if info["status"] in ('PENDING', 'FAILED', 'MAPPED'):
            processable.append((sid, info["file_path"], info["status"]))
    return sorted(processable, key=lambda x: x[0])


def get_status_counts(db_path: str) -> dict[str, int]:
    """Count submissions grouped by status."""
    data = _load_db(db_path)
    counts = {}
    for info in data.values():
        status = info["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def get_total_count(db_path: str) -> int:
    """Return the total number of tracked submissions."""
    data = _load_db(db_path)
    return len(data)


# ═══════════════════════════════════════════════════════════════════════════════
# File Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_submission_filenames(
    target_dir: str,
    master_filename: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str | None]:
    """
    Rename every ``.pdf`` in *target_dir* whose name contains whitespace,
    collapsing whitespace runs to a single underscore. Resolves any
    collision the rename creates by appending a numeric suffix.

    Returns ``(renamed, skipped, master_filename)`` where *renamed* is a
    list of ``(old_name, new_name)``, *skipped* is ``(name, error)`` for
    files that couldn't be renamed (e.g. open in another program), and
    *master_filename* is the (possibly updated) master filename to keep
    the caller's exclusion check working if the master itself got renamed.
    """
    existing = {f.lower() for f in os.listdir(target_dir) if f.lower().endswith(".pdf")}
    renamed: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    updated_master = master_filename

    for fname in sorted(os.listdir(target_dir)):
        if not fname.lower().endswith(".pdf") or not re.search(r"\s", fname):
            continue

        stem, ext = os.path.splitext(fname)
        new_stem = re.sub(r"\s+", "_", stem.strip())
        candidate = f"{new_stem}{ext}"
        n = 2
        while candidate.lower() in existing:
            candidate = f"{new_stem}_{n}{ext}"
            n += 1

        try:
            os.rename(os.path.join(target_dir, fname), os.path.join(target_dir, candidate))
        except OSError as exc:
            skipped.append((fname, str(exc)))
            continue

        existing.discard(fname.lower())
        existing.add(candidate.lower())
        renamed.append((fname, candidate))
        if master_filename and fname == master_filename:
            updated_master = candidate

    return renamed, skipped, updated_master


def discover_student_pdfs(
    target_dir: str,
    master_filename: str | None = None,
) -> list[tuple[str, str]]:
    """
    Scan *target_dir* for ``.pdf`` files, extract ``student_id`` from the
    filename stem, and return a sorted list of ``(student_id, abs_path)``.
    The master solution (if present) is excluded.
    """
    results: list[tuple[str, str]] = []
    seen_counts: dict[str, int] = {}
    # Sorted, not raw os.listdir() order: which file is "primary" vs
    # "[duplicate]" for a double-uploaded student depends on iteration
    # order, and os.listdir() order isn't guaranteed stable across runs —
    # a shift would desync tracker.json (and its cached _map.json) from
    # which physical file a given ID actually refers to.
    for fname in sorted(os.listdir(target_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        if master_filename and fname == master_filename:
            continue

        stem = os.path.splitext(fname)[0]

        # Heuristic to extract student ID:
        # Split by typical separators (underscore, dash, space)
        parts = re.split(r'[_\-\s]+', stem)
        # Find parts that are strictly digits
        digit_parts = [p for p in parts if p.isdigit()]

        if digit_parts:
            # Pick the longest digit-only part as the student ID
            student_id = max(digit_parts, key=len)
        else:
            # Fallback to the whole stem if no purely numeric part is found
            student_id = stem

        # Two filenames can resolve to the same ID (e.g. "1234_v1.pdf" and
        # "1234_final.pdf"). Suffix later occurrences instead of silently
        # overwriting each other in the tracker.
        seen_counts[student_id] = seen_counts.get(student_id, 0) + 1
        occurrence = seen_counts[student_id]
        if occurrence > 1:
            suffix = "[duplicate]" if occurrence == 2 else f"[duplicate{occurrence - 1}]"
            student_id = f"{student_id}{suffix}"

        results.append((student_id, os.path.join(target_dir, fname)))
    return sorted(results, key=lambda x: x[0])


def find_duplicate_uploads(student_pdfs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return (id_a, id_b) for every pair of student IDs whose PDF files are
    byte-identical.

    Filename-based student IDs (see discover_student_pdfs) trust the
    filename to say who a submission belongs to — they have no way to catch
    the case where the same file was accidentally uploaded twice under two
    different students' names (observed live: "24141084_RONY_MIAH.pdf" and
    "24141084_Saiful_Rony.pdf" were an exact byte-for-byte copy of a third
    student's submission — note these two even shared the SAME filename ID,
    just different apparent names, which is exactly the case this must not
    miss). A byte-identical match is real students' handwritten answers
    essentially never hashing the same by coincidence, so any match — same
    nominal ID or not — is worth surfacing rather than silently trusted:
    flag both sides and let a human/grader confirm identity, rather than
    assume a same-ID match is always a harmless self-resubmission.
    """
    hashes: dict[str, list[str]] = {}
    for student_id, path in student_pdfs:
        with open(path, "rb") as fh:
            digest = hashlib.md5(fh.read()).hexdigest()
        hashes.setdefault(digest, []).append(student_id)

    pairs: list[tuple[str, str]] = []
    for ids in hashes.values():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((ids[i], ids[j]))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _preflight_check(api_key: str, model_id: str | list[str] | None) -> str | None:
    """One cheap call to confirm the API key/model actually work.

    Without this, a bad key silently burns through full per-model
    retry+fallback for every single submission in the batch before the user
    notices anything is wrong. Returns an error string, or None if OK.
    """
    try:
        call_nvidia_structured(
            api_key=api_key,
            model=model_id or get_pool("evaluation"),
            system_prompt="Reply with exactly one word: OK",
            user_prompt="OK?",
            schema=None,
        )
        return None
    except Exception as exc:
        return str(exc)


def _process_student(
    api_key: str,
    model_id: str | list[str],
    question_blueprint: dict,
    master_answers: dict,
    active_criteria: list[str],
    reports_dir: str,
    student_id: str,
    file_path: str,
    current_status: str,
) -> tuple[str, dict | None, dict | None, Exception | None]:
    """Runs Stage 1 + Stage 2 for one student. Called from a worker thread —
    must not print or write files itself; the caller persists results and
    drives progress output on the main thread. Returns
    (student_id, stage1_result, stage2_result, error).
    """
    noop_log = lambda _msg: None
    try:
        map_path = os.path.join(reports_dir, f"{student_id}_map.json")
        if current_status in ("PENDING", "FAILED"):
            stage1_result = run_student_mapping(api_key, file_path, question_blueprint, noop_log, model_id)
        elif os.path.isfile(map_path):
            with open(map_path, "r", encoding="utf-8") as fh:
                stage1_result = json.load(fh)
        else:
            stage1_result = run_student_mapping(api_key, file_path, question_blueprint, noop_log, model_id)

        student_answers = stage1_result.get("structured_student_solution", stage1_result)

        stage2_result = run_stage2(
            api_key, model_id, master_answers, student_answers,
            active_criteria, noop_log,
            question_context=question_blueprint,
        )
        return student_id, stage1_result, stage2_result, None
    except Exception as exc:
        return student_id, None, None, exc


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def _load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isfile(API_KEY_FILE):
        key = open(API_KEY_FILE, encoding="utf-8").read().strip()
        if key:
            return key
    sys.exit(f"No API key found. Pass --api-key or put one in {API_KEY_FILE}")


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ── Interactive wizard (step-by-step prompts instead of flags) ──

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default


def _prompt_path(msg: str) -> str:
    while True:
        val = _prompt(msg)
        if val and os.path.isfile(val):
            return val
        print(f"  (!) file not found: {val!r} — try again.")


def _prompt_dir(msg: str) -> str:
    while True:
        val = _prompt(msg)
        if val and os.path.isdir(val):
            return val
        print(f"  (!) directory not found: {val!r} — try again.")


def _prompt_yes_no(msg: str, default: bool = True) -> bool:
    val = input(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not val else val.startswith("y")


def _prompt_criteria() -> list[str]:
    print("  Default criteria:")
    for c in DEFAULT_CRITERIA:
        print(f"    - {c}")
    criteria = list(DEFAULT_CRITERIA) if _prompt_yes_no("Use these?", default=True) else []
    print("  Add custom criteria one at a time, blank line to stop.")
    while True:
        c = input(f"    Custom criterion #{len(criteria) + 1} (blank to finish): ").strip()
        if not c:
            break
        criteria.append(c)
    return criteria


def _run_wizard() -> argparse.Namespace:
    print("=== Assignment Grader — Step-by-Step Setup ===")
    question_pdf = _prompt_path("Question PDF path")
    master_pdf = _prompt_path("Master solution PDF path")
    target_dir = _prompt_dir("Student submissions folder")
    api_key = _prompt("NVIDIA API key (blank = use api_key.txt)") or None
    model_id = _prompt("Model override (blank = default qwen/qwen3.5-397b-a17b)") or None
    criteria = _prompt_criteria()
    workers = int(_prompt("Concurrent workers", str(BATCH_MAX_WORKERS)))
    limit_raw = _prompt("Limit to first N pending submissions (blank = whole batch)")
    limit = int(limit_raw) if limit_raw else None

    print("\n--- Ready ---")
    print(f"  Question PDF : {question_pdf}")
    print(f"  Master PDF   : {master_pdf}")
    print(f"  Target dir   : {target_dir}")
    print(f"  Criteria     : {criteria}")
    print(f"  Workers      : {workers}")
    print(f"  Limit        : {limit or 'none (full batch)'}")
    if not _prompt_yes_no("Proceed?", default=True):
        sys.exit("Aborted.")

    return argparse.Namespace(
        question_pdf=question_pdf, master_pdf=master_pdf, target_dir=target_dir,
        api_key=api_key, model_id=model_id, criteria=criteria or None,
        workers=workers, limit=limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-grade student PDFs against a master solution.")
    parser.add_argument("--question-pdf", default=None, help="Assignment question paper PDF")
    parser.add_argument("--master-pdf", default=None, help="Master solution / answer key PDF")
    parser.add_argument("--target-dir", default=None, help="Folder containing student submission PDFs")
    parser.add_argument("--api-key", default=None, help="Overrides api_key.txt")
    parser.add_argument("--model-id", default=None, help="Single model override; default = qwen/qwen3.5-397b-a17b for every task")
    parser.add_argument(
        "--criteria", action="append", default=None,
        help="Evaluation criterion; repeatable. Default = the 3 standard criteria.",
    )
    parser.add_argument("--workers", type=int, default=BATCH_MAX_WORKERS, help="Concurrent submissions in flight")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending submissions (testing)")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Step-by-step prompts instead of flags (also triggered automatically with no arguments)",
    )
    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        args = _run_wizard()
    elif not (args.question_pdf and args.master_pdf and args.target_dir):
        parser.error(
            "--question-pdf, --master-pdf and --target-dir are required "
            "(or run with no arguments, or --interactive, for step-by-step setup)"
        )

    api_key = _load_api_key(args.api_key)
    criteria = args.criteria or DEFAULT_CRITERIA
    target_dir = args.target_dir
    reports_dir = os.path.join(target_dir, REPORTS_DIR)
    os.makedirs(reports_dir, exist_ok=True)

    print("Pre-flight check…", flush=True)
    preflight_error = _preflight_check(api_key, args.model_id)
    if preflight_error:
        sys.exit(f"Pre-flight check failed — verify your API key/model: {preflight_error}")

    print("Scanning target directory…", flush=True)
    renamed, skipped, master_filename = sanitize_submission_filenames(
        target_dir, os.path.basename(args.master_pdf)
    )
    for old, new in renamed:
        print(f"  renamed: {old} -> {new}", flush=True)
    for name, err in skipped:
        print(f"  (!) could not rename {name}: {err}", flush=True)

    student_pdfs = discover_student_pdfs(target_dir, master_filename)
    if not student_pdfs:
        sys.exit("No student PDFs found in target directory.")
    dupes = [sid for sid, _ in student_pdfs if "[duplicate" in sid]
    if dupes:
        print(f"  (!) filename collisions suffixed: {', '.join(dupes)}", flush=True)
    print(f"Found {len(student_pdfs)} student PDF(s).", flush=True)

    db_path = init_db(target_dir)
    for sid, fpath in student_pdfs:
        upsert_submission(db_path, sid, fpath)

    print("Checking for duplicate-upload cross-contamination…", flush=True)
    content_dupes = find_duplicate_uploads(student_pdfs)
    duplicate_content_ids: set[str] = set()
    for id_a, id_b in content_dupes:
        duplicate_content_ids.add(id_a)
        duplicate_content_ids.add(id_b)
    # DUPLICATE_CONTENT, not MISMATCH: a byte-identical match is real but
    # ambiguous on its own — it could be cross-student contamination (two
    # slots ended up with a third student's file) or a harmless same-student
    # double-upload (same person submitted their own PDF twice). A hash
    # match can't tell those apart (they look structurally identical —
    # same-ID and cross-ID collisions both just mean "two files, same
    # bytes"), so this only marks it for identity verification rather than
    # asserting either outcome. Something with an actual view of the PDF
    # content (a human, or the grade-assignments skill's subagents) decides
    # whether to promote it to MISMATCH or clear it back to PENDING.
    for sid in sorted(duplicate_content_ids):
        update_status(db_path, sid, "DUPLICATE_CONTENT")
    if content_dupes:
        for id_a, id_b in content_dupes:
            print(
                f"  (!) {id_a} and {id_b} are byte-identical PDFs. Both marked "
                f"DUPLICATE_CONTENT (excluded from grading) pending identity "
                f"verification — could be cross-student contamination or a "
                f"harmless same-student re-upload.",
                flush=True,
            )

    def log(msg: str) -> None:
        print(f"  {msg}", flush=True)

    # ── Question blueprint (cached) ──
    question_map_path = os.path.join(reports_dir, "_question_blueprint.json")
    if os.path.isfile(question_map_path):
        print("Loading existing question blueprint…", flush=True)
        with open(question_map_path, "r", encoding="utf-8") as fh:
            question_blueprint = json.load(fh)
    else:
        print("Building question blueprint…", flush=True)
        try:
            question_blueprint = build_question_blueprint(api_key, args.question_pdf, log, args.model_id)
        except Exception as exc:
            sys.exit(f"Failed to build question blueprint after retries: {exc}\nRerun the same command to try again.")
        with open(question_map_path, "w", encoding="utf-8") as fh:
            json.dump(question_blueprint, fh, indent=2, ensure_ascii=False)

    # ── Master solution blueprint (cached) ──
    master_map_path = os.path.join(reports_dir, "_master_blueprint.json")
    if os.path.isfile(master_map_path):
        print("Loading existing master solution blueprint…", flush=True)
        with open(master_map_path, "r", encoding="utf-8") as fh:
            master_result = json.load(fh)
    else:
        print("Extracting master solution answers…", flush=True)
        try:
            master_result = run_student_mapping(
                api_key, args.master_pdf, question_blueprint, log, args.model_id, is_master=True
            )
        except Exception as exc:
            sys.exit(f"Failed to map master solution after retries: {exc}\nRerun the same command to try again.")
        with open(master_map_path, "w", encoding="utf-8") as fh:
            json.dump(master_result, fh, indent=2, ensure_ascii=False)
    master_answers = master_result.get("structured_student_solution", master_result)

    # ── Batch loop ──
    to_process = get_processable(db_path)
    total_files = get_total_count(db_path)
    if args.limit is not None:
        to_process = to_process[: args.limit]

    if not to_process:
        print("All submissions already evaluated.", flush=True)
    else:
        print(f"Processing {len(to_process)} submission(s) with {args.workers} worker(s)…", flush=True)
        evaluated_before = get_status_counts(db_path).get("EVALUATED", 0)
        completed = 0
        batch_start = time.time()

        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {
                executor.submit(
                    _process_student, api_key, args.model_id, question_blueprint,
                    master_answers, criteria, reports_dir, student_id, file_path, current_status,
                ): student_id
                for student_id, file_path, current_status in to_process
            }

            for future in as_completed(futures):
                student_id, stage1_result, stage2_result, error = future.result()

                if error is not None:
                    update_status(db_path, student_id, "FAILED")
                    print(f"  [FAIL] {student_id}: {error}", flush=True)
                else:
                    with open(os.path.join(reports_dir, f"{student_id}_map.json"), "w", encoding="utf-8") as fh:
                        json.dump(stage1_result, fh, indent=2, ensure_ascii=False)
                    update_status(db_path, student_id, "MAPPED")

                    with open(os.path.join(reports_dir, f"{student_id}_evaluation.json"), "w", encoding="utf-8") as fh:
                        json.dump(stage2_result, fh, indent=2, ensure_ascii=False)
                    update_status(db_path, student_id, "EVALUATED")
                    print(f"  [OK]   {student_id}", flush=True)

                completed += 1
                total_evaluated = evaluated_before + completed
                elapsed = time.time() - batch_start
                avg_per_item = elapsed / completed
                remaining = len(to_process) - completed
                eta_str = _fmt_eta(avg_per_item * remaining) if remaining else "0s"
                print(
                    f"  [{completed}/{len(to_process)}]  total evaluated: "
                    f"{total_evaluated}/{total_files}  ETA: {eta_str}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\nInterrupted — already-graded submissions are kept in tracker.json. Rerun to resume.", flush=True)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    final_counts = get_status_counts(db_path)
    summary_path = export_summary_csv(reports_dir)
    print(f"\nFinished — {final_counts}", flush=True)
    print(f"Summary CSV -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
