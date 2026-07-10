"""
Builds a per-student summary CSV (opens directly in Excel) from the
Stage 2 evaluation reports produced by the pipeline.
"""

import csv
import json
import os
import re
from collections import Counter
from typing import Any

_DUPLICATE_SUFFIX_RE = re.compile(r"\[duplicate\d*\]$")


def full_comment(data: dict) -> str:
    """Concatenate every question's full feedback text, verbatim (not summarised)."""
    comments = data.get("question_wise_comments", [])
    return " | ".join(f"{c.get('question_id', '?')}: {c.get('comment', '')}" for c in comments)


def _sort_key(student_id: str) -> tuple[int, Any]:
    return (0, int(student_id)) if student_id.isdigit() else (1, student_id)


def _base_student_id(student_id: str) -> str:
    """Strip the `[duplicate]`/`[duplicateN]` suffix discover_student_pdfs
    adds when two filenames extract to the same student ID."""
    return _DUPLICATE_SUFFIX_RE.sub("", student_id)


def _status_counts(data: dict) -> tuple[int, int, int]:
    """Tally each sub-question's `status` -> (Complete, Missing, Partial)."""
    comments = data.get("question_wise_comments", [])
    counts = Counter(c.get("status", "Unknown") for c in comments)
    return counts.get("Complete", 0), counts.get("Missing", 0), counts.get("Partial", 0)


def _correct_counts(data: dict) -> tuple[int, int]:
    """Tally each sub-question's `is_correct` -> (Correct, Incorrect)."""
    comments = data.get("question_wise_comments", [])
    correct = sum(1 for c in comments if c.get("is_correct"))
    return correct, len(comments) - correct


def _correctness_score(data: dict) -> float:
    """Fraction of question_wise_comments marked correct — always numeric,
    unlike the free-text quality/remarks field, so it's safe to compare
    across duplicate submissions of the same student."""
    comments = data.get("question_wise_comments", [])
    if not comments:
        return 0.0
    return sum(1 for c in comments if c.get("is_correct")) / len(comments)


def _coverage_breadth(data: dict) -> int:
    """Count distinct top-level questions covered in question_wise_comments.

    A malformed grading response (duplicate parent+sub-part entries that
    stop partway through, e.g. covers only questions 1-4 of 7) can still
    score artificially high on `_correctness_score` — most of its entries
    marked correct just means it never reached the harder later questions.
    Breadth catches that: it's how many distinct top-level questions were
    actually addressed, so a complete-but-lower-scoring submission still
    outranks an incomplete-but-higher-scoring one when picking between
    duplicate uploads of the same student."""
    covered = set()
    for c in data.get("question_wise_comments", []):
        m = re.match(r"(\d+)", str(c.get("question_id", "")))
        if m:
            covered.add(int(m.group(1)))
    return len(covered)


def _expected_coverage(breadths: list[int]) -> int:
    """The most common coverage breadth across this batch's winning submissions.

    Self-calibrates to whatever the real top-level question count is for
    this assignment (7, 12, whatever) purely from the data at hand, no need
    to thread the question blueprint through report_export. Assumes most
    students' grading succeeded cleanly — true as long as malformed results
    are the minority, which is the case this flag exists to catch."""
    if not breadths:
        return 0
    return Counter(breadths).most_common(1)[0][0]


def _review_flag(data: dict, expected_total: int) -> str:
    """Return a manual-review reason, or '' if the submission looks structurally sound.

    Catches the two shapes of malformed Stage 2 output seen in practice:
    a response that stops partway through (fewer top-level questions
    covered than the rest of the batch), and one with duplicate parent +
    sub-part entries for the same question (more raw entries than distinct
    questions covered) — either can still look fine at a glance (schema-valid,
    plausible correctness score) so this has to check structure, not content.
    """
    comments = data.get("question_wise_comments", [])
    raw_count = len(comments)
    covered = set()
    for c in comments:
        m = re.match(r"(\d+)", str(c.get("question_id", "")))
        if m:
            covered.add(int(m.group(1)))
    breadth = len(covered)

    reasons = []
    if expected_total and breadth < expected_total:
        missing = sorted(set(range(1, expected_total + 1)) - covered)
        reasons.append(f"incomplete: missing question(s) {', '.join(map(str, missing))}")
    if raw_count > breadth:
        reasons.append(f"duplicate entries: {raw_count} entries for only {breadth} question(s)")
    return "; ".join(reasons)


_MISMATCH_STATUS = "MISMATCH"
_DUPLICATE_CONTENT_STATUS = "DUPLICATE_CONTENT"
# Tracker statuses whose row must always reflect the tracker, never a stale
# evaluation file left over from before the status was set.
_OVERRIDE_STATUSES = {_MISMATCH_STATUS, _DUPLICATE_CONTENT_STATUS}


def _never_graded_ids(reports_dir: str, graded_base_ids: set[str]) -> list[tuple[str, str]]:
    """Cross-check tracker.json for students with zero successful evaluation.

    A student whose every attempt failed (bad OCR, model derailment, etc.)
    has no '<id>_evaluation.json' file at all, so the file-scan loop above
    never sees them — they'd be silently absent from the summary rather
    than flagged. Returns [(base_id, last_status), ...] for anyone tracked
    but not present in graded_base_ids. Best-effort: if tracker.json is
    missing (e.g. in tests that don't set one up), returns nothing.
    """
    tracker_path = os.path.join(reports_dir, "tracker.json")
    if not os.path.isfile(tracker_path):
        return []
    with open(tracker_path, "r", encoding="utf-8") as fh:
        tracker = json.load(fh)

    missing: dict[str, str] = {}
    for student_id, info in tracker.items():
        base_id = _base_student_id(student_id)
        if base_id in graded_base_ids:
            continue
        missing[base_id] = info.get("status", "UNKNOWN")
    return sorted(missing.items())


def _override_status_base_ids(reports_dir: str) -> dict[str, str]:
    """base_id -> status for every tracker entry whose status is in
    _OVERRIDE_STATUSES — read directly, independent of whether an
    '<id>_evaluation.json' happens to exist. A stale evaluation file can be
    sitting on disk from before the status was set (it was generated while
    the wrong PDF was still believed to belong to this student, or before a
    duplicate-content collision was noticed); these statuses must override
    that stale grade in the summary, not silently lose to it just because a
    file already exists.
    """
    tracker_path = os.path.join(reports_dir, "tracker.json")
    if not os.path.isfile(tracker_path):
        return {}
    with open(tracker_path, "r", encoding="utf-8") as fh:
        tracker = json.load(fh)
    result: dict[str, str] = {}
    for student_id, info in tracker.items():
        status = info.get("status")
        if status in _OVERRIDE_STATUSES:
            result[_base_student_id(student_id)] = status
    return result


def _never_graded_reason(status: str) -> str:
    """Human-readable Review Flag text for a tracker status with no evaluation file.

    MISMATCH is set (by hand, or by a grader that checks submission identity)
    when the name/ID written inside the PDF doesn't match the filename it was
    uploaded under — e.g. two students' submission slots both ended up
    containing a copy of a third student's PDF. That's a source-data problem,
    not a grading failure, so it gets its own distinct, actionable label
    instead of the generic "never successfully graded" wording.

    DUPLICATE_CONTENT is the weaker, purely mechanical signal that precedes
    it: two submissions are byte-identical (see cli.find_duplicate_uploads),
    which could mean cross-student contamination (-> MISMATCH once someone
    confirms it) OR a harmless same-student double-upload (-> back to
    PENDING once confirmed) — a hash match alone can't tell those apart
    (same-ID and cross-ID collisions look structurally identical), so it
    waits for identity verification instead of asserting either outcome.
    """
    if status == _MISMATCH_STATUS:
        return "Submitted Other's Assignment"
    if status == _DUPLICATE_CONTENT_STATUS:
        return "duplicate content detected (byte-identical to another submission) — pending identity verification"
    return f"never successfully graded (tracker status: {status})"


def export_summary_csv(reports_dir: str, output_path: str | None = None) -> str:
    """Scan *reports_dir* for '<id>_evaluation.json' files and write a summary CSV.

    Duplicate-suffixed IDs (two filenames that resolved to the same student,
    e.g. an accidental double-upload) are collapsed to one row using the
    most complete submission (by top-level question coverage, then by
    correctness score) — otherwise the same student shows up twice.

    Every row is checked for structurally malformed grading output (see
    `_review_flag`) and flagged for manual review rather than silently
    reported as-is; students with no successful evaluation at all are added
    as their own flagged rows via `_never_graded_ids` instead of vanishing
    from the summary.
    """
    if output_path is None:
        report_dir = os.path.join(reports_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        output_path = os.path.join(report_dir, "summary.csv")

    best: dict[str, tuple] = {}       # base_id -> (rank, comment, complete, missing, partial, correct, incorrect, answered_all, answers_correct, remarks)
    best_data: dict[str, dict] = {}   # base_id -> winning raw evaluation JSON, for the review-flag pass below
    for fname in os.listdir(reports_dir):
        if not fname.endswith("_evaluation.json"):
            continue
        student_id = fname[: -len("_evaluation.json")]
        with open(os.path.join(reports_dir, fname), "r", encoding="utf-8") as fh:
            data = json.load(fh)

        base_id = _base_student_id(student_id)
        rank = (_coverage_breadth(data), _correctness_score(data))
        if base_id not in best or rank > best[base_id][0]:
            complete, missing, partial = _status_counts(data)
            correct, incorrect = _correct_counts(data)
            best[base_id] = (
                rank,
                full_comment(data),
                len(data.get("question_wise_comments", [])),
                complete,
                missing,
                partial,
                correct,
                incorrect,
                data.get("answered_all_questions", ""),
                data.get("answers_are_correct", ""),
                data.get("overall_remarks_a_b_c", ""),
            )
            best_data[base_id] = data

    expected_total = _expected_coverage([_coverage_breadth(d) for d in best_data.values()])
    override_ids = _override_status_base_ids(reports_dir)

    rows = [
        (base_id, *rest[1:], _review_flag(best_data[base_id], expected_total))
        for base_id, rest in best.items()
        if base_id not in override_ids
    ]
    for base_id, status in _never_graded_ids(reports_dir, set(best.keys()) | set(override_ids.keys())):
        rows.append((base_id, "", 0, 0, 0, 0, 0, 0, "", "", "", _never_graded_reason(status)))
    for base_id, status in override_ids.items():
        rows.append((base_id, "", 0, 0, 0, 0, 0, 0, "", "", "", _never_graded_reason(status)))
    rows.sort(key=lambda row: _sort_key(row[0]))

    # utf-8-sig so Excel auto-detects encoding instead of mangling special chars.
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Student ID", "Full Comment", "Total Sub-Questions", "Complete", "Missing", "Partial",
            "Correct", "Incorrect", "Answered All Questions", "Answers Correct",
            "Overall Remarks (A/B/C)", "Review Flag",
        ])
        writer.writerows(rows)

    return output_path
