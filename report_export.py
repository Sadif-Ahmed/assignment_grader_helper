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


def export_summary_csv(reports_dir: str, output_path: str | None = None) -> str:
    """Scan *reports_dir* for '<id>_evaluation.json' files and write a summary CSV.

    Duplicate-suffixed IDs (two filenames that resolved to the same student,
    e.g. an accidental double-upload) are collapsed to one row using the
    most complete submission (by top-level question coverage, then by
    correctness score) — otherwise the same student shows up twice.
    """
    if output_path is None:
        report_dir = os.path.join(reports_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        output_path = os.path.join(report_dir, "summary.csv")

    best: dict[str, tuple] = {}  # base_id -> (rank, comment, complete, missing, partial, correct, incorrect, answered_all, answers_correct, remarks)
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

    rows = [
        (base_id, *rest[1:])
        for base_id, rest in best.items()
    ]
    rows.sort(key=lambda row: _sort_key(row[0]))

    # utf-8-sig so Excel auto-detects encoding instead of mangling special chars.
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Student ID", "Full Comment", "Total Sub-Questions", "Complete", "Missing", "Partial",
            "Correct", "Incorrect", "Answered All Questions", "Answers Correct",
            "Overall Remarks (A/B/C)",
        ])
        writer.writerows(rows)

    return output_path
