"""
Builds a per-student summary CSV (opens directly in Excel) from the
Stage 2 evaluation reports produced by the pipeline.
"""

import csv
import json
import os
import re
from typing import Any

_DUPLICATE_SUFFIX_RE = re.compile(r"\[duplicate\d*\]$")

_QUALITY_KEYWORDS = ("remark", "grade", "rating", "score")


def _find_field(data: dict, keywords: tuple[str, ...], want_type: type) -> Any:
    """Find the first field whose (sanitized) name contains one of *keywords*."""
    for key, value in data.items():
        if isinstance(value, want_type) and any(kw in key.lower() for kw in keywords):
            return value
    return None


def _quality(data: dict) -> str:
    """Return the overall quality/grade for one evaluation JSON."""
    comments = data.get("question_wise_comments", [])
    quality = _find_field(data, _QUALITY_KEYWORDS, str)
    if quality is None:
        correct_count = sum(1 for c in comments if c.get("is_correct"))
        quality = f"{correct_count}/{len(comments)} correct" if comments else "N/A"
    return quality


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


def _correctness_score(data: dict) -> float:
    """Fraction of question_wise_comments marked correct — always numeric,
    unlike the free-text quality/remarks field, so it's safe to compare
    across duplicate submissions of the same student."""
    comments = data.get("question_wise_comments", [])
    if not comments:
        return 0.0
    return sum(1 for c in comments if c.get("is_correct")) / len(comments)


def export_summary_csv(reports_dir: str, output_path: str | None = None) -> str:
    """Scan *reports_dir* for '<id>_evaluation.json' files and write a summary CSV.

    Duplicate-suffixed IDs (two filenames that resolved to the same student,
    e.g. an accidental double-upload) are collapsed to one row using the
    higher-scoring submission — otherwise the same student shows up twice.
    """
    output_path = output_path or os.path.join(reports_dir, "summary.csv")

    best: dict[str, tuple] = {}  # base_id -> (score, comment, quality, answered_all, answers_correct, remarks)
    for fname in os.listdir(reports_dir):
        if not fname.endswith("_evaluation.json"):
            continue
        student_id = fname[: -len("_evaluation.json")]
        with open(os.path.join(reports_dir, fname), "r", encoding="utf-8") as fh:
            data = json.load(fh)

        base_id = _base_student_id(student_id)
        score = _correctness_score(data)
        if base_id not in best or score > best[base_id][0]:
            best[base_id] = (
                score,
                full_comment(data),
                _quality(data),
                data.get("answered_all_questions", ""),
                data.get("answers_are_correct", ""),
                data.get("overall_remarks_a_b_c", ""),
            )

    rows = [
        (base_id, comment, quality, answered_all, answers_correct, remarks)
        for base_id, (_, comment, quality, answered_all, answers_correct, remarks) in best.items()
    ]
    rows.sort(key=lambda row: _sort_key(row[0]))

    # utf-8-sig so Excel auto-detects encoding instead of mangling special chars.
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Student ID", "Full Comment", "Overall Quality",
            "Answered All Questions", "Answers Correct", "Overall Remarks (A/B/C)",
        ])
        writer.writerows(rows)

    return output_path
