"""
Self-check for report_export's duplicate-collapsing in export_summary_csv.

Pins the fix for real duplicates seen in A1/submissions (e.g. student
23101132 uploaded twice under different filenames, both get graded
independently as "23101132" and "23101132[duplicate]"): the summary must
show one row per real student, using the higher-scoring submission.
"""

import csv
import json
import os
import shutil
import tempfile

from report_export import export_summary_csv


def _write_eval(reports_dir: str, student_id: str, correct: int, total: int, remarks: str = "C") -> None:
    data = {
        "answered_all_questions": True,
        "answers_are_correct": correct == total,
        "overall_remarks_a_b_c": remarks,
        "question_wise_comments": [
            {"question_id": str(i), "comment": f"comment {i}", "status": "Complete", "is_correct": i < correct}
            for i in range(total)
        ]
    }
    with open(os.path.join(reports_dir, f"{student_id}_evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def demo() -> None:
    reports_dir = tempfile.mkdtemp()
    try:
        _write_eval(reports_dir, "23101132", correct=2, total=7, remarks="F")             # worse attempt
        _write_eval(reports_dir, "23101132[duplicate]", correct=6, total=7, remarks="A")  # better attempt
        _write_eval(reports_dir, "23201333", correct=1, total=7)                          # no duplicate

        out_path = export_summary_csv(reports_dir)
        with open(out_path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))

        ids = [r["Student ID"] for r in rows]
        assert ids == ["23101132", "23201333"], f"expected one collapsed row per student, got {ids}"
        assert "duplicate" not in rows[0]["Student ID"], rows[0]
        assert rows[0]["Overall Quality"] == "A", (
            f"expected the higher-scoring duplicate's remarks to win, got {rows[0]['Overall Quality']}"
        )
        # The winning duplicate's own fields must come along, not the loser's.
        assert rows[0]["Overall Remarks (A/B/C)"] == "A", rows[0]
        assert rows[0]["Answered All Questions"] == "True", rows[0]
        assert rows[0]["Answers Correct"] == "False", rows[0]  # 6/7, not all correct
        assert "comment 0" in rows[0]["Full Comment"], rows[0]

        print("OK: duplicate-suffixed IDs collapse to the higher-scoring submission, all columns follow the winner.")
    finally:
        shutil.rmtree(reports_dir)


if __name__ == "__main__":
    demo()
