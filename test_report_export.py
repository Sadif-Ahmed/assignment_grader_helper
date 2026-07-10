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
    comments = [
        {"question_id": str(i), "comment": f"comment {i}", "status": "Complete", "is_correct": i < correct}
        for i in range(total)
    ]
    _write_eval_raw(reports_dir, student_id, comments, remarks=remarks, answers_correct=(correct == total))


def _write_eval_raw(reports_dir: str, student_id: str, comments: list, remarks: str = "C", answers_correct: bool = True) -> None:
    data = {
        "answered_all_questions": True,
        "answers_are_correct": answers_correct,
        "overall_remarks_a_b_c": remarks,
        "question_wise_comments": comments,
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
        # The winning duplicate's own fields must come along, not the loser's.
        assert rows[0]["Overall Remarks (A/B/C)"] == "A", (
            f"expected the higher-scoring duplicate's remarks to win, got {rows[0]['Overall Remarks (A/B/C)']}"
        )
        assert rows[0]["Total Sub-Questions"] == "7", rows[0]
        assert rows[0]["Answered All Questions"] == "True", rows[0]
        assert rows[0]["Answers Correct"] == "False", rows[0]  # 6/7, not all correct
        assert "comment 0" in rows[0]["Full Comment"], rows[0]

        print("OK: duplicate-suffixed IDs collapse to the higher-scoring submission, all columns follow the winner.")
    finally:
        shutil.rmtree(reports_dir)


def demo_breadth_beats_score() -> None:
    """A malformed response that only covers questions 0-3 of 7 (but scores
    100% on those few) must lose to a complete 7/7 response that scores
    lower — coverage breadth outranks raw correctness score. Pins the real
    bug hit on student 23201214: a truncated-but-'perfect' duplicate was
    winning over the complete, honestly-graded submission."""
    reports_dir = tempfile.mkdtemp()
    try:
        _write_eval(reports_dir, "99999999", correct=3, total=4)              # malformed: stops at Q3, but 3/4 correct = 0.75
        _write_eval(reports_dir, "99999999[duplicate]", correct=4, total=7)   # complete: 4/7 correct = 0.571

        out_path = export_summary_csv(reports_dir)
        with open(out_path, encoding="utf-8-sig") as fh:
            rows = {r["Student ID"]: r for r in csv.DictReader(fh)}

        assert rows["99999999"]["Total Sub-Questions"] == "7", (
            f"expected the complete (breadth=7) submission to win over the higher-scoring but "
            f"truncated one, got {rows['99999999']}"
        )
        print("OK: coverage breadth outranks raw correctness score when picking between duplicates.")
    finally:
        shutil.rmtree(reports_dir)


def demo_review_flag() -> None:
    """Pins the actual bug hit on students 22201813 and 23201214[duplicate]:
    a response with duplicate parent+sub-part entries ('1' and '1a') that
    stops partway through (questions 5-7 never appear) must be flagged for
    manual review, even though it's schema-valid and scores plausibly.
    A student with zero successful evaluation (every attempt FAILED in
    tracker.json, no *_evaluation.json at all) must still show up in the
    summary as its own flagged row instead of silently vanishing."""
    reports_dir = tempfile.mkdtemp()
    try:
        # Five clean 7/7 students establish the batch's "normal" coverage (mode=7).
        for i, sid in enumerate(["10001", "10002", "10003", "10004", "10005"]):
            _write_eval(reports_dir, sid, correct=4 + i % 3, total=7)

        # The real malformed shape: parent ("1") + sub-part ("1a","1b") duplicate
        # entries for questions 1-4, nothing for 5-7. breadth=4, raw_count=9.
        malformed_comments = []
        for q in range(1, 5):
            malformed_comments.append({"question_id": str(q), "comment": "parent", "status": "Complete", "is_correct": True})
            malformed_comments.append({"question_id": f"{q}a", "comment": "sub a", "status": "Complete", "is_correct": True})
        malformed_comments.append({"question_id": "4b", "comment": "sub b", "status": "Complete", "is_correct": True})
        _write_eval_raw(reports_dir, "10006", malformed_comments)

        # 10009: a MISMATCH discovered AFTER an evaluation file already existed
        # (e.g. graded before anyone noticed the PDF belonged to someone else).
        # The stale file must not win just because it exists.
        _write_eval(reports_dir, "10009", correct=7, total=7, remarks="A")

        # A student tracked but with zero successful evaluation — no *_evaluation.json.
        # 10008 is the identity-mismatch case: filename says one student, the PDF
        # content is someone else's submission (e.g. a duplicate-upload mixup).
        # 10010: cli.find_duplicate_uploads found a byte-identical pair but nobody
        # has confirmed yet whether it's cross-student contamination or a
        # harmless same-student double-upload — the weaker, unconfirmed status.
        tracker = {
            "10007": {"file_path": "x.pdf", "status": "FAILED", "last_updated": "2026-01-01T00:00:00"},
            "10008": {"file_path": "y.pdf", "status": "MISMATCH", "last_updated": "2026-01-01T00:00:00"},
            "10009": {"file_path": "z.pdf", "status": "MISMATCH", "last_updated": "2026-01-01T00:00:00"},
            "10010": {"file_path": "w.pdf", "status": "DUPLICATE_CONTENT", "last_updated": "2026-01-01T00:00:00"},
        }
        with open(os.path.join(reports_dir, "tracker.json"), "w", encoding="utf-8") as fh:
            json.dump(tracker, fh)

        out_path = export_summary_csv(reports_dir)
        with open(out_path, encoding="utf-8-sig") as fh:
            rows = {r["Student ID"]: r for r in csv.DictReader(fh)}

        for sid in ["10001", "10002", "10003", "10004", "10005"]:
            assert rows[sid]["Review Flag"] == "", f"expected clean student {sid} to have no review flag, got {rows[sid]}"

        flag = rows["10006"]["Review Flag"]
        assert "incomplete" in flag and "5" in flag and "6" in flag and "7" in flag, (
            f"expected incomplete-coverage flag naming missing questions 5,6,7, got {flag!r}"
        )
        assert "duplicate entries" in flag, f"expected duplicate-entries flag too, got {flag!r}"

        assert "10007" in rows, "student with zero successful evaluation must still appear in the summary"
        assert "never successfully graded" in rows["10007"]["Review Flag"], rows["10007"]
        assert "FAILED" in rows["10007"]["Review Flag"], rows["10007"]

        assert rows["10008"]["Review Flag"] == "Submitted Other's Assignment", (
            f"expected MISMATCH tracker status to produce the identity-mismatch flag, got {rows['10008']}"
        )

        assert rows["10009"]["Review Flag"] == "Submitted Other's Assignment", (
            f"MISMATCH must override a stale pre-existing evaluation file, not lose to it, got {rows['10009']}"
        )
        assert rows["10009"]["Total Sub-Questions"] == "0", (
            f"the stale (now-known-wrong) evaluation data must not leak into the row, got {rows['10009']}"
        )

        assert rows["10010"]["Review Flag"] == (
            "duplicate content detected (byte-identical to another submission) — pending identity verification"
        ), rows["10010"]

        print("OK: malformed/incomplete, never-graded, identity-mismatch, and duplicate-content submissions are flagged for manual review.")
    finally:
        shutil.rmtree(reports_dir)


if __name__ == "__main__":
    demo()
    demo_breadth_beats_score()
    demo_review_flag()
