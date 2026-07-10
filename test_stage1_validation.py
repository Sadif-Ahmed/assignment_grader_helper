"""
Self-check for stage1._assert_question_count.

Pins the completeness-validation fix: a restructuring call that returns
syntactically valid JSON but silently truncates (declared total_questions
doesn't match how many questions actually came back — observed live on
student 23201333: declared 7, delivered 1) must raise, so it's retried via
nvidia_client's existing per-model retry + pool fallback instead of being
accepted silently.
"""

from stage1 import _assert_question_count, _assert_nonempty_ocr


def demo() -> None:
    # Complete: no error.
    _assert_question_count({"total_questions": 2, "questions": [{}, {}]})

    # The exact shape of the observed bug: declared 7, only 1 delivered.
    try:
        _assert_question_count({"total_questions": 7, "questions": [{}]})
        raise AssertionError("expected ValueError for truncated output")
    except ValueError:
        pass

    # Declared 0 (e.g. empty question paper) must not false-positive.
    _assert_question_count({"total_questions": 0, "questions": []})

    # OCR guard: non-empty content passes, blank/whitespace-only is rejected.
    _assert_nonempty_ocr({"content": "Problem 1\n(a) Yield is..."})
    for bad in ({"content": ""}, {"content": "   \n  "}, {}):
        try:
            _assert_nonempty_ocr(bad)
            raise AssertionError(f"expected ValueError for blank OCR content: {bad!r}")
        except ValueError:
            pass

    print("OK: truncated restructuring output is rejected, complete output passes.")


if __name__ == "__main__":
    demo()
