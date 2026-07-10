"""
Self-check for stage2._assert_full_coverage.

Pins the completeness-validation fix: an evaluation call that returns
syntactically valid JSON but only covers some of the top-level questions
(observed live: covers questions 1-4 of 7, with duplicate parent+sub-part
entries for 1-4) must raise, so it's retried instead of accepted silently.
"""

from stage2 import _assert_full_coverage


def demo() -> None:
    # Complete: no error.
    _assert_full_coverage(
        {"question_wise_comments": [{"question_id": str(i)} for i in range(1, 8)]},
        expected_total=7,
    )

    # The exact shape of the observed bug: covers only 1-4 of 7, with
    # duplicate parent ("1") and sub-part ("1a") entries.
    try:
        _assert_full_coverage(
            {"question_wise_comments": [
                {"question_id": qid} for qid in
                ["1", "1a", "1b", "2", "2a", "3", "3a", "4", "4a"]
            ]},
            expected_total=7,
        )
        raise AssertionError("expected ValueError for incomplete coverage")
    except ValueError:
        pass

    # expected_total=0 (unknown blueprint) must not false-positive.
    _assert_full_coverage({"question_wise_comments": []}, expected_total=0)

    print("OK: incomplete question coverage is rejected, complete coverage passes.")


if __name__ == "__main__":
    demo()
