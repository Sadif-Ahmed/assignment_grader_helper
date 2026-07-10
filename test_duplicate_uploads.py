"""
Self-check for cli.find_duplicate_uploads.

Pins the real bug hit in A1/submissions: "24141084_RONY_MIAH.pdf" and
"24141084_Saiful_Rony.pdf" were byte-identical to EACH OTHER (both actually
containing a third student's, 23201214's, work) despite sharing the same
filename-derived ID — a duplicate-upload mixup where two clearly different
apparent names ("RONY_MIAH" vs "Saiful_Rony") pointed at identical bytes.
An earlier version of this check excluded same-ID matches as "probably a
harmless self-resubmission", which would have missed this exact case —
so same-ID matches must be flagged too, not just cross-ID ones.
"""

import os
import shutil
import tempfile

from cli import find_duplicate_uploads


def _write_pdf(target_dir: str, filename: str, content: bytes) -> str:
    path = os.path.join(target_dir, filename)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def demo() -> None:
    target_dir = tempfile.mkdtemp()
    try:
        mixup_bytes = b"%PDF-1.4 fake content actually belonging to 23201214"
        farhans_real_bytes = b"%PDF-1.4 23201214's own distinct real submission"
        other_bytes = b"%PDF-1.4 a completely different submission"

        student_pdfs = [
            # 23201214's own real, distinct file — must NOT be flagged.
            ("23201214", _write_pdf(target_dir, "23201214_-_FARHAN_TANVIR.pdf", farhans_real_bytes)),
            # Same nominal ID (24141084 / 24141084[duplicate]), byte-identical
            # to each other, but under two different apparent names — this is
            # the exact real-world shape that must be caught.
            ("24141084", _write_pdf(target_dir, "24141084_-_RONY_MIAH.pdf", mixup_bytes)),
            ("24141084[duplicate]", _write_pdf(target_dir, "24141084_-_Saiful_Rony.pdf", mixup_bytes)),
            # Unrelated student, unrelated content — must NOT be flagged.
            ("20201178", _write_pdf(target_dir, "20201178_-_AR_AKASH.pdf", other_bytes)),
        ]

        pairs = find_duplicate_uploads(student_pdfs)

        assert set(pairs) == {("24141084", "24141084[duplicate]")}, (
            f"expected exactly the same-ID mixup pair flagged, got {pairs}"
        )

        print("OK: byte-identical PDFs are flagged even when they share the same nominal student ID.")
    finally:
        shutil.rmtree(target_dir)


if __name__ == "__main__":
    demo()
