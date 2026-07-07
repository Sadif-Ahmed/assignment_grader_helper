"""
Self-check for stage1.sanitize_llm_text.

Pins the exact character set observed to truncate a Stage 1 mapping call
mid-JSON on a real submission (student 23201333): the model emitted only
question 1a/1b out of 7 once it hit superscripts/pi/therefore in the raw
OCR text, despite max_tokens being nowhere near the limit. Every char in
`_TROUBLE_CHARS` must come out as plain ASCII so it can't reproduce.
"""

from stage1 import sanitize_llm_text

_TROUBLE_CHARS = "≈“”‘’×÷−–—…→•⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁αβγδΔθλμπσΣΩφ∴·√±≤≥≠°∞∑∫½¼¾"


def demo() -> None:
    for ch in _TROUBLE_CHARS:
        out = sanitize_llm_text(f"before {ch} after")
        assert out.isascii(), f"{ch!r} (U+{ord(ch):04X}) survived sanitization: {out!r}"

    # Real snippet from the truncating submission.
    sample = "∴ Wafer area = π(11)^2 = 380.1336 cm²  10⁻¹² Hz"
    out = sanitize_llm_text(sample)
    assert out.isascii(), out
    assert "pi" in out and "therefore" in out, out

    # Accented characters (e.g. names) transliterate via the NFKD fallback
    # instead of being silently dropped whole.
    assert sanitize_llm_text("café") == "cafe"

    print("OK: all known trouble characters sanitize to plain ASCII.")


if __name__ == "__main__":
    demo()
