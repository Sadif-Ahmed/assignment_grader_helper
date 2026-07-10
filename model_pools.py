"""
Centralized LLM model configuration for the assignment checker pipeline.

Pinned to a single strongest model — no weaker-model fallback. A fallback
pool (Qwen -> Kimi -> DeepSeek) previously masked Qwen server errors by
silently degrading to weaker models, which produced malformed/truncated
structured output (incomplete question coverage, duplicated entries) that
passed schema validation but was semantically wrong. Better to fail loudly
on Qwen and retry than to silently grade with a weaker model.
"""

_STRONGEST_MODEL = "qwen/qwen3.5-397b-a17b"

MODEL_POOLS: dict[str, str] = {
    # Task 1: Extract printed text from question PDF pages (vision required)
    "question_extraction": _STRONGEST_MODEL,

    # Task 2: Parse handwritten student answers from PDF pages (vision required)
    "student_parsing": _STRONGEST_MODEL,

    # Task 3: Restructure scattered answers into question hierarchy (text-only)
    "restructuring": _STRONGEST_MODEL,

    # Task 4: Grade student answers against master solution (text-only)
    "evaluation": _STRONGEST_MODEL,
}


def get_pool(task_name: str, override: str | list[str] | None = None) -> str | list[str]:
    """
    Return the model for a given task.

    If `override` is provided (e.g. from --model-id CLI flag), that single
    model is used for all tasks instead of the configured default.
    """
    if override:
        return override
    return MODEL_POOLS[task_name]
