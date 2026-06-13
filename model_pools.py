"""
Centralized LLM model pool configuration for the assignment checker pipeline.

Each task gets a list of 3 models tried in order. When a model fails (after
its own internal retries in nvidia_client.py), the next model in the pool is
attempted automatically via `call_nvidia_structured(model=pool_list, ...)`.
"""

_GLOBAL_POOL = [
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2.6",
    "deepseek-ai/deepseek-v4-flash",
]

MODEL_POOLS: dict[str, list[str]] = {
    # Task 1: Extract printed text from question PDF pages (vision required)
    "question_extraction": _GLOBAL_POOL,

    # Task 2: Parse handwritten student answers from PDF pages (vision required)
    "student_parsing": _GLOBAL_POOL,

    # Task 3: Restructure scattered answers into question hierarchy (text-only)
    "restructuring": _GLOBAL_POOL,

    # Task 4: Grade student answers against master solution (text-only)
    "evaluation": _GLOBAL_POOL,
}


def get_pool(task_name: str, override: str | list[str] | None = None) -> str | list[str]:
    """
    Return the model pool for a given task.

    If `override` is provided (e.g. from --model-id CLI flag), that single
    model is used for all tasks instead of the pool.
    """
    if override:
        return override
    return MODEL_POOLS[task_name]
