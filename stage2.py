import functools
import json
import logging
import re
from typing import Any
from pydantic import BaseModel, Field, create_model
from nvidia_client import call_nvidia_structured
from model_pools import get_pool
from stage1 import sanitize_llm_text, sanitize_json_strings

logger = logging.getLogger(__name__)

class QuestionComment(BaseModel):
    """Per-question feedback entry included in every evaluation."""
    question_id: str = Field(description="Specific sub-question identifier (e.g. '1a', '2c' instead of just 'Problem 2')")
    comment: str = Field(description="Specific feedback detailing what is missing, partially correct, or completely correct.")
    status: str = Field(description="Must be exactly one of: 'Complete', 'Partial', 'Missing'")
    is_correct: bool = Field(description="Whether the question was fully and correctly answered")


_MAX_FIELD_NAME_LEN = 50


def _sanitize_field_name(criterion: str) -> str:
    """Convert a human-readable criterion string into a valid Python identifier."""
    name = criterion.lower().strip().rstrip("?").strip()
    chars: list[str] = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            chars.append(ch)
        elif ch in (" ", "-", "/", ",", "(", ")"):
            chars.append("_")
    name = "".join(chars)
    # Collapse repeated underscores
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    if not name or not name[0].isalpha():
        name = "criterion_" + name
    # ponytail: hard cap so a long custom criterion can't produce an
    # unbounded schema key (was overflowing the schema preview). Full
    # text is preserved in the field's description either way.
    return name[:_MAX_FIELD_NAME_LEN].rstrip("_")


def build_evaluation_schema(criteria: list[str]) -> type[BaseModel]:
    """
    Build a Pydantic model at runtime whose fields mirror the active criteria.

    • Yes/no questions (ending with ``?``) → ``bool``
    • Remark / grade criteria → ``str``
    • Always includes ``question_wise_comments: list[QuestionComment]``
    """
    fields: dict[str, Any] = {}

    for criterion in criteria:
        field_name = _sanitize_field_name(criterion)
        if field_name in fields:
            # Two different criteria truncated to the same key — disambiguate
            # instead of one silently overwriting the other in the schema.
            suffix = 2
            while f"{field_name}_{suffix}" in fields:
                suffix += 1
            field_name = f"{field_name}_{suffix}"
        lower = criterion.lower()

        if any(kw in lower for kw in ("remarks", "grade", "rating", "score")):
            fields[field_name] = (str, Field(description=criterion))
        elif criterion.strip().endswith("?"):
            fields[field_name] = (bool, Field(description=criterion))
        else:
            fields[field_name] = (str, Field(description=criterion))

    # Consolidated per-question feedback
    fields["question_wise_comments"] = (
        list[QuestionComment],
        Field(description="Detailed per-question feedback bundling all specific comments"),
    )

    return create_model("EvaluationResult", **fields)


def build_criteria_prompt(criteria: list[str]) -> str:
    """Render the criteria list as numbered prompt text for Stage 2."""
    lines = ["Apply the following evaluation criteria strictly:"]
    for i, c in enumerate(criteria, 1):
        lines.append(f"  {i}. {c}")
    lines.append("")
    lines.append(
        "For each criterion, provide your assessment in the corresponding field. "
        "Additionally, bundle all per-question feedback into the "
        "question_wise_comments array. You MUST evaluate at the granular sub-question "
        "level (e.g., '1a', '1b' instead of clumping them into 'Problem 1'). For each "
        "sub-question, categorize its status as 'Complete', 'Partial', or 'Missing'."
    )
    return "\n".join(lines)


def _assert_full_coverage(result: dict, expected_total: int) -> None:
    """Raise if question_wise_comments doesn't cover every top-level question.

    Observed live: the grader sometimes emits both a parent-level entry
    ('1') and sub-part entries ('1a', '1b', ...) for the same question, then
    stops partway through (e.g. covers only questions 1-4 of 7) with no
    truncation/length signal — schema-valid but semantically incomplete.
    Passed as `validate=` so nvidia_client's existing retry handles recovery.
    """
    if not expected_total:
        return
    covered = set()
    for c in result.get("question_wise_comments", []):
        m = re.match(r"(\d+)", str(c.get("question_id", "")))
        if m:
            covered.add(int(m.group(1)))
    missing = sorted(set(range(1, expected_total + 1)) - covered)
    if missing:
        raise ValueError(
            f"Incomplete evaluation: missing top-level question(s) {missing} "
            f"of {expected_total} in question_wise_comments"
        )


_STAGE2_SYSTEM = (
    "Act as a strict and granular grader. Compare the student's extracted answers against "
    "the master solution. Apply the provided evaluation criteria strictly to "
    "generate a final report. Be thorough and fair. Both the student answers "
    "and master solution are provided as structured text extracted from their "
    "original handwritten PDFs. "
    "CRITICAL: Address all feedback directly to the student in the second person (e.g. 'You missed...', 'Your answer is correct', 'You need to...'). "
    "CRITICAL: Always evaluate and report on the specific sub-question level (e.g. '2a', '3b') "
    "rather than grouping feedback under the parent question (e.g. 'Problem 2'). "
    "If an answer is missing or incomplete, explicitly flag it. "
    "CRITICAL: Ensure your global evaluation criteria responses are strictly logically consistent "
    "with your sub-question status. For example, if you mark all sub-questions as correct, "
    "the overall answers_are_correct field MUST be true. "
    "CRITICAL: Allow up to 5% deviation between a student's final numerical answer and the "
    "master solution's value before counting it as an error — small differences from rounding "
    "or intermediate-precision choices (e.g. carrying 3 vs 5 decimal places through a multi-step "
    "calculation) are NOT mistakes. Only mark a numerical answer wrong if it deviates by more "
    "than 5%, or if the method/formula itself is incorrect regardless of how close the final "
    "number lands."
)


def run_stage2(
    api_key: str,
    model_id: str | list[str] | None,
    master_map: dict,
    student_map: dict,
    criteria: list[str],
    log,
    question_context: dict | str = "",
) -> dict:
    """
    Stage 2: Fully text-based evaluation.

    Compares the student's Stage 1 structural map against the master solution's
    Stage 1 structural map. No images are needed — both inputs are text-based
    JSON maps extracted by Stage 1.

    If question_context is provided (extracted from the question PDF), it is
    included so the grader knows exactly what was asked.
    """
    log("📊 Stage 2 → Evaluation & Grading (text-based comparison)…")

    expected_total = question_context.get("total_questions", 0) if isinstance(question_context, dict) else 0

    # Defensive sanitization: master_map/student_map/question_context are
    # LLM-generated (from stage1) and criteria may be free-typed by the user
    # (e.g. pasted with smart quotes). Both land verbatim in this prompt.
    criteria = [sanitize_llm_text(c) for c in criteria]
    master_map = sanitize_json_strings(master_map)
    student_map = sanitize_json_strings(student_map)
    if isinstance(question_context, dict):
        question_context = sanitize_json_strings(question_context)
    elif isinstance(question_context, str) and question_context:
        question_context = sanitize_llm_text(question_context)

    eval_schema_model = build_evaluation_schema(criteria)
    eval_schema = eval_schema_model.model_json_schema()
    criteria_text = build_criteria_prompt(criteria)

    # Build the prompt with optional question context
    prompt_parts = []

    if question_context:
        context_str = json.dumps(question_context, indent=2) if isinstance(question_context, dict) else str(question_context)
        prompt_parts.append(
            "## Original Assignment Questions:\n"
            f"```\n{context_str}\n```\n"
        )

    prompt_parts.append(
        "## Master Solution (Answer Key):\n"
        f"```json\n{json.dumps(master_map, indent=2)}\n```\n"
    )
    prompt_parts.append(
        "## Student Submission (Extracted Answers):\n"
        f"```json\n{json.dumps(student_map, indent=2)}\n```\n"
    )
    prompt_parts.append(
        f"## {criteria_text}\n\n"
        "Compare the student's answers against the master solution thoroughly. "
        "Provide your structured evaluation."
    )

    user_prompt = "\n".join(prompt_parts)

    try:
        evaluation_model = get_pool("evaluation", model_id)
        sys_prompt = _STAGE2_SYSTEM + " Output strictly valid JSON matching the provided schema."
        result = call_nvidia_structured(
            api_key=api_key,
            model=evaluation_model,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            image_b64=None,  # No images — fully text-based
            schema=eval_schema,
            validate=functools.partial(_assert_full_coverage, expected_total=expected_total),
        )
        return result
    except Exception as exc:
        logger.exception(f"Stage 2 evaluation failed")
        log(f"❌ Stage 2 failed: {exc}")
        raise

