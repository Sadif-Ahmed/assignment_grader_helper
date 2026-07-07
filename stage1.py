import json
import logging
import unicodedata
from pydantic import BaseModel, Field
from nvidia_client import call_nvidia_structured, pdf_to_base64_images
from model_pools import get_pool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class Stage1SubQuestion(BaseModel):
    id: str = Field(description="Sub-question ID (e.g., '1a', '2.1')")
    content: str = Field(description="The FULL, verbatim text of the student's answer for this sub-question. Must include all paragraphs, formulas, and text. DO NOT SUMMARIZE.")

class Stage1Question(BaseModel):
    question_id: str = Field(description="Main question ID (e.g., '1', '2')")
    raw_content: str = Field(description="The FULL, verbatim text of the student's answer for this main question (if any). Must include all paragraphs, formulas, and text. DO NOT SUMMARIZE.")
    sub_questions: list[Stage1SubQuestion] = Field(default_factory=list)

class Stage1Response(BaseModel):
    total_questions: int = Field(description="Total number of main questions found")
    questions: list[Stage1Question] = Field(description="Array of parsed questions")

class RawPageContent(BaseModel):
    page_num: int
    text: str

class RawOCRResponse(BaseModel):
    pages: list[RawPageContent]

# ─────────────────────────────────────────────────────────────────────────────
# Unicode Sanitization
# ─────────────────────────────────────────────────────────────────────────────

# Symbols observed to derail LLM generation (e.g. "≈" can trigger a premature
# stop mid-JSON, well before any max_tokens limit — the model just stops
# emitting further questions). Normalize them to plain ASCII before any text
# reaches a prompt. STEM assignments reliably produce superscripts (10⁻¹²),
# Greek letters (α, β, π), and math operators (∴, ·, ≤) in OCR'd handwriting,
# so this list keeps growing — see the NFKD catch-all below for anything new.
_UNICODE_SANITIZE_MAP = {
    "≈": " approx ",  # ≈
    "“": '"', "”": '"',  # “ ”
    "‘": "'", "’": "'",  # ‘ ’
    "×": "x",    # ×
    "÷": "/",    # ÷
    "−": "-",    # − minus sign
    "–": "-",    # – en dash
    "—": "--",   # — em dash
    "…": "...",  # …
    "→": "->",   # →
    "•": "-",    # •
    # Superscripts (exponents, e.g. 10⁻¹² cm², x³)
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "⁺": "^+", "⁻": "^-", "⁼": "^=", "⁽": "^(", "⁾": "^)", "ⁿ": "^n",
    # Subscripts (e.g. CPI_alpha written as CPIα, x₁)
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
    # Greek letters common in engineering/math notation
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "Delta",
    "θ": "theta", "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma",
    "Σ": "Sigma", "Ω": "Omega", "φ": "phi",
    # Other math/measurement symbols
    "∴": "therefore", "·": "*", "√": "sqrt", "±": "+/-",
    "≤": "<=", "≥": ">=", "≠": "!=", "°": " deg", "∞": "infinity",
    "∑": "sum", "∫": "integral",
    "½": "1/2", "¼": "1/4", "¾": "3/4",
}

def sanitize_llm_text(text: str) -> str:
    """Strip unicode characters known to trigger premature LLM generation stops.

    First applies the curated map above (semantic replacements for symbols
    we've actually seen cause trouble). Anything still non-ASCII afterwards
    (accented names, a symbol not yet catalogued) is transliterated via NFKD
    decomposition, or dropped if it has no ASCII form — better than letting
    an uncaught symbol silently truncate an entire evaluation again.
    """
    for char, repl in _UNICODE_SANITIZE_MAP.items():
        text = text.replace(char, repl)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def sanitize_json_strings(obj):
    """Recursively apply sanitize_llm_text to every string in a dict/list structure.

    LLM-generated JSON (structuring/mapping output) can reintroduce the same
    problem symbols even when its input was clean, and that output is what
    gets embedded verbatim into the NEXT prompt (mapping, stage 2). Run this
    on every LLM output before it's re-embedded elsewhere.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_strings(v) for v in obj]
    if isinstance(obj, str):
        return sanitize_llm_text(obj)
    return obj

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_EXTRACT_SYSTEM = (
    "Act as a document transcriber. Extract ALL text content from this page "
    "accurately. Preserve structure, headings, numbering, and formatting. "
    "Return the extracted text as plain text."
)

_STUDENT_OCR_SYSTEM = (
    "Act as an advanced document transcriber. You are shown a SINGLE PAGE of a "
    "student submission. Your task is to transcribe ALL handwritten "
    "and printed content visible on this page completely and accurately. "
    "Preserve structure, headings, equations, code, and explanations verbatim. "
    "Do NOT evaluate correctness. Do NOT omit any text. Return the transcribed text as plain text."
)

_QUESTION_STRUCTURE_SYSTEM = (
    "Act as a meticulous data organizer. You are given the raw transcribed text of a master question paper. "
    "Your job is to read the text and determine the true hierarchy of questions and sub-questions. "
    "Create a perfectly structured JSON representing the assignment's question hierarchy. "
    "Return structured JSON matching the provided schema. Expected JSON skeleton:\n"
    "```json\n"
    "{\n"
    "  \"total_questions\": 3, // IMPORTANT: Update with the actual total number of questions found\n"
    "  \"questions\": [\n"
    "    {\n"
    "      \"question_id\": \"1\",\n"
    "      \"raw_content\": \"Text of the main question...\",\n"
    "      \"sub_questions\": [\n"
    "        {\n"
    "          \"id\": \"1a\",\n"
    "          \"content\": \"Text of the sub-question...\"\n"
    "        }\n"
    "        // ... other sub-questions for question 1\n"
    "      ]\n"
    "    }\n"
    "    // ... ALL other questions (e.g., question 2, question 3, etc.) MUST be extracted\n"
    "  ]\n"
    "}\n"
    "```"
)

_STUDENT_MAPPING_SYSTEM = (
    "Act as a meticulous data organizer. You are given the perfectly structured hierarchy of an assignment (Question Context) "
    "and the verbatim raw text of a student's submission (Raw Student Answers). "
    "Your job is to take ALL the text from the raw student answers and slot it exactly into the structured hierarchy. "
    "CRITICAL RULE 1: The 'raw_content' and 'content' fields in your output MUST contain the full student's actual answers from the 'Raw Student Answers' text (including all paragraphs, equations, and explanations). Do NOT just extract headers or summaries. "
    "CRITICAL RULE 2: Do NOT evaluate correctness. Do NOT change, summarize, or discard ANY of "
    "the student's content from the raw text. EVERY piece of text from the raw text "
    "MUST be preserved in the final output. If a student answer doesn't fit neatly, put it "
    "in the closest matching question or as a new question, but NEVER delete it. "
    "Return structured JSON matching the provided schema. Expected JSON skeleton:\n"
    "```json\n"
    "{\n"
    "  \"total_questions\": 3, // IMPORTANT: Update with the actual total number of questions from the Blueprint\n"
    "  \"questions\": [\n"
    "    {\n"
    "      \"question_id\": \"1\",\n"
    "      \"raw_content\": \"Student's answer for the main question 1...\",\n"
    "      \"sub_questions\": [\n"
    "        {\n"
    "          \"id\": \"1a\",\n"
    "          \"content\": \"Student's answer for sub-question 1a...\"\n"
    "        }\n"
    "        // ... other sub-questions for question 1\n"
    "      ]\n"
    "    }\n"
    "    // ... ALL other questions (e.g., question 2, question 3, etc.) MUST be included\n"
    "  ]\n"
    "}\n"
    "```"
)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Raw OCR Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw_ocr(
    api_key: str,
    model_id: str | list[str],
    pdf_path: str,
    system_prompt: str,
    log,
) -> dict:
    """
    Process a PDF page-by-page, calling the LLM once per page to extract text.
    Returns a dictionary matching RawOCRResponse: {"pages": [{"page_num": 1, "text": "..."}, ...]}
    """
    page_images = pdf_to_base64_images(pdf_path)
    log(f"   📄 Extracting raw OCR from {len(page_images)} page(s)…")

    pages_data = []

    for i, img_b64 in enumerate(page_images):
        log(f"   📃 Page {i + 1}/{len(page_images)} OCR…")
        result = call_nvidia_structured(
            api_key=api_key,
            model=model_id,
            system_prompt=system_prompt,
            user_prompt=f"Extract all text from page {i + 1} of this document.",
            image_b64=img_b64,
            schema=None,  # free-form text response
        )
        page_text = sanitize_llm_text(result.get("content", "").strip())
        if page_text:
            pages_data.append({
                "page_num": i + 1,
                "text": page_text
            })
            
    return {"pages": pages_data}


def _assert_question_count(result: dict) -> None:
    """Raise if the model under-delivered relative to its own declared count.

    A restructuring call can return syntactically valid JSON that's
    semantically truncated — total_questions: 7 but only 1 question actually
    populated (observed live: the model stops emitting further questions
    without any truncation/length signal). Passed as `validate=` so
    nvidia_client's existing per-model retry + pool fallback handle recovery;
    no new retry logic needed here.
    """
    declared = result.get("total_questions", 0)
    actual = len(result.get("questions", []))
    if declared and actual < declared:
        raise ValueError(
            f"Incomplete restructuring output: declared total_questions={declared} "
            f"but only {actual} question(s) present"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Question Structuring
# ─────────────────────────────────────────────────────────────────────────────

def structure_questions(
    api_key: str,
    model_id: str | list[str],
    raw_question_json: dict,
    log,
) -> dict:
    log("📋 Structuring question paper hierarchy…")
    
    # Concatenate all question pages into a single string for structuring
    question_text = ""
    for page in raw_question_json.get("pages", []):
        question_text += f"--- Page {page['page_num']} ---\n{page['text']}\n\n"
        
    if not question_text.strip():
        log("   ⚠️ No question text found, returning empty structure.")
        return {"total_questions": 0, "questions": []}
        
    schema = Stage1Response.model_json_schema()
    user_prompt = (
        "## Raw Question Paper Text:\n"
        f"```\n{question_text}\n```\n\n"
        "Please extract the questions and sub-questions from this text and "
        "output them as a perfectly structured JSON matching the schema."
    )
    
    try:
        result = call_nvidia_structured(
            api_key=api_key,
            model=model_id,
            system_prompt=_QUESTION_STRUCTURE_SYSTEM,
            user_prompt=user_prompt,
            image_b64=None,
            schema=schema,
            max_tokens=32768,
            validate=_assert_question_count,
        )
        result = sanitize_json_strings(result)
        log(f"   ✅ Question hierarchy created with {result.get('total_questions', 0)} top-level question(s).")
        return result
    except Exception as exc:
        logger.error(f"Failed to structure questions: {exc}")
        log(f"   ⚠️ Failed to structure questions: {exc}")
        return {"total_questions": 0, "questions": []}


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Student Answer Mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_student_answers(
    api_key: str,
    model_id: str | list[str],
    structured_question: dict,
    raw_student_json: dict,
    log,
) -> dict:
    log("🔄 Mapping student answers to question hierarchy…")
    
    # Concatenate all student pages into a single string for mapping
    student_text = ""
    for page in raw_student_json.get("pages", []):
        student_text += f"--- Page {page['page_num']} ---\n{page['text']}\n\n"

    if not student_text.strip():
        log("   ⚠️ No student text found.")
        return {"total_questions": 0, "questions": []}
        
    if not structured_question.get("questions"):
        log("   ⚠️ No structured question hierarchy available. Mapping skipped.")
        # Fallback: Just return a single question containing all raw text
        return {
            "total_questions": 1,
            "questions": [{
                "question_id": "All",
                "raw_content": student_text,
                "sub_questions": []
            }]
        }
        
    schema = Stage1Response.model_json_schema()
    user_prompt = (
        "## Question Hierarchy (Blueprint):\n"
        "Use this ONLY as a structural guide to know the IDs of the questions and sub-questions. Do NOT copy its text into your output.\n"
        f"```json\n{json.dumps(structured_question, indent=2)}\n```\n\n"
        "## Raw Student Answers:\n"
        f"```\n{student_text}\n```\n\n"
        "Please slot ALL the Raw Student Answers exactly into the provided Question Hierarchy. "
        "Place the student's full paragraphs, equations, and explanations verbatim into the appropriate 'raw_content' or 'content' fields. "
        "DO NOT just extract the headers. Extract the ACTUAL answers written by the student.\n"
        "Ensure no student text is omitted. Output valid JSON matching the schema."
    )
    
    try:
        result = call_nvidia_structured(
            api_key=api_key,
            model=model_id,
            system_prompt=_STUDENT_MAPPING_SYSTEM,
            user_prompt=user_prompt,
            image_b64=None,
            schema=schema,
            max_tokens=32768,
            validate=_assert_question_count,
        )
        result = sanitize_json_strings(result)
        log(f"   ✅ Student mapping complete: {result.get('total_questions', 0)} top-level question(s) mapped.")
        return result
    except Exception as exc:
        logger.error(f"Failed to map student answers: {exc}")
        log(f"   ⚠️ Failed to map student answers: {exc}")
        # Fallback
        return {
            "total_questions": 1,
            "questions": [{
                "question_id": "All (Fallback)",
                "raw_content": student_text,
                "sub_questions": []
            }]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Stage 1 Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def build_question_blueprint(
    api_key: str,
    question_pdf_path: str,
    log,
    model_id_override: str | list[str] | None = None,
) -> dict:
    """Extracts raw OCR from the question PDF and structures it into a hierarchy."""
    log("📤 Building Question Blueprint…")
    question_extraction_model = get_pool("question_extraction", model_id_override)
    restructuring_model = get_pool("restructuring", model_id_override)

    log("📋 Part 1a: Extracting raw OCR from question paper…")
    raw_question_json = extract_raw_ocr(
        api_key=api_key,
        model_id=question_extraction_model,
        pdf_path=question_pdf_path,
        system_prompt=_QUESTION_EXTRACT_SYSTEM,
        log=log
    )

    log("📐 Part 2: Structuring Question Hierarchy…")
    if raw_question_json.get("pages"):
        structured_question = structure_questions(
            api_key=api_key,
            model_id=restructuring_model,
            raw_question_json=raw_question_json,
            log=log
        )
    else:
        log("   ℹ️ Skipping question structuring (no question text).")
        structured_question = {}
        
    return structured_question

def run_student_mapping(
    api_key: str,
    student_pdf_path: str,
    structured_question: dict,
    log,
    model_id_override: str | list[str] | None = None,
    is_master: bool = False,
) -> dict:
    """Extracts OCR from the student PDF and maps it using the provided structured hierarchy."""
    if is_master:
        log("📤 Processing Master Solution…")
    else:
        log("📤 Processing Student Submission…")
        
    student_parsing_model = get_pool("student_parsing", model_id_override)
    restructuring_model = get_pool("restructuring", model_id_override)

    if is_master:
        log("📝 Part 1b: Extracting raw OCR from master solution…")
    else:
        log("📝 Part 1b: Extracting raw OCR from student submission…")
    raw_student_json = extract_raw_ocr(
        api_key=api_key,
        model_id=student_parsing_model,
        pdf_path=student_pdf_path,
        system_prompt=_STUDENT_OCR_SYSTEM,
        log=log
    )

    if is_master:
        log("🧩 Part 3: Mapping Master Solution to Hierarchy…")
    else:
        log("🧩 Part 3: Mapping Student Answers to Hierarchy…")
    
    structured_student_solution = map_student_answers(
        api_key=api_key,
        model_id=restructuring_model,
        structured_question=structured_question,
        raw_student_json=raw_student_json,
        log=log
    )

    return {
        "structured_student_solution": structured_student_solution,
        "raw_student_solution": raw_student_json,
    }

def run_stage1(
    api_key: str,
    model_id: str | list[str] | None,
    student_pdf_path: str,
    question_pdf_path: str | None,
    log,
) -> dict:
    """
    Legacy Stage 1 wrapper: 3-Part Pipeline
    
    Part 1: Raw OCR Extraction (for both student and question PDFs)
    Part 2: Question Structuring
    Part 3: Student Answer Mapping
    
    Returns a dict with:
      - "structured_question": the structured JSON of the questions
      - "structured_student_solution": the structured JSON of the student answers
      - "raw_student_solution": the raw OCR JSON of the student answers
    """
    log("📤 Stage 1 → Legacy 3-Part Pipeline Wrapper…")
    
    if question_pdf_path:
        structured_question = build_question_blueprint(api_key, question_pdf_path, log, model_id)
    else:
        log("   ℹ️ No question paper provided.")
        structured_question = {}
        
    student_result = run_student_mapping(api_key, student_pdf_path, structured_question, log, model_id)
    
    log("✅ Stage 1 complete!")
    
    return {
        "structured_question": structured_question,
        "structured_student_solution": student_result["structured_student_solution"],
        "raw_student_solution": student_result["raw_student_solution"],
    }
