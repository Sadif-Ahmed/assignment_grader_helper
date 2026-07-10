import argparse
import json
import logging
import sys
import io
import os
from stage1 import build_question_blueprint, run_student_mapping
from stage2 import run_stage2
from model_pools import MODEL_POOLS

logging.basicConfig(level=logging.INFO)

DEFAULT_CRITERIA = [
    "Answered all questions?",
    "Answers are correct?",
    "Overall remarks (A/B/C)?",
]

def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 (Evaluation) of the checker pipeline.")
    parser.add_argument("--api-key", required=True, help="NVIDIA API Key")
    parser.add_argument("--model-id", default=None, help="Optional: override all tasks with a single model ID.")
    parser.add_argument("--question-json", required=True, help="Path to pre-computed question blueprint JSON")
    parser.add_argument("--solution-json", help="Path to pre-computed master solution blueprint JSON")
    parser.add_argument("--master-pdf", help="Path to master solution PDF (if solution-json is not provided)")
    parser.add_argument("--student-json", required=True, help="Path to pre-computed student JSON (stage1 output)")
    parser.add_argument("--output", default="final_evaluation.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    if not args.solution_json and not args.master_pdf:
        parser.error("Must provide either --solution-json or --master-pdf")
    
    if args.model_id:
        print(f"Using single model override: {args.model_id}")
    else:
        print("Using per-task models:")
        for task, model in MODEL_POOLS.items():
            if task in ("student_parsing", "evaluation", "question_extraction", "restructuring"):
                print(f"  {task}: {model}")

    def log(msg):
        print(msg)
        
    # ── Load Question Blueprint ──
    print(f"\nLoading question blueprint from {args.question_json}...")
    try:
        with open(args.question_json, "r", encoding="utf-8") as f:
            question_blueprint = json.load(f)
    except Exception as e:
        print(f"Error loading question blueprint: {e}")
        return

    # ── Load or Build Solution Blueprint ──
    solution_blueprint = {}
    if args.solution_json and os.path.exists(args.solution_json):
        print(f"Loading pre-computed master solution from {args.solution_json}...")
        try:
            with open(args.solution_json, "r", encoding="utf-8") as f:
                solution_data = json.load(f)
                solution_blueprint = solution_data.get("structured_student_solution", solution_data)
        except Exception as e:
            print(f"Error loading solution blueprint: {e}")
            return
    elif args.master_pdf:
        print(f"Building solution blueprint from master PDF: {args.master_pdf}...")
        try:
            master_result = run_student_mapping(
                api_key=args.api_key,
                student_pdf_path=args.master_pdf,
                structured_question=question_blueprint,
                log=log,
                model_id_override=args.model_id,
            )
            solution_blueprint = master_result.get("structured_student_solution", {})
            
            # Save it for future runs
            blueprint_file = "solution_blueprint.json"
            with open(blueprint_file, "w", encoding="utf-8") as f:
                json.dump(master_result, f, indent=2, ensure_ascii=False)
            print(f"Solution blueprint saved to {blueprint_file}.\n")
        except Exception as e:
            print(f"Error building solution blueprint: {e}")
            return

    # ── Load Student Solution ──
    print(f"Loading student submission from {args.student_json}...")
    try:
        with open(args.student_json, "r", encoding="utf-8") as f:
            student_result = json.load(f)
        student_answers = student_result.get("structured_student_solution", student_result)
        print("Student submission loaded successfully.\n")
    except Exception as e:
        print(f"Error loading student json: {e}")
        return
        
    # ── Stage 2: Evaluate (text-based) ──
    print("Running Stage 2 (evaluation)...")
    try:
        result = run_stage2(
            api_key=args.api_key,
            model_id=args.model_id or MODEL_POOLS["evaluation"],
            master_map=solution_blueprint,
            student_map=student_answers,
            criteria=DEFAULT_CRITERIA,
            log=log,
            question_context=question_blueprint,
        )
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Stage 2 completed successfully! Evaluation saved to {args.output}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Fix Windows console encoding for emoji output — only when run as a
    # script, so importing this module doesn't hijack the caller's stdout.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
