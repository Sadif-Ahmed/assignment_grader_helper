import argparse
import json
import logging
import sys
import io
from stage1 import build_question_blueprint, run_student_mapping
from model_pools import MODEL_POOLS

logging.basicConfig(level=logging.INFO)

# Fix Windows console encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description="Run only Stage 1 of the checker pipeline.")
    parser.add_argument("--api-key", required=True, help="NVIDIA API Key")
    parser.add_argument("--model-id", default=None, help="Optional: override all tasks with a single model ID.")
    parser.add_argument("--student-pdf", required=True, help="Path to student PDF")
    parser.add_argument("--question-pdf", help="Path to optional question PDF")
    parser.add_argument("--question-json", help="Path to pre-computed question blueprint JSON")
    parser.add_argument("--output", default="stage1_output.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    if args.model_id:
        print(f"Using single model override: {args.model_id}")
    else:
        print("Using per-task models:")
        for task, model in MODEL_POOLS.items():
            if task in ("question_extraction", "student_parsing", "restructuring"):
                print(f"  {task}: {model}")
    
    def log(msg):
        print(msg)
        
    print(f"\nStarting Stage 1 processing for {args.student_pdf}...")
    try:
        # ── Part 1: Blueprint Generation ──
        structured_question = {}
        if args.question_json:
            print(f"Loading pre-computed question blueprint from {args.question_json}...")
            with open(args.question_json, "r", encoding="utf-8") as f:
                structured_question = json.load(f)
        elif args.question_pdf:
            print(f"Building question blueprint from {args.question_pdf}...")
            structured_question = build_question_blueprint(
                api_key=args.api_key,
                question_pdf_path=args.question_pdf,
                log=log,
                model_id_override=args.model_id
            )
            blueprint_file = "question_blueprint.json"
            with open(blueprint_file, "w", encoding="utf-8") as f:
                json.dump(structured_question, f, indent=2, ensure_ascii=False)
            print(f"Question blueprint saved to {blueprint_file}")
        
        # ── Part 2: Student Mapping ──
        print(f"\nProcessing student submission...")
        student_result = run_student_mapping(
            api_key=args.api_key,
            student_pdf_path=args.student_pdf,
            structured_question=structured_question,
            log=log,
            model_id_override=args.model_id
        )
        
        # Combine the output format to match legacy behavior
        final_output = {
            "structured_question": structured_question,
            "structured_student_solution": student_result["structured_student_solution"],
            "raw_student_solution": student_result["raw_student_solution"],
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        print(f"\nStage 1 completed successfully! Output saved to {args.output}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
