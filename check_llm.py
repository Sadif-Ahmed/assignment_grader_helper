import argparse
import json
import logging
import sys

from nvidia_client import call_nvidia_structured, pdf_to_base64_images

logging.basicConfig(
    filename="error.log",
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test NVIDIA NIM connection and PDF upload.")
    parser.add_argument("pdf_path", help="Path to the PDF file to test.")
    parser.add_argument("--api-key", required=True, help="NVIDIA API Key")
    parser.add_argument("--model", default="meta/llama-3.2-90b-vision-instruct", help="NVIDIA NIM Model ID")

    args = parser.parse_args()

    print(f"Testing Model: {args.model}")
    print(f"PDF File: {args.pdf_path}")
    print("Sending request to LLM (with PDF to Image conversion)...")

    system_prompt = '''Act as an advanced document parser. Extract a comprehensive textual map of this submission. Identify total answered questions, accurately label sub-questions, and transcribe handwritten text/code completely.
    Do NOT evaluate correctness. Return a structured JSON object.'''
    user_prompt = "Please provide a brief summary of this document."

    try:
        # Convert PDF to images and send only the first page as a quick test
        page_images = pdf_to_base64_images(args.pdf_path)
        print(f"PDF has {len(page_images)} page(s). Sending page 1 for test…")

        result = call_nvidia_structured(
            api_key=args.api_key,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_b64=page_images[0] if page_images else None,
            schema=None, # No schema for this simple test
        )
        print("\n✅ Success! Response:")
        print("=" * 40)
        print(result.get("content", json.dumps(result, indent=2)))
        print("=" * 40)
    except Exception as e:
        logger.exception("check_llm.py test failed")
        print(f"\n❌ Failed: {e}")
        print("Check error.log for full details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

