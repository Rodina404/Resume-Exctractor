import json
from pathlib import Path
from src.pdf_parser import extract_text_from_pdf

RESUME_DIR = Path("data/raw/resumes")
OUTPUT_PATH = Path("data/interim/parsed_resumes.jsonl")

def simple_tokenize(text: str):
    # simple starter tokenizer
    return text.replace("\n", " \n ").split()

def parse_resumes():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, pdf_file in enumerate(sorted(RESUME_DIR.glob("*.pdf")), start=1):
            text = extract_text_from_pdf(str(pdf_file))
            tokens = simple_tokenize(text)

            record = {
                "id": f"resume_{i}",
                "file_name": pdf_file.name,
                "text": text,
                "tokens": tokens
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved parsed resumes to {OUTPUT_PATH}")

if __name__ == "__main__":
    parse_resumes()