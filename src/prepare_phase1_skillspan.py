import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_PATH = Path("data/interim/skillspan_converted.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def convert_skillspan_to_jsonl():
    ds = load_dataset("jjzha/skillspan")

    total = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            for row in ds[split]:
                # Adjust keys if dataset format differs
                tokens = row["tokens"]
                ner_tags = row["tags"] if "tags" in row else row["ner_tags"]

                # If tags are integers, you must map them first.
                # Here we assume they are already labels or can be mapped externally.
                if all(isinstance(x, int) for x in ner_tags):
                    continue

                example = {
                    "id": f"skillspan_{split}_{total}",
                    "tokens": tokens,
                    "ner_tags": ner_tags
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1

    print(f"Saved {total} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    convert_skillspan_to_jsonl()