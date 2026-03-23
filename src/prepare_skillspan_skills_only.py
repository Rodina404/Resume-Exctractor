import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/processed/skills_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_skillspan_skills_only():
    """
    Downloads jjzha/skillspan, maps all valid NER tags to generic B-SKILL / I-SKILL or O,
    and segregates into strict splits.
    """
    ds = load_dataset("jjzha/skillspan")

    for split in ["train", "validation", "test"]:
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                tokens = row["tokens"]
                # Huggingface dataset dictates it usually as tags or ner_tags
                original_tags = row.get("tags", row.get("ner_tags", []))
                
                # Check mapping dictionary if stored in dataset features
                features = ds[split].features
                label_feature = features.get("tags") or features.get("ner_tags")
                
                mapped_tags = []
                for tag in original_tags:
                    if hasattr(label_feature, "int2str"):
                        tag_str = label_feature.int2str(tag).upper()
                    else:
                        tag_str = str(tag).upper()
                    
                    # Strictly binary isolate to Skill classes, destroying Knowledge/other tags
                    if tag_str == "O" or tag == 0:
                        mapped_tags.append("O")
                    elif tag_str.startswith("B-"):
                        mapped_tags.append("B-SKILL")
                    elif tag_str.startswith("I-"):
                        mapped_tags.append("I-SKILL")
                    else:
                        mapped_tags.append("O")
                        
                example = {
                    "id": f"skillspan_{split}_{total}",
                    "tokens": tokens,
                    "ner_tags": mapped_tags
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1
                
        print(f"Saved {total} examples to {output_path}")

def normalize_tags(tags):
    """
    Normalizes dataset tags (integers or strings) to strictly binary B-SKILL, I-SKILL, or O.
    jjzha/skillspan tags_skill feature contains: 0 (O), 1 (B-Skill), 2 (I-Skill).
    """
    mapped = []
    for tag in tags:
        if isinstance(tag, int):
            if tag == 1:
                mapped.append("B-SKILL")
            elif tag == 2:
                mapped.append("I-SKILL")
            else:
                mapped.append("O")
        else:
            tag_str = str(tag).upper()
            if tag_str.startswith("B-"):
                mapped.append("B-SKILL")
            elif tag_str.startswith("I-"):
                mapped.append("I-SKILL")
            else:
                mapped.append("O")
    return mapped
print(ds.keys())
def convert_split(rows, output_file, split_name):
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            tokens = row["tokens"]
            tags = row["tags_skill"]  # حسب dataset card
            ner_tags = normalize_tags(tags)

            if len(tokens) != len(ner_tags):
                continue

            example = {
                "id": f"{split_name}_{idx}",
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"{split_name}: saved {count} rows to {output_file}")

def main():
    ds = load_dataset("jjzha/skillspan")

    # dataset card بيعرض train / dev / test
    convert_split(ds["train"], OUTPUT_DIR / "train.jsonl", "train")
    convert_split(ds["dev"], OUTPUT_DIR / "valid.jsonl", "valid")
    convert_split(ds["test"], OUTPUT_DIR / "test.jsonl", "test")

if __name__ == "__main__":
    main()