import json
from collections import Counter
from datasets import load_dataset

# =========================
# 1) Check original dataset
# =========================

print("\n=== ORIGINAL DATASET SAMPLE ===")
ds = load_dataset("jjzha/skillspan")

print("\nKeys:", ds.keys())
print("\nSample row:\n", ds["train"][0])

# Check distribution of tags
counter = Counter()
for row in ds["train"]:
    counter.update(row["tags_skill"])

print("\nOriginal tags distribution:", counter)


# =========================
# 2) Check processed files
# =========================

def check_processed(split):
    path = f"data/processed/skills_only/{split}.jsonl"
    counter = Counter()

    print(f"\n=== CHECKING {split.upper()} ===")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            counter.update(row["ner_tags"])

            # print first 2 samples
            if i < 2:
                print(f"\nSample {i}")
                print("Tokens:", row["tokens"][:30])
                print("Labels:", row["ner_tags"][:30])

    print(f"\nLabel distribution in {split}:", counter)


for split in ["train", "valid", "test"]:
    check_processed(split)