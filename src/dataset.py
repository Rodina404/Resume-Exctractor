import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH
from src.labels import LABEL2ID

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_dataset(train_path, valid_path, test_path):
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    test_rows = load_jsonl(test_path)

    ds = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(valid_rows),
        "test": Dataset.from_list(test_rows),
    })
    return ds

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    aligned_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[labels[word_idx]])
            else:
                current = labels[word_idx]
                if current.startswith("B-"):
                    current = current.replace("B-", "I-")
                label_ids.append(LABEL2ID[current])
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized