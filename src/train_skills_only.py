import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src.labels_skills_only import LABEL_LIST, LABEL2ID, ID2LABEL
from src.utils_metrics import compute_metrics_factory

MODEL_NAME = "jjzha/jobbert-base-cased"
DATA_DIR = Path("data/processed/skills_only")
OUTPUT_DIR = Path("outputs/jobbert_skills_only")
MAX_LENGTH = 256


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_dataset():
    return DatasetDict({
        "train": Dataset.from_list(load_jsonl(DATA_DIR / "train.jsonl")),
        "validation": Dataset.from_list(load_jsonl(DATA_DIR / "valid.jsonl")),
        "test": Dataset.from_list(load_jsonl(DATA_DIR / "test.jsonl")),
    })


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    all_labels = []

    for i, doc_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[doc_labels[word_idx]])
            else:
                current = doc_labels[word_idx]
                if current.startswith("B-"):
                    current = current.replace("B-", "I-")
                label_ids.append(LABEL2ID[current])
            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def main():
    ds = build_dataset()
    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_factory(LABEL_LIST),
        processing_class=tokenizer,
    )

    trainer.train()

    print("\n=== TEST RESULTS ===")
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    print(test_metrics)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()