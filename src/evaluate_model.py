from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from src.config import OUTPUT_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE
from src.dataset import build_dataset, tokenize_and_align_labels
from src.utils import compute_metrics

def main():
    dataset = build_dataset(TRAIN_FILE, VALID_FILE, TEST_FILE)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(OUTPUT_DIR))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp_eval", report_to="none"),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(metrics)

if __name__ == "__main__":
    main()