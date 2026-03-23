from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from src.config import MODEL_NAME, OUTPUT_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE
from src.dataset import build_dataset, tokenize_and_align_labels
from src.labels import LABEL_LIST, LABEL2ID, ID2LABEL
from src.utils import compute_metrics

def main():
    dataset = build_dataset(TRAIN_FILE, VALID_FILE, TEST_FILE)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    print("TEST METRICS:", test_metrics)

if __name__ == "__main__":
    main()