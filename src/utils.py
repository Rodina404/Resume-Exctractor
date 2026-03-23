import numpy as np
import evaluate
from src.labels import LABEL_LIST

seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_labels = []
        gold_labels = []
        for pred_id, gold_id in zip(pred_seq, label_seq):
            if gold_id == -100:
                continue
            pred_labels.append(LABEL_LIST[pred_id])
            gold_labels.append(LABEL_LIST[gold_id])
        true_predictions.append(pred_labels)
        true_labels.append(gold_labels)

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }