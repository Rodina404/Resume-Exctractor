from transformers import pipeline
from src.config import OUTPUT_DIR
from src.pdf_parser import extract_text_from_pdf
from src.postprocess import merge_entities, to_final_output

def load_pipeline():
    return pipeline(
        "token-classification",
        model=str(OUTPUT_DIR),
        tokenizer=str(OUTPUT_DIR),
        aggregation_strategy="simple",
    )

def predict_resume(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    ner = load_pipeline()
    raw_preds = ner(text)
    entities = []

    for p in raw_preds:
        entities.append({
            "entity_group": p["entity_group"],
            "word": p["word"],
            "score": p["score"],
            "start": p["start"],
            "end": p["end"],
        })

    merged = merge_entities(entities)
    result = to_final_output(merged)
    return result

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1]
    result = predict_resume(pdf_path)
    print(result)