from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jobbert_resume_ner"

MODEL_NAME = "jjzha/jobbert-base-cased"
MAX_LENGTH = 256

TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"