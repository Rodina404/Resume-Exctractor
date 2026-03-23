# Resume Extractor

A Named Entity Recognition (NER) system for extracting structured information (Skills, Experience, Education) from Resume PDFs.

## 🚀 Architecture & Flow
1. **PDF Parsing:** Extracts text from uploaded resumes using `pdfplumber` (`src/pdf_parser.py`).
2. **Model Inference:** Text is passed to a domain-adapted language model, `jobbert-base-cased` using a HuggingFace Token Classification pipeline (`src/predict_pdf.py`).
3. **Post-processing:** Token-level predictions are aggregated and unified into readable entities (`src/postprocess.py`).
4. **Web UI:** A Streamlit application is provided as a friendly user interface for immediate result visualization (`app.py`).

## 📊 Training & Data Results
The model is fine-tuned on parsed dataset containing resumes specifically tailored for skill extraction (`data/processed/skills_only`).
- **Base Model:** `jjzha/jobbert-base-cased`
- **Training Epochs:** 3
- **Batch Size:** 8
- **Learning Rate:** 2e-5

**Final Evaluation Metrics (Epoch 3.0 / Step 1800)**:
- **Accuracy:** 95.14%
- **F1 Score:** 0.4828
- **Precision:** 0.4455
- **Recall:** 0.5271

## 📁 Data Preparation
Data generation scripts are included in `src/` to handle preprocessing of datasets (`prepare_phase1_skillspan.py`, `prepare_phase2_resumes.py`, `prepare_skillspan_skills_only.py`). Preprocessed sets are formatted in JSONL format for HuggingFace Dataset consumption.

## 🌱 Future Work (To-Be-Continued)
- Improve model F1 Score and Precision across underrepresented labels (e.g. multi-word complex skills, long sentences).
- Integrate continuous training pipeline or human-in-the-loop to evaluate user corrections.
- Extend capability to seamlessly extract and map contact information, locations, and soft skills with higher confidence.
- Expand data augmentation approaches to improve recall of rare technical skills.

## ⚙️ Setup and Usage

1. Create a virtual environment and install dependencies:
```bash
python -m venv env
env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. Run the Web User Interface via Streamlit:
```bash
streamlit run app.py
```
