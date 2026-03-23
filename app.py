import streamlit as st
from src.predict_pdf import predict_resume

st.set_page_config(page_title="Resume Extractor", layout="wide")

st.title("Resume Extraction System")

uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])

if uploaded_file is not None:
    temp_path = "temp_resume.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    result = predict_resume(temp_path)

    st.subheader("Skills")
    st.json(result["Skills"])

    st.subheader("Experience")
    st.json(result["experience"])

    st.subheader("Education")
    st.json(result["education"])