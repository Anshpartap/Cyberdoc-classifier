import streamlit as st
import os
import numpy as np
import joblib
from classify import classify_document  # Import function from classify.py
from tempfile import NamedTemporaryFile

# Streamlit page configuration
st.set_page_config(page_title="Cyber Document Classifier", page_icon="🛡️", layout="wide")

# App Title
st.title("🛡️ Cyber Document Classifier")
st.write("Upload a PDF or DOCX file, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

# Sensitivity threshold input
threshold = st.slider("Select Sensitivity Threshold", 0.0, 1.0, 0.6)

if uploaded_file is not None:
    st.success(f"✅ File {uploaded_file.name} uploaded successfully!")

    # Save file temporarily
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    # Classify the document
    with st.spinner("🔍 Analyzing document... Please wait."):
        result = classify_document(temp_path, threshold)

    # Display classification results
    st.subheader("📊 Classification Result:")
    st.write(result)

    # Clean up temp file
    os.remove(temp_path)
