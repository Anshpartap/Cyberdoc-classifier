import streamlit as st
import os
import tempfile
from classify import classify_document  # Import classify function

# Streamlit page configuration
st.set_page_config(page_title="Cyber Document Classifier", page_icon="🛡️")

# App Title
st.title("🛡️ Cyber Document Classifier")
st.write("Upload a PDF or DOCX file, and the model will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

# Threshold Input
threshold = st.slider("Select Sensitivity Threshold", 0.0, 1.0, 0.6)

# Process uploaded file
if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    st.success(f"✅ File {uploaded_file.name} uploaded successfully!")

    # Perform classification
    result = classify_document(temp_path, threshold)

    # Display classification result
    st.subheader("📊 Classification Result:")
    st.write(result)
