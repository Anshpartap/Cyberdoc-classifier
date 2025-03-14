import os
import numpy as np
import torch
import docx
import PyPDF2
import joblib
import shap
import xgboost as xgb
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer

# Paths to trained models
MODEL_PATH = "./xgb_classifier.pkl"
LDA_MODEL_PATH = "./lda_model.pkl"
VECTORIZER_PATH = "./vectorizer.pkl"
TOKENIZER_PATH = "bert-base-uncased"
BERT_MODEL_PATH = "bert-base-uncased"

# Check if all required model files exist
for model_file in [MODEL_PATH, LDA_MODEL_PATH, VECTORIZER_PATH]:
    if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
        raise FileNotFoundError(f"❌ Required model file {model_file} is missing or empty. Retrain the model first.")

# Load trained models
xgb_classifier = joblib.load(MODEL_PATH)
lda_model = joblib.load(LDA_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

# Load SHAP Explainer for feature interpretation
explainer = shap.TreeExplainer(xgb_classifier)

# Function to extract text from PDF or DOCX
def extract_text(file_path):
    """Extracts text from PDF or DOCX file."""
    text = ""
    try:
        if file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() if page.extract_text() else ''
    except Exception as e:
        print(f"⚠️ Error extracting text from {file_path}: {e}")
    return text.strip()

# Extract BERT embeddings
def get_bert_embedding(text):
    """Extracts BERT embeddings for input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

# Extract LDA topic features
def get_lda_features(text):
    """Extracts LDA topic distribution features."""
    text_vectorized = vectorizer.transform([text])
    return lda_model.transform(text_vectorized)

# Generate SHAP Explanation (Fixed)
def explain_shap(features):
    """Generate SHAP explanation for model prediction."""
    features = np.array(features).reshape(1, -1)  # Ensure correct shape
    shap_values = explainer.shap_values(features)

    # ✅ Fix: Handle different SHAP formats
    if isinstance(explainer.expected_value, np.ndarray):
        expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value
    else:
        expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return shap.force_plot(expected_value, shap_values, feature_names=[f'Feature {i+1}' for i in range(features.shape[1])])

# Classify a single document
def classify_document(file_path, threshold):
    """Classifies a single file as Sensitive or Normal."""
    if not file_path.endswith((".pdf", ".docx")):
        print("⚠️ Unsupported file type. Please provide a PDF or DOCX file.")
        return

    text = extract_text(file_path)
    if not text:
        print("⚠️ Document contains no readable text.")
        return

    # Extract features
    bert_embedding = get_bert_embedding(text)
    lda_feature = get_lda_features(text)

    # ✅ FIX: Ensure consistent shape before concatenation
    features = np.hstack((bert_embedding.reshape(1, -1), lda_feature))

    # Predict probability
    probability = xgb_classifier.predict_proba(features)[0][1]  # Probability of Sensitive
    category = "Sensitive" if probability >= threshold else "Normal"

    print("\n==================================================")
    print(f"📄 File: {os.path.basename(file_path)}")
    print(f"📌 Category: {category}")
    print(f"🔢 Probability: {probability:.2f} (Threshold: {threshold})")
    print("==================================================")

    # Generate SHAP explanation
    shap_plot = explain_shap(features)
    shap.save_html(f"shap_explanation_{os.path.basename(file_path)}.html", shap_plot)
    print(f"✅ SHAP explanation saved as shap_explanation_{os.path.basename(file_path)}.html")

# Classify all files in a folder
def classify_folder(folder_path, threshold):
    """Scans a folder and classifies each document."""
    print(f"\n📂 Scanning folder: {folder_path}...\n")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith((".pdf", ".docx")):
            classify_document(file_path, threshold)

# Main script execution
if __name__ == "__main__":
    path = input("Enter file or folder path to classify: ").strip()
    threshold = float(input("Enter sensitivity threshold (e.g., 0.6 for 60%): "))

    if os.path.isdir(path):
        classify_folder(path, threshold)
    elif os.path.isfile(path) and path.endswith((".pdf", ".docx")):
        classify_document(path, threshold)
    else:
        print("⚠️ Invalid input. Please provide a valid file or folder path.")
