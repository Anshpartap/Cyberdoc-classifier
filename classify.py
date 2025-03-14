def classify_document(file_path, threshold):
    """Classifies a single file as Sensitive or Normal."""
    if not file_path.endswith((".pdf", ".docx")):
        return "⚠️ Unsupported file type. Please provide a PDF or DOCX file."

    text = extract_text(file_path)
    if not text:
        return "⚠️ Document contains no readable text."

    # Extract features
    bert_embedding = get_bert_embedding(text)
    lda_feature = get_lda_features(text)

    # Ensure consistent shape before concatenation
    bert_embedding = np.reshape(bert_embedding, (1, -1))
    lda_feature = np.reshape(lda_feature, (1, -1))

    # Combine features
    features = np.hstack((bert_embedding, lda_feature))

    # Predict probability
    probability = xgb_classifier.predict_proba(features)[0][1]  # Probability of Sensitive
    category = "Sensitive" if probability >= threshold else "Normal"

    # Return classification result as a string
    return f"📄 **File:** {os.path.basename(file_path)}\n" \
           f"📌 **Category:** {category}\n" \
           f"🔢 **Probability:** {probability:.2f} (Threshold: {threshold})"
