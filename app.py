import streamlit as st
import joblib
import os

st.set_page_config(page_title="Spam Classifier", page_icon="📨", layout="centered")

MODEL_PATH = os.path.join("models", "spam_nb_pipeline.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.title("📨 Spam Email/SMS Classifier")
st.write("Type a message and the model will classify it as spam or ham.")

user_text = st.text_area("Message", height=150, placeholder="Enter SMS or email content here...")
if st.button("Classify"):
    if not user_text.strip():
        st.warning("Please enter a message.")
    else:
        pred = pipeline.predict([user_text])[0]
        proba = pipeline.predict_proba([user_text])[0] if hasattr(pipeline, "predict_proba") else None

        if pred == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM")

        if proba is not None:
            # Show probability nicely
            labels = pipeline.classes_
            probs = {labels[i]: float(proba[i]) for i in range(len(labels))}
            st.write("Confidence:", probs)
