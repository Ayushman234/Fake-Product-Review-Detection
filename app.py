"""
app.py — Streamlit Web Interface
Fake Product Review Detection System

A simple web interface for detecting fake product reviews.

Usage:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import joblib

# ──────────────────────────────────────────────
# Setup paths
# ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

MODEL_PATH = os.path.join(script_dir, "models", "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "models", "tfidf_vectorizer.pkl")

from src.preprocess import clean_text


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered",
)


# ──────────────────────────────────────────────
# Load Model (cached so it only loads once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved model and vectorizer."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found at: {MODEL_PATH}\n\n"
            f"Please run `python src/train.py` first to train the model."
        )
        st.stop()
    
    if not os.path.exists(VECTORIZER_PATH):
        st.error(
            f"❌ Vectorizer file not found at: {VECTORIZER_PATH}\n\n"
            f"Please run `python src/train.py` first to train the model."
        )
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ──────────────────────────────────────────────
# App UI
# ──────────────────────────────────────────────
def main():
    # Header
    st.title("🔍 Fake Product Review Detector")
    st.markdown(
        "Detect whether a product review is **Fake** or **Genuine** "
        "using Machine Learning (NLP + TF-IDF + Logistic Regression)."
    )
    st.markdown("---")
    
    # Load model
    model, vectorizer = load_model()
    
    # Input area
    st.subheader("📝 Enter a Product Review")
    review_text = st.text_area(
        label="Type or paste a product review below:",
        height=150,
        placeholder="Example: This product is absolutely amazing and works perfectly. Highly recommended!",
    )
    
    # Predict button
    predict_clicked = st.button("🔎 Detect Review", type="primary", use_container_width=True)
    
    if predict_clicked:
        if not review_text.strip():
            st.warning("⚠️ Please enter a review before clicking Detect.")
        else:
            # Clean and predict
            cleaned = clean_text(review_text)
            
            if not cleaned.strip():
                st.warning("⚠️ Review was empty after cleaning. Please enter a valid review.")
            else:
                text_tfidf = vectorizer.transform([cleaned])
                prediction = model.predict(text_tfidf)[0]
                probabilities = model.predict_proba(text_tfidf)[0]
                confidence = probabilities[prediction] * 100
                
                st.markdown("---")
                st.subheader("📊 Result")
                
                if prediction == 1:
                    # Fake review
                    st.error("🚨 **FAKE REVIEW DETECTED**")
                    st.metric(
                        label="Prediction",
                        value="Fake Review",
                        delta=f"{confidence:.1f}% confidence",
                        delta_color="inverse",
                    )
                else:
                    # Genuine review
                    st.success("✅ **GENUINE REVIEW**")
                    st.metric(
                        label="Prediction",
                        value="Genuine Review",
                        delta=f"{confidence:.1f}% confidence",
                        delta_color="normal",
                    )
                
                # Details expander
                with st.expander("🔬 Details"):
                    st.write(f"**Original Review:** {review_text[:500]}")
                    st.write(f"**Cleaned Text:** {cleaned[:500]}")
                    st.write(f"**Fake Probability:** {probabilities[1] * 100:.2f}%")
                    st.write(f"**Genuine Probability:** {probabilities[0] * 100:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Logistic Regression + TF-IDF Vectorization | "
        "Fake Product Review Detection System"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
