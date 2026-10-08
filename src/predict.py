"""
predict.py — Interactive Prediction Script
Fake Product Review Detection System

This script loads the saved model and TF-IDF vectorizer,
then lets you enter reviews and get Fake/Genuine predictions.

Usage:
    python src/predict.py
"""

import os
import sys
import joblib

# ──────────────────────────────────────────────
# Setup paths
# ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

MODEL_PATH = os.path.join(project_dir, "models", "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(project_dir, "models", "tfidf_vectorizer.pkl")

# Import our text cleaning function
from src.preprocess import clean_text


def load_model_and_vectorizer():
    """
    Load the saved Logistic Regression model and TF-IDF vectorizer.
    
    Returns:
        model: Trained Logistic Regression model.
        vectorizer: Fitted TF-IDF vectorizer.
    """
    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model file not found at: {MODEL_PATH}")
        print(f"   Please run 'python src/train.py' first to train the model.")
        sys.exit(1)
    
    # Check vectorizer file exists
    if not os.path.exists(VECTORIZER_PATH):
        print(f"\n❌ Vectorizer file not found at: {VECTORIZER_PATH}")
        print(f"   Please run 'python src/train.py' first to train the model.")
        sys.exit(1)
    
    print("📦 Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!\n")
    
    return model, vectorizer


def predict_review(review_text, model, vectorizer):
    """
    Predict whether a review is Fake or Genuine.
    
    Args:
        review_text (str): The review to classify.
        model: Trained Logistic Regression model.
        vectorizer: Fitted TF-IDF vectorizer.
    
    Returns:
        prediction (str): 'Fake' or 'Genuine'.
        confidence (float): Prediction confidence (0-100).
    """
    # Clean the text using the same preprocessing as training
    cleaned_text = clean_text(review_text)
    
    if not cleaned_text.strip():
        return "Unknown", 0.0
    
    # Transform using the saved TF-IDF vectorizer
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get label and confidence
    label = "Fake" if prediction == 1 else "Genuine"
    confidence = probabilities[prediction] * 100
    
    return label, confidence


def main():
    """Interactive prediction loop."""
    print("\n" + "═" * 60)
    print("  🔍 FAKE PRODUCT REVIEW DETECTION SYSTEM")
    print("  Prediction Mode")
    print("═" * 60 + "\n")
    
    # Load model
    model, vectorizer = load_model_and_vectorizer()
    
    print("─" * 60)
    print("Enter a product review to check if it's Fake or Genuine.")
    print("Type 'quit' or 'exit' to stop.")
    print("─" * 60)
    
    while True:
        print()
        review = input("📝 Enter your review:\n> ").strip()
        
        # Check for exit
        if review.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Check for empty input
        if not review:
            print("⚠️  Please enter a non-empty review.")
            continue
        
        # Predict
        label, confidence = predict_review(review, model, vectorizer)
        
        if label == "Unknown":
            print("⚠️  Review was empty after cleaning. Please enter a valid review.")
            continue
        
        # Display result
        print()
        print("┌─────────────────────────────────────────┐")
        print(f"│ Review:                                 │")
        
        # Word-wrap the review for display
        display_text = review[:100] + "..." if len(review) > 100 else review
        print(f"│   {display_text}")
        print(f"│                                         │")
        
        # Color-coded result
        if label == "Fake":
            print(f"│ 🚨 Prediction:  FAKE REVIEW             │")
        else:
            print(f"│ ✅ Prediction:  GENUINE REVIEW           │")
        
        print(f"│ 📊 Confidence:  {confidence:.2f}%                  │")
        print("└─────────────────────────────────────────┘")
    
    print()


if __name__ == "__main__":
    main()
