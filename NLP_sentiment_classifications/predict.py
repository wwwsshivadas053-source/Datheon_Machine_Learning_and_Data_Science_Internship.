"""
Interactive Sentiment Predictor CLI
===================================
Utility to predict sentiment (positive, neutral, negative) for custom customer reviews
using the trained TF-IDF vectorizer and classification model.

Usage:
    python predict.py "The product exceeded my expectations and works great!"
    python predict.py --interactive
"""

import sys
import re
import os
import argparse
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_FILE = os.path.join(os.path.dirname(__file__), "sentiment_classifier.joblib")
VECTORIZER_FILE = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.joblib")

# Initialize Lemmatizer if NLTK is installed
lemmatizer = None
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("tests")
except Exception:
    lemmatizer = None

NEGATION_WORDS = {"no", "nor", "not", "never", "n't"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

def preprocess_text(text: str) -> str:
    """Normalize input review preserving negation words."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        if token in STOP_WORDS:
            continue
        if lemmatizer is not None:
            try:
                token = lemmatizer.lemmatize(token)
            except Exception:
                pass
        tokens.append(token)
    return " ".join(tokens)

def load_artifacts():
    """Load model and TF-IDF vectorizer from disk."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(
            f"Artifacts missing! Please ensure '{MODEL_FILE}' and '{VECTORIZER_FILE}' exist."
        )
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

def predict_review(text: str, model, vectorizer):
    """Run prediction on a review string and display probabilities."""
    cleaned = preprocess_text(text)
    if not cleaned:
        print("[!] Warning: Review text became empty after stopword removal and cleaning.")
        return None

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    print("\n" + "=" * 60)
    print(f"Review: \"{text}\"")
    print(f"Cleaned Tokens: {cleaned}")
    print(f"Predicted Sentiment: {prediction.upper()}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        print("-" * 60)
        print("Confidence Breakdown:")
        for cls, prob in zip(model.classes_, probabilities):
            bar = "#" * int(prob * 30)
            print(f"  - {cls.capitalize():<8}: {prob:>6.2%}  [{bar:<30}]")
    print("=" * 60 + "\n")
    return prediction

def main():
    parser = argparse.ArgumentParser(description="NLP Sentiment Classification Inference CLI")
    parser.add_argument("review", nargs="?", help="Review text to classify (wrap in quotes)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prompt loop")
    args = parser.parse_args()

    model, vectorizer = load_artifacts()

    if args.interactive or not args.review:
        print("\n--- Interactive Customer Review Sentiment Predictor ---")
        print("Type your review and press Enter. Type 'exit' or 'q' to quit.\n")
        while True:
            try:
                user_input = input("Enter review >> ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting...")
                    break
                if not user_input:
                    continue
                predict_review(user_input, model, vectorizer)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
    else:
        predict_review(args.review, model, vectorizer)

if __name__ == "__main__":
    main()
