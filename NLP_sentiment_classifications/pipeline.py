"""
NLP Sentiment Classification Pipeline (End-to-End Training & Evaluation)
========================================================================
Downloads the customer reviews dataset, performs text normalization,
extracts TF-IDF n-gram features, trains Multinomial Naive Bayes and
Logistic Regression models, reports evaluation metrics, generates visual
plots, and persists model artifacts.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Dataset Acquisition
DATA_URL = (
    "https://huggingface.co/datasets/davidberg/sentiment-reviews/"
    "resolve/main/EcoPreprocessed.csv"
)

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    print(f"[*] Fetching dataset from: {url}")
    df = pd.read_csv(url)
    print(f"[+] Loaded {len(df):,} records with columns: {df.columns.tolist()}")
    return df

# 2. Text Normalization & Lemmatization Setup
lemmatizer = None
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("reviews")
    print("[+] NLTK WordNet Lemmatizer successfully initialized.")
except Exception as exc:
    print(f"[!] Warning: Lemmatizer fallback enabled ({exc})")
    lemmatizer = None

NEGATION_WORDS = {"no", "nor", "not", "never", "n't"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

def preprocess_text(text: str) -> str:
    """Normalize text while preserving critical negation semantics."""
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

def main():
    # Load and clean
    df = load_data()
    data = df[["review", "division"]].dropna().copy()
    data["review"] = data["review"].astype(str).str.strip()
    data["division"] = data["division"].astype(str).str.strip().str.lower()
    data = data[data["review"].ne("") & data["division"].ne("")]
    data = data.drop_duplicates(subset=["review", "division"]).reset_index(drop=True)
    print(f"[+] Cleaned unique review records: {len(data):,}")

    data["clean_review"] = data["review"].apply(preprocess_text)
    data = data[data["clean_review"].str.strip().ne("")].reset_index(drop=True)
    print(f"[+] Valid reviews after normalization: {len(data):,}")

    # Stratified Train/Test Split
    X = data["clean_review"]
    y = data["division"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[+] Partitioned: {len(X_train):,} train instances, {len(X_test):,} test instances.")

    # TF-IDF Vectorization
    print("[*] Fitting TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model Training
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}
    predictions = {}
    results = []

    print("[*] Training candidate classifiers...")
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        trained_models[name] = model
        predictions[name] = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0)
        })

    results_df = pd.DataFrame(results).sort_values(by="F1-score (macro)", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 50)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 50)
    print(results_df.to_string(index=False))

    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]
    best_preds = predictions[best_model_name]
    labels = sorted(y.unique())

    print("\n" + "=" * 50)
    print(f"CLASSIFICATION REPORT ({best_model_name})")
    print("=" * 50)
    print(classification_report(y_test, best_preds, labels=labels, zero_division=0))

    # Persist artifacts
    model_path = os.path.join(BASE_DIR, "sentiment_classifier.joblib")
    vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"[+] Saved serialized model: {model_path}")
    print(f"[+] Saved serialized vectorizer: {vectorizer_path}")
    print("[+] Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
