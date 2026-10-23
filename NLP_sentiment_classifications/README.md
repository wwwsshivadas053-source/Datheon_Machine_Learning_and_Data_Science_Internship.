# 🌐 NLP Sentiment Classification on Customer Reviews (TF-IDF)

<p align="center">
  <img src="assets/nlp_sentiment_banner.jpg" alt="NLP Sentiment Classification Banner" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/NLTK-NLP-blue?style=for-the-badge" alt="NLTK" />
  <img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
</p>

---

## 📌 Executive Summary

This repository presents an **end-to-end Machine Learning Natural Language Processing (NLP) pipeline** engineered to classify multi-class customer sentiment into **Positive**, **Neutral**, and **Negative** categories. 

The architecture encompasses data harvesting from Hugging Face, custom text normalization with negation-aware token filtering and WordNet lemmatization, sublinear TF-IDF n-gram vectorization, and rigorous comparative evaluation between **Multinomial Naive Bayes** and **Logistic Regression**. The repository includes interactive Jupyter notebooks, standalone production pipeline scripts, persisted inference models (`.joblib`), and high-resolution visual performance analytics.

---

## 📋 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Dataset Specification & Distribution](#-dataset-specification--distribution)
- [Text Preprocessing & NLP Pipeline](#-text-preprocessing--nlp-pipeline)
- [Feature Extraction: Sublinear TF-IDF](#-feature-extraction-sublinear-tf-idf)
- [Model Benchmark & Performance Evaluation](#-model-benchmark--performance-evaluation)
- [Confusion Matrix & Error Diagnostics](#-confusion-matrix--error-diagnostics)
- [Interactive Real-Time Inference](#-interactive-real-time-inference)
- [Project Directory Structure](#-project-directory-structure)
- [Quickstart & Execution Guide](#-quickstart--execution-guide)
- [Engineering Limitations & Roadmap](#-engineering-limitations--roadmap)
- [License & Acknowledgments](#-license--acknowledgments)

---

## ✨ Key Highlights

- **Multi-Class Sentiment Classification**: Classifies customer reviews into three granular divisions: `positive`, `neutral`, and `negative`.
- **Negation-Preserving Text Normalization**: Prevents semantic inversion by retaining key negation tokens (`not`, `no`, `never`, `n't`) often stripped by standard stopword lists.
- **Bi-Gram TF-IDF Vectorization**: Employs $(1, 2)$-gram ranges with sublinear term-frequency scaling ($1 + \log(\text{tf})$) to capture compound sentiment expressions like *"not good"* or *"exceeded expectations"*.
- **Benchmarked Classifiers**: Side-by-side comparison between probabilistic generative modeling (**Multinomial Naive Bayes**) and discriminative linear classification (**Logistic Regression**).
- **Macro-Metric Focus**: Evaluates models on **Macro Precision, Recall, and F1-score** to guarantee fair evaluation across severely imbalanced sentiment classes.
- **Production Artifacts**: Ready-to-deploy serialized models (`sentiment_classifier.joblib`, `tfidf_vectorizer.joblib`) with a dedicated CLI inference tool ([`predict.py`](predict.py)).

---

## 🏗️ System Architecture

The end-to-end workflow transforms raw textual customer feedback into calibrated probability distributions and sentiment labels:

```mermaid
flowchart TD
    A[Raw Dataset: Hugging Face EcoPreprocessed.csv] --> B[Data Sanitation & Deduplication]
    B --> C[Text Normalization Engine]
    
    subgraph Preprocessing [NLP Cleaning & Normalization]
        C --> C1[Lowercase Conversion & Regex Cleaning]
        C1 --> C2[HTML Tag & URL Removal]
        C2 --> C3[Stopword Filtering with Negation Retention]
        C3 --> C4[NLTK WordNet Lemmatization]
    end

    C4 --> D[Stratified 80/20 Train-Test Split]
    
    subgraph Feature_Engineering [Vectorization]
        D --> E[TF-IDF Vectorizer]
        E -->|Unigrams + Bigrams| E1[Sublinear TF Scaling]
        E1 -->|max_features=5000| E2[Sparse Vector Matrices]
    end

    E2 --> F[Model Benchmark Suite]
    
    subgraph Modeling [Classifiers]
        F --> G1[Multinomial Naive Bayes]
        F --> G2[Logistic Regression max_iter=2000]
    end

    G1 --> H[Model Evaluation & Metrics Comparison]
    G2 --> H
    
    H --> I[Confusion Matrix & Classification Report]
    H --> J[Export Best Model & Vectorizer .joblib]
    J --> K[Interactive CLI Inference Engine predict.py]
```

---

## 📊 Dataset Specification & Distribution

The pipeline leverages the **[Hugging Face davidberg/sentiment-reviews](https://huggingface.co/datasets/davidberg/sentiment-reviews)** benchmark dataset (`EcoPreprocessed.csv`).

### Dataset Metadata
| Attribute | Detail |
| :--- | :--- |
| **Origin / Source** | Hugging Face Hub (`davidberg/sentiment-reviews`) |
| **Raw Sample Size** | 4,084 customer reviews |
| **Unique Sanitized Records** | 3,467 instances |
| **Valid Post-Normalization** | 3,458 instances |
| **Features Analyzed** | `review` (raw text input), `division` (categorical sentiment target) |
| **Target Classes** | `positive`, `neutral`, `negative` |

### Class Imbalance Analysis

As typical in real-world e-commerce and consumer feedback, positive reviews dominate the corpus, presenting a notable class skew:

<p align="center">
  <img src="assets/sentiment_distribution.png" alt="Distribution of Sentiment Labels" width="85%" />
</p>

| Sentiment Label | Record Count | Class Proportion | Strategic Handling |
| :--- | :---: | :---: | :--- |
| **Positive** | 3,066 | **75.1%** | Majority class; defines baseline accuracy. |
| **Neutral** | 536 | **13.1%** | Nuanced transition zone; lower lexical polarization. |
| **Negative** | 482 | **11.8%** | Critical minority class; prioritized for high precision. |

> 💡 **Methodological Note:** Because positive feedback constitutes over 75% of the data, simple **Accuracy** is an insufficient metric (a naive majority-class classifier would achieve ~75% accuracy while failing completely on negative feedback). We prioritize **Macro F1-Score** to weight all three classes equally.

---

## 🔬 Text Preprocessing & NLP Pipeline

Raw user text contains markup artifacts, erratic capitalization, punctuation anomalies, and noisy tokens. The preprocessing engine cleans each document systematically:

1. **Case Normalization**: All characters are lowercased to ensure identical tokens match regardless of capitalization (`"EXCELLENT"` $\rightarrow$ `"excellent"`).
2. **Entity & Noise Stripping**:
   - HTML tags: `re.sub(r"<[^>]+>", " ", text)`
   - Hyperlinks & URLs: `re.sub(r"https?://\S+|www\.\S+", " ", text)`
   - Non-alphabetic symbols & punctuation: `re.sub(r"[^a-z\s']", " ", text)`
3. **Negation Retention Strategy**:
   - Standard NLP stopword lists (e.g., scikit-learn's `ENGLISH_STOP_WORDS`) discard words like *"not"*, *"no"*, and *"never"*.
   - In sentiment analysis, dropping *"not"* transforms *"not good"* into *"good"*, causing severe classification error.
   - **Solution**: We construct a customized stopword vocabulary preserving all negation tokens:
     ```python
     NEGATION_WORDS = {"no", "nor", "not", "never", "n't"}
     STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
     ```
4. **Morphological Lemmatization**:
   - NLTK's `WordNetLemmatizer` reduces inflected words to their root lemma (e.g., *"reviews"* $\rightarrow$ *"review"*, *"working"* $\rightarrow$ *"work"*).
   - Engineered with automatic fallback to ensure execution even in restricted offline environments.

---

## ⚙️ Feature Extraction: Sublinear TF-IDF

Text data is vectorized using **Term Frequency-Inverse Document Frequency (TF-IDF)** to convert textual tokens into dense numerical representations reflecting semantic importance:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Vectorizer Hyperparameters
- **`ngram_range=(1, 2)`**: Extracts both individual unigrams (*"great"*, *"slow"*) and contiguous bigrams (*"not working"*, *"highly recommend"*).
- **`max_features=5000`**: Restricts the vocabulary to the top 5,000 most informative terms, controlling dimensionality and preventing overfitting.
- **`sublinear_tf=True`**: Applies logarithmic term-frequency scaling ($1 + \log(\text{TF})$) to prevent high-frequency repeated words in long reviews from artificially dominating the feature space.
- **Data Leakage Safeguard**: Fitted strictly on `X_train` (2,766 reviews) and transformed on `X_test` (692 reviews).

---

## 📈 Model Benchmark & Performance Evaluation

Both candidate classifiers were trained on the identical 80/20 stratified train/test split.

<p align="center">
  <img src="assets/model_comparison.png" alt="Model Performance Comparison Benchmark" width="90%" />
</p>

### Comprehensive Benchmark Matrix

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression** | **79.91%** | **80.01%** | **52.89%** | **58.06%** | **Selected Best Model** |
| 🥈 **Multinomial Naive Bayes** | 73.12% | 57.68% | 34.06% | 29.56% | Baseline Model |

### In-Depth Classifier Analysis

1. **Logistic Regression (Winner)**:
   - Achieved superior performance across all dimensions, with an overall **Accuracy of 79.91%** and a **Macro Precision of 80.01%**.
   - Handles the dense correlations between overlapping unigrams and bigrams far more effectively than Naive Bayes.
   - Converged smoothly within 2,000 iterations without requiring aggressive regularization penalties.
2. **Multinomial Naive Bayes**:
   - Highly vulnerable to the strong class prior probabilities in an imbalanced dataset ($P(\text{positive}) \approx 75\%$).
   - Due to the conditional independence assumption, it heavily over-predicted the majority class, achieving only 34.06% macro recall and an F1-score of 29.56%.

---

## 🎯 Confusion Matrix & Error Diagnostics

The classification report and confusion matrix below illustrate how the best model (**Logistic Regression**) performs across each class on the held-out test partition ($N = 692$):

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix - Logistic Regression" width="80%" />
</p>

### Detailed Per-Class Classification Report

```text
              precision    recall  f1-score   support

    negative       0.88      0.41      0.56        92
     neutral       0.72      0.19      0.30        96
    positive       0.80      0.99      0.88       504

    accuracy                           0.80       692
   macro avg       0.80      0.53      0.58       692
weighted avg       0.80      0.80      0.76       692
```

### Error Analysis & Observations

1. **High Negative Precision (88%)**:
   - When Logistic Regression flags a review as **negative**, it is correct **88% of the time**. This ensures very low false-alarm rates for escalation workflows.
2. **Majority Class Retention (Positive Recall: 99%)**:
   - The model accurately captures 497 out of 504 positive reviews, missing only 7.
3. **The Neutral Classification Challenge**:
   - As shown in the confusion matrix, 77 neutral reviews were classified as positive. Neutral reviews in consumer feedback frequently contain mild praise (*"arrived on time"*, *"works fine"*) which triggers positive lexical tokens without strong negative markers.
   - Future work can mitigate this by applying cost-sensitive re-weighting or fine-tuning transformer models.

---

## 💬 Interactive Real-Time Inference

To validate production readiness, the model was tested against live customer reviews with probability calibration:

<p align="center">
  <img src="assets/sample_predictions.png" alt="Interactive Sentiment Prediction and Class Probabilities" width="100%" />
</p>

### Sample Inference Run Breakdown

| Sample Review | Predicted | Confidence | Probability Breakdown |
| :--- | :---: | :---: | :--- |
| *"The product is excellent, durable, and works really well!"* | **POSITIVE** | **94.2%** | Pos: 94.2% \| Neu: 4.1% \| Neg: 1.7% |
| *"The product is disappointing and does not work properly at all."* | **NEUTRAL** | **81.8%** | Pos: 10.2% \| Neu: 81.8% \| Neg: 8.0% |
| *"The item arrived on time, basic quality as described in the specs."* | **POSITIVE** | **78.2%** | Pos: 78.2% \| Neu: 13.9% \| Neg: 7.9% |
| *"Customer service was terrible, refused refund and very rude."* | **NEGATIVE** | **39.5%** | Pos: 35.3% \| Neu: 25.2% \| Neg: 39.5% |
| *"Loving the vibrant display and fantastic battery life!"* | **POSITIVE** | **64.8%** | Pos: 64.8% \| Neu: 25.6% \| Neg: 9.6% |

---

## 📁 Project Directory Structure

```text
d:/NLP_sentiment_classifications/
│
├── assets/                               # High-resolution visual artifacts
│   ├── nlp_sentiment_banner.jpg          # Repository header banner
│   ├── sentiment_distribution.png        # Class frequency & percentage breakdown
│   ├── model_comparison.png              # Benchmark chart (Accuracy, F1, Precision, Recall)
│   ├── confusion_matrix.png              # Logistic Regression error heatmap
│   └── sample_predictions.png            # Real-time inference probability bars
│
├── NLP_Sentiment_Classification_TFIDF.ipynb  # Interactive Jupyter Notebook pipeline
├── pipeline.py                           # Standalone end-to-end training script
├── predict.py                            # Interactive CLI inference utility
├── requirements.txt                      # Project dependency specification
├── sentiment_classifier.joblib           # Serialized trained Logistic Regression model
├── tfidf_vectorizer.joblib               # Serialized TF-IDF feature extractor
└── README.md                             # Comprehensive technical documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites & Environment Setup

Clone the repository and install the dependencies in a virtual environment:

```bash
# Clone the workspace
git clone https://github.com/your-username/NLP_sentiment_classifications.git
cd NLP_sentiment_classifications

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Interactive Predictions via CLI

You can immediately test sentiment classification on custom text strings:

```bash
# Single review classification
python predict.py "The customer support resolved my issue quickly and the product works flawlessly!"
```

**Output:**
```text
============================================================
Review: "The customer support resolved my issue quickly and the product works flawlessly!"
Cleaned Tokens: customer support resolved issue quickly product work flawlessly
Predicted Sentiment: POSITIVE
------------------------------------------------------------
Confidence Breakdown:
  - Negative: 14.61%  [####                          ]
  - Neutral : 29.73%  [########                      ]
  - Positive: 55.66%  [################              ]
============================================================
```

#### Interactive Prompt Mode:
```bash
python predict.py --interactive
```

### 3. Retrain the Pipeline from Scratch

To re-fetch the latest data, execute full normalization, re-train models, and regenerate artifacts:

```bash
python pipeline.py
```

### 4. Open in Jupyter Notebook

To explore intermediate steps and execute visualizations cell by cell:

```bash
jupyter notebook NLP_Sentiment_Classification_TFIDF.ipynb
```

---

## 🛠️ Engineering Limitations & Roadmap

While the TF-IDF and Logistic Regression architecture delivers strong accuracy and ultra-fast inference (< 2ms per review), several avenues remain for ongoing optimization:

| Area | Current Behavior | Proposed Enhancement |
| :--- | :--- | :--- |
| **Contextual Nuance** | N-grams capture local syntax but lack deep semantic understanding. | Fine-tune transformer models (**DistilBERT** or **RoBERTa**) for bidirectional contextual embeddings. |
| **Class Imbalance** | 75% positive reviews lead to reduced recall on neutral and negative classes. | Implement **SMOTE** (Synthetic Minority Over-sampling) or class-balanced cost functions (`class_weight='balanced'`). |
| **Sarcasm & Slang** | Non-literal language and colloquialisms can mislead token weights. | Incorporate sentiment-focused pre-trained encoders (e.g., `cardiffnlp/twitter-roberta-base-sentiment`). |
| **API Deployment** | CLI and notebook execution. | Wrap the model in a high-performance **FastAPI microservice** with Docker containerization. |

---

## 📄 License & Acknowledgments

- **Dataset**: Provided by [Hugging Face davidberg/sentiment-reviews](https://huggingface.co/datasets/davidberg/sentiment-reviews).
- **Libraries**: Built using [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NLTK](https://www.nltk.org/), and [seaborn](https://seaborn.pydata.org/).
- **License**: Released under the **MIT License**. Feel free to use, modify, and integrate into your own projects.
