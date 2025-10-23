# 🎬 Classical NLP Sentiment Analysis — IMDb Reviews

> **End-to-end NLP project** showcasing text preprocessing, TF-IDF, Naive Bayes, Logistic Regression, topic modeling, and interactive Plotly visualizations.

---

## 🧩 Project Overview

This project demonstrates a **complete classical NLP pipeline** using the [IMDb 50K Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The notebook covers every step from raw text to interpretable insights — ideal for understanding how traditional NLP techniques still power many real-world sentiment applications.

---

## ⚙️ Pipeline Overview

| Stage | Description | Tools |
|:------|:-------------|:------|
| **Data Extraction** | Load IMDb dataset (50k labeled reviews) | pandas |
| **Data Cleaning** | HTML removal, lowercasing, regex normalization | re, unicodedata |
| **Preprocessing** | Tokenization, stopword removal, POS-aware lemmatization | NLTK |
| **Feature Engineering** | Bag of Words, TF-IDF vectorization | scikit-learn |
| **Modeling** | Multinomial Naive Bayes, Logistic Regression | scikit-learn |
| **Unsupervised Learning** | LDA topic modeling, KMeans clustering | scikit-learn |
| **Evaluation** | Precision, recall, F1, confusion matrix, ROC | scikit-learn |
| **Visualization** | Interactive charts and topic bars | Plotly |
| **Similarity Search** | Cosine similarity for semantic closeness | sklearn.metrics |

---

## 📊 Results Summary

| Model | Accuracy | F1-Score | Notes |
|:------|:---------:|:--------:|:------|
| **Naive Bayes** | 0.85 | 0.85 | Lightweight & interpretable baseline |
| **Logistic Regression** | **0.89** | **0.88** | Best performing linear model |
| **KMeans (unsupervised)** | ARI ≈ 0.41 | — | Clusters correlate with sentiment |
| **LDA Topics** | — | — | Extracts interpretable review themes |

---

## 📈 Visual Insights

Key interactive visualizations (powered by Plotly):

- **Sentiment distribution** — shows dataset balance  
- **Top word bars** — most predictive positive/negative terms  
- **Confusion matrices & ROC curves** — model evaluation  
- **LDA topics** — interpretable semantic clusters  
- **2-D TF-IDF projection** — visualizing review embeddings  
- **Cosine similarity explorer** — find similar reviews  

---

## 🧠 Key Observations

- TF-IDF remains a **strong baseline** for text representation.  
- Logistic Regression outperforms Naive Bayes by ~4 % on accuracy.  
- LDA successfully groups reviews into interpretable topics such as *acting*, *storyline*, *direction*, and *music*.  
- Interactive visualizations (Plotly) significantly improve insight communication.  

---

## 🚀 How to Run

###  Environment Setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
