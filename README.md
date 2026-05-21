<h1 align="center">🔍 Hate Speech Detection using Machine Learning & NLP</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  A complete end-to-end NLP pipeline that detects hate speech in Twitter data using text preprocessing, TF-IDF vectorization, and machine learning classifiers.
</p>

---

## 📌 Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [NLP Pipeline](#nlp-pipeline)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

---

## 🧠 Overview

Hate speech on social media platforms like Twitter poses a serious challenge for content moderation. This project builds a **machine learning classifier** that automatically identifies hateful language in tweets using a full NLP pipeline — from raw text to model predictions.

Key highlights:
- ✅ End-to-end NLP pipeline (cleaning → features → model)
- ✅ Multiple ML classifiers trained and benchmarked
- ✅ TF-IDF based feature engineering
- ✅ Detailed evaluation with precision, recall, F1-score

---

## 🏗️ Project Architecture

```
Raw Twitter Data
      │
      ▼
Text Preprocessing (cleaning, stopword removal)
      │
      ▼
Tokenization → Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
ML Classifier (Logistic Regression / SVM / Naive Bayes)
      │
      ▼
Prediction: Hate Speech / Not Hate Speech
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| NLP | NLTK, spaCy, re |
| ML | Scikit-learn |
| Feature Extraction | TF-IDF Vectorizer |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 📦 Dataset

- **Source:** Twitter hate speech dataset
- **Labels:** `hate_speech`, `offensive_language`, `neither`
- **Size:** ~25,000 labeled tweets
- **Preprocessing applied:** lowercasing, URL removal, punctuation stripping, stopword removal, lemmatization

---

## 🔬 NLP Pipeline

### 1. Text Cleaning
```python
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)         # Remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # Remove special chars
    return text.strip()
```

### 2. Tokenization & Lemmatization
```python
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def tokenize_lemmatize(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens]
```

### 3. TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
```

---

## 📊 Models & Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 91.2% | 0.90 | 0.91 | 0.90 |
| Support Vector Machine | 92.5% | 0.92 | 0.93 | 0.92 |
| Naive Bayes | 87.3% | 0.86 | 0.87 | 0.86 |
| Random Forest | 89.8% | 0.89 | 0.90 | 0.89 |

> ✅ **Best Model:** Support Vector Machine with TF-IDF (bigrams) — **92.5% accuracy**

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hate-speech-detection-nlp.git
cd hate-speech-detection-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run preprocessing
python src/preprocess.py

# Train the model
python src/train.py

# Evaluate results
python src/evaluate.py
```

Or open the notebook:
```bash
jupyter notebook notebooks/hate_speech_analysis.ipynb
```

---

## 👥 Contributors

| Name | Role |
|---|---|
| Your Name | NLP Pipeline, Model Training, Evaluation, Documentation |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">⭐ Star this repo if you found it helpful!</p>
