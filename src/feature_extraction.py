import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(X_train, max_features=10000, ngram_range=(1, 2)):
    """
    Build and fit TF-IDF Vectorizer.
    Uses bigrams (1,2) for better context capture.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True       # Apply log normalization
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Save vectorizer for future use
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    
    print(f"✅ TF-IDF fitted | Vocabulary size: {len(tfidf.vocabulary_)}")
    return tfidf, X_train_tfidf

def transform_text(tfidf, X):
    """Transform new text using fitted TF-IDF."""
    return tfidf.transform(X)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/tweets_processed.csv")
    tfidf, X = build_tfidf(df["processed_text"])
    print("Feature matrix shape:", X.shape)
