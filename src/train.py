import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import build_tfidf

def train_models(X_train_tfidf, y_train):
    """Train multiple classifiers and return results."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained[name] = model
        print(f"✅ Trained: {name}")
    return trained

if __name__ == "__main__":
    df = pd.read_csv("data/processed/tweets_processed.csv")
    X = df["processed_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf, X_train_tfidf = build_tfidf(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    trained_models = train_models(X_train_tfidf, y_train)

    # Save best model
    with open("models/svm_classifier.pkl", "wb") as f:
        pickle.dump(trained_models["Support Vector Machine"], f)
    print("✅ Best model saved!")
