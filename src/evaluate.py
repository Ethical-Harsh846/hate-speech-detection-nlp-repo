import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print classification report and plot confusion matrix."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {model_name} — Accuracy: {acc:.4f}")
    print('='*40)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Hate", "Hate"],
                yticklabels=["Not Hate", "Hate"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.show()
    print(f"✅ Confusion matrix saved to results/")

if __name__ == "__main__":
    import pickle
    with open("models/svm_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    df = pd.read_csv("data/processed/tweets_processed.csv")
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42
    )
    X_test_tfidf = tfidf.transform(X_test)
    evaluate_model(model, X_test_tfidf, y_test, "Support Vector Machine")
