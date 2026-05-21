import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Remove URLs, mentions, hashtags, special characters."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)             # Remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # Remove extra spaces
    return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text."""
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def preprocess_pipeline(df, text_column="tweet"):
    """Full preprocessing pipeline."""
    df["cleaned_text"] = df[text_column].apply(clean_text)
    df["processed_text"] = df["cleaned_text"].apply(tokenize_and_lemmatize)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/tweets.csv")
    df = preprocess_pipeline(df)
    df.to_csv("data/processed/tweets_processed.csv", index=False)
    print("✅ Preprocessing complete!")
