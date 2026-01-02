import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_nb_pipeline.joblib")

def load_data(path: str) -> pd.DataFrame:
    # The Kaggle CSV often has extra unused columns and latin-1 encoding
    df = pd.read_csv(path, encoding="latin-1")
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    return df

def split_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

def build_pipeline() -> Pipeline:
    # TF-IDF + Multinomial Naive Bayes is simple and strong for text
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',     # simple stopword removal
            ngram_range=(1, 2),       # unigrams + bigrams improve performance
            max_df=0.95,              # drop very frequent terms
            min_df=2                  # drop very rare terms
        )),
        ('clf', MultinomialNB())
    ])
    return pipeline

def train_and_evaluate(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return pipeline

def save_model(pipeline: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved model to {path}")

def main():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline()
    pipeline = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
    save_model(pipeline, MODEL_PATH)

if __name__ == "__main__":
    main()
