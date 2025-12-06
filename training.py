import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# ---- 1. NLTK setup (stopwords) ----
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---- 2. Utility: text cleaning function ----
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()              # extra spaces hatana
    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---- 3. Load dataset from Fake.csv + True.csv ----
DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
    raise FileNotFoundError(
        f"Fake.csv / True.csv nahi mil rahe. Please ensure both files exist in '{DATA_DIR}' folder."
    )

print(f"Loading Fake.csv from: {FAKE_PATH}")
fake_df = pd.read_csv(FAKE_PATH)

print(f"Loading True.csv from: {TRUE_PATH}")
true_df = pd.read_csv(TRUE_PATH)

# In dono files mein columns: title, text, subject, date hote hain
# Ab hum label column manually add karenge
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine both
df = pd.concat([fake_df, true_df], ignore_index=True)

# title + text ko merge karke ek hi text column banate hain
df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

# Sirf text + label rakhte hain
df = df[["text", "label"]].dropna()

print("Before cleaning:", df.shape)
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""]
print("After cleaning:", df.shape)

X = df["clean_text"].values
y = df["label"].values

# ---- 4. Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 5. Pipeline: TF-IDF + Logistic Regression ----
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)  # unigrams + bigrams
    )),
    ("clf", LogisticRegression(max_iter=200))
])

print("Training model...")
model.fit(X_train, y_train)

# ---- 6. Evaluation ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---- 7. Save model ----
MODEL_PATH = "fake_news_model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
