import streamlit as st
import joblib
import os

# ---- 1. Load trained model (pipeline: TF-IDF + LogisticRegression) ----
MODEL_PATH = "fake_news_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Please run training.py first."
        )
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ---- 2. Streamlit UI ----
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.caption(
    "Note: This is a machine learning demo trained on a public dataset. "
    "Predictions may be wrong and should not be used as a sole source of truth."
)

st.write(
    "Paste any news headline or article below and the model will predict "
    "whether it's likely **FAKE** or **REAL**."
)

user_input = st.text_area(
    "Enter news text:",
    height=200,
    placeholder="Paste a news article or headline here..."
)

if st.button("Check Authenticity"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        class_index = list(model.classes_).index(prediction)
        confidence = proba[class_index]

        threshold = 0.6  # 60% confidence threshold

        if confidence < threshold:
            st.info(
                f"🤔 I'm not very confident about this prediction "
                f"(confidence: {confidence*100:.2f}%). "
                "Please verify this news from reliable sources."
            )
        else:
            if prediction.lower() == "fake":
                st.error(f"⚠️ This news looks **FAKE** (confidence: {confidence*100:.2f}%)")
            else:
                st.success(f"✅ This news looks **REAL** (confidence: {confidence*100:.2f}%)")

        with st.expander("Show raw prediction details"):
            st.write("Predicted label:", prediction)
            st.write("Class probabilities:")
            for cls, p in zip(model.classes_, proba):
                st.write(f"- {cls}: {p*100:.2f}%")
