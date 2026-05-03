import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# تحميل stopwords
nltk.download('stopwords')

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="CV Classifier", page_icon="📄", layout="centered")

# =========================
# 🎨 CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.main {
    background-color: #0E1117;
}
textarea {
    background-color: #1E1E2F !important;
    color: white !important;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #45a049;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1f4037;
    color: white;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 📦 LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =========================
# 🧹 CLEAN FUNCTION
# =========================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# 🖥️ UI
# =========================
st.title("📄 CV Classification App")
st.write("Upload or paste your CV and classify it using Machine Learning 🚀")

# Upload file
uploaded_file = st.file_uploader("📎 Upload your CV (txt file)", type=["txt"])

user_input = ""

if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")

# Text area
user_input = st.text_area("✍️ Paste your CV here:", user_input, height=200)

# =========================
# 🔮 PREDICTION
# =========================
if st.button("🔍 Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter or upload a CV")
    else:
        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        confidence = max(probs)

        # =========================
        # 🎯 RESULT
        # =========================
        st.markdown(f"""
        <div class="result-box">
        Predicted Category: <b>{prediction}</b><br>
        Confidence: <b>{confidence:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # 📊 TOP 3
        # =========================
        st.subheader("🏆 Top 3 Predictions")

        top3_idx = np.argsort(probs)[-3:][::-1]

        for i in top3_idx:
            cat = label_encoder.inverse_transform([i])[0]
            st.write(f"👉 {cat} : {probs[i]:.2f}")

        # =========================
        # 📈 CHART
        # =========================
        st.subheader("📊 Prediction Probabilities")

        df_probs = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        st.bar_chart(df_probs.set_index("Category"))

        # =========================
        # 🧹 CLEAN TEXT
        # =========================
        st.subheader("🧹 Cleaned Text (NLP Step)")
        st.write(cleaned)
