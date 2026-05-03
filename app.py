import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import time

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher", page_icon="📄", layout="centered")

# ======================
# CSS
# ======================
st.markdown("""
<style>
body {background-color: #0E1117;}
textarea {background-color: #1E1E2F !important; color:white !important;}
.result-box {
    padding:15px;
    border-radius:10px;
    background: linear-gradient(90deg,#1f4037,#99f2c8);
    color:black;
    text-align:center;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD
# ======================
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

# ======================
# CLEAN
# ======================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ======================
# PDF READER
# ======================
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + " "
    return text

# ======================
# UI
# ======================
st.title("🚀 CV Matching & Classification App")

# CV Upload
cv_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])

# Job Description
job_desc = st.text_area("💼 Paste Job Description here:", height=200)

cv_text = ""

if cv_file:
    cv_text = read_pdf(cv_file)

# ======================
# BUTTON
# ======================
if st.button("🔍 Analyze"):

    if cv_text == "" or job_desc.strip() == "":
        st.warning("⚠️ Please upload CV and add job description")
    else:
        # Animation loading
        with st.spinner("⏳ Analyzing..."):
            time.sleep(1.5)

        # CLEAN
        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_desc)

        # VECTOR
        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])

        # ======================
        # 🔗 SIMILARITY
        # ======================
        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # ======================
        # 🤖 PREDICTION
        # ======================
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # ======================
        # 🎯 RESULT
        # ======================
        st.markdown(f"""
        <div class="result-box">
        🎯 Category: <b>{pred}</b><br>
        📊 Confidence: <b>{confidence:.2f}</b><br>
        🔗 Matching Score: <b>{similarity:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        # ======================
        # 📊 BAR CHART
        # ======================
        st.subheader("📊 Prediction Probabilities")

        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        st.bar_chart(df.set_index("Category"))

        # ======================
        # 🏆 TOP 3
        # ======================
        st.subheader("🏆 Top Predictions")

        top3 = np.argsort(probs)[-3:][::-1]

        for i in top3:
            cat = label_encoder.inverse_transform([i])[0]
            st.write(f"{cat} : {probs[i]:.2f}")

        # ======================
        # 🧠 INTERPRETATION
        # ======================
        if similarity > 0.7:
            st.success("🔥 Great match!")
        elif similarity > 0.4:
            st.info("🙂 متوسط")
        else:
            st.error("❌ ضعيف")
