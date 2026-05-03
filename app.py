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
from collections import Counter

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher Pro", page_icon="🚀", layout="wide")

# ======================
# CSS PRO UI
# ======================
st.markdown("""
<style>
body {background-color: #0E1117;}
textarea {background-color: #1E1E2F !important; color:white !important;}

.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg,#1f4037,#99f2c8);
    color:black;
    text-align:center;
    margin:10px;
    font-size:18px;
}

.badge {
    background-color:#4CAF50;
    padding:5px 10px;
    border-radius:8px;
    margin:3px;
    display:inline-block;
    color:white;
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
# PDF
# ======================
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

# ======================
# KEYWORDS
# ======================
def extract_keywords(text, top_n=10):
    words = text.split()
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

# ======================
# UI
# ======================
st.title("🚀 CV Matching & Analysis Dashboard")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("📄 Upload CV (PDF)", type=["pdf"])

with col2:
    job_desc = st.text_area("💼 Job Description", height=150)

cv_text = ""

if cv_file:
    cv_text = read_pdf(cv_file)

# ======================
# BUTTON
# ======================
if st.button("🔍 Analyze CV"):

    if cv_text == "" or job_desc.strip() == "":
        st.warning("⚠️ Please upload CV and add job description")
    else:
        with st.spinner("⏳ Processing..."):
            time.sleep(1.5)

        # CLEAN
        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_desc)

        # VECTOR
        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])

        # SIMILARITY
        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # MODEL
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # ======================
        # DASHBOARD CARDS
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 Category<br><b>{pred}</b></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 Confidence<br><b>{confidence:.2f}</b></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 Matching<br><b>{similarity:.2f}</b></div>", unsafe_allow_html=True)

        # ======================
        # TOP 3
        # ======================
        st.subheader("🏆 Top Predictions")

        top3 = np.argsort(probs)[-3:][::-1]

        for i in top3:
            cat = label_encoder.inverse_transform([i])[0]
            st.write(f"👉 {cat} : {probs[i]:.2f}")

        # ======================
        # CHART
        # ======================
        st.subheader("📊 Prediction Distribution")

        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        st.bar_chart(df.set_index("Category"))

        # ======================
        # KEYWORDS
        # ======================
        st.subheader("🧠 Extracted Keywords")

        keywords = extract_keywords(cv_clean, 15)

        for k in keywords:
            st.markdown(f"<span class='badge'>{k}</span>", unsafe_allow_html=True)

        # ======================
        # SKILLS MATCHING
        # ======================
        st.subheader("🎯 Matching Skills")

        cv_words = set(cv_clean.split())
        job_words = set(job_clean.split())

        matched = cv_words.intersection(job_words)

        if matched:
            for m in list(matched)[:20]:
                st.markdown(f"<span class='badge'>{m}</span>", unsafe_allow_html=True)
        else:
            st.write("No strong matching keywords")

        # ======================
        # INTERPRETATION
        # ======================
        st.subheader("📌 Interpretation")

        if similarity > 0.7:
            st.success("🔥 Excellent Match")
        elif similarity > 0.4:
            st.info("🙂 متوسط")
        else:
            st.error("❌ ضعيف")

        # ======================
        # CLEAN TEXT (OPTIONAL)
        # ======================
        with st.expander("🧹 Show Cleaned CV"):
            st.write(cv_clean)
