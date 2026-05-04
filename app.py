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

# sentence transformers (safe)
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher PRO MAX 🚀", layout="wide")

# ======================
# 🔥 CSS ULTRA PRO
# ======================
st.markdown("""
<style>
body {background:#0E1117;}
.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    text-align:center;
    font-size:20px;
    margin:10px;
    box-shadow:0px 4px 20px rgba(0,0,0,0.4);
}
.skill {
    background:#1f4037;
    padding:6px 12px;
    border-radius:20px;
    margin:4px;
    display:inline-block;
    color:#99f2c8;
    font-size:13px;
}
.match {
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD
# ======================
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

@st.cache_resource
def load_st():
    if ST_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

st_model = load_st()

# ======================
# CLEAN
# ======================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = list(set(words))  # remove duplicates
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ======================
# PDF
# ======================
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for p in pdf.pages:
        if p.extract_text():
            text += p.extract_text() + " "
    return text

# ======================
# KEYWORDS
# ======================
def get_keywords(text):
    words = text.split()
    words = [w for w in words if len(w) > 3]
    return list(set(words))

# ======================
# UI
# ======================
st.title("🚀 CV Matcher PRO MAX")

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
if st.button("🔍 Analyze"):

    if cv_text == "" or job_desc.strip() == "":
        st.warning("⚠️ Fill all inputs")
    else:
        with st.spinner("⏳ Analyzing..."):
            time.sleep(1)

        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_desc)

        # ======================
        # MODEL
        # ======================
        cv_vec = vectorizer.transform([cv_clean])
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # ======================
        # MATCHING (SMART)
        # ======================
        try:
            if st_model:
                cv_emb = st_model.encode([cv_clean])
                job_emb = st_model.encode([job_clean])
                similarity = cosine_similarity(cv_emb, job_emb)[0][0]
            else:
                raise Exception()
        except:
            job_vec = vectorizer.transform([job_clean])
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # ======================
        # KPIs
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {similarity:.2f}</div>", unsafe_allow_html=True)

        # ======================
        # MATCH QUALITY
        # ======================
        st.subheader("🎯 Match Quality")

        if similarity > 0.75:
            st.success("🔥 Strong Match")
        elif similarity > 0.5:
            st.info("🙂 Medium Match")
        else:
            st.error("❌ Weak Match")

        st.progress(float(similarity))

        # ======================
        # 📊 CHART
        # ======================
        st.subheader("📊 Prediction Distribution")

        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Prob": probs
        })

        st.bar_chart(df.set_index("Category"))

        # ======================
        # 🧠 KEYWORDS
        # ======================
        st.subheader("🧠 Keywords")

        cv_keywords = get_keywords(cv_clean)
        job_keywords = get_keywords(job_clean)

        for k in cv_keywords[:15]:
            st.markdown(f"<span class='skill'>{k}</span>", unsafe_allow_html=True)

        # ======================
        # 🔥 COMMON SKILLS
        # ======================
        st.subheader("🔥 Common Skills")

        common = list(set(cv_keywords) & set(job_keywords))

        if common:
            for w in common[:10]:
                st.markdown(f"<span class='skill'>{w}</span>", unsafe_allow_html=True)
        else:
            st.write("No common skills")

        # ======================
        # 🚨 MISSING SKILLS
        # ======================
        st.subheader("🚨 Missing Skills (Recommendations)")

        missing = list(set(job_keywords) - set(cv_keywords))

        if missing:
            for w in missing[:10]:
                st.markdown(f"<span class='skill'>{w}</span>", unsafe_allow_html=True)
        else:
            st.write("Your CV matches all requirements 🎉")

        # ======================
        # 📊 MATCH VISUAL
        # ======================
        st.subheader("📊 Matching Comparison")

        match_df = pd.DataFrame({
            "Type": ["Matching Score"],
            "Score": [similarity]
        })

        st.bar_chart(match_df.set_index("Type"))

        # ======================
        # 📏 TEXT STATS
        # ======================
        st.subheader("📏 Analysis")

        st.write(f"CV words: {len(cv_clean.split())}")
        st.write(f"Job words: {len(job_clean.split())}")

        # ======================
        # CLEAN VIEW
        # ======================
        with st.expander("🧹 Cleaned CV"):
            st.write(cv_clean)
