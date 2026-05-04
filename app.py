import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time

# sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher GOD MODE 🚀", layout="wide")

# ======================
# 🌐 LANGUAGE
# ======================
lang = st.sidebar.selectbox("🌐 Language", ["English", "Français"])

def t(en, fr):
    return en if lang == "English" else fr

# ======================
# 🎨 CSS PRO MAX
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
}
.skill {
    background:#1f4037;
    padding:6px 12px;
    border-radius:20px;
    margin:4px;
    display:inline-block;
    color:#99f2c8;
}
.highlight {
    background:#FFD700;
    color:black;
    padding:2px 4px;
    border-radius:5px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS
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
    words = list(set(text.split()))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ======================
# PDF READER
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
    return list(set([w for w in text.split() if len(w) > 3]))

# ======================
# HIGHLIGHT
# ======================
def highlight_text(text, keywords):
    for word in keywords:
        text = re.sub(f"\\b{word}\\b",
                      f"<span class='highlight'>{word}</span>",
                      text)
    return text

# ======================
# RADAR
# ======================
def radar(common, missing):
    labels = ["Match", "Missing"]
    values = [len(common), len(missing)]
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots()
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    return fig

# ======================
# AI EXPLANATION
# ======================
def explain(sim, common, missing):
    if sim > 0.75:
        return "Strong match. Your CV aligns very well."
    elif sim > 0.5:
        return "Medium match. Improve some skills."
    else:
        return "Weak match. Many important skills missing."

# ======================
# UI
# ======================
st.title(t("🚀 CV Matcher GOD MODE", "🚀 Analyseur CV GOD MODE"))

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("📄 Upload CV (PDF)", type=["pdf"])

with col2:
    job_file = st.file_uploader("💼 Upload Job (PDF)", type=["pdf"])
    job_desc = st.text_area("OR paste job description")

cv_text = ""
job_text = ""

if cv_file:
    cv_text = read_pdf(cv_file)

if job_file:
    job_text = read_pdf(job_file)
elif job_desc:
    job_text = job_desc

# ======================
# ANALYZE
# ======================
if st.button("🔍 Analyze"):

    if cv_text == "" or job_text == "":
        st.warning("⚠️ Fill all inputs")
    else:
        with st.spinner("⏳ AI Thinking..."):
            time.sleep(1)

        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_text)

        # MODEL
        cv_vec = vectorizer.transform([cv_clean])
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # MATCH
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

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {similarity:.2f}</div>", unsafe_allow_html=True)

        # SKILLS
        cv_k = get_keywords(cv_clean)
        job_k = get_keywords(job_clean)

        common = list(set(cv_k) & set(job_k))
        missing = list(set(job_k) - set(cv_k))

        # MATCH
        st.subheader("🎯 Match")
        st.progress(float(similarity))

        # EXPLANATION
        st.subheader("🧠 AI Explanation")
        st.info(explain(similarity, common, missing))

        # CHART
        st.subheader("📊 Distribution")
        df = pd.DataFrame({"Category": label_encoder.classes_, "Prob": probs})
        st.bar_chart(df.set_index("Category"))

        # RADAR
        st.subheader("📊 Skills Radar")
        st.pyplot(radar(common, missing))

        # COMMON
        st.subheader("🔥 Common Skills")
        for w in common[:15]:
            st.markdown(f"<span class='skill'>{w}</span>", unsafe_allow_html=True)

        # MISSING
        st.subheader("🚨 Missing Skills")
        for w in missing[:15]:
            st.markdown(f"<span class='skill'>{w}</span>", unsafe_allow_html=True)

        # HIGHLIGHT CV
        st.subheader("✨ Highlighted CV")
        highlighted = highlight_text(cv_clean, common)
        st.markdown(highlighted, unsafe_allow_html=True)

        # CLEAN TEXT
        with st.expander("🧹 Cleaned CV"):
            st.write(cv_clean)
