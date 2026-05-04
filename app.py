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

# ⚠️ try import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher PRO", page_icon="🚀", layout="wide")

# ======================
# LIGHT CSS (SAFE)
# ======================
st.markdown("""
<style>
body {background-color: #0E1117;}
textarea {background-color:#1E1E2F !important; color:white !important;}
.card {
    padding:15px;
    border-radius:12px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    text-align:center;
    margin:10px;
}
.badge {
    background:#4CAF50;
    padding:5px 10px;
    border-radius:10px;
    margin:4px;
    display:inline-block;
    color:white;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS
# ======================
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

# ======================
# LOAD SENTENCE MODEL (SAFE)
# ======================
@st.cache_resource
def load_st_model():
    if ST_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    return None

st_model = load_st_model()

# ======================
# CLEAN
# ======================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = list(set(words))
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
st.title("🚀 CV Matcher (Stable Version)")

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
        with st.spinner("⏳ Processing..."):
            time.sleep(1)

        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_desc)

        # ======================
        # CLASSIFICATION
        # ======================
        cv_vec = vectorizer.transform([cv_clean])
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # ======================
        # SAFE MATCHING (🔥)
        # ======================
        try:
            if st_model:
                cv_emb = st_model.encode([cv_clean], show_progress_bar=False)
                job_emb = st_model.encode([job_clean], show_progress_bar=False)
                similarity = cosine_similarity(cv_emb, job_emb)[0][0]
            else:
                raise Exception("ST not available")
        except:
            # fallback TF-IDF
            job_vec = vectorizer.transform([job_clean])
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # ======================
        # CARDS
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {similarity:.2f}</div>", unsafe_allow_html=True)

        # ======================
        # PROGRESS
        # ======================
        st.subheader("📈 Matching Score")
        st.progress(float(similarity))

        # ======================
        # TOP 3
        # ======================
        st.subheader("🏆 Top Predictions")

        top3 = np.argsort(probs)[-3:][::-1]

        for i in top3:
            cat = label_encoder.inverse_transform([i])[0]
            st.write(f"{cat}: {probs[i]:.2f}")

        # ======================
        # CHART
        # ======================
        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        st.bar_chart(df.set_index("Category"))

        # ======================
        # KEYWORDS
        # ======================
        st.subheader("🧠 Keywords")

        for k in extract_keywords(cv_clean):
            st.markdown(f"<span class='badge'>{k}</span>", unsafe_allow_html=True)

        # ======================
        # INTERPRETATION
        # ======================
        st.subheader("📌 Result")

        if similarity > 0.7:
            st.success("🔥 Excellent Match")
        elif similarity > 0.4:
            st.info("🙂 Medium Match")
        else:
            st.error("❌ Weak Match")

        # ======================
        # CLEAN TEXT
        # ======================
        with st.expander("🧹 Cleaned CV"):
            st.write(cv_clean)
