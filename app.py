import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import time

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher Final Boss 🚀", layout="wide")

# ======================
# 🌐 LANGUAGE
# ======================
st.sidebar.title("⚙️ Settings / Paramètres")
lang = st.sidebar.selectbox("🌐 Language / Langue", ["English", "Français"])

def t(en, fr):
    return en if lang == "English" else fr

# ======================
# 🎨 CSS ULTRA PRO
# ======================
st.markdown("""
<style>
body {background-color:#0E1117;}

.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.3);
}

.metric-title {font-size:14px;opacity:0.8;}
.big-number {font-size:30px;font-weight:bold;}

.skill {
    display:inline-block;
    background:#1f4037;
    padding:6px 12px;
    margin:5px;
    border-radius:20px;
    color:white;
}

.match {background:#00FFAA;color:black;}
.missing {background:#FF4B4B;color:white;}

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
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = list(set(text.split()))
    words = [w for w in words if w not in stop_words]
    return words

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
# UI
# ======================
st.title("🚀 CV Matcher FINAL BOSS")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("📄 Upload CV", type=["pdf"])

with col2:
    job_file = st.file_uploader("💼 Upload Job (PDF)", type=["pdf"])
    job_desc = st.text_area("OR paste job description")

cv_text = read_pdf(cv_file) if cv_file else ""
job_text = read_pdf(job_file) if job_file else job_desc

# ======================
# ANALYZE
# ======================
if st.button("🔥 Analyze Like a Boss"):

    if cv_text == "" or job_text == "":
        st.warning("Fill all inputs")
    else:
        with st.spinner("AI thinking..."):
            time.sleep(1)

        cv_words = clean_text(cv_text)
        job_words = clean_text(job_text)

        cv_clean = " ".join(cv_words)
        job_clean = " ".join(job_words)

        # ======================
        # PREDICTION
        # ======================
        cv_vec = vectorizer.transform([cv_clean])
        pred_index = model.predict(cv_vec)[0]
        pred = label_encoder.inverse_transform([pred_index])[0]

        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # ======================
        # MATCHING
        # ======================
        if ST_AVAILABLE:
            cv_emb = st_model.encode([cv_clean])
            job_emb = st_model.encode([job_clean])
            similarity = cosine_similarity(cv_emb, job_emb)[0][0]
        else:
            job_vec = vectorizer.transform([job_clean])
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # ======================
        # DASHBOARD
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'><div class='metric-title'>🎯 Category</div><div class='big-number'>{pred}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><div class='metric-title'>📊 Confidence</div><div class='big-number'>{confidence:.2f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><div class='metric-title'>🔗 Matching</div><div class='big-number'>{similarity:.2f}</div></div>", unsafe_allow_html=True)

        # ======================
        # GAUGE
        # ======================
        st.subheader("🎯 Matching Score")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity*100,
            gauge={
                'axis': {'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':'red'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'green'}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # BAR
        # ======================
        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        st.subheader("📊 Predictions")
        st.bar_chart(df.set_index("Category"))

        # ======================
        # SKILLS MATCHING
        # ======================
        st.subheader("🧠 Skills Analysis")

        cv_set = set(cv_words)
        job_set = set(job_words)

        common = list(cv_set & job_set)[:15]
        missing = list(job_set - cv_set)[:15]

        st.write("✅ Matching Skills")
        for s in common:
            st.markdown(f"<span class='skill match'>{s}</span>", unsafe_allow_html=True)

        st.write("❌ Missing Skills")
        for s in missing:
            st.markdown(f"<span class='skill missing'>{s}</span>", unsafe_allow_html=True)

        # ======================
        # INTERPRETATION
        # ======================
        st.subheader("🧠 AI Interpretation")

        if similarity > 0.75:
            st.success("🔥 Strong Match - Great CV!")
        elif similarity > 0.5:
            st.info("🙂 Medium Match - Improve skills")
        else:
            st.error("❌ Weak Match - Needs improvement")

        # ======================
        # KEYWORDS
        # ======================
        st.subheader("💡 Keywords")

        for w in cv_words[:20]:
            st.markdown(f"<span class='skill'>{w}</span>", unsafe_allow_html=True)
