import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ======================
# INIT
# ======================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="CV Matcher AI PRO", layout="wide")

# ======================
# LANGUAGE
# ======================
lang = st.sidebar.selectbox("🌐 Language", ["English", "Français"])

def t(en, fr):
    return en if lang == "English" else fr

st.sidebar.title(t("Settings", "Paramètres"))

# ======================
# STYLE
# ======================
st.markdown("""
<style>
.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    text-align:center;
    font-size:20px;
    font-weight:bold;
}
.skill {
    display:inline-block;
    padding:8px 12px;
    margin:5px;
    border-radius:20px;
    font-weight:bold;
}
.match {background:#00FFAA;color:black;}
.missing {background:#FF4B4B;color:white;}
.highlight {background:yellow;color:black;padding:2px;}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ======================
# FUNCTIONS
# ======================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = list(set(text.split()))
    words = [w for w in words if w not in stop_words]
    return words

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

def highlight_text(text, keywords):
    for w in keywords:
        text = re.sub(rf"\b{w}\b",
                      f"<span class='highlight'>{w}</span>",
                      text,
                      flags=re.IGNORECASE)
    return text

def generate_pdf(pred, score, similarity, common, missing):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Category: {pred}", styles["Normal"]),
        Paragraph(f"Score: {score}/10", styles["Normal"]),
        Paragraph(f"Matching: {round(similarity*100,1)}%", styles["Normal"]),
        Paragraph("Skills: " + ", ".join(common), styles["Normal"]),
        Paragraph("Missing: " + ", ".join(missing), styles["Normal"])
    ]

    doc.build(content)

# ======================
# UI
# ======================
st.title("🚀 CV Matcher AI PRO")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader(t("Upload CV (PDF)", "Uploader CV (PDF)"), type=["pdf"])
    cv_text_input = st.text_area(t("Paste your CV", "Coller votre CV"))

with col2:
    job_file = st.file_uploader(t("Upload Job (PDF)", "Uploader Offre (PDF)"), type=["pdf"])
    job_text_input = st.text_area(t("Paste Job Description", "Coller offre d'emploi"))

cv_text = read_pdf(cv_file) if cv_file else cv_text_input
job_text = read_pdf(job_file) if job_file else job_text_input

# ======================
# ANALYZE
# ======================
if st.button("🔥 Analyze"):

    if cv_text.strip() == "" or job_text.strip() == "":
        st.warning(t("Please fill all fields", "Remplir tous les champs"))

    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

        # CLEAN
        cv_words = clean_text(cv_text)
        job_words = clean_text(job_text)

        cv_clean = " ".join(cv_words)
        job_clean = " ".join(job_words)

        # VECTORIZATION
        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])

        # PREDICTION
        pred_idx = model.predict(cv_vec)[0]
        pred = label_encoder.inverse_transform([pred_idx])[0]

        probs = model.predict_proba(cv_vec)[0]
        confidence = float(np.max(probs))

        # MATCHING
        similarity = float(cosine_similarity(cv_vec, job_vec)[0][0])

        # SCORE
        score10 = round(similarity * 10, 1)
        percent = round(similarity * 100, 1)

        # SKILLS
        cv_set = set(cv_words)
        job_set = set(job_words)

        common = list(cv_set & job_set)[:15]
        missing = list(job_set - cv_set)[:15]

        # ======================
        # DASHBOARD
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {percent}%</div>", unsafe_allow_html=True)

        # ======================
        # GAUGE
        # ======================
        st.subheader("🎯 Matching Score")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': 'red'},
                    {'range': [40, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'green'}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # ======================
        # RADAR
        # ======================
        st.subheader("🧠 Skills Radar")

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=[len(common), len(missing), score10],
            theta=["Match", "Missing", "Score"],
            fill='toself'
        ))

        st.plotly_chart(radar, use_container_width=True)

        # ======================
        # SKILLS
        # ======================
        st.subheader("✅ Matching Skills")

        cols = st.columns(5)
        for i, s in enumerate(common):
            cols[i % 5].markdown(f"<div class='skill match'>{s}</div>", unsafe_allow_html=True)

        st.subheader("❌ Missing Skills")

        cols = st.columns(5)
        for i, s in enumerate(missing):
            cols[i % 5].markdown(f"<div class='skill missing'>{s}</div>", unsafe_allow_html=True)

        # ======================
        # HIGHLIGHT
        # ======================
        st.subheader("📄 CV Highlight")

        st.markdown(highlight_text(cv_text, common[:10]), unsafe_allow_html=True)

        # ======================
        # AI EXPLANATION
        # ======================
        st.subheader("🤖 AI Explanation")

        st.info(
            f"Strong match: {', '.join(common[:5])}\n\n"
            f"Missing skills: {', '.join(missing[:5])}\n\n"
            f"Recommendation: Learn {', '.join(missing[:3])}"
        )

        # ======================
        # EXPORT PDF
        # ======================
        generate_pdf(pred, score10, similarity, common, missing)

        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f)
