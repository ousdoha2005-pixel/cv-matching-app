import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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
    font-size:18px;
    font-weight:bold;
}
.skill {
    display:inline-block;
    padding:8px 14px;
    margin:6px;
    border-radius:25px;
    font-weight:bold;
    font-size:14px;
}
.match {background:#00FFAA;color:black;}
.missing {background:#FF4B4B;color:white;}
.highlight {background:yellow;color:black;padding:2px;}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
label_encoder = pickle.load(open("label_encoder.pkl","rb"))

# ======================
# SKILLS DATABASE (SMART)
# ======================
SKILLS_DB = [
    "python","java","c++","sql","machine learning","deep learning",
    "nlp","data analysis","pandas","numpy","scikit-learn",
    "tensorflow","pytorch","power bi","excel","docker","aws"
]

# ======================
# FUNCTIONS
# ======================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = list(set(text.split()))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

def extract_skills(text):
    text = text.lower()
    return list(set([s for s in SKILLS_DB if s in text]))

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

def generate_pdf(pred, score, percent, common, missing):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Category: {pred}", styles["Normal"]),
        Paragraph(f"Score: {score}/10", styles["Normal"]),
        Paragraph(f"Match: {percent}%", styles["Normal"]),
        Paragraph("Matching Skills: " + ", ".join(common), styles["Normal"]),
        Paragraph("Missing Skills: " + ", ".join(missing), styles["Normal"])
    ]

    doc.build(content)

def animate_value(label, value):
    placeholder = st.empty()
    for i in np.linspace(0, value, 30):
        placeholder.markdown(f"<div class='card'>{label}<br>{round(i,2)}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

# ======================
# UI
# ======================
st.title("🚀 CV Matcher AI PRO")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader(t("Upload CV (PDF)", "Uploader CV"), type=["pdf"])
    cv_text_input = st.text_area(t("Paste your CV", "Coller CV"))

with col2:
    job_file = st.file_uploader(t("Upload Job (PDF)", "Uploader Offre"), type=["pdf"])
    job_text_input = st.text_area(t("Paste Job Description", "Coller Offre"))

cv_text = read_pdf(cv_file) if cv_file else cv_text_input
job_text = read_pdf(job_file) if job_file else job_text_input

# ======================
# ANALYZE
# ======================
if st.button("🔥 Analyze"):

    if cv_text.strip() == "" or job_text.strip() == "":
        st.warning("⚠️ Fill all fields")

    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

        # CLEAN
        cv_words = clean_text(cv_text)
        job_words = clean_text(job_text)

        cv_clean = " ".join(cv_words)
        job_clean = " ".join(job_words)

        # MODEL
        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])

        pred_idx = model.predict(cv_vec)[0]
        pred = label_encoder.inverse_transform([pred_idx])[0]

        probs = model.predict_proba(cv_vec)[0]
        confidence = float(np.max(probs))

        similarity = float(cosine_similarity(cv_vec, job_vec)[0][0])

        score10 = round(similarity * 10, 1)
        percent = round(similarity * 100, 1)

        # SKILLS (SMART)
        cv_skills = extract_skills(cv_text)
        job_skills = extract_skills(job_text)

        common = list(set(cv_skills) & set(job_skills))
        missing = list(set(job_skills) - set(cv_skills))

        # ======================
        # DASHBOARD
        # ======================
        st.subheader("📊 AI Dashboard")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"<div class='card'>🎯<br>{pred}</div>", unsafe_allow_html=True)

        with c2:
            animate_value("📊 Confidence", confidence)

        with c3:
            animate_value("🔗 Match %", percent)

        st.progress(int(percent))

        # ======================
        # GAUGE
        # ======================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            gauge={'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # ======================
        # BAR
        # ======================
        df = pd.DataFrame({
            "Type": ["Matching", "Missing"],
            "Count": [len(common), len(missing)]
        })
        st.plotly_chart(px.bar(df, x="Type", y="Count", color="Type", text="Count"))

        # ======================
        # RADAR
        # ======================
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=[len(common), len(missing), score10],
            theta=["Match", "Missing", "Score"],
            fill='toself'
        ))
        st.plotly_chart(radar)

        # ======================
        # SKILLS UI
        # ======================
        st.subheader("✅ Matching Skills")
        for s in common:
            st.markdown(f"<span class='skill match'>{s}</span>", unsafe_allow_html=True)

        st.subheader("❌ Missing Skills")
        for s in missing:
            st.markdown(f"<span class='skill missing'>{s}</span>", unsafe_allow_html=True)

        # ======================
        # HIGHLIGHT
        # ======================
        st.subheader("📄 Highlight CV")
        st.markdown(highlight_text(cv_text, common), unsafe_allow_html=True)

        # ======================
        # AI EXPLANATION
        # ======================
        st.subheader("🤖 AI Explanation")

        if percent > 70:
            st.success("🔥 Strong match")
        elif percent > 40:
            st.info("🙂 Medium match")
        else:
            st.error("❌ Weak match")

        # ======================
        # PDF
        # ======================
        generate_pdf(pred, score10, percent, common, missing)

        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f)
