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
import time

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

nltk.download('stopwords')

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="CV Matcher PRO", layout="wide")

# ======================
# CSS
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
}

.skill {
    display:inline-block;
    padding:10px;
    margin:5px;
    border-radius:20px;
    font-weight:bold;
}

.match {background:#00FFAA;color:black;}
.missing {background:#FF4B4B;color:white;}

.highlight {
    background:yellow;
    color:black;
    padding:2px;
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
# HIGHLIGHT
# ======================
def highlight_text(text, keywords):
    for w in keywords:
        text = re.sub(f"\\b{w}\\b", f"<span class='highlight'>{w}</span>", text, flags=re.IGNORECASE)
    return text

# ======================
# UI
# ======================
st.title("🚀 CV Matcher PRO FINAL")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("📄 Upload CV", type=["pdf"])

with col2:
    job_desc = st.text_area("💼 Job Description")

cv_text = read_pdf(cv_file) if cv_file else ""

# ======================
# ANALYZE
# ======================
if st.button("🔥 Analyze"):

    if cv_text == "" or job_desc == "":
        st.warning("Fill inputs")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

        cv_words = clean_text(cv_text)
        job_words = clean_text(job_desc)

        cv_clean = " ".join(cv_words)
        job_clean = " ".join(job_words)

        # Prediction
        cv_vec = vectorizer.transform([cv_clean])
        pred_index = model.predict(cv_vec)[0]
        pred = label_encoder.inverse_transform([pred_index])[0]

        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # Matching
        job_vec = vectorizer.transform([job_clean])
        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # Score
        score = round(similarity * 10, 1)

        # Skills
        cv_set = set(cv_words)
        job_set = set(job_words)

        common = list(cv_set & job_set)[:15]
        missing = list(job_set - cv_set)[:15]

        # ======================
        # CARDS
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {similarity:.2f}</div>", unsafe_allow_html=True)

        # ======================
        # SCORE
        # ======================
        st.subheader("🏆 CV Score")
        st.metric("Score /10", score)

        # ======================
        # GAUGE
        # ======================
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
        # RADAR CHART
        # ======================
        st.subheader("🧠 Skills Radar")

        values = [len(common), len(missing), score]
        labels = ["Match Skills", "Missing Skills", "Score"]

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself'
        ))

        st.plotly_chart(radar)

        # ======================
        # SKILLS GRID
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
        # HIGHLIGHT TEXT
        # ======================
        st.subheader("📄 CV Highlight")

        highlighted = highlight_text(cv_text, common[:10])
        st.markdown(highlighted, unsafe_allow_html=True)

        # ======================
        # WHY MATCH
        # ======================
        st.subheader("🤖 Why Match")

        st.info(f"""
Matching based on shared skills: {', '.join(common[:5])}

Missing important skills: {', '.join(missing[:5])}
""")

        # ======================
        # EXPORT PDF
        # ======================
        def generate_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph(f"Category: {pred}", styles["Normal"]))
            content.append(Paragraph(f"Score: {score}/10", styles["Normal"]))
            content.append(Paragraph(f"Matching: {similarity:.2f}", styles["Normal"]))
            content.append(Paragraph("Skills: " + ", ".join(common), styles["Normal"]))
            content.append(Paragraph("Missing: " + ", ".join(missing), styles["Normal"]))

            doc.build(content)

        generate_pdf()

        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download PDF Report", f)
