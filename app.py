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
st.set_page_config(page_title="CV Matcher Dashboard 🚀", layout="wide")

# ======================
# 🌐 LANGUAGE
# ======================
st.sidebar.title("⚙️ Settings / Paramètres")

lang = st.sidebar.selectbox("🌐 Language / Langue", ["English", "Français"])

def t(en, fr):
    return en if lang == "English" else fr

# ======================
# CSS
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
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = list(set(text.split()))
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
# UI
# ======================
st.title(t("🚀 CV Matcher Dashboard", "🚀 Tableau de bord CV"))

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader(t("Upload CV", "Importer CV"), type=["pdf"])

with col2:
    job_desc = st.text_area(t("Job Description", "Description du poste"))

cv_text = read_pdf(cv_file) if cv_file else ""

# ======================
# ANALYZE
# ======================
if st.button(t("Analyze", "Analyser")):

    if cv_text == "" or job_desc == "":
        st.warning(t("Fill inputs", "Remplir les champs"))
    else:
        with st.spinner("..."):
            time.sleep(1)

        cv_clean = clean_text(cv_text)
        job_clean = clean_text(job_desc)

        # Prediction
        cv_vec = vectorizer.transform([cv_clean])
        pred = model.predict(cv_vec)[0]
        probs = model.predict_proba(cv_vec)[0]
        confidence = max(probs)

        # Matching
        if ST_AVAILABLE:
            cv_emb = st_model.encode([cv_clean])
            job_emb = st_model.encode([job_clean])
            similarity = cosine_similarity(cv_emb, job_emb)[0][0]
        else:
            job_vec = vectorizer.transform([job_clean])
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # ======================
        # 🎯 KPI DASHBOARD
        # ======================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"<div class='card'>🎯 {pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>📊 {confidence:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>🔗 {similarity:.2f}</div>", unsafe_allow_html=True)

        # ======================
        # 🎯 GAUGE CHART
        # ======================
        st.subheader("🎯 Matching Score")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity * 100,
            title={'text': "Match %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)

        # ======================
        # 📊 BAR CHART
        # ======================
        st.subheader("📊 Prediction Distribution")

        df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": probs
        })

        fig_bar = px.bar(df, x="Category", y="Probability",
                         color="Probability",
                         title="Model Confidence")

        st.plotly_chart(fig_bar, use_container_width=True)

        # ======================
        # 📈 TOP 5
        # ======================
        st.subheader("🏆 Top Predictions")

        top5 = np.argsort(probs)[-5:][::-1]
        for i in top5:
            st.write(f"{label_encoder.inverse_transform([i])[0]} → {probs[i]:.2f}")

        # ======================
        # 📊 PIE CHART
        # ======================
        st.subheader("📊 Category Share")

        fig_pie = px.pie(df, names="Category", values="Probability")
        st.plotly_chart(fig_pie, use_container_width=True)

        # ======================
        # 🧠 MATCH INTERPRETATION
        # ======================
        st.subheader("🧠 Interpretation")

        if similarity > 0.75:
            st.success("🔥 Strong Match")
        elif similarity > 0.5:
            st.info("🙂 Medium Match")
        else:
            st.error("❌ Weak Match")
