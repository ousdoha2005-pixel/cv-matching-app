import streamlit as st
import pickle
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go

# =========================
# LOAD MODEL FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

try:
    le = pickle.load(open("label_encoder.pkl", "rb"))
except:
    le = None

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="CV Matching PRO", page_icon="🚀")
st.title("🚀 CV Matching App PRO")

uploaded_file = st.file_uploader("📄 Upload your CV", type=["pdf"])
cv_text_input = st.text_area("✍️ Or paste your CV")
job_desc = st.text_area("🧾 Paste Job Description")

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Skills DB
skills_db = [
    "python","java","c++","sql","machine learning","deep learning",
    "data science","data analysis","nlp","computer vision",
    "pandas","numpy","tensorflow","keras","pytorch",
    "flask","django","react","javascript","html","css",
    "aws","docker","git","linux"
]

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in skills_db if skill in text]))

# =========================
# MAIN
# =========================
if st.button("🚀 Analyze"):

    # CV INPUT
    if uploaded_file:
        cv = extract_text_from_pdf(uploaded_file)
    else:
        cv = cv_text_input

    if not cv:
        st.warning("⚠️ Please add your CV")
        st.stop()

    cv_clean = clean_text(cv)
    cv_vec = vectorizer.transform([cv_clean])

    st.info(f"📄 CV Length: {len(cv_clean.split())} words")

    # =========================
    # CLASSIFICATION
    # =========================
    pred = model.predict(cv_vec)

    if le:
        try:
            job_name = le.inverse_transform(pred)[0]
        except:
            job_name = str(pred[0])
    else:
        job_name = str(pred[0])

    st.success(f"🎯 Predicted Job: {job_name}")

    # =========================
    # TOP JOBS
    # =========================
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(cv_vec)[0]
        top_idx = probs.argsort()[-5:][::-1]

        if le:
            jobs = [le.classes_[i] for i in top_idx]
        else:
            jobs = [str(i) for i in top_idx]

        scores = [probs[i]*100 for i in top_idx]

        df = pd.DataFrame({"Job": jobs, "Score": scores})

        st.subheader("🏆 Top Matching Jobs")

        fig = px.bar(df, x="Score", y="Job", orientation="h")
        st.plotly_chart(fig)

        st.plotly_chart(px.pie(df, names="Job", values="Score"))

    # =========================
    # MATCHING
    # =========================
    if job_desc:

        job_clean = clean_text(job_desc)
        job_vec = vectorizer.transform([job_clean])

        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # Skills
        cv_skills = extract_skills(cv)
        job_skills = extract_skills(job_desc)

        matched = list(set(cv_skills) & set(job_skills))

        # Boost score
        if job_skills:
            skill_score = len(matched) / len(job_skills)
            similarity = 0.6 * similarity + 0.4 * skill_score

        score = similarity * 100

        # =========================
        # DASHBOARD
        # =========================
        st.divider()
        st.header("📊 Matching Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Match Score", f"{score:.2f}%")
            st.progress(int(score))

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "green"},
                    ]
                }
            ))
            st.plotly_chart(fig_gauge)

            if score > 75:
                st.success("🔥 Strong Match")
            elif score > 50:
                st.warning("⚠️ Medium Match")
            else:
                st.error("❌ Weak Match")

        with col2:
            keywords = list(set(cv_clean.split()) & set(job_clean.split()))
            st.subheader("🔑 Keywords Match")
            st.write(keywords[:15])

        # =========================
        # SKILLS ANALYSIS
        # =========================
        st.subheader("🧠 Skills Analysis")

        st.write("📄 CV Skills:", cv_skills)
        st.write("🧾 Job Skills:", job_skills)

        st.subheader("🎯 Matched Skills")
        st.success(matched if matched else "No match")

        # =========================
        # COVERAGE
        # =========================
        if job_skills:
            coverage = (len(matched) / len(job_skills)) * 100
            st.progress(int(coverage))
            st.write(f"Coverage: {coverage:.1f}%")

        # =========================
        # SUGGESTIONS
        # =========================
        missing = list(set(job_skills) - set(cv_skills))

        st.subheader("💡 Suggestions")

        if missing:
            st.warning("Add these skills:")
            st.write(missing[:5])
        else:
            st.success("Perfect match 🎉")
