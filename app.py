import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import re

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

try:
    le = pickle.load(open("label_encoder.pkl", "rb"))
except:
    le = None

# =========================
# UI
# =========================
st.set_page_config(page_title="CV Matcher PRO", page_icon="🚀")
st.title("🚀 CV Matching App PRO")

uploaded_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])
cv_text_input = st.text_area("✍️ Or paste your CV")
job_desc = st.text_area("🧾 Paste Job Description")

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# =========================
# MAIN BUTTON
# =========================
if st.button("🚀 Analyze"):

    # =========================
    # GET CV
    # =========================
    if uploaded_file:
        cv = extract_text_from_pdf(uploaded_file)
    else:
        cv = cv_text_input

    if not cv or len(cv.strip()) == 0:
        st.warning("⚠️ Please upload or paste your CV")
        st.stop()

    # =========================
    # CLEAN + VECTORIZE
    # =========================
    cv_clean = clean_text(cv)
    cv_vec = vectorizer.transform([cv_clean])

    st.info(f"📄 CV Length: {len(cv_clean.split())} words")

    # =========================
    # PREDICTION
    # =========================
    pred = model.predict(cv_vec)

    if le:
        try:
            job_name = le.inverse_transform(pred)[0]
        except:
            job_name = str(pred[0])
    else:
        job_name = str(pred[0])

    st.success(f"✅ Predicted job: {job_name}")

    # =========================
    # TOP JOBS
    # =========================
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(cv_vec)[0]
        top_indices = probs.argsort()[-5:][::-1]

        if le:
            jobs = [le.classes_[i] for i in top_indices]
        else:
            jobs = [str(i) for i in top_indices]

        scores = [probs[i]*100 for i in top_indices]

        df = pd.DataFrame({"Job": jobs, "Score": scores})

        st.subheader("🏆 Top Matching Jobs")

        chart = alt.Chart(df).mark_bar().encode(
            x="Score",
            y=alt.Y("Job", sort='-x'),
            color="Job"
        )
        st.altair_chart(chart, use_container_width=True)

        fig_pie = px.pie(df, names="Job", values="Score")
        st.plotly_chart(fig_pie)

    # =========================
    # MATCHING SCORE
    # =========================
    if job_desc:

        job_clean = clean_text(job_desc)
        job_vec = vectorizer.transform([job_clean])

        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        # Skills
        skills = [
            "python","java","sql","machine learning","deep learning",
            "data analysis","django","flask","aws","docker"
        ]

        matched_skills = [
            skill for skill in skills
            if skill in cv_clean and skill in job_clean
        ]

        # Boost score
        similarity += len(matched_skills) * 0.03
        similarity = min(similarity, 1)

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

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Match Score"},
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

            # Messages
            if score > 75:
                st.success("🔥 Strong Match")
            elif score > 50:
                st.warning("⚠️ Medium Match")
            else:
                st.error("❌ Weak Match")

            st.info(f"""
🧠 Details:
- Similarity: {round(similarity*100,2)}%
- Matched Skills: {len(matched_skills)}
- CV Words: {len(cv_clean.split())}
""")

        with col2:
            stop_words = ["the","a","to","in","of","and","with","for","on"]

            keywords = [
                w for w in set(cv_clean.split()) & set(job_clean.split())
                if w not in stop_words and len(w) > 2
            ]

            st.subheader("🔑 Keywords Match")
            st.write(keywords[:15])

        # =========================
        # SKILLS DISPLAY
        # =========================
        st.subheader("🧠 Matched Skills")

        if matched_skills:
            st.success(matched_skills)

            df_skills = pd.DataFrame({
                "Skill": matched_skills,
                "Value": [1]*len(matched_skills)
            })

            fig = px.pie(df_skills, names="Skill", values="Value")
            st.plotly_chart(fig)
        else:
            st.warning("No strong skill match")

        # =========================
        # SUGGESTIONS
        # =========================
        st.subheader("💡 Suggestions")

        if score < 50:
            st.write("👉 Add more relevant skills from job description")
        elif score < 75:
            st.write("👉 Improve your CV wording")
        else:
            st.write("🎉 Excellent match!")
