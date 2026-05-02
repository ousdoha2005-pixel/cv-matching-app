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
job_desc = st.text_area("🧾 Paste Job Description (optional)")

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
# BUTTON
# =========================
if st.button("🚀 Analyze"):

    # get CV
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
        job = le.inverse_transform(pred)
        job_name = job[0]
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

        fig_pie = px.pie(df, names="Job", values="Score", title="Job Distribution")
        st.plotly_chart(fig_pie)

    # =========================
    # MATCHING
    # =========================
    if job_desc:

        job_clean = clean_text(job_desc)
        job_vec = vectorizer.transform([job_clean])

        similarity = cosine_similarity(cv_vec, job_vec)[0][0]

        skills = ["python","java","sql","machine learning","django","flask","aws","docker"]

        match_count = 0
        matched_skills = []

        for skill in skills:
            if skill in cv_clean and skill in job_clean:
                match_count += 1
                matched_skills.append(skill)

        similarity += match_count * 0.05
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

            st.info(f"""
🧠 Why this score?

- Similarity: {round(similarity*100,2)}%
- Matched Skills: {len(matched_skills)}
- CV Length: {len(cv_clean.split())}
""")

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

            if score > 70:
                st.success("🔥 Strong Match")
            elif score > 40:
                st.warning("⚠️ Medium Match")
            else:
                st.error("❌ Weak Match")

        with col2:
            stop_words = ["the","a","to","in","of","and","with","for","on"]

            common_words = [
                w for w in set(cv_clean.split()) & set(job_clean.split())
                if w not in stop_words and len(w) > 2
            ]

            st.write("🔑 Keywords")
            st.write(common_words[:15])

        # =========================
        # SKILLS
        # =========================
        st.subheader("🧠 Matched Skills")

        if matched_skills:
            st.success(matched_skills)

            skill_df = pd.DataFrame({
                "Skill": matched_skills,
                "Value": [1]*len(matched_skills)
            })

            fig_skill = px.pie(skill_df, names="Skill", values="Value")
            st.plotly_chart(fig_skill)

        else:
            st.warning("No strong skill match")

        # =========================
        # SUGGESTIONS
        # =========================
        st.subheader("💡 Suggestions")

        if score < 50:
            st.write("👉 Add more skills from job description")
        elif score < 70:
            st.write("👉 Improve your CV")
        else:
            st.write("🎉 Excellent match!")
