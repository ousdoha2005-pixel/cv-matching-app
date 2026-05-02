import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
import altair as alt

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="CV Matcher PRO", page_icon="🚀", layout="centered")

st.title("🚀 CV Matching App PRO")
st.markdown("### 🔍 Smart CV Analysis & Job Matching")

# =========================
# INPUTS
# =========================
uploaded_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])
cv_text_input = st.text_area("✍️ Or paste your CV")
job_desc = st.text_area("🧾 Paste Job Description (optional)")

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    return text.lower().strip()

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

    # CV SOURCE
    if uploaded_file:
        cv = extract_text_from_pdf(uploaded_file)
    else:
        cv = cv_text_input

    if cv:

        cv_clean = clean_text(cv)
        cv_vec = vectorizer.transform([cv_clean])

        # =========================
        # 🎯 PREDICTION
        # =========================
        pred = model.predict(cv_vec)
        job = le.inverse_transform(pred)

        st.success(f"✅ Predicted job: **{job[0]}**")

        # =========================
        # 🏆 TOP JOBS
        # =========================
        probs = model.predict_proba(cv_vec)[0]
        top_indices = probs.argsort()[-5:][::-1]

        st.subheader("🏆 Top Matching Jobs")

        jobs = [le.classes_[i] for i in top_indices]
        scores = [probs[i]*100 for i in top_indices]

        df = pd.DataFrame({"Job": jobs, "Score": scores})

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Score", title="Probability (%)"),
            y=alt.Y("Job", sort='-x'),
            color="Job"
        )

        st.altair_chart(chart, use_container_width=True)

        # =========================
        # 🔥 IF JOB DESCRIPTION EXISTS
        # =========================
        if job_desc:

            job_clean = clean_text(job_desc)
            job_vec = vectorizer.transform([job_clean])

            # similarity
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

            # skills boost
            skills = ["python", "java", "sql", "machine learning", "django", "flask", "aws", "docker"]

            matched_skills = []
            for skill in skills:
                if skill in cv_clean and skill in job_clean:
                    similarity += 0.1
                    matched_skills.append(skill)

            similarity = min(similarity, 1)
            score = similarity * 100

            # =========================
            # 📊 DASHBOARD
            # =========================
            st.divider()
            st.header("📊 Matching Dashboard")

            col1, col2 = st.columns(2)

            # 🔹 SCORE
            with col1:
                st.metric("Match Score", f"{score:.2f}%")
                st.progress(int(score))

                if score > 70:
                    st.success("🔥 Strong Match")
                elif score > 40:
                    st.warning("⚠️ Medium Match")
                else:
                    st.error("❌ Weak Match")

            # 🔹 COMMON KEYWORDS
            with col2:
                common_words = list(set(cv_clean.split()) & set(job_clean.split()))
                st.write("🔑 Common Keywords")
                st.write(common_words[:15])

            # =========================
            # 🧠 SKILLS
            # =========================
            st.subheader("🧠 Matched Skills")

            if matched_skills:
                st.success(matched_skills)

                skill_df = pd.DataFrame({
                    "Skill": matched_skills,
                    "Value": [1]*len(matched_skills)
                })

                skill_chart = alt.Chart(skill_df).mark_bar().encode(
                    x="Skill",
                    y="Value",
                    color="Skill"
                )

                st.altair_chart(skill_chart, use_container_width=True)

            else:
                st.warning("No strong skill match")

            # =========================
            # 💡 SUGGESTIONS
            # =========================
            st.subheader("💡 Suggestions")

            if score < 50:
                st.write("👉 Add more relevant skills from the job description.")
            elif score < 70:
                st.write("👉 Improve your CV by adding more technologies.")
            else:
                st.write("🎉 Excellent match!")

    else:
        st.warning("⚠️ Please upload or paste your CV")
