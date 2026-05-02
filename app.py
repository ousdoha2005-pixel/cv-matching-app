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
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
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

    if uploaded_file:
        cv = extract_text_from_pdf(uploaded_file)
    else:
        cv = cv_text_input

    if cv:

        cv_clean = clean_text(cv)
        cv_vec = vectorizer.transform([cv_clean])

        st.info(f"📄 CV Length: {len(cv_clean.split())} words")

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

        jobs = [le.classes_[i] for i in top_indices]
        scores = [probs[i]*100 for i in top_indices]

        df = pd.DataFrame({"Job": jobs, "Score": scores})

        st.subheader("🏆 Top Matching Jobs")

        # BAR (altair)
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Score", title="Probability (%)"),
            y=alt.Y("Job", sort='-x'),
            color="Job"
        )
        st.altair_chart(chart, use_container_width=True)

        # 🔥 PIE CHART
        fig_pie = px.pie(df, names="Job", values="Score", title="Job Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        # =========================
        # 🔥 MATCHING (IF JD EXISTS)
        # =========================
        if job_desc:

            job_clean = clean_text(job_desc)
            job_vec = vectorizer.transform([job_clean])
            similarity = cosine_similarity(cv_vec, job_vec)[0][0]

            match_count = 0
            matched_skills = []

            for skill in skills:
                 if skill in cv_clean and skill in job_clean:
                       match_count += 1
                       matched_skills.append(skill)

                similarity += match_count * 0.08
                similarity = min(similarity, 1)
                score = similarity * 100


            st.divider()
            st.header("📊 Matching Dashboard")

            col1, col2 = st.columns(2)

            # 🔹 SCORE + GAUGE
            with col1:
                st.metric("Match Score", f"{score:.2f}%")
                st.progress(int(score))
                # 👇 هنا زيد Why this score
                st.caption("Explanation of how the score is calculated")
                st.info(f"""
               🧠 **Why this score?**

                - 🔹 Similarity (NLP): {round(similarity*100,2)}%
                - 🔹 Matched Skills: {len(matched_skills)}
                - 🔹 CV Length: {len(cv_clean.split())} words
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

            # 🔹 CLEAN KEYWORDS
            with col2:
                stop_words = ["the","a","to","in","of","and","with","for","on","at","is","are",
                              "this","that","be","as","an","by","from","or","it","we","you"]

                common_words = [
                    w for w in set(cv_clean.split()) & set(job_clean.split())
                    if w not in stop_words and len(w) > 2
                ]

                st.write("🔑 Clean Keywords")
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

                # 🔥 PIE SKILLS
                fig_skill = px.pie(skill_df, names="Skill", values="Value", title="Matched Skills")
                st.plotly_chart(fig_skill, use_container_width=True)

            else:
                st.warning("No strong skill match")

            # =========================
            # 💡 EXPLANATION
            # =========================
            st.info("💡 Matching is calculated using NLP cosine similarity + skill matching.")

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
