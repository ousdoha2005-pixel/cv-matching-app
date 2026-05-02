import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# UI
st.set_page_config(page_title="CV Matcher PRO", page_icon="🚀")
st.title("🚀 CV Matching App PRO")

# upload CV
uploaded_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])

# text input fallback
cv_text_input = st.text_area("✍️ Or paste your CV")

job_desc = st.text_area("🧾 Paste Job Description")

# clean
def clean_text(text):
    return text.lower().strip()

# extract PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# button
if st.button("🚀 Analyze"):
    
    if uploaded_file:
        cv = extract_text_from_pdf(uploaded_file)
    else:
        cv = cv_text_input

    if cv and job_desc:

        cv_clean = clean_text(cv)
        job_clean = clean_text(job_desc)

        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])

        # prediction
        pred = model.predict(cv_vec)
        job = le.inverse_transform(pred)

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

        # common words
        common_words = list(set(cv_clean.split()) & set(job_clean.split()))

        # 🔥 TOP JOBS RANKING
        probs = model.predict_proba(cv_vec)[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_jobs = [(le.classes_[i], probs[i]) for i in top_indices]

        # UI
        st.success(f"✅ Predicted job: {job[0]}")

        st.subheader("📊 Matching Score")
        st.progress(int(score))

        if score > 70:
            st.success(f"🔥 Strong Match: {score:.2f}%")
        elif score > 40:
            st.warning(f"⚠️ Medium Match: {score:.2f}%")
        else:
            st.error(f"❌ Weak Match: {score:.2f}%")

        st.subheader("🔑 Common Keywords")
        st.write(common_words)

        st.subheader("🧠 Matched Skills")
        if matched_skills:
            st.success(matched_skills)
        else:
            st.warning("No strong skill match")

        # 🔥 TOP JOBS
        st.subheader("🏆 Top Matching Jobs")
        for job_name, prob in top_jobs:
            st.write(f"{job_name} → {prob*100:.2f}%")

        # suggestions
        st.subheader("💡 Suggestions")
        if score < 50:
            st.write("👉 Add more relevant skills from job description.")
        elif score < 70:
            st.write("👉 Improve your CV with more technologies.")
        else:
            st.write("🎉 Excellent match!")

    else:
        st.warning("⚠️ Please provide CV and Job Description")
