import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# UI
st.set_page_config(page_title="CV Matching App", page_icon="💼", layout="centered")

st.title("💼 CV Matching App")
st.markdown("### 🔍 Match your CV with Job Description")

# inputs
cv = st.text_area("📄 Paste your CV")
job_desc = st.text_area("🧾 Paste Job Description")

# clean function
def clean_text(text):
    return text.lower().strip()

# button
if st.button("🚀 Analyze"):
    if cv and job_desc:
        
        # cleaning
        cv_clean = clean_text(cv)
        job_clean = clean_text(job_desc)
        
        # transform
        cv_vec = vectorizer.transform([cv_clean])
        job_vec = vectorizer.transform([job_clean])
        
        # prediction
        pred = model.predict(cv_vec)
        job = le.inverse_transform(pred)
        
        # similarity
        similarity = cosine_similarity(cv_vec, job_vec)[0][0]
        
        # boost skills
        common_skills = ["python", "java", "sql", "machine learning", "django", "flask", "aws", "docker"]

        matched_skills = []
        for skill in common_skills:
            if skill in cv_clean and skill in job_clean:
                similarity += 0.1
                matched_skills.append(skill)
        
        # boost extra
        if "developer" in cv_clean and "developer" in job_clean:
            similarity += 0.2
        
        similarity = min(similarity, 1)
        score = similarity * 100
        
        # common words
        common_words = list(set(cv_clean.split()) & set(job_clean.split()))
        
        # 🎯 RESULTS UI
        st.success(f"✅ Predicted job: **{job[0]}**")
        
        # progress bar
        st.subheader("📊 Matching Score")
        st.progress(int(score))
        
        if score > 70:
            st.success(f"🔥 Strong Match: {score:.2f}%")
        elif score > 40:
            st.warning(f"⚠️ Medium Match: {score:.2f}%")
        else:
            st.error(f"❌ Weak Match: {score:.2f}%")
        
        # keywords
        st.subheader("🔑 Common Keywords")
        st.write(common_words)
        
        # matched skills
        st.subheader("🧠 Matched Skills")
        if matched_skills:
            st.success(matched_skills)
        else:
            st.warning("No strong skill match found")
        
        # suggestions
        st.subheader("💡 Suggestions")
        if score < 50:
            st.write("👉 Add more relevant skills from the job description to your CV.")
        elif score < 70:
            st.write("👉 Improve your CV by adding more specific technologies.")
        else:
            st.write("🎉 Your CV is well aligned with the job!")
    
    else:
        st.warning("⚠️ Please enter both CV and Job Description")
