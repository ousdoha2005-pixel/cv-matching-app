import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# title
st.title("CV Matching App")

# inputs
cv = st.text_area("Paste your CV")
job_desc = st.text_area("Paste Job Description")

# clean function
def clean_text(text):
    return text.lower().strip()

# button
if st.button("Analyze"):
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
        
        # boost
        if "python" in cv_clean and "python" in job_clean:
            similarity += 0.2
        if "developer" in cv_clean and "developer" in job_clean:
            similarity += 0.2
        
        similarity = min(similarity, 1)
        score = similarity * 100
        
        # results
        st.success(f"Predicted job: {job[0]}")
        
        if score > 70:
            st.success(f"🔥 Strong Match: {score:.2f}%")
        elif score > 40:
            st.warning(f"⚠️ Medium Match: {score:.2f}%")
        else:
            st.error(f"❌ Weak Match: {score:.2f}%")
    
    else:
        st.warning("Please enter both CV and Job Description")
