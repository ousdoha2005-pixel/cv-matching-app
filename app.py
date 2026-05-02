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

# button
if st.button("Analyze"):
    if cv and job_desc:
        
        # transform
        cv_vec = vectorizer.transform([cv])
        job_vec = vectorizer.transform([job_desc])
        
        # prediction
        pred = model.predict(cv_vec)
        job = le.inverse_transform(pred)
        
        # similarity
        similarity = cosine_similarity(cv_vec, job_vec)[0][0]
        score = similarity * 100
        
        # show result
        st.success(f"Predicted job: {job[0]}")
        
        # matching score display
        if score > 70:
            st.success(f"🔥 Strong Match: {score:.2f}%")
        elif score > 40:
            st.warning(f"⚠️ Medium Match: {score:.2f}%")
        else:
            st.error(f"❌ Weak Match: {score:.2f}%")
    
    else:
        st.warning("Please enter both CV and Job Description")
