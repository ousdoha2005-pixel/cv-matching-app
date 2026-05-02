import streamlit as st
import pickle

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("CV Matching App")

cv = st.text_area("Paste your CV")

if st.button("Analyze"):
    if cv:
        X = vectorizer.transform([cv])
        pred = model.predict(X)
        job = le.inverse_transform(pred)
        st.success(f"Predicted job: {job[0]}")
    else:
        st.warning("Please enter CV text")
