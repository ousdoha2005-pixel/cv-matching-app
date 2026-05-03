import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# تحميل stopwords مرة وحدة
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# cleaning function (نفس اللي درتي ف training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# UI
st.title("📄 CV Classification App")

user_input = st.text_area("Paste your CV here:")

if st.button("Predict"):
    if user_input.strip() != "":
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)
        category = le.inverse_transform(pred)

        st.success(f"Predicted Category: {category[0]}")
    else:
        st.warning("Please enter some text")
