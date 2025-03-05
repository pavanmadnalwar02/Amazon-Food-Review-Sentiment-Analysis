# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 18:40:54 2025

@author: pavan
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
with open(r"C:\Users\pavan\Documents\Data science\AI\Sentiment Analysis\naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\pavan\Documents\Data science\AI\Sentiment Analysis\tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    text = " ".join([word for word in words if word not in stop_words])
    return text

# Streamlit UI
st.title("Amazon Fine Food Review Sentiment Analysis")
st.write("This app classifies Amazon product reviews as **Positive** or **Negative** using a Naïve Bayes model.")

# User Input
user_input = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_input = preprocess_text(user_input)
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.subheader("Prediction Result:")
        st.write(f"### {sentiment}")
    else:
        st.warning("Please enter a valid review before analyzing.")
