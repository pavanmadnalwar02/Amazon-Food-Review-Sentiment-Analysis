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
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove numbers and special characters
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    # Return cleaned text only if it contains at least one valid word
    return " ".join(filtered_words) if filtered_words else None  

# Streamlit UI
st.title("🚀 Welcome to Pavan Madnalwar Sentiment Analysis App! 🎉")
st.subheader("🔍Amazon Fine Food Review Sentiment Analysis")
st.write("📝 Enter a review, and our model will determine whether it is **Positive** 😊 or **Negative** 😠.")


# User Input
user_input = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    cleaned_input = preprocess_text(user_input)

    if cleaned_input:  # Check if the cleaned input is valid
        transformed_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_input)[0]
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.subheader("Prediction Result:")
        st.write(f"### {sentiment}")
    else:
        st.warning("Please enter a valid review with meaningful words before analyzing.")