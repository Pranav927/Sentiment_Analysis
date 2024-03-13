import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords

# Load the IMDb movie reviews dataset
@st.cache_data
def load_data():
    data_path = r"C:\Users\prana\Downloads\Aditi\Pranav\Portfolio_Projects\Sentiment Analysis\IMDB_Dataset.csv"  # Update with the path to your dataset
    data = pd.read_csv(data_path)
    return data

data = load_data()

# Data cleaning
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stop words
    return text

data['cleaned_review'] = data['review'].apply(clean_text)

# Preprocessing
X = data['cleaned_review']
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training and evaluation
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for model_name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"{model_name} accuracy: {accuracy}")
