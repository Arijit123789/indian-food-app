import streamlit as st
from modules.data_loader import load_data
from modules.eda import plot_state_distribution, plot_diet_distribution_by_state
from modules.model import preprocess_features, train_classifier, recommend_dishes
import os

# --- Corrections Start Here ---

# Build a path relative to the current script file (app.py)
# This is much more reliable on platforms like Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'IndianFood.csv')

# --- Corrections End Here ---

st.title('Indian Food Classifier & Recommender')

# Data load
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Error: The data file was not found at {DATA_PATH}")
    st.error("Please make sure 'IndianFood.csv' is in the 'data' directory.")
    st.stop()


st.header('Exploratory Data Analysis')
if st.button("Show State Distribution"):
    plot_state_distribution(df)
if st.button("Show Diet Distribution by State"):
    plot_diet_distribution_by_state(df)

st.header('Dish Recommendation')
query = st.text_input('Describe the dish or ingredients:')
if query:
    X, vectorizer = preprocess_features(df)
    recommended = recommend_dishes(df, query, vectorizer)
    st.write(recommended)

st.header('State Classifier Accuracy')
if st.button('Train Classifier'):
    X, _ = preprocess_features(df)
    clf, acc = train_classifier(df, X)
    st.success(f"Classifier Accuracy: {acc * 100:.2f}%")