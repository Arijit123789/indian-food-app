import streamlit as st
from modules.data_loader import load_data
from modules.eda import plot_state_distribution, plot_diet_distribution_by_state
from modules.model import preprocess_features, train_classifier, recommend_dishes
import os
import pandas as pd

# --- Page Configuration (Sets a professional theme) ---
st.set_page_config(
    page_title="Indian Food Recommender",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- File Path (Same as before) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'indian_food.csv')

# --- Custom CSS (To make it look awesome) ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Title style */
    .stTitle {
        font-weight: bold;
        color: #FF4B4B; /* Streamlit's red */
    }

    /* Container styles */
    .st-emotion-cache-z5fcl4 { /* This is a common class for containers */
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Button style */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #E03C3C;
        color: white;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #FFF;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading (Cached for performance) ---
@st.cache_data
def load_and_preprocess_data(path):
    try:
        df = load_data(path)
        X, vectorizer = preprocess_features(df)
        return df, X, vectorizer
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at {DATA_PATH}")
        st.error("Please make sure 'indian_food.csv' is in the 'data' directory.")
        st.stop()

# Load all data at the start
df, X, vectorizer = load_and_preprocess_data(DATA_PATH)

# ==================================================================================
# --- SIDEBAR: Navigation ---
# ==================================================================================
with st.sidebar:
    st.image("https://i.imgur.com/H1F10H4.png", width=100) # A placeholder logo
    st.title("🍲 Indian Food App")
    
    st.markdown("---")
    
    app_section = st.radio(
        "Choose a Function",
        ("Dish Recommender", "Exploratory Data Analysis (EDA)", "State Classifier")
    )
    
    st.markdown("---")
    st.info("This app recommends Indian dishes based on ingredients and classifies them by state.")


# ==================================================================================
# --- MAIN PAGE: Content based on sidebar navigation ---
# ==================================================================================

# --- Function 1: Dish Recommender (Default Page) ---
if app_section == "Dish Recommender":
    st.title("Dish Recommender")
    st.markdown("Describe the kind of dish or ingredients you're looking for, and we'll find the top 5 matches!")
    
    with st.container(border=True):
        query = st.text_input('Describe the dish or ingredients (e.g., "chicken spicy" or "rice milk sweet"):')
        
        if query:
            st.markdown("### Here are your recommendations:")
            recommended = recommend_dishes(df, query, vectorizer)
            
            # Display results in a much cleaner table
            st.dataframe(
                recommended,
                column_config={
                    "name": "Dish Name",
                    "state": "State",
                    "ingredients": "Ingredients"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Enter some ingredients to get started.")

# --- Function 2: Exploratory Data Analysis (EDA) ---
elif app_section == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis")
    st.markdown("Explore the dataset to understand the distribution of dishes.")
    
    with st.container(border=True):
        st.header("Distribution of Dishes by State")
        st.markdown("This chart shows how many dishes in the dataset come from each state.")
        plot_state_distribution(df)

    with st.container(border=True):
        st.header("Diet Distribution by State")
        st.markdown("This chart shows the breakdown of vegetarian vs. non-vegetarian dishes across states.")
        plot_diet_distribution_by_state(df)

# --- Function 3: State Classifier ---
elif app_section == "State Classifier":
    st.title("State Classifier Model")
    st.markdown("Train a Machine Learning model to predict a dish's state of origin based on its ingredients.")

    with st.container(border=True):
        st.header("Train Classifier")
        st.markdown("Click the button to train a Random Forest Classifier on the dataset. This may take a moment.")
        
        if st.button('Train Classifier'):
            with st.spinner('Training model...'):
                clf, acc = train_classifier(df, X)
                st.success(f"Classifier Accuracy: {acc * 100:.2f}%")
                st.markdown("This accuracy score represents how well the model learned to predict the state from the ingredients on a hidden test set.")
