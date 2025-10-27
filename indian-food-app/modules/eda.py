import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_state_distribution(df):
    fig, ax = plt.subplots()
    df['state'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

def plot_diet_distribution_by_state(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(df['state'], df['diet']).plot(kind='bar', ax=ax)
    st.pyplot(fig)
