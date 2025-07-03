
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(path_or_url:str=None):
    try:
        if path_or_url:
            return pd.read_csv(path_or_url)
        return pd.read_csv("data/cloud_kitchen_survey_synthetic_clean.csv")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()
