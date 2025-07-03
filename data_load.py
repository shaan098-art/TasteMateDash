import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Loads CSV data. If `path` is None it tries the default demo file.
    Caches the result for fast reloads.
    """
    default_path = "cloud_kitchen_survey_synthetic_clean.csv"
    try:
        src = path or default_path
        df = pd.read_csv(src)
        # Basic sanity fix: convert semicolon lists to str in case of NaNs
        list_cols = ["order_windows", "fav_cuisines", "allergens", "liked_features"]
        for c in list_cols:
            df[c] = df[c].astype(str).fillna("")
        return df
    except Exception as e:
        st.error(f"❌ Could not load data → {e}")
        return pd.DataFrame()
