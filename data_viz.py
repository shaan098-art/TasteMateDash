# data_load.py

import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Attempts to load the survey CSV from:
      1) the provided path
      2) root folder: 'cloud_kitchen_survey_synthetic_clean.csv'
      3) data subfolder: 'data/cloud_kitchen_survey_synthetic_clean.csv'
    If none are found, returns an empty DataFrame.
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        "cloud_kitchen_survey_synthetic_clean.csv",
        os.path.join("data", "cloud_kitchen_survey_synthetic_clean.csv")
    ]

    for fp in candidates:
        if fp and os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                # ensure multi-select columns are strings
                for c in ["order_windows","fav_cuisines","allergens","liked_features"]:
                    if c in df.columns:
                        df[c] = df[c].astype(str).fillna("")
                return df
            except Exception as e:
                st.error(f"Found '{fp}' but failed to load: {e}")
                return pd.DataFrame()

    # If we get here, no file was found
    st.warning(
        "⚠️ No default data file found.\n"
        "Please upload your CSV using the **Upload / Download** tab."
    )
    return pd.DataFrame()
