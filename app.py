
import streamlit as st
import pandas as pd
from src import data_load, data_viz, classification, clustering, association_rules, regression

st.set_page_config(page_title="TasteMateDash", page_icon="🍲", layout="wide")

st.sidebar.title("TasteMateDash")
tab = st.sidebar.radio("Navigate", ("Data Visualisation", "Classification", "Clustering", "Association Rule Mining", "Regression", "Upload/Download"))

if "data" not in st.session_state:
    st.session_state["data"] = data_load.load_data()

df = st.session_state["data"]

if tab == "Data Visualisation":
    data_viz.main_viz(df)
elif tab == "Classification":
    classification.classification_tab(df)
elif tab == "Clustering":
    clustering.clustering_tab(df)
elif tab == "Association Rule Mining":
    association_rules.association_rule_tab(df)
elif tab == "Regression":
    regression.regression_tab(df)
else:
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        st.session_state["data"] = pd.read_csv(uploaded)
        st.success("Data replaced!")
    st.download_button("Download current data", df.to_csv(index=False), "TasteMateDash_data.csv")
