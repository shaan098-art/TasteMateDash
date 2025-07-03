
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Local module imports (all expected to sit in the SAME directory as app.py)
# -----------------------------------------------------------------------------
import data_load            # handles loading CSV from repo or upload
import data_viz             # tab 1 – descriptive insights
import classification       # tab 2 – KNN, DT, RF, GBRT
import clustering           # tab 3 – k-means personas
import association_rules    # tab 4 – apriori mining
import regression           # tab 5 – linear, ridge, lasso, DT regressors
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="TasteMateDash",
    page_icon="🍲",
    layout="wide"
)

st.sidebar.title("TasteMateDash")
page = st.sidebar.radio(
    "Navigate",
    (
        "Data Visualisation",
        "Classification",
        "Clustering",
        "Association Rule Mining",
        "Regression",
        "Upload / Download"
    )
)

# -----------------------------------------------------------------------------#
#  Load data once and store in session_state
# -----------------------------------------------------------------------------#
if "data" not in st.session_state:
    st.session_state["data"] = data_load.load_data()

df = st.session_state["data"]

# -----------------------------------------------------------------------------#
#  Page router
# -----------------------------------------------------------------------------#
if page == "Data Visualisation":
    data_viz.main_viz(df)

elif page == "Classification":
    classification.classification_tab(df)

elif page == "Clustering":
    clustering.clustering_tab(df)

elif page == "Association Rule Mining":
    association_rules.association_rule_tab(df)

elif page == "Regression":
    regression.regression_tab(df)

elif page == "Upload / Download":
    st.header("Upload new data or download the current dataset")

    uploaded = st.file_uploader(
        "Upload a **clean** survey CSV (same column structure).",
        type="csv"
    )
    if uploaded is not None:
        st.session_state["data"] = pd.read_csv(uploaded)
        st.success("📈 New data loaded! Switch tabs to analyse it.")
        df = st.session_state["data"]

    st.download_button(
        "Download current data",
        df.to_csv(index=False),
        file_name="TasteMateDash_data.csv",
        mime="text/csv"
    )
    st.write(
        "Use the other tabs to build models or visualise insights, then come "
        "back here to download the results."
    )
