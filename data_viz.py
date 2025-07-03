# data_viz.py

import streamlit as st
import pandas as pd

def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    """Explode a semicolon-separated column into value counts."""
    return df[col].str.split(";").explode().str.strip().value_counts()

def main_viz(df: pd.DataFrame) -> None:
    st.header("📊 Exploratory Data Visualisation (Streamlit + pandas only)")

    # 1. Orders per week by age group
    st.subheader("Orders per Week by Age Group")
    crosstab = pd.crosstab(df["orders_per_week"], df["age_group"])
    st.bar_chart(crosstab)

    # 2. Average spend by income bracket
    st.subheader("Average Spend by Income Bracket")
    spend_means = df.groupby("income_bracket")["avg_spend_aed"].mean().sort_index()
    st.bar_chart(spend_means)

    # 3. Favourite cuisines
    st.subheader("Favourite Cuisines")
    vc = _explode_counts(df, "fav_cuisines")
    st.bar_chart(vc)

    # 4. Eco-packaging importance
    st.subheader("Eco-friendly Packaging Importance")
    sustain_counts = df["pack_sustain_score"].value_counts().sort_index()
    st.bar_chart(sustain_counts)

    # 5. Liked features
    st.subheader("Most Liked App Features")
    feat_counts = _explode_counts(df, "liked_features")
    st.bar_chart(feat_counts)

    # 6. Numeric correlations (as table)
    st.subheader("Numeric Feature Correlations")
    num_cols = df.select_dtypes("number").columns
    corr = df[num_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn"))

    # 7. Ordering windows preference
    st.subheader("Ordering Windows Preference")
    ow_counts = _explode_counts(df, "order_windows")
    st.bar_chart(ow_counts)

    # 8. Diet style by age group
    st.subheader("Diet Style by Age Group")
    diet_age = df.groupby(["age_group", "diet_style"]).size().unstack(fill_value=0)
    st.bar_chart(diet_age)

    # 9. Tip percentage distribution
    st.subheader("Tip Percentage Distribution (counts by bin)")
    tip_bins = pd.cut(df["tip_pct"], bins=10, include_lowest=True)
    tip_counts = tip_bins.value_counts().sort_index()
    st.bar_chart(tip_counts)

    # 10. NPS score distribution
    st.subheader("NPS Score Distribution")
    nps_counts = df["nps"].value_counts().sort_index()
    st.bar_chart(nps_counts)
