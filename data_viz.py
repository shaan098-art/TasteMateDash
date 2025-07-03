# data_viz.py

import streamlit as st
import pandas as pd

def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].str.split(";").explode().str.strip().value_counts()

def main_viz(df: pd.DataFrame) -> None:
    st.header("📊 Exploratory Data Visualisation")

    # 1) Check for required columns up front
    required = [
        "orders_per_week", "age_group", "income_bracket", "avg_spend_aed",
        "fav_cuisines", "pack_sustain_score", "liked_features",
        "order_windows", "tip_pct", "nps"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"Missing columns for Data Visualization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
        return

    # 2) Orders per week by age group
    st.subheader("Orders per Week by Age Group")
    crosstab = pd.crosstab(df["orders_per_week"], df["age_group"])
    st.bar_chart(crosstab)

    # 3) Average spend by income bracket
    st.subheader("Average Spend by Income Bracket")
    spend_means = df.groupby("income_bracket")["avg_spend_aed"].mean().sort_index()
    st.bar_chart(spend_means)

    # 4) Favourite cuisines
    st.subheader("Favourite Cuisines")
    vc = _explode_counts(df, "fav_cuisines")
    st.bar_chart(vc)

    # 5) Eco-packaging importance
    st.subheader("Eco-friendly Packaging Importance")
    sustain_counts = df["pack_sustain_score"].value_counts().sort_index()
    st.bar_chart(sustain_counts)

    # 6) Liked features
    st.subheader("Most Liked App Features")
    feat_counts = _explode_counts(df, "liked_features")
    st.bar_chart(feat_counts)

    # 7) Ordering windows preference
    st.subheader("Ordering Windows Preference")
    ow_counts = _explode_counts(df, "order_windows")
    st.bar_chart(ow_counts)

    # 8) Tip percentage distribution
    st.subheader("Tip Percentage Distribution")
    tip_bins = pd.cut(df["tip_pct"], bins=10, include_lowest=True)
    tip_counts = tip_bins.value_counts().sort_index()
    st.bar_chart(tip_counts)

    # 9) NPS score distribution
    st.subheader("NPS Score Distribution")
    nps_counts = df["nps"].value_counts().sort_index()
    st.bar_chart(nps_counts)

    # 10) Correlation matrix (table)
    st.subheader("Numeric Feature Correlations")
    num_cols = df.select_dtypes("number").columns
    corr = df[num_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn"))
