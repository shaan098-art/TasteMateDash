# data_viz.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    """Explode a semicolon-separated column into value counts."""
    return df[col].str.split(";").explode().str.strip().value_counts()

def main_viz(df: pd.DataFrame) -> None:
    st.header("📊 Exploratory Data Visualisation (Matplotlib Only)")

    # 1. Orders per week by age group
    st.subheader("Orders per Week by Age Group")
    ct = pd.crosstab(df["orders_per_week"], df["age_group"])
    st.bar_chart(ct)

    # 2. Average spend by income bracket
    st.subheader("Average Spend by Income Bracket")
    spend_means = df.groupby("income_bracket")["avg_spend_aed"].mean().sort_index()
    st.bar_chart(spend_means)

    # 3. Favourite cuisines
    st.subheader("Favourite Cuisines")
    vc = _explode_counts(df, "fav_cuisines")
    st.bar_chart(vc)

    # 4. Packaging sustainability score distribution
    st.subheader("Eco-friendly Packaging Importance")
    fig, ax = plt.subplots()
    ax.hist(df["pack_sustain_score"], bins=5, edgecolor="black")
    ax.set_xlabel("Importance (1–5)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 5. Liked features
    st.subheader("Most Liked App Features")
    vc2 = _explode_counts(df, "liked_features")
    st.bar_chart(vc2)

    # 6. Correlation heatmap with matplotlib
    st.subheader("Numeric Feature Correlations")
    num_cols = df.select_dtypes("number").columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    st.pyplot(fig)

    # 7. Ordering windows
    st.subheader("Ordering Windows Preference")
    vc3 = _explode_counts(df, "order_windows")
    st.bar_chart(vc3)

    # 8. Diet style by age (line chart)
    st.subheader("Diet Style by Age Group")
    diet_age = df.groupby(["age_group", "diet_style"]).size().unstack(fill_value=0)
    st.line_chart(diet_age)

    # 9. Tip percentage distribution
    st.subheader("Tip Percentage Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["tip_pct"], bins=20, edgecolor="black")
    ax.set_xlabel("Tip %")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 10. NPS score distribution
    st.subheader("NPS Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["nps"], bins=11, edgecolor="black")
    ax.set_xlabel("NPS (0–10)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
