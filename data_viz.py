# data_viz.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].str.split(";").explode().str.strip().value_counts()

def main_viz(df: pd.DataFrame) -> None:
    st.header("📊 Exploratory Data Visualisation (Matplotlib / Seaborn)")

    # 1. Orders per week (count plot)
    st.subheader("Orders per Week by Age Group")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="orders_per_week", hue="age_group", ax=ax)
    st.pyplot(fig)

    # 2. Average spend vs. income (boxplot)
    st.subheader("Average Spend by Income Bracket")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="income_bracket", y="avg_spend_aed", ax=ax)
    st.pyplot(fig)

    # 3. Favourite cuisines (bar chart)
    st.subheader("Favourite Cuisines")
    vc = _explode_counts(df, "fav_cuisines")
    st.bar_chart(vc)

    # 4. Eco-packaging importance (histogram)
    st.subheader("Eco-friendly Packaging Importance")
    fig, ax = plt.subplots()
    sns.histplot(df["pack_sustain_score"], bins=5, kde=False, ax=ax)
    ax.set_xlabel("Importance (1–5)")
    st.pyplot(fig)

    # 5. Liked features (bar chart)
    st.subheader("Most Liked App Features")
    vc2 = _explode_counts(df, "liked_features")
    st.bar_chart(vc2)

    # 6. Correlation heatmap
    st.subheader("Numeric Feature Correlations")
    num_cols = df.select_dtypes("number").columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # 7. Ordering windows (bar chart)
    st.subheader("Ordering Windows Preference")
    vc3 = _explode_counts(df, "order_windows")
    st.bar_chart(vc3)

    # 8. Diet style by age (grouped barplot)
    st.subheader("Diet Style by Age Group")
    diet_age = df.groupby(["age_group","diet_style"]).size().unstack(fill_value=0)
    st.line_chart(diet_age)

    # 9. Tip percentage distribution (histogram)
    st.subheader("Tip Percentage Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["tip_pct"], bins=20, kde=False, ax=ax)
    ax.set_xlabel("Tip %")
    st.pyplot(fig)

    # 10. NPS distribution (histogram)
    st.subheader("NPS Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["nps"], bins=11, kde=False, ax=ax)
    ax.set_xlabel("NPS (0–10)")
    st.pyplot(fig)
