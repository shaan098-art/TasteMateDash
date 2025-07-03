import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    "Helper to explode a semicolon list column into value counts."
    return df[col].str.split(";").explode().str.strip().value_counts()

def main_viz(df: pd.DataFrame) -> None:
    st.header("📊 Exploratory Data Visualisation")

    # 1. Orders per week
    fig = px.histogram(df, x="orders_per_week", color="age_group", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 2. Average spend vs. income
    fig = px.box(df, x="income_bracket", y="avg_spend_aed", color="income_bracket")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Favourite cuisines
    st.subheader("Favourite Cuisines")
    st.bar_chart(_explode_counts(df, "fav_cuisines"))

    # 4. Packaging sustainability score
    fig = px.histogram(df, x="pack_sustain_score", nbins=5, color="diet_style")
    st.plotly_chart(fig, use_container_width=True)

    # 5. Liked features
    st.subheader("Most Liked App Features")
    st.bar_chart(_explode_counts(df, "liked_features"))

    # 6. Correlation heat-map
    st.subheader("Numeric Feature Correlations")
    num_cols = df.select_dtypes("number").columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # 7. Ordering windows
    st.bar_chart(_explode_counts(df, "order_windows"))

    # 8. Diet style by age group
    diet_age = df.groupby(["age_group", "diet_style"]).size().reset_index(name="count")
    fig = px.bar(diet_age, x="age_group", y="count", color="diet_style", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # 9. Tip percentage
    fig = px.histogram(df, x="tip_pct", nbins=20, color="gender_id")
    st.plotly_chart(fig, use_container_width=True)

    # 10. NPS distribution
    fig = px.histogram(df, x="nps", nbins=11, color="adoption_timing")
    st.plotly_chart(fig, use_container_width=True)
