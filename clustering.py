import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def _prep_X(df: pd.DataFrame):
    X = df.select_dtypes("number").copy()
    mlb = MultiLabelBinarizer()
    for col in ["order_windows", "fav_cuisines", "liked_features"]:
        X = pd.concat([X, pd.DataFrame(
            mlb.fit_transform(df[col].str.split(";")),
            columns=[f"{col}__{v}" for v in mlb.classes_])],
            axis=1
        )
    for col in ["gender_id", "income_bracket", "adoption_timing"]:
        X[col] = LabelEncoder().fit_transform(df[col])
    return StandardScaler().fit_transform(X.fillna(0))

def clustering_tab(df: pd.DataFrame) -> None:
    st.header("🎯 K-means Clustering")
    X = _prep_X(df)

    # Elbow chart
    sse = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in range(2, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), sse, marker="o"); ax.set_xlabel("k"); ax.set_ylabel("SSE")
    st.pyplot(fig)

    k = st.slider("Number of clusters", 2, 10, 4)
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    df_lab = df.copy(); df_lab["cluster"] = labels

    # Simple persona summary
    persona = df_lab.groupby("cluster").agg(
        Age=("age_group", lambda x: x.value_counts().idxmax()),
        Income=("income_bracket", lambda x: x.value_counts().idxmax()),
        OrdersWeek=("orders_per_week", lambda x: x.value_counts().idxmax()),
        FavCuisine=("fav_cuisines", lambda x: x.mode().iat[0]),
        SpendMean=("avg_spend_aed", "mean"),
        NPS=("nps", "mean")
    ).round(1)
    st.subheader("Cluster Personas")
    st.dataframe(persona)

    st.download_button("Download data w/ cluster", df_lab.to_csv(index=False),
                       "clustered_data.csv", "text/csv")
