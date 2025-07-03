# clustering.py

import streamlit as st

# Attempt to import dependencies
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
    from sklearn.cluster import KMeans
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

if not _sklearn_available:
    def clustering_tab(df):
        st.error(
            "🚨 **scikit-learn** is not installed.\n\n"
            "Add it to `requirements.txt`:\n\n"
            "```\nscikit-learn\n```\n"
            "and redeploy to enable clustering."
        )
else:
    def _prep_X(df: pd.DataFrame) -> np.ndarray:
        X = df.select_dtypes("number").copy()
        mlb = MultiLabelBinarizer()
        for col in ["order_windows", "fav_cuisines", "liked_features"]:
            arr = mlb.fit_transform(df[col].str.split(";"))
            X = pd.concat([X, pd.DataFrame(arr, columns=[f"{col}__{v}" for v in mlb.classes_])], axis=1)
        for col in ["gender_id", "income_bracket", "adoption_timing"]:
            X[col] = LabelEncoder().fit_transform(df[col])
        return StandardScaler().fit_transform(X.fillna(0))
    
    def clustering_tab(df: pd.DataFrame) -> None:
        st.header("🎯 K-means Customer Segmentation")

        X = _prep_X(df)

        # Elbow chart
        sse = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            sse.append(km.inertia_)
        st.subheader("Elbow Method")
        st.line_chart(pd.Series(sse, index=range(2, 11)))

        # Cluster slider
        k = st.slider("Select number of clusters (k)", 2, 10, 4)
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = km.labels_
        df2 = df.copy()
        df2["cluster"] = labels

        # Persona table
        persona = df2.groupby("cluster").agg({
            "age_group": lambda x: x.mode().iat[0],
            "income_bracket": lambda x: x.mode().iat[0],
            "orders_per_week": lambda x: x.mode().iat[0],
            "fav_cuisines": lambda x: x.mode().iat[0],
            "avg_spend_aed": "mean",
            "nps": "mean"
        }).round(2)
        persona.rename(columns={
            "age_group": "Common Age",
            "income_bracket": "Common Income",
            "orders_per_week": "Common Orders/Week",
            "fav_cuisines": "Top Cuisine",
            "avg_spend_aed": "Avg Spend",
            "nps": "Avg NPS"
        }, inplace=True)

        st.subheader("Customer Personas by Cluster")
        st.dataframe(persona)

        csv = df2.to_csv(index=False)
        st.download_button("Download clustered data", csv, "clustered_data.csv", "text/csv")
