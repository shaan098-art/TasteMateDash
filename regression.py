import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def _prep(df: pd.DataFrame, target: str):
    X = df.select_dtypes("number").copy()
    mlb = MultiLabelBinarizer()
    for col in ["order_windows", "fav_cuisines", "liked_features"]:
        X = pd.concat([X, pd.DataFrame(
            mlb.fit_transform(df[col].str.split(";")),
            columns=[f"{col}__{v}" for v in mlb.classes_])],
            axis=1
        )
    for col in ["gender_id", "income_bracket", "adoption_timing", "diet_style"]:
        X[col] = LabelEncoder().fit_transform(df[col])
    X = StandardScaler().fit_transform(X.fillna(0))
    y = df[target].values
    return train_test_split(X, y, test_size=0.25, random_state=42)

def regression_tab(df: pd.DataFrame) -> None:
    st.header("📈 Regression Insights")
    target = st.selectbox(
        "Target variable",
        ["avg_spend_aed", "tip_pct", "nps", "max_wait_min", "commute_minutes"]
    )
    Xtr, Xte, ytr, yte = _prep(df, target)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    rows, preds = [], {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        yp = mdl.predict(Xte)
        preds[name] = yp
        rows.append([name,
                     mean_squared_error(yte, yp),
                     r2_score(yte, yp)])
    st.dataframe(pd.DataFrame(rows, columns=["Model", "MSE", "R²"]).round(3))

    # Example scatter (Decision Tree)
    fig, ax = plt.subplots()
    ax.scatter(yte, preds["Decision Tree"], alpha=0.5)
    ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    st.pyplot(fig)
