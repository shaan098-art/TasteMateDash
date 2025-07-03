# regression.py

import streamlit as st
import pandas as pd

def regression_tab(df: pd.DataFrame) -> None:
    st.header("📈 Regression Insights")

    # Lazy import of scikit-learn and matplotlib
    try:
        import numpy as np
        from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
    except ImportError:
        st.error(
            "🚨 **scikit-learn** (and/or **matplotlib**) is not installed.\n\n"
            "Add to your `requirements.txt`:\n\n"
            "    scikit-learn\n"
            "    matplotlib\n\n"
            "Then redeploy to enable regression analysis."
        )
        return

    # Select target variable
    target = st.selectbox(
        "Select target for regression",
        ["avg_spend_aed", "tip_pct", "nps", "max_wait_min", "commute_minutes"]
    )

    # Data prep
    def _prep(df, target):
        X = df.select_dtypes("number").copy()
        mlb = MultiLabelBinarizer()
        for col in ["order_windows", "fav_cuisines", "liked_features"]:
            arr = mlb.fit_transform(df[col].str.split(";"))
            X = pd.concat([X, pd.DataFrame(arr, columns=[f"{col}__{v}" for v in mlb.classes_])], axis=1)
        for col in ["gender_id", "income_bracket", "adoption_timing", "diet_style"]:
            X[col] = LabelEncoder().fit_transform(df[col])
        X_scaled = StandardScaler().fit_transform(X.fillna(0))
        y = df[target].values
        return train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    Xtr, Xte, ytr, yte = _prep(df, target)

    # Define models
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    # Train & evaluate
    results, preds = [], {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        yp = mdl.predict(Xte)
        preds[name] = yp
        mse = mean_squared_error(yte, yp)
        r2 = r2_score(yte, yp)
        results.append([name, mse, r2])

    st.subheader("Model Performance (MSE and R²)")
    st.dataframe(pd.DataFrame(results, columns=["Model", "MSE", "R²"]).round(3))

    # Scatter plot for Decision Tree
    st.subheader("Prediction vs. True (Decision Tree)")
    fig, ax = plt.subplots()
    ax.scatter(yte, preds["Decision Tree"], alpha=0.5)
    ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    st.pyplot(fig)

    # Show top linear coefficients
    if hasattr(models["Linear"], "coef_"):
        st.subheader("Top 10 Feature Coefficients (Linear Regression)")
        coefs = pd.Series(models["Linear"].coef_, 
                          index=[f"feat_{i}" for i in range(len(models["Linear"].coef_))])
        st.bar_chart(coefs.abs().sort_values(ascending=False).head(10))
