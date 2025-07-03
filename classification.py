import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Data preparation helpers
# ------------------------------------------------------------------ #
def _binarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(df[col].str.split(";"))
    return pd.DataFrame(arr, columns=[f"{col}__{v}" for v in mlb.classes_])

def _prep_X_y(df: pd.DataFrame, target: str = "diet_style"):
    df = df.copy()
    X = df.select_dtypes("number").copy()

    for col in ["order_windows", "fav_cuisines", "liked_features"]:
        X = pd.concat([X, _binarize(df, col)], axis=1)

    for col in ["gender_id", "income_bracket", "adoption_timing"]:
        X[col] = LabelEncoder().fit_transform(df[col])

    y = LabelEncoder().fit_transform(df[target])
    X = StandardScaler().fit_transform(X.fillna(0))
    return X, y

# ------------------------------------------------------------------ #
#  Streamlit tab
# ------------------------------------------------------------------ #
def classification_tab(df: pd.DataFrame) -> None:
    st.header("🤖 Classification – Predict Dietary Style")
    X, y = _prep_X_y(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    metrics, probs, preds = [], {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        yp = mdl.predict(Xte)
        preds[name] = yp
        if hasattr(mdl, "predict_proba"):
            probs[name] = mdl.predict_proba(Xte)
        metrics.append([
            name,
            accuracy_score(yte, yp),
            precision_score(yte, yp, average="weighted", zero_division=0),
            recall_score(yte, yp, average="weighted", zero_division=0),
            f1_score(yte, yp, average="weighted", zero_division=0)
        ])

    st.subheader("Model Performance")
    st.dataframe(
        pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).round(3)
    )

    # Confusion matrix toggle
    algo = st.selectbox("Show confusion matrix for:", list(models.keys()))
    cm = confusion_matrix(yte, preds[algo])
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{algo} – Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    # ROC curves (one-vs-rest, class 0)
    st.subheader("ROC Curve (class 0 vs rest)")
    fig, ax = plt.subplots()
    for name, pr in probs.items():
        fpr, tpr, _ = roc_curve(yte, pr[:, 0], pos_label=0)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    # Prediction on uploaded data
    st.subheader("Batch-predict new data")
    up = st.file_uploader("Upload CSV (same columns, no diet_style)", type="csv")
    if up:
        new_df = pd.read_csv(up)
        Xnew, _ = _prep_X_y(new_df)   # y ignored
        mdl_name = st.selectbox("Model to use", list(models.keys()), key="predsel")
        pred = models[mdl_name].predict(Xnew)
        le = LabelEncoder().fit(df["diet_style"])
        new_df["predicted_diet_style"] = le.inverse_transform(pred)
        st.write(new_df.head())
        st.download_button(
            "Download predictions",
            new_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )
