import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def _basket(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts specified semicolon-list columns into one-hot basket format.
    """
    basket = pd.DataFrame(index=df.index)
    for col in cols:
        exploded = df[col].str.split(";").explode().str.strip()
        dummies = pd.get_dummies(exploded).groupby(level=0).max()
        basket = basket.join(dummies, how="outer")
    return basket.fillna(0).astype(int)

def association_rule_tab(df: pd.DataFrame) -> None:
    st.header("🕸️ Association-Rule Mining (Apriori)")
    multi_cols = [c for c in df.columns if ";" in ";".join(df[c].astype(str))]

    cols = st.multiselect("Columns to include", multi_cols, default=multi_cols[:2])
    if len(cols) < 2:
        st.info("Select at least two columns.")
        return

    basket = _basket(df, cols)
    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.4, 0.05)

    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)

    if rules.empty:
        st.warning("No rules found with current thresholds.")
    else:
        st.dataframe(rules[["antecedents", "consequents", "support",
                            "confidence", "lift"]])
