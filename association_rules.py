# association_rules.py

import streamlit as st
import pandas as pd

def _build_basket(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert semicolon-list columns into one-hot basket format."""
    basket = pd.DataFrame(index=df.index)
    for col in cols:
        exploded = df[col].str.split(";").explode().str.strip()
        dummies = pd.get_dummies(exploded).groupby(level=0).max()
        basket = basket.join(dummies, how="outer")
    return basket.fillna(0).astype(int)

def association_rule_tab(df: pd.DataFrame) -> None:
    st.header("🕸️ Association Rule Mining")

    # Lazy import to avoid import-time errors
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        st.error(
            "🚨 **mlxtend** is not installed.\n\n"
            "To enable association-rule mining, add to your requirements.txt:\n\n"
            "    mlxtend\n\n"
            "and then redeploy."
        )
        return

    # Identify multi-select columns (semicolon-separated)
    multi_cols = [c for c in df.columns if ";" in " ".join(df[c].astype(str))]
    if len(multi_cols) < 2:
        st.warning("Need at least two multi-select columns to mine rules.")
        return

    cols = st.multiselect(
        "Select columns for Apriori mining",
        multi_cols, 
        default=multi_cols[:2]
    )
    if len(cols) < 2:
        return

    basket = _build_basket(df, cols)

    min_sup = st.slider("Minimum support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.4, 0.05)

    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)

    if rules.empty:
        st.info("No association rules found with those thresholds.")
    else:
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
        for _, r in rules.iterrows():
            st.write(
                f"• If a user selects **{', '.join(r['antecedents'])}**, "
                f"they also tend to select **{', '.join(r['consequents'])}** "
                f"(conf={r['confidence']:.2f}, lift={r['lift']:.2f})"
            )
