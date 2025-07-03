# association_rules.py

import streamlit as st

# Attempt to import mlxtend; stub if unavailable
try:
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    _mlxtend_available = True
except ImportError:
    _mlxtend_available = False

if not _mlxtend_available:
    def association_rule_tab(df):
        st.error(
            "🚨 **mlxtend** is not installed.\n\n"
            "To enable association-rule mining, add the following to your `requirements.txt`:\n\n"
            "```txt\n"
            "mlxtend\n"
            "```\n"
            "and then redeploy your app."
        )
else:
    def _get_basket(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Convert semicolon-list columns into one-hot (basket) format.
        """
        basket = pd.DataFrame(index=df.index)
        for col in cols:
            exploded = df[col].str.split(";").explode().str.strip()
            dummies = pd.get_dummies(exploded).groupby(level=0).max()
            basket = basket.join(dummies, how="outer")
        return basket.fillna(0).astype(int)

    def association_rule_tab(df: pd.DataFrame) -> None:
        st.header("🕸️ Association Rule Mini
