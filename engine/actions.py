"""The prescriptive layer: turn churn risk x customer value into this week's
visit list, with reason codes and talking-point SKU recommendations."""
from __future__ import annotations

from .recommender import recommend


def action_list(churn_out, rec, n: int = 15) -> list:
    df = churn_out["scores"].copy()
    df["priority"] = df["churn_p"] * df["value_52w"]
    df = df.sort_values("priority", ascending=False).head(n)
    items = []
    for _, row in df.iterrows():
        oid = row["outlet_id"]
        items.append({
            "outlet_id": oid,
            "churn_p": float(row["churn_p"]),
            "value_52w": float(row["value_52w"]),
            "priority": float(row["priority"]),
            "segment": row["segment"],
            "reasons": churn_out["reasons"][oid],
            "pitch": recommend(rec, oid, k=3),
        })
    return items
