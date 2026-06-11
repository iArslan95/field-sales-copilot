"""Churn model: gradient boosting on behavioural snapshot features.

Honest setup: train on two historic snapshots (weeks 58 and 64), evaluate
out-of-time on week 70 (labels = the six weeks after each snapshot), then
score "now" (week 78) for the live risk list. Ground-truth fields like
`churn_start` are never used as features — only observable behaviour.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import churn_label, features, is_active

TRAIN_SNAPSHOTS = (58, 64)
TEST_SNAPSHOT = 70
NUMERIC = ("recency_w", "orders_4w", "orders_12w", "freq_trend", "basket_12w",
           "basket_trend", "cats_4w", "cats_lost", "issues_8w", "shock_share",
           "tenure_w", "value_52w")


def _frame(scenario, asof):
    rows = []
    for o in scenario["outlets"]:
        if o.start_week > asof - 12 or not is_active(scenario, o.id, asof):
            continue
        f = features(scenario, o.id, asof)
        f["label"] = churn_label(scenario, o.id, asof)
        f["outlet_id"] = o.id
        rows.append(f)
    return pd.DataFrame(rows)


def _design(df):
    x = df[list(NUMERIC)].copy()
    for seg in ("Buurtsuper", "Avondwinkel", "Toko & speciaalzaak", "Horeca",
                "Tankstation"):
        x[f"seg_{seg}"] = (df["segment"] == seg).astype(int)
    x["size_n"] = df["size"].map({"S": 0, "M": 1, "L": 2})
    return x


def train(scenario) -> dict:
    train_df = pd.concat([_frame(scenario, w) for w in TRAIN_SNAPSHOTS],
                         ignore_index=True)
    test_df = _frame(scenario, TEST_SNAPSHOT)

    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=250,
        l2_regularization=1.0, random_state=scenario["seed"],
    )
    clf.fit(_design(train_df), train_df["label"])
    auc = roc_auc_score(test_df["label"], clf.predict_proba(_design(test_df))[:, 1])

    now = scenario["now"]
    rows = []
    for o in scenario["outlets"]:
        if not is_active(scenario, o.id, now):
            continue
        f = features(scenario, o.id, now)
        f["outlet_id"] = o.id
        rows.append(f)
    now_df = pd.DataFrame(rows)
    now_df["churn_p"] = clf.predict_proba(_design(now_df))[:, 1]

    seg_median = now_df.groupby("segment")[list(NUMERIC)].median()
    reasons = {
        r["outlet_id"]: _reasons(r, seg_median.loc[r["segment"]])
        for _, r in now_df.iterrows()
    }
    return {
        "auc": auc,
        "train_rows": len(train_df),
        "base_rate": float(train_df["label"].mean()),
        "scores": now_df,
        "reasons": reasons,
    }


def _reasons(row, norm) -> list:
    """Plain-language reason codes: where does this outlet deviate from its
    segment's normal behaviour?"""
    found = []
    if row["recency_w"] >= max(3, 2 * max(norm["recency_w"], 1)):
        found.append((row["recency_w"],
                      f"no order for {int(row['recency_w'])} weeks "
                      f"(segment norm ~{int(norm['recency_w'])})"))
    if row["freq_trend"] < 0.6:
        found.append((2.5, f"order frequency at {row['freq_trend']:.0%} of its "
                           "own 12-week pace"))
    if 0 < row["basket_trend"] < 0.75:
        found.append((2.0, f"basket value down to {row['basket_trend']:.0%} of "
                           "its 12-week average"))
    if row["cats_lost"] >= 2:
        found.append((1.8, f"dropped {int(row['cats_lost'])} categories from "
                           "the recent baskets"))
    if row["issues_8w"] >= 2:
        found.append((1.6, f"{int(row['issues_8w'])} delivery issues in the "
                           "last 8 weeks"))
    if row["shock_share"] > 0.35:
        found.append((1.2, f"{row['shock_share']:.0%} of its basket sits in "
                           "the price-increased categories"))
    found.sort(key=lambda t: -t[0])
    return [msg for _, msg in found[:3]] or ["behaviour broadly normal — "
                                             "risk driven by weak overall cadence"]
