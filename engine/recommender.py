"""Next-best-SKU recommender: item-item collaborative filtering on the
outlet x SKU purchase matrix (recency-weighted), with plain-language
"because" explanations.

Honest setup: similarities are learned on weeks <= 70 only; evaluation
checks whether the top-5 recommendations hit the genuinely NEW SKUs each
outlet adopted in weeks 71-78, against a popularity baseline ("recommend
what everyone buys"). Hit-rate@5 on adopters is reported in the app.
"""
from __future__ import annotations

import numpy as np

from .data import is_active

TRAIN_ASOF = 70
DECAY = 0.97


def _matrix(scenario, asof, window=32):
    outlets = [o for o in scenario["outlets"]]
    skus = list(scenario["skus"])
    o_idx = {o.id: i for i, o in enumerate(outlets)}
    s_idx = {s.id: j for j, s in enumerate(skus)}
    m = np.zeros((len(outlets), len(skus)))
    for oid, weeks in scenario["orders"].items():
        for w, basket in weeks.items():
            if asof - window < w <= asof:
                wgt = DECAY ** (asof - w)
                for sid, qty in basket.items():
                    m[o_idx[oid], s_idx[sid]] += qty * wgt
    return m, o_idx, s_idx, outlets, skus


def build(scenario) -> dict:
    m, o_idx, s_idx, outlets, skus = _matrix(scenario, scenario["now"])
    norms = np.linalg.norm(m, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    unit = m / norms
    sim = unit.T @ unit
    np.fill_diagonal(sim, 0.0)
    return {"m": m, "sim": sim, "o_idx": o_idx, "s_idx": s_idx,
            "outlets": outlets, "skus": skus}


def recommend(rec, oid: str, k: int = 3) -> list:
    i = rec["o_idx"][oid]
    owned = rec["m"][i]
    scores = rec["sim"] @ owned
    scores[owned > 0] = -1.0  # only SKUs the outlet does not buy yet
    order = np.argsort(-scores)[:k]
    out = []
    for j in order:
        if scores[j] <= 0:
            break
        contrib = rec["sim"][j] * owned
        top = np.argsort(-contrib)[:2]
        because = [rec["skus"][t].name for t in top if contrib[t] > 0]
        out.append({
            "sku": rec["skus"][j].name,
            "category": rec["skus"][j].category,
            "price": rec["skus"][j].price,
            "because": because,
        })
    return out


def evaluate(scenario, k: int = 5) -> dict:
    """Hit-rate@k on holdout adopters vs a popularity baseline."""
    m_train, o_idx, s_idx, outlets, skus = _matrix(scenario, TRAIN_ASOF)
    norms = np.linalg.norm(m_train, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    unit = m_train / norms
    sim = unit.T @ unit
    np.fill_diagonal(sim, 0.0)
    popularity = m_train.sum(axis=0)

    hits_model = hits_pop = n_eval = 0
    for o in outlets:
        if not is_active(scenario, o.id, TRAIN_ASOF):
            continue
        i = o_idx[o.id]
        owned = m_train[i]
        weeks_bought = {}
        for w in range(TRAIN_ASOF + 1, scenario["now"] + 1):
            for sid in scenario["orders"][o.id].get(w, {}):
                weeks_bought.setdefault(sid, set()).add(w)
        # True adoption = bought in 2+ separate holdout weeks (not one-off noise).
        new = ({s_idx[sid] for sid, ws in weeks_bought.items() if len(ws) >= 2}
               - set(np.nonzero(owned)[0]))
        if not new:
            continue
        n_eval += 1
        scores = sim @ owned
        scores[owned > 0] = -1.0
        top_model = set(np.argsort(-scores)[:k])
        pop = popularity.copy()
        pop[owned > 0] = -1.0
        top_pop = set(np.argsort(-pop)[:k])
        hits_model += bool(top_model & new)
        hits_pop += bool(top_pop & new)

    return {
        "hitrate_model": hits_model / n_eval if n_eval else 0.0,
        "hitrate_popularity": hits_pop / n_eval if n_eval else 0.0,
        "n_adopters": n_eval,
        "k": k,
    }
