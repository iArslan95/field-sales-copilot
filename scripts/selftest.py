"""Smoke test for the ShelfMate engine: data integrity, churn-model quality
out-of-time, recommender lift vs popularity, and a sane action list.

    python scripts/selftest.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engine import actions, churn, data, recommender  # noqa: E402

HEADER = (
    f"{'seed':>4} {'outlets':>7} {'active':>6} {'AUC':>6} {'base':>6} "
    f"{'HR@5':>6} {'HRpop':>6} {'lift':>5} {'adopt':>5} {'risk10':>7}"
)


def main():
    print(HEADER)
    for seed in range(1, 7):
        sc = data.generate(seed, n_outlets=300)
        ch = churn.train(sc)
        ev = recommender.evaluate(sc)
        rec = recommender.build(sc)
        acts = actions.action_list(ch, rec, n=15)

        lift = (ev["hitrate_model"] / ev["hitrate_popularity"]
                if ev["hitrate_popularity"] else float("inf"))
        top10_p = sum(a["churn_p"] for a in acts[:10]) / 10
        print(
            f"{seed:>4} {len(sc['outlets']):>7} {len(ch['scores']):>6} "
            f"{ch['auc']:>6.3f} {ch['base_rate']:>6.2%} "
            f"{ev['hitrate_model']:>6.2%} {ev['hitrate_popularity']:>6.2%} "
            f"{lift:>5.2f} {ev['n_adopters']:>5} {top10_p:>7.2%}"
        )
        assert ch["auc"] >= 0.70, f"AUC too weak: {ch['auc']:.3f}"
        assert ev["hitrate_model"] >= ev["hitrate_popularity"], \
            "recommender should beat popularity"
        assert len(acts) == 15 and all(a["reasons"] for a in acts)
        assert all(len(a["pitch"]) >= 1 for a in acts[:5])
    print("ALL OK")


if __name__ == "__main__":
    main()
