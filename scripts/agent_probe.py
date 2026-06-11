"""End-to-end probe of the agent loop with real Groq calls: two turns that
should trigger different tools. Needs a key in .streamlit/secrets.toml or
the GROQ_API_KEY environment variable.

    python scripts/agent_probe.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
from engine import churn, data, recommender  # noqa: E402


def main():
    key = agent.get_api_key()
    if not key:
        sys.exit("No GROQ_API_KEY found in secrets or environment.")

    scenario = data.generate(2, n_outlets=300)
    ctx = {
        "scenario": scenario,
        "churn": churn.train(scenario),
        "rec": recommender.build(scenario),
        "evals": recommender.evaluate(scenario),
        "risk_threshold": 0.45,
    }

    q1 = ("Plan my top 5 visits for this week and give me one line on why "
          "number 1 matters.")
    a1, arts1 = agent.agent_turn(key, ctx, [{"role": "user", "content": q1}])
    tools1 = [a["tool"] for a in arts1]
    print("Q1 tools used:", tools1)
    print("A1:", a1[:400].replace("\n", " "))
    assert "get_action_list" in tools1, "expected the action-list tool"
    assert len(a1) > 60

    top = max(ctx["churn"]["scores"].itertuples(), key=lambda r: r.churn_p)
    name = next(o.name for o in scenario["outlets"] if o.id == top.outlet_id)
    q2 = f"Why is {name} at risk, and what should I pitch there?"
    a2, arts2 = agent.agent_turn(
        key, ctx,
        [{"role": "user", "content": q1}, {"role": "assistant", "content": a1},
         {"role": "user", "content": q2}])
    tools2 = [a["tool"] for a in arts2]
    print("Q2 tools used:", tools2)
    print("A2:", a2[:400].replace("\n", " "))
    assert any(t in tools2 for t in ("get_outlet_profile",
                                     "get_recommendations")), \
        "expected an outlet-level tool"
    assert len(a2) > 60

    print("AGENT PROBE OK")


if __name__ == "__main__":
    main()
