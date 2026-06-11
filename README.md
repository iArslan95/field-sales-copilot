# 🛒 ShelfMate — a GenAI field-sales copilot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-churn%20%2B%20CF-green)
![Agent](https://img.shields.io/badge/Groq-function--calling%20agent-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An **agentic AI demo for CPG customer development**: a copilot that helps a
field-sales account manager start the day. It plans the week from
**churn risk × customer value**, explains *why* each outlet is slipping, and
arms every visit with **next-best-SKU** talking points — through an LLM agent
(Groq, Llama 3.3 70B) that picks its own tools via function calling, grounded
in models trained on the data.

## Why

An account manager with 300 small outlets can't watch them all. By the time a
buurtsuper or toko silently stops ordering, the relationship is already cold.
The classic answer is a dashboard; the better answer is a colleague: *"these
19 outlets are slipping, this one matters most, here's why, and here's what
to pitch when you walk in."* That is churn modelling + a recommendation
engine + prioritisation, delivered through a conversation.

## How it works

```
account manager ──chat──▶ LLM agent (Groq function calling)
                              │  picks tools autonomously
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
      churn model       recommender         action list
   (HistGradient-     (item-item CF,     (risk × value, with
    Boosting, AUC      hit-rate@5 vs      reason codes and
    out-of-time)       popularity)        pitch SKUs)
            └─────────────────┴──────────────────┘
                     78 weeks of synthetic orders,
                ~300 outlets, 48 SKUs in 6 categories
```

- **The agent decides** which tool to call (`get_action_list`,
  `get_outlet_profile`, `get_recommendations`, `search_outlets`,
  `get_briefing`, `get_model_card`); tool output is rendered as tables and
  cards under each answer, so the user gets text *and* working artifacts
  (including a CSV download of the visit list).
- **Churn model** — gradient boosting on observable behaviour (recency,
  frequency trend, basket trend, categories lost, delivery issues,
  price-increase exposure). Trained on two historic snapshots, evaluated
  **out-of-time**; no ground-truth leakage.
- **Recommender** — recency-weighted item-item collaborative filtering;
  evaluated on genuinely adopted SKUs in an 8-week holdout against a
  popularity baseline.
- **Honest model card in the app** — AUC, hit-rate@5 vs baseline, holdout
  design and caveats, one click away.

All data is **synthetic with planted structure** (segment-level preferences,
seasonality, a price shock, delivery issues, gradual decay), so the models
have something real to learn and the evaluation is meaningful. Outlet and
brand names are fictional.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your Groq key
streamlit run app.py
```

Sanity-check the engine and the agent:

```bash
python scripts/selftest.py     # data, churn AUC, recommender lift, action list
python scripts/ui_test.py      # headless UI test (no network)
python scripts/agent_probe.py  # real Groq tool-calling round trip
```

## The LLM key

The copilot uses Groq's free tier (`llama-3.3-70b-versatile`). Get a key at
[console.groq.com](https://console.groq.com/keys), put it in
`.streamlit/secrets.toml` locally (gitignored) or in **App settings →
Secrets** on Streamlit Cloud:

```toml
GROQ_API_KEY = "gsk_..."
```

Without a key the app still renders (KPIs, briefing, model card); only the
conversation is disabled.

## Project structure

```
app.py                  Streamlit UI: KPI strip, copilot, artifacts, model card
agent.py                Groq function-calling loop + tool registry
engine/
  data.py               synthetic territory generator (with planted structure)
  churn.py              snapshot features, HistGB model, reason codes
  recommender.py        item-item CF + holdout evaluation vs popularity
  actions.py            visit list: churn risk x value + pitch SKUs
scripts/selftest.py     multi-seed engine test (AUC & lift thresholds)
scripts/ui_test.py      headless AppTest of the UI
scripts/agent_probe.py  end-to-end agent test with real tool calls
```

## From demo to production

- Retrain on real sell-out (POS/SAP), with proper backtesting windows
- **Uplift modelling**: target who is *saveable*, not just who is at risk
- Campaign holdouts to measure incremental revenue of visits and pitches
- Industrialisation on Databricks: Delta tables, MLflow tracking & registry,
  scheduled retraining, monitoring for drift
- Agent guardrails and evaluation: tool-call success rates, grounding checks,
  red-team prompts

## Disclaimer

Educational portfolio project, inspired by customer-development work in CPG
(field sales, perfect store). All data is synthetic; all outlet, brand and
product names are fictional. Not affiliated with any retailer or CPG company.
