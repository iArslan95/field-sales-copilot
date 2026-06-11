"""ShelfMate agent: a Groq function-calling loop over the engine.

This is the agentic layer: the LLM receives a small set of tools (action
list, outlet profile, recommendations, search, model card), decides itself
which to call, the app executes them against the trained models, and the
final answer is grounded in those results. Tool payloads double as visual
artifacts: the app renders them as tables/cards under the agent's reply.
"""
from __future__ import annotations

import json
import os

import requests

from engine import churn  # noqa: F401  (re-exported context builders below)
from engine.actions import action_list
from engine.recommender import recommend

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MSGS = 10
MAX_USER_MESSAGES = 25

SYSTEM_PROMPT = """\
You are ShelfMate, a GenAI copilot for CPG field-sales account managers. You
are embedded in a portfolio demo built by Ismail Arslan: one synthetic sales
territory (Rotterdam region, ~300 small retail outlets: neighbourhood
supermarkets, night shops, tokos, food-service, forecourt shops) buying a
fictional FMCG portfolio (tea, bouillon & soup, sauces, laundry, personal
care, ice cream) over 78 weeks. Week 78 = today. All data is synthetic.

YOUR TOOLS (always ground answers in them — never invent outlets or numbers):
- get_briefing: territory KPIs and the top risks. Use for "how are we doing",
  start-of-day, or summary questions.
- get_action_list: this week's prioritised visit list (churn risk x annual
  value), with reason codes and pitch SKUs. Use for "plan my week/day".
- get_outlet_profile: everything about one outlet (risk, drivers, behaviour,
  what to pitch). Use when the user names an outlet.
- get_recommendations: next-best-SKUs for one outlet with the "because".
- search_outlets: filter outlets by segment, district or risk.
- get_model_card: honest model quality (out-of-time AUC, hit-rate@5 vs a
  popularity baseline, holdout design). Use for "can I trust this".

HOW TO ANSWER
- Practical account-manager language; short paragraphs or tight bullets.
- Mirror the user's language: English in -> English out, Dutch in -> Dutch out.
- Money like EUR 1,234. Percentages without decimals unless meaningful.
- The app renders your tool results as tables/cards UNDER your message —
  summarise the top 2-3 and the insight; do not repeat whole tables in text.
- Churn probabilities are model estimates on synthetic data — say "risk",
  not certainty. If asked something outside this territory, say so briefly.
- Lead with the answer, then the why. End with one concrete next step when
  it helps the user act.
"""

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_briefing",
        "description": "Territory KPIs: active outlets, number at risk, value "
                       "at risk, top-5 risk outlets, model health.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_action_list",
        "description": "Prioritised visit list for this week: churn risk x "
                       "annual value, with reasons and pitch SKUs per outlet.",
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "number of visits (1-25)",
                  "default": 10}}},
    }},
    {"type": "function", "function": {
        "name": "get_outlet_profile",
        "description": "Full profile of one outlet by (fuzzy) name: risk, "
                       "reason codes, behaviour, top products, what to pitch.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "outlet name, may be "
                                                      "partial"}},
            "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "get_recommendations",
        "description": "Next-best-SKU recommendations for one outlet.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "k": {"type": "integer", "default": 3}},
            "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "search_outlets",
        "description": "Find outlets by segment, district (wijk) or name text;"
                       " optionally only above a churn-risk threshold.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "segment, district or "
                                                       "name text; empty = all"},
            "min_risk": {"type": "number", "default": 0.0},
            "n": {"type": "integer", "default": 10}},
            },
    }},
    {"type": "function", "function": {
        "name": "get_model_card",
        "description": "Honest evaluation of the underlying models: holdout "
                       "design, churn AUC, recommender hit-rate vs popularity.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


def get_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


# ------------------------------------------------------------------ tools

def _match_outlet(ctx, name: str):
    needle = name.lower().strip()
    scored = []
    for o in ctx["scenario"]["outlets"]:
        hay = o.name.lower()
        if needle == hay:
            return o, []
        if needle in hay:
            scored.append((len(hay) - len(needle), o))
        else:
            hits = sum(1 for tok in needle.split() if tok in hay)
            if hits:
                scored.append((100 - hits, o))
    scored.sort(key=lambda t: t[0])
    if not scored:
        return None, []
    return scored[0][1], [o.name for _, o in scored[1:4]]


def _risk_row(ctx, oid):
    df = ctx["churn"]["scores"]
    row = df[df["outlet_id"] == oid]
    return row.iloc[0] if len(row) else None


def make_tools(ctx):
    scenario, ch, rec, ev = (ctx["scenario"], ctx["churn"], ctx["rec"],
                             ctx["evals"])
    outlets = {o.id: o for o in scenario["outlets"]}

    def get_briefing():
        df = ch["scores"]
        thr = ctx["risk_threshold"]
        risk = df[df["churn_p"] >= thr].sort_values("churn_p", ascending=False)
        return {
            "week": scenario["now"],
            "active_outlets": int(len(df)),
            "risk_threshold": thr,
            "outlets_at_risk": int(len(risk)),
            "annual_value_at_risk_eur": float(risk["value_52w"].sum()),
            "top_risks": [
                {"outlet": outlets[r.outlet_id].name,
                 "churn_risk": round(float(r.churn_p), 2),
                 "annual_value_eur": float(r.value_52w)}
                for r in risk.head(5).itertuples()
            ],
            "model_health": {
                "churn_auc_out_of_time": round(ch["auc"], 3),
                "recommender_hitrate_at_5": round(ev["hitrate_model"], 3),
                "popularity_baseline": round(ev["hitrate_popularity"], 3),
            },
        }

    def get_action_list(n=10):
        n = max(1, min(int(n or 10), 25))
        items = action_list(ch, rec, n=n)
        return {"visits": [
            {"rank": i + 1,
             "outlet": outlets[a["outlet_id"]].name,
             "segment": a["segment"],
             "district": outlets[a["outlet_id"]].wijk,
             "churn_risk": round(a["churn_p"], 2),
             "annual_value_eur": a["value_52w"],
             "reasons": a["reasons"],
             "pitch": [f"{p['sku']} (because they buy "
                       f"{' & '.join(p['because'][:2])})" for p in a["pitch"]]}
            for i, a in enumerate(items)
        ]}

    def get_outlet_profile(name):
        outlet, alts = _match_outlet(ctx, name)
        if outlet is None:
            return {"error": f"no outlet matching '{name}'"}
        from engine.data import features
        f = features(scenario, outlet.id, scenario["now"])
        row = _risk_row(ctx, outlet.id)
        recs = recommend(rec, outlet.id, k=3)
        m = rec["m"][rec["o_idx"][outlet.id]]
        top_owned = [rec["skus"][j].name
                     for j in list(reversed(m.argsort()))[:5] if m[j] > 0]
        return {
            "outlet": outlet.name, "segment": outlet.segment,
            "district": outlet.wijk, "size": outlet.size,
            "churn_risk": round(float(row["churn_p"]), 2) if row is not None
            else "inactive (already lapsed)",
            "risk_reasons": ch["reasons"].get(outlet.id, []),
            "annual_value_eur": f["value_52w"],
            "weeks_since_last_order": f["recency_w"],
            "orders_last_12w": f["orders_12w"],
            "avg_basket_eur": f["basket_12w"],
            "basket_trend_vs_12w": round(f["basket_trend"], 2),
            "active_categories": f["cats_4w"],
            "top_products": top_owned,
            "recommended_pitch": recs,
            "similar_outlets_hint": alts,
        }

    def get_recommendations(name, k=3):
        outlet, _ = _match_outlet(ctx, name)
        if outlet is None:
            return {"error": f"no outlet matching '{name}'"}
        return {"outlet": outlet.name,
                "recommendations": recommend(rec, outlet.id,
                                             k=max(1, min(int(k or 3), 6)))}

    def search_outlets(query="", min_risk=0.0, n=10):
        q = (query or "").lower()
        df = ch["scores"]
        rows = []
        for r in df.itertuples():
            o = outlets[r.outlet_id]
            hay = f"{o.name} {o.segment} {o.wijk}".lower()
            if q and q not in hay:
                continue
            if r.churn_p < float(min_risk or 0):
                continue
            rows.append({"outlet": o.name, "segment": o.segment,
                         "district": o.wijk,
                         "churn_risk": round(float(r.churn_p), 2),
                         "annual_value_eur": float(r.value_52w)})
        rows.sort(key=lambda x: -x["churn_risk"])
        return {"matches": rows[:max(1, min(int(n or 10), 25))],
                "total_matches": len(rows)}

    def get_model_card():
        return {
            "data": "synthetic territory, 78 weeks of weekly orders, "
                    f"{len(scenario['outlets'])} outlets, "
                    f"{len(scenario['skus'])} SKUs in 6 categories",
            "churn_model": {
                "type": "HistGradientBoosting on behavioural snapshot features",
                "training": "snapshots at weeks 58 & 64, label = no orders in "
                            "the following 6 weeks",
                "evaluation": "out-of-time snapshot at week 70",
                "auc": round(ch["auc"], 3),
                "churn_base_rate": round(ch["base_rate"], 3),
                "features_note": "only observable behaviour (recency, "
                                 "frequency trend, basket trend, categories "
                                 "lost, delivery issues, price-increase "
                                 "exposure) — no ground-truth leakage",
            },
            "recommender": {
                "type": "item-item collaborative filtering, recency-weighted",
                "evaluation": "similarities learned on weeks <=70; tested on "
                              "SKUs newly adopted (bought in 2+ weeks) in "
                              "weeks 71-78",
                "hitrate_at_5": round(ev["hitrate_model"], 3),
                "popularity_baseline_at_5": round(ev["hitrate_popularity"], 3),
                "evaluated_adopters": ev["n_adopters"],
            },
            "caveats": "synthetic data with planted structure; in production "
                       "this would be retrained on real POS/SAP sell-out data "
                       "with uplift-based targeting and holdout campaigns",
        }

    return {
        "get_briefing": get_briefing,
        "get_action_list": get_action_list,
        "get_outlet_profile": get_outlet_profile,
        "get_recommendations": get_recommendations,
        "search_outlets": search_outlets,
        "get_model_card": get_model_card,
    }


# ------------------------------------------------------------- agent loop

def _post(api_key, messages, use_tools=True):
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 900,
    }
    if use_tools:
        payload["tools"] = TOOLS_SPEC
        payload["tool_choice"] = "auto"
    resp = requests.post(GROQ_URL, json=payload, timeout=60,
                         headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]


def agent_turn(api_key, ctx, history):
    """Run one user turn through the tool loop.

    Returns (answer_text, artifacts) where artifacts = [{tool, args, data}]
    for everything the agent looked up — the app renders these visually.
    """
    tools = make_tools(ctx)
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + history[-MAX_HISTORY_MSGS:])
    artifacts = []

    for _ in range(MAX_TOOL_ROUNDS):
        msg = _post(api_key, messages)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or "", artifacts
        messages.append({"role": "assistant",
                         "content": msg.get("content") or "",
                         "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                fn = tools[name]
                data = fn(**args)
            except Exception as exc:  # surface errors to the model, not the app
                data = {"error": str(exc)}
            artifacts.append({"tool": name, "args": args, "data": data})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": name,
                             "content": json.dumps(data)[:7000]})

    final = _post(api_key, messages, use_tools=False)
    return final.get("content") or "", artifacts
