"""ShelfMate — a GenAI field-sales copilot for CPG account managers.

Agent-first demo: the chat is the primary interface. A Groq-hosted LLM with
function calling decides which tool to use (action list, outlet profile,
recommendations, search, model card); the tools run against a churn model and
an item-item recommender trained on synthetic-but-structured order data, and
every tool result is rendered as a table or card under the agent's answer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import agent
from engine import churn, data, recommender

st.set_page_config(
    page_title="ShelfMate — field sales copilot",
    page_icon="🛒",
    layout="wide",
)

DEFAULT_SEED = 3

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1150px;}
.hero {
  background: #ffffff;
  border: 1px solid #e7e5e4; border-left: 4px solid #0f766e;
  border-radius: 14px; padding: 22px 28px; margin-bottom: 16px;
}
.hero h1 {margin: 0; font-size: 1.7rem; color: #1c1917; letter-spacing: -0.01em;}
.hero p {margin: 7px 0 0; color: #78716c; font-size: 0.96rem; max-width: 90ch;}
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e7e5e4;
  border-radius: 12px; padding: 12px 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.stButton button {
  font-size: 0.85rem; text-align: left; width: 100%;
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 10px;
  color: #44403c; padding: 7px 12px;
}
.stButton button:hover {border-color: #0f766e; color: #0f766e;}
[data-testid="stChatMessage"] {background: transparent;}
.profile-card {
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 12px;
  padding: 12px 16px; margin: 4px 0; font-size: 0.9rem; color: #44403c;
}
.profile-card b {color: #1c1917;}
.footer {color: #a8a29e; font-size: 0.85rem; margin-top: 26px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def eur(x: float) -> str:
    return f"€ {x:,.0f}"


@st.cache_data(show_spinner="Training models on this territory…")
def pipeline(seed: int, n_outlets: int):
    scenario = data.generate(seed, n_outlets=n_outlets)
    ch = churn.train(scenario)
    ev = recommender.evaluate(scenario)
    rec = recommender.build(scenario)
    return {"scenario": scenario, "churn": ch, "rec": rec, "evals": ev}


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛒 ShelfMate")
    st.caption("GenAI copilot demo — synthetic territory, real models.")
    st.markdown("#### Territory")
    n_outlets = st.slider("Outlets in territory", 150, 400, 300, 50,
                          help="Size of the synthetic territory. Models retrain "
                               "automatically (a few seconds).")
    seed = st.number_input("Random seed", 1, 999, DEFAULT_SEED,
                           help="Same seed = same territory. Change it for a "
                                "fresh book of business.")
    with st.expander("⚙️ Advanced"):
        risk_thr = st.slider("At-risk threshold", 0.10, 0.60, 0.25, 0.05,
                             help="Churn probability above which an outlet "
                                  "counts as 'at risk' in the KPIs and briefing. "
                                  "The base churn rate is only ~4-5%, so 25% "
                                  "already means five-times-elevated risk.")
    st.caption("The agent picks its own tools: action list, outlet profiles, "
               "recommendations, search and the model card.")

ctx = pipeline(int(seed), int(n_outlets))
ctx = dict(ctx, risk_threshold=float(risk_thr))
scores = ctx["churn"]["scores"]
outlets_by_id = {o.id: o for o in ctx["scenario"]["outlets"]}

risk_df = scores[scores["churn_p"] >= risk_thr]
top_risk = scores.sort_values("churn_p", ascending=False).iloc[0]
top_risk_name = outlets_by_id[top_risk["outlet_id"]].name

# ------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <h1>🛒 ShelfMate — your field-sales copilot</h1>
      <p>A GenAI agent for the account manager: it plans your week from
      <b>churn risk × customer value</b>, explains <i>why</i> each outlet is
      slipping, and arms you with <b>next-best-SKU</b> talking points — every
      answer grounded in real model output via function calling.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active outlets", f"{len(scores)}",
          help="Outlets with at least one order in the last 8 weeks.")
c2.metric("At risk", f"{len(risk_df)}",
          help=f"Outlets above {risk_thr:.0%} churn probability.")
c3.metric("Annual value at risk", eur(risk_df["value_52w"].sum()),
          help="Sum of trailing-52-week revenue of all at-risk outlets.")
c4.metric("Model health",
          f"AUC {ctx['churn']['auc']:.2f} · "
          f"{ctx['evals']['hitrate_model'] / max(ctx['evals']['hitrate_popularity'], 1e-9):.1f}× pop",
          help="Churn AUC on an out-of-time snapshot · recommender hit-rate@5 "
               "as a multiple of the popularity baseline. Details in the model "
               "card below.")

# ------------------------------------------------------------- chat state
chat_key = (int(seed), int(n_outlets), float(risk_thr))
if st.session_state.get("chat_key") != chat_key:
    st.session_state["chat_key"] = chat_key
    st.session_state["chat"] = [{
        "role": "assistant",
        "content": (
            f"Good morning ☀️ Territory check for week {ctx['scenario']['now']}: "
            f"**{len(risk_df)} outlets** sit above {risk_thr:.0%} churn risk, "
            f"with {eur(risk_df['value_52w'].sum())} of annual value at stake. "
            f"Biggest concern right now: **{top_risk_name}** "
            f"({top_risk['churn_p']:.0%} risk). "
            "Want me to plan your week, or dig into a specific outlet?"
        ),
        "artifacts": [],
    }]

history = st.session_state["chat"]


def render_artifacts(arts, msg_idx):
    for j, art in enumerate(arts):
        tool, payload = art["tool"], art["data"]
        if "error" in payload:
            continue
        if tool == "get_action_list":
            rows = [{
                "#": v["rank"], "Outlet": v["outlet"], "Segment": v["segment"],
                "District": v["district"], "Risk": f"{v['churn_risk']:.0%}",
                "Value/yr": eur(v["annual_value_eur"]),
                "Why": " · ".join(v["reasons"]),
                "Pitch": " · ".join(p.split(" (")[0] for p in v["pitch"]),
            } for v in payload["visits"]]
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button(
                "⬇️ Download visit list (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="shelfmate_visits.csv", mime="text/csv",
                key=f"dl_{msg_idx}_{j}",
            )
        elif tool == "search_outlets":
            rows = [{
                "Outlet": m["outlet"], "Segment": m["segment"],
                "District": m["district"], "Risk": f"{m['churn_risk']:.0%}",
                "Value/yr": eur(m["annual_value_eur"]),
            } for m in payload["matches"]]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             width="stretch")
        elif tool == "get_outlet_profile":
            recs = "".join(
                f"<br>• <b>{r['sku']}</b> <span style='color:#78716c'>"
                f"(because they buy {' & '.join(r['because'][:2])})</span>"
                for r in payload.get("recommended_pitch", []))
            reasons = "".join(f"<br>• {r}" for r in payload.get("risk_reasons", []))
            st.markdown(
                f"""<div class="profile-card"><b>{payload['outlet']}</b> ·
                {payload['segment']} · {payload['district']} ·
                size {payload['size']}<br>
                Risk <b>{payload['churn_risk'] if isinstance(payload['churn_risk'], str)
                         else f"{payload['churn_risk']:.0%}"}</b> ·
                value {eur(payload['annual_value_eur'])}/yr ·
                last order {payload['weeks_since_last_order']} w ago ·
                basket trend {payload['basket_trend_vs_12w']:.0%}
                <br><br><b>Why at risk</b>{reasons}
                <br><br><b>What to pitch</b>{recs}</div>""",
                unsafe_allow_html=True,
            )
        elif tool == "get_recommendations":
            recs = "".join(
                f"<br>• <b>{r['sku']}</b> <span style='color:#78716c'>"
                f"({r['category']}, € {r['price']:.2f} — because they buy "
                f"{' & '.join(r['because'][:2])})</span>"
                for r in payload.get("recommendations", []))
            st.markdown(f"""<div class="profile-card"><b>Pitch for
                {payload['outlet']}</b>{recs}</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- copilot
box = st.container(height=520, border=True)
with box:
    for i, msg in enumerate(history):
        avatar = "🛒" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_artifacts(msg.get("artifacts", []), i)

api_key = agent.get_api_key()
if not api_key:
    st.info("Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (locally) or the "
            "app's Secrets on Streamlit Cloud to wake the copilot.")
    st.stop()

n_user = sum(1 for m in history if m["role"] == "user")
if n_user >= agent.MAX_USER_MESSAGES:
    st.warning("Chat limit reached for this session — change the territory or "
               "refresh to start over.")
    st.stop()

if n_user == 0:
    chips = [
        "Plan my top-10 visits for this week",
        f"Why is {top_risk_name} at risk, and what should I pitch there?",
        "How good are these models, honestly?",
    ]
    cols = st.columns(3)
    for col, q in zip(cols, chips):
        with col:
            if st.button(q, key=f"chip_{hash(q) % 9999}"):
                st.session_state["pending_q"] = q

user_msg = st.chat_input("Ask your copilot — plan the week, probe an outlet, "
                         "challenge the models…")
user_msg = user_msg or st.session_state.pop("pending_q", None)

if user_msg:
    history.append({"role": "user", "content": user_msg, "artifacts": []})
    with box:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant", avatar="🛒"):
            with st.spinner("ShelfMate is checking the models…"):
                api_history = [{"role": m["role"], "content": m["content"]}
                               for m in history]
                try:
                    answer, artifacts = agent.agent_turn(api_key, ctx, api_history)
                except Exception as exc:
                    answer, artifacts = f"⚠️ The copilot hit an error: {exc}", []
            st.markdown(answer)
            render_artifacts(artifacts, len(history))
    history.append({"role": "assistant", "content": answer,
                    "artifacts": artifacts})
    st.rerun()

# ------------------------------------------------------ supporting detail
with st.expander("📊 Model card — how good is this, honestly?"):
    ev, ch = ctx["evals"], ctx["churn"]
    st.markdown(f"""
**Churn model** — HistGradientBoosting on behavioural snapshot features
(recency, frequency trend, basket trend, categories lost, delivery issues,
price-increase exposure — no ground-truth leakage). Trained on snapshots at
weeks 58 & 64; label = no orders in the next 6 weeks; evaluated
**out-of-time** at week 70: **AUC {ch['auc']:.3f}** (churn base rate
{ch['base_rate']:.1%}).

**Recommender** — item-item collaborative filtering (recency-weighted) learned
on weeks ≤ 70, tested on SKUs genuinely adopted (bought in 2+ weeks) in weeks
71–78: **hit-rate@5 {ev['hitrate_model']:.0%}** vs popularity baseline
{ev['hitrate_popularity']:.0%} across {ev['n_adopters']} adopters —
**{ev['hitrate_model'] / max(ev['hitrate_popularity'], 1e-9):.1f}× lift**.

**Prescriptive layer** — visit priority = churn probability × trailing-52-week
value; pitch = top next-best-SKUs with their "because" explanation.

*Roadmap to production: retrain on real POS/SAP sell-out, uplift-based
targeting (who is saveable, not just who is at risk), campaign holdouts,
Databricks/MLflow industrialisation, and agent guardrails + evaluation.*
""")

with st.expander("📦 Data peek — the synthetic territory"):
    peek = scores.copy()
    peek["Outlet"] = peek["outlet_id"].map(lambda i: outlets_by_id[i].name)
    peek = peek[["Outlet", "segment", "size", "churn_p", "value_52w",
                 "recency_w", "orders_12w", "basket_12w"]]
    peek.columns = ["Outlet", "Segment", "Size", "Churn risk", "Value 52w (€)",
                    "Recency (w)", "Orders 12w", "Avg basket (€)"]
    st.dataframe(peek.sort_values("Churn risk", ascending=False),
                 hide_index=True, width="stretch", height=300)
    skus = pd.DataFrame([{"SKU": s.name, "Category": s.category,
                          "Price": s.price} for s in ctx["scenario"]["skus"]])
    st.dataframe(skus, hide_index=True, width="stretch", height=240)

st.markdown(
    """<div class="footer">ShelfMate · Groq (Llama 3.3 70B) function-calling
    agent + scikit-learn + Streamlit · all data synthetic · inspired by CPG
    customer-development work (field sales, perfect store) · built by Ismail
    Arslan as a portfolio demo — not affiliated with any retailer or CPG
    company.</div>""",
    unsafe_allow_html=True,
)
