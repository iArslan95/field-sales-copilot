"""ShelfMate — a GenAI field-sales copilot for CPG account managers.

Agent-first, model-transparent: the copilot tab is the front door (a Groq
function-calling agent over churn, recommendation and prioritisation tools),
while each model gets its own intel page with visuals, insights and a focused
specialist chat. Data is synthetic with planted structure, Unilever-style
portfolio for recognisability; every number is fictional.

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import agent
from engine import churn, data, recommender
from engine.recommender import recommend

st.set_page_config(
    page_title="ShelfMate — field sales copilot",
    page_icon="🛒",
    layout="wide",
)

DEFAULT_SEED = 3

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1180px;}
.hero {
  background: #ffffff;
  border: 1px solid #e7e5e4; border-left: 4px solid #0f766e;
  border-radius: 14px; padding: 22px 28px; margin-bottom: 16px;
}
.hero h1 {margin: 0; font-size: 1.7rem; color: #1c1917; letter-spacing: -0.01em;}
.hero p {margin: 7px 0 0; color: #78716c; font-size: 0.96rem; max-width: 95ch;}
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e7e5e4;
  border-radius: 12px; padding: 12px 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.stButton button {
  font-size: 0.84rem; text-align: left; width: 100%;
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 10px;
  color: #44403c; padding: 7px 12px;
}
.stButton button:hover {border-color: #0f766e; color: #0f766e;}
.stTabs [data-baseweb="tab-list"] {gap: 8px; padding: 2px 0 10px;}
.stTabs [data-baseweb="tab"] {
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 10px;
  padding: 9px 18px; font-weight: 600; font-size: 0.97rem; color: #57534e;
}
.stTabs [data-baseweb="tab"]:hover {border-color: #0f766e; color: #0f766e;}
.stTabs [aria-selected="true"] {background: #0f766e; border-color: #0f766e; color: #ffffff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
[data-testid="stChatMessage"] {background: transparent;}
.profile-card {
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 12px;
  padding: 12px 16px; margin: 4px 0; font-size: 0.9rem; color: #44403c;
}
.profile-card b {color: #1c1917;}
.insight {
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 12px;
  padding: 10px 16px; margin: 2px 0 10px; font-size: 0.9rem; color: #44403c;
}
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


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#57534e",
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
    )
    fig.update_xaxes(gridcolor="#e7e5e4", zeroline=False)
    fig.update_yaxes(gridcolor="#e7e5e4")
    return fig


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛒 ShelfMate")
    st.caption("GenAI copilot demo — synthetic territory, real models. "
               "Unilever brand names as recognisable flavour only.")
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
    st.caption("The copilot picks its own tools; the intel tabs add visuals "
               "plus a focused specialist chat per model.")

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
      slipping, and arms you with <b>next-best-SKU</b> talking points across a
      Knorr–Unox–Hellmann's–Dove portfolio — every answer grounded in real
      model output via function calling.</p>
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
          help="Sum of trailing-52-week wholesale revenue of all at-risk outlets.")
c4.metric("Model health",
          f"AUC {ctx['churn']['auc']:.2f} · "
          f"{ctx['evals']['hitrate_model'] / max(ctx['evals']['hitrate_popularity'], 1e-9):.1f}× pop",
          help="Churn AUC pooled over two out-of-time snapshots · recommender "
               "hit-rate@5 as a multiple of the popularity baseline. Details "
               "in Data & method.")

# ------------------------------------------------------------- chat state
chat_key = (int(seed), int(n_outlets), float(risk_thr))
if st.session_state.get("chat_key") != chat_key:
    st.session_state["chat_key"] = chat_key
    for k in ("chat_main", "chat_churn", "chat_rec"):
        st.session_state.pop(k, None)

BRIEFING = (
    f"Good morning ☀️ Territory check for week {ctx['scenario']['now']}: "
    f"**{len(risk_df)} outlets** sit above {risk_thr:.0%} churn risk, with "
    f"{eur(risk_df['value_52w'].sum())} of annual value at stake. Biggest "
    f"concern right now: **{top_risk_name}** ({top_risk['churn_p']:.0%} risk). "
    "Want me to plan your week, or dig into a specific outlet?"
)


def render_artifacts(arts, slot, msg_idx):
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
                key=f"dl_{slot}_{msg_idx}_{j}",
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
            risk = payload["churn_risk"]
            risk_txt = risk if isinstance(risk, str) else f"{risk:.0%}"
            st.markdown(
                f"""<div class="profile-card"><b>{payload['outlet']}</b> ·
                {payload['segment']} · {payload['district']} ·
                size {payload['size']}<br>
                Risk <b>{risk_txt}</b> ·
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
                f"({r['category']}, € {r['price']:.2f}/case — because they buy "
                f"{' & '.join(r['because'][:2])})</span>"
                for r in payload.get("recommendations", []))
            st.markdown(f"""<div class="profile-card"><b>Pitch for
                {payload['outlet']}</b>{recs}</div>""", unsafe_allow_html=True)


def total_user_messages():
    return sum(
        sum(1 for m in st.session_state.get(k, []) if m["role"] == "user")
        for k in ("chat_main", "chat_churn", "chat_rec")
    )


def chat_panel(slot, height, chips, intro=None, focus=None,
               placeholder="Ask your copilot…"):
    """A copilot panel with persistent suggestion chips (they never disappear)."""
    hist_key = f"chat_{slot}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = (
            [{"role": "assistant", "content": intro, "artifacts": []}]
            if intro else [])
    history = st.session_state[hist_key]

    box = st.container(height=height, border=True)
    with box:
        if not history:
            st.markdown("<small style='color:#a8a29e'>Ask the specialist — "
                        "or use a suggestion below.</small>",
                        unsafe_allow_html=True)
        for i, msg in enumerate(history):
            avatar = "🛒" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                render_artifacts(msg.get("artifacts", []), slot, i)

    api_key = agent.get_api_key()
    if not api_key:
        st.info("Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (locally) or "
                "the app's Secrets on Streamlit Cloud to wake the copilot.")
        return

    cols = st.columns(len(chips))
    for col, q in zip(cols, chips):
        with col:
            if st.button(q, key=f"chip_{slot}_{abs(hash(q)) % 99999}"):
                st.session_state[f"pending_{slot}"] = q

    user_msg = st.chat_input(placeholder, key=f"input_{slot}")
    user_msg = user_msg or st.session_state.pop(f"pending_{slot}", None)
    if not user_msg:
        return
    if total_user_messages() >= agent.MAX_USER_MESSAGES:
        st.warning("Chat limit reached for this session — change the territory "
                   "or refresh to start over.")
        return

    history.append({"role": "user", "content": user_msg, "artifacts": []})
    with box:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant", avatar="🛒"):
            with st.spinner("ShelfMate is checking the models…"):
                api_history = [{"role": m["role"], "content": m["content"]}
                               for m in history]
                try:
                    answer, artifacts = agent.agent_turn(
                        api_key, ctx, api_history, focus=focus)
                except Exception as exc:
                    answer, artifacts = f"⚠️ The copilot hit an error: {exc}", []
            st.markdown(answer)
            render_artifacts(artifacts, slot, len(history))
    history.append({"role": "assistant", "content": answer,
                    "artifacts": artifacts})
    st.rerun()


# -------------------------------------------------------------------- tabs
tab_main, tab_churn, tab_rec, tab_method = st.tabs(
    ["🤖 Copilot", "📉 Churn intel", "🧺 Next-best-SKU", "📚 Data & method"]
)

with tab_main:
    chat_panel(
        "main", 470,
        chips=[
            "Plan my top-10 visits for this week",
            f"Why is {top_risk_name} at risk, and what should I pitch there?",
            "How good are these models, honestly?",
        ],
        intro=BRIEFING,
        placeholder="Plan the week, probe an outlet, challenge the models…",
    )

# -------------------------------------------------------------- churn tab
with tab_churn:
    seg_stats = (scores.assign(at_risk=scores["churn_p"] >= risk_thr)
                 .groupby("segment")
                 .agg(outlets=("outlet_id", "count"),
                      at_risk=("at_risk", "sum"),
                      avg_risk=("churn_p", "mean"),
                      value_at_risk=("value_52w",
                                     lambda v: v[scores.loc[v.index, "churn_p"]
                                                 >= risk_thr].sum()))
                 .sort_values("value_at_risk", ascending=False))
    worst_seg = seg_stats.index[0]
    st.markdown(
        f"""<div class="insight">📌 <b>{len(risk_df)} outlets</b> above
        {risk_thr:.0%} risk, worth {eur(risk_df['value_52w'].sum())}/yr.
        Most value bleeding in <b>{worst_seg}</b>
        ({eur(seg_stats.loc[worst_seg, 'value_at_risk'])}). Sharpest case:
        <b>{top_risk_name}</b> at {top_risk['churn_p']:.0%}.</div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.35, 1])
    with col1:
        plot_df = scores.copy()
        plot_df["Outlet"] = plot_df["outlet_id"].map(
            lambda i: outlets_by_id[i].name)
        fig = px.scatter(
            plot_df, x="churn_p", y="value_52w", color="segment",
            hover_name="Outlet", title="Risk × value — who matters most",
            labels={"churn_p": "churn risk", "value_52w": "value €/yr",
                    "segment": ""},
            color_discrete_sequence=("#0f766e", "#e09f3e", "#7c6ba0",
                                     "#b56576", "#457b9d"),
        )
        fig.add_vline(x=risk_thr, line_dash="dot", line_color="#a8a29e")
        st.plotly_chart(style_fig(fig, 360))
    with col2:
        reason_kind = {"no order": "Gone quiet", "frequency": "Ordering less",
                       "basket": "Smaller baskets", "categories": "Dropping categories",
                       "delivery": "Delivery issues", "price-increased": "Price-shock exposure"}
        counts = {}
        for oid in risk_df["outlet_id"]:
            for r in ctx["churn"]["reasons"].get(oid, []):
                for kw, label in reason_kind.items():
                    if kw in r:
                        counts[label] = counts.get(label, 0) + 1
                        break
        rdf = pd.DataFrame(sorted(counts.items(), key=lambda t: t[1]),
                           columns=["Driver", "At-risk outlets"])
        fig = px.bar(rdf, x="At-risk outlets", y="Driver", orientation="h",
                     title="What drives the risk (reason codes)",
                     color_discrete_sequence=["#0f766e"])
        st.plotly_chart(style_fig(fig, 360))

    st.dataframe(
        seg_stats.reset_index().rename(columns={
            "segment": "Segment", "outlets": "Outlets", "at_risk": "At risk",
            "avg_risk": "Avg risk", "value_at_risk": "Value at risk (€/yr)"}),
        hide_index=True, width="stretch",
        column_config={"Avg risk": st.column_config.NumberColumn(format="%.0f%%")},
    )

    st.markdown("##### 💬 Ask the churn specialist")
    chat_panel(
        "churn", 360,
        chips=[
            "Which segment is bleeding the most value, and why?",
            "Build a save plan for the riskiest outlet",
            "What drives churn most across the territory?",
        ],
        focus="The user is on the churn-intel page. Act as the churn "
              "specialist: lean on get_action_list, get_outlet_profile, "
              "search_outlets and get_model_card, and frame answers around "
              "retention.",
        placeholder="Ask about risk, drivers, segments, save plans…",
    )

# ---------------------------------------------------------- recommender tab
with tab_rec:
    ev = ctx["evals"]
    rec = ctx["rec"]
    push = {}
    for o in ctx["scenario"]["outlets"]:
        if o.id not in rec["o_idx"]:
            continue
        for r in recommend(rec, o.id, k=3):
            push[r["sku"]] = push.get(r["sku"], 0) + 1
    push_df = pd.DataFrame(sorted(push.items(), key=lambda t: t[1])[-10:],
                           columns=["SKU", "Outlets where recommended"])
    top_push = push_df.iloc[-1]["SKU"] if len(push_df) else "—"
    st.markdown(
        f"""<div class="insight">📌 Hit-rate@5 on holdout adopters:
        <b>{ev['hitrate_model']:.0%}</b> vs {ev['hitrate_popularity']:.0%} for
        the popularity baseline
        (<b>{ev['hitrate_model'] / max(ev['hitrate_popularity'], 1e-9):.1f}×
        lift</b>, n={ev['n_adopters']}). Biggest whitespace right now:
        <b>{top_push}</b> — recommended at
        {int(push_df.iloc[-1]['Outlets where recommended']) if len(push_df) else 0}
        outlets.</div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.35])
    with col1:
        eval_df = pd.DataFrame({
            "Approach": ["Item-item CF", "Popularity baseline"],
            "Hit-rate@5": [ev["hitrate_model"], ev["hitrate_popularity"]],
        })
        fig = px.bar(eval_df, x="Approach", y="Hit-rate@5", text_auto=".0%",
                     title="Holdout evaluation (weeks 71–78 adopters)",
                     color="Approach",
                     color_discrete_sequence=("#0f766e", "#d6d3d1"))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 340))
    with col2:
        fig = px.bar(push_df, x="Outlets where recommended", y="SKU",
                     orientation="h", title="Territory whitespace — push list",
                     color_discrete_sequence=["#e09f3e"])
        st.plotly_chart(style_fig(fig, 340))

    st.markdown("##### 🔍 Outlet explorer")
    names = sorted(o.name for o in ctx["scenario"]["outlets"]
                   if o.id in set(scores["outlet_id"]))
    chosen = st.selectbox("Pick an outlet", names, label_visibility="collapsed")
    chosen_o = next(o for o in ctx["scenario"]["outlets"] if o.name == chosen)
    m = rec["m"][rec["o_idx"][chosen_o.id]]
    owned = [rec["skus"][j].name for j in list(reversed(m.argsort()))[:5]
             if m[j] > 0]
    recs = recommend(rec, chosen_o.id, k=3)
    rec_html = "".join(
        f"<br>• <b>{r['sku']}</b> <span style='color:#78716c'>"
        f"({r['category']}, € {r['price']:.2f}/case — because they buy "
        f"{' & '.join(r['because'][:2])})</span>" for r in recs)
    st.markdown(
        f"""<div class="profile-card"><b>{chosen}</b> · {chosen_o.segment} ·
        {chosen_o.wijk}<br><br><b>Buys most</b>: {', '.join(owned) or '—'}
        <br><br><b>Next best SKUs</b>{rec_html or '<br>• portfolio fully covered'}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("##### 💬 Ask the assortment specialist")
    chat_panel(
        "rec", 360,
        chips=[
            f"What should I pitch at {top_risk_name}?",
            "Which SKUs deserve a territory-wide push, and at which outlets?",
            "How does this beat just selling the bestsellers everywhere?",
        ],
        focus="The user is on the recommender-intel page. Act as the "
              "assortment specialist: lean on get_recommendations, "
              "get_outlet_profile, search_outlets and get_model_card, and "
              "frame answers around distribution and whitespace.",
        placeholder="Ask about pitches, whitespace, the evaluation…",
    )

# -------------------------------------------------------------- method tab
with tab_method:
    ev, ch = ctx["evals"], ctx["churn"]
    st.markdown(f"""
**The data** — synthetic territory: {len(ctx['scenario']['outlets'])} outlets,
{len(ctx['scenario']['skus'])} SKUs in 6 categories of a Unilever-style
portfolio (Knorr, Unox, Hellmann's, Calvé, Conimex, Cif, Domestos, Sun,
Robijn, Omo, Dove, Axe, Rexona, Andrélon), 78 weeks of weekly wholesale
orders in cases. Planted structure: segment-level brand preferences,
winter/summer seasonality, a price increase at week 58, delivery issues and
gradual ordering decay. Brand names are recognisable flavour — volumes,
prices and outlets are fictional.

**Churn model** — HistGradientBoosting on behavioural snapshot features
(recency, frequency trend, basket trend, categories lost, delivery issues,
price-increase exposure — no ground-truth leakage). Trained on snapshots at
weeks 50, 58 & 64; label = no orders in the next 6 weeks; evaluated
**out-of-time, pooled over snapshots 68 & 71**: **AUC {ch['auc']:.3f}**
(churn base rate {ch['base_rate']:.1%}).

**Recommender** — item-item collaborative filtering (recency-weighted)
learned on weeks ≤ 70, tested on SKUs genuinely adopted (bought in 2+ weeks)
in weeks 71–78: **hit-rate@5 {ev['hitrate_model']:.0%}** vs popularity
baseline {ev['hitrate_popularity']:.0%} across {ev['n_adopters']} adopters —
**{ev['hitrate_model'] / max(ev['hitrate_popularity'], 1e-9):.1f}× lift**.

**Prescriptive layer** — visit priority = churn probability ×
trailing-52-week value; pitch = next-best-SKUs with their "because".

**The agent** — Groq (Llama 3.3 70B) with function calling over six tools;
every answer is grounded in tool output, and the app renders that output as
tables and cards under the reply.

*Roadmap to production: retrain on real POS/SAP sell-out, uplift-based
targeting (who is saveable, not just at risk), campaign holdouts,
Databricks/MLflow industrialisation, agent guardrails & evaluation.*
""")
    peek = scores.copy()
    peek["Outlet"] = peek["outlet_id"].map(lambda i: outlets_by_id[i].name)
    peek = peek[["Outlet", "segment", "size", "churn_p", "value_52w",
                 "recency_w", "orders_12w", "basket_12w"]]
    peek.columns = ["Outlet", "Segment", "Size", "Churn risk", "Value 52w (€)",
                    "Recency (w)", "Orders 12w", "Avg basket (€)"]
    st.dataframe(peek.sort_values("Churn risk", ascending=False),
                 hide_index=True, width="stretch", height=280)
    skus = pd.DataFrame([{"SKU": s.name, "Category": s.category,
                          "Case price": s.price}
                         for s in ctx["scenario"]["skus"]])
    st.dataframe(skus, hide_index=True, width="stretch", height=240)

st.markdown(
    """<div class="footer">ShelfMate · Groq (Llama 3.3 70B) function-calling
    agent + scikit-learn + Streamlit · all data synthetic — Unilever brand
    names appear as recognisable flavour only; volumes, prices and outlets are
    fictional · inspired by CPG customer-development work (field sales,
    perfect store) · built by Ismail Arslan as a portfolio demo — not
    affiliated with or endorsed by Unilever or any retailer.</div>""",
    unsafe_allow_html=True,
)
