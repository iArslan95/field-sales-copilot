"""Synthetic CPG order data for a field-sales territory (Rotterdam region).

One scenario = 78 weeks of weekly orders from a few hundred small retail
outlets (neighbourhood supermarkets, night shops, tokos, food-service,
forecourt shops) buying a fictional CPG portfolio across six categories.

The data is generated with real latent structure, so the models have
something genuine to learn:
- outlets prefer the SKUs popular in their segment (collaborative signal),
- winter/summer categories follow a seasonal curve,
- a price increase hits two categories at week 58 and price-sensitive
  outlets cut those baskets,
- some outlets suffer delivery issues, and a share of outlets starts a
  gradual ordering decay — the churn the model must catch early.

All names are fictional; no real retailer or brand data is used.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

WEEKS = 78          # week 78 = "now"
PRICE_SHOCK_WEEK = 58
SHOCKED_CATEGORIES = ("Sauzen", "Personal care")

SEGMENTS = {
    # segment: (share, weekly order prob, basket scale)
    "Buurtsuper": (0.30, 0.85, 1.00),
    "Avondwinkel": (0.20, 0.70, 0.60),
    "Toko & speciaalzaak": (0.20, 0.65, 0.70),
    "Horeca": (0.15, 0.75, 1.40),
    "Tankstation": (0.15, 0.60, 0.50),
}

SIZES = {"S": 0.6, "M": 1.0, "L": 1.6}

WIJKEN = (
    "Delfshaven", "Kralingen", "Charlois", "Feijenoord", "Blijdorp",
    "Overschie", "IJsselmonde", "Hillegersberg", "Ommoord", "Pendrecht",
    "Katendrecht", "Noord", "Kop van Zuid", "Schiebroek", "Lombardijen",
    "Hoogvliet", "Capelle", "Schiedam", "Vlaardingen", "Barendrecht",
)

NAME_POOLS = {
    "Buurtsuper": ("Buurtsuper De Linde", "Buurtsuper 't Hoekje", "Supermarkt Van Dijk",
                   "Buurtsuper De Eik", "Supermarkt Nieuw West", "Buurtsuper Morgenster",
                   "Supermarkt De Brink", "Buurtsuper Parkzicht"),
    "Avondwinkel": ("Avondwinkel Nova", "Avondwinkel Luna", "Nachtwinkel City",
                    "Avondwinkel 24Zeven", "Avondwinkel De Ster", "Nachtshop Centraal"),
    "Toko & speciaalzaak": ("Toko Bandung", "Toko Surabaya", "Toko Sari",
                            "Bazar Mevlana", "Toko Anatolia", "Mercado Lisboa",
                            "Toko Casablanca", "Delicatessen Adria"),
    "Horeca": ("Grandcafé De Brug", "Eetcafé Zuid", "Lunchroom Vers",
               "Cafetaria Smul", "Snackbar De Punt", "Brasserie Maaszicht"),
    "Tankstation": ("Tank & Shop A20", "Servico Zuidplein", "Tankshop De Vaan",
                    "Motorpoort Shop", "Tank & Go Ring"),
}

# category: (season, price range) — season bends the demand curve.
CATEGORIES = {
    "Thee": ("winter", (2.2, 3.5)),
    "Bouillon & soep": ("winter", (1.8, 3.0)),
    "Sauzen": ("summer", (2.0, 4.0)),
    "Wasmiddel": (None, (6.0, 11.0)),
    "Personal care": (None, (3.0, 6.0)),
    "IJs": ("summer", (4.0, 7.0)),
}

SKU_NAMES = {
    "Thee": ("Bremer Earl Grey", "Bremer Groene Thee", "Bremer Rooibos",
             "Bremer Citroen", "Thejo Zwarte Thee", "Thejo Munt",
             "Thejo Kamille", "Thejo Gember"),
    "Bouillon & soep": ("Goudbouillon Kip", "Goudbouillon Runder",
                        "Goudbouillon Groente", "Goudbouillon Vis",
                        "SoepNu Tomaat", "SoepNu Champignon", "SoepNu Erwten",
                        "SoepNu Kip"),
    "Sauzen": ("Mayolin Mayonaise", "Mayolin Fritessaus", "Mayolin Truffelmayo",
               "Delisaus Curry", "Delisaus Ketchup", "Delisaus Knoflook",
               "Delisaus Samurai", "Delisaus Piri Piri"),
    "Wasmiddel": ("Wasko Color", "Wasko Wit", "Wasko Zwart", "Wasko Sport",
                  "Linnea Vloeibaar", "Linnea Pods", "Linnea Sensitive",
                  "Linnea Wol"),
    "Personal care": ("Fresqo Deo Roller", "Fresqo Deo Spray", "Fresqo Douchegel",
                      "Fresqo Scrub", "Velura Shampoo", "Velura Conditioner",
                      "Velura Handzeep", "Velura Bodylotion"),
    "IJs": ("Polario Vanille", "Polario Chocolade", "Polario Aardbei",
            "Polario Hazelnoot", "Scoopy Cookies", "Scoopy Caramel",
            "Scoopy Mango", "Scoopy Pistache"),
}


@dataclass(frozen=True)
class Sku:
    id: str
    name: str
    category: str
    price: float


@dataclass(frozen=True)
class Outlet:
    id: str
    name: str
    segment: str
    wijk: str
    size: str
    start_week: int
    churn_start: int        # 0 = healthy; hidden ground truth, never a feature
    price_sensitive: bool
    issue_weeks: tuple      # weeks with delivery problems (observable)


def _season_factor(category: str, week: int) -> float:
    season = CATEGORIES[category][0]
    yw = (week - 1) % 52 + 1
    if season == "winter":
        return max(0.35, 1 + 0.40 * math.cos(2 * math.pi * (yw - 1) / 52))
    if season == "summer":
        return max(0.30, 1 + 0.55 * math.cos(2 * math.pi * (yw - 26) / 52))
    return 1.0


def _make_skus(rng: random.Random):
    skus = []
    for cat, names in SKU_NAMES.items():
        lo, hi = CATEGORIES[cat][1]
        for k, name in enumerate(names):
            skus.append(Sku(id=f"SKU-{len(skus):03d}", name=name, category=cat,
                            price=round(rng.uniform(lo, hi), 2)))
    return skus


def generate(seed: int, n_outlets: int = 300) -> dict:
    rng = random.Random(seed)
    skus = _make_skus(rng)
    by_cat = {cat: [s for s in skus if s.category == cat] for cat in CATEGORIES}

    # Segment-level popular subsets: the collaborative-filtering signal.
    seg_popular = {
        seg: {cat: rng.sample(by_cat[cat], 3) for cat in CATEGORIES}
        for seg in SEGMENTS
    }

    seg_names = list(SEGMENTS)
    seg_weights = [SEGMENTS[s][0] for s in seg_names]
    used_names = set()
    outlets, prefs, affinities, adoptions = [], {}, {}, {}

    for i in range(n_outlets):
        seg = rng.choices(seg_names, weights=seg_weights, k=1)[0]
        for _ in range(40):
            name = f"{rng.choice(NAME_POOLS[seg])} · {rng.choice(WIJKEN)}"
            if name not in used_names:
                break
            name = None
        if name is None:
            name = f"{rng.choice(NAME_POOLS[seg])} · {rng.choice(WIJKEN)} {i}"
        used_names.add(name)

        roll = rng.random()
        churn_start = 0
        if roll < 0.18:
            churn_start = rng.randint(52, 74)      # the decay we must catch
        elif roll < 0.24:
            churn_start = rng.randint(20, 45)      # historic churners (training)

        outlet = Outlet(
            id=f"OUT-{1000 + i}", name=name, segment=seg,
            wijk=name.split("·")[-1].strip(),
            size=rng.choices(("S", "M", "L"), weights=(0.35, 0.45, 0.20), k=1)[0],
            start_week=1 if rng.random() < 0.85 else rng.randint(10, 50),
            churn_start=churn_start,
            price_sensitive=rng.random() < 0.35,
            issue_weeks=tuple(sorted(rng.sample(range(56, WEEKS + 1),
                                                rng.randint(2, 5))))
            if rng.random() < 0.12 else (),
        )
        outlets.append(outlet)

        # 2 preferred SKUs per category, mostly from the segment's popular set.
        prefs[outlet.id] = {
            cat: rng.sample(seg_popular[seg][cat], 2) if rng.random() < 0.8
            else rng.sample(by_cat[cat], 2)
            for cat in CATEGORIES
        }
        affinities[outlet.id] = {
            cat: 0.0 if rng.random() < 0.15
            else SEGMENT_AFFINITY[seg][cat] * rng.uniform(0.6, 1.4)
            for cat in CATEGORIES
        }

        # Assortment adoption: over time an outlet picks up the remaining
        # segment favourite in a few categories. Predictable from segment
        # co-occurrence — exactly what the recommender should catch.
        candidates = [
            cat for cat in CATEGORIES
            if affinities[outlet.id][cat] > 0
            and any(s not in prefs[outlet.id][cat] for s in seg_popular[seg][cat])
        ]
        events = {}
        for cat in rng.sample(candidates, k=min(len(candidates), rng.randint(1, 3))):
            extra = next(s for s in seg_popular[seg][cat]
                         if s not in prefs[outlet.id][cat])
            events[cat] = (rng.randint(24, 76), extra)
        adoptions[outlet.id] = events

    orders = {o.id: {} for o in outlets}
    for o in outlets:
        _, order_p, basket_scale = SEGMENTS[o.segment]
        size_mult = SIZES[o.size]
        for w in range(o.start_week, WEEKS + 1):
            p = order_p
            if o.churn_start and w >= o.churn_start:
                p *= 0.80 ** (w - o.churn_start)
            if w in o.issue_weeks:
                p *= 0.45
            if rng.random() > p:
                continue
            basket = {}
            for cat in CATEGORIES:
                lam = affinities[o.id][cat] * _season_factor(cat, w)
                if (o.price_sensitive and w >= PRICE_SHOCK_WEEK
                        and cat in SHOCKED_CATEGORIES):
                    lam *= 0.50
                if lam <= 0 or rng.random() > min(0.95, lam):
                    continue
                chosen = list(prefs[o.id][cat])
                adopted = adoptions[o.id].get(cat)
                if adopted and w >= adopted[0]:
                    chosen.append(adopted[1])
                if rng.random() < 0.12:  # occasional exploration
                    pool = seg_popular[o.segment][cat] if rng.random() < 0.7 \
                        else by_cat[cat]
                    chosen.append(rng.choice(pool))
                for sku in chosen:
                    if rng.random() < 0.75:
                        qty = max(1, int(rng.uniform(1, 4) * size_mult
                                         * basket_scale))
                        basket[sku.id] = basket.get(sku.id, 0) + qty
            if basket:
                orders[o.id][w] = basket

    return {"seed": seed, "outlets": tuple(outlets), "skus": tuple(skus),
            "orders": orders, "now": WEEKS}


# Baseline appetite of each segment per category (0..1-ish).
SEGMENT_AFFINITY = {
    "Buurtsuper": {"Thee": 0.75, "Bouillon & soep": 0.70, "Sauzen": 0.80,
                   "Wasmiddel": 0.65, "Personal care": 0.70, "IJs": 0.60},
    "Avondwinkel": {"Thee": 0.35, "Bouillon & soep": 0.30, "Sauzen": 0.60,
                    "Wasmiddel": 0.25, "Personal care": 0.50, "IJs": 0.80},
    "Toko & speciaalzaak": {"Thee": 0.80, "Bouillon & soep": 0.75, "Sauzen": 0.70,
                            "Wasmiddel": 0.30, "Personal care": 0.45, "IJs": 0.35},
    "Horeca": {"Thee": 0.65, "Bouillon & soep": 0.85, "Sauzen": 0.90,
               "Wasmiddel": 0.10, "Personal care": 0.15, "IJs": 0.70},
    "Tankstation": {"Thee": 0.30, "Bouillon & soep": 0.35, "Sauzen": 0.55,
                    "Wasmiddel": 0.15, "Personal care": 0.55, "IJs": 0.85},
}


# ----------------------------------------------------------------- features

def order_value(scenario, oid: str, week: int) -> float:
    prices = {s.id: s.price for s in scenario["skus"]}
    basket = scenario["orders"][oid].get(week, {})
    return sum(qty * prices[sid] for sid, qty in basket.items())


def is_active(scenario, oid: str, asof: int, window: int = 8) -> bool:
    return any(w in scenario["orders"][oid]
               for w in range(max(1, asof - window + 1), asof + 1))


def churn_label(scenario, oid: str, asof: int, horizon: int = 6) -> int:
    """1 = no orders in the `horizon` weeks after `asof`."""
    return int(not any(w in scenario["orders"][oid]
                       for w in range(asof + 1, asof + horizon + 1)))


def features(scenario, oid: str, asof: int) -> dict:
    """Observable behaviour features at a snapshot week (no ground truth)."""
    outlet = next(o for o in scenario["outlets"] if o.id == oid)
    weeks = sorted(w for w in scenario["orders"][oid] if w <= asof)
    cats = {s.id: s.category for s in scenario["skus"]}

    def in_window(a, b):
        return [w for w in weeks if a < w <= b]

    last = weeks[-1] if weeks else 0
    o4, o12 = in_window(asof - 4, asof), in_window(asof - 12, asof)
    val4 = sum(order_value(scenario, oid, w) for w in o4)
    val12 = sum(order_value(scenario, oid, w) for w in o12)
    basket4 = val4 / len(o4) if o4 else 0.0
    basket12 = val12 / len(o12) if o12 else 0.0

    def cats_in(ws):
        out = set()
        for w in ws:
            out |= {cats[sid] for sid in scenario["orders"][oid][w]}
        return out

    cats_recent = cats_in(o4)
    cats_before = cats_in(in_window(asof - 16, asof - 4))
    pre = in_window(asof - 26, asof - 13)
    shock_val = sum(qty * 1 for w in pre
                    for sid, qty in scenario["orders"][oid][w].items()
                    if cats[sid] in SHOCKED_CATEGORIES)
    all_val = sum(qty * 1 for w in pre
                  for qty in scenario["orders"][oid][w].values()) or 1

    return {
        "recency_w": min(26, asof - last) if weeks else 26,
        "orders_4w": len(o4),
        "orders_12w": len(o12),
        "freq_trend": len(o4) / max(len(o12) / 3.0, 0.5),
        "basket_12w": round(basket12, 1),
        "basket_trend": basket4 / basket12 if basket12 else 0.0,
        "cats_4w": len(cats_recent),
        "cats_lost": max(0, len(cats_before - cats_recent)),
        "issues_8w": sum(1 for w in outlet.issue_weeks if asof - 8 < w <= asof),
        "shock_share": shock_val / all_val,
        "tenure_w": asof - outlet.start_week,
        "value_52w": round(sum(order_value(scenario, oid, w)
                               for w in in_window(asof - 52, asof)), 0),
        "segment": outlet.segment,
        "size": outlet.size,
    }
