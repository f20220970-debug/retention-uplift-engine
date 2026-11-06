"""Retention & reactivation engine on simulated customers with a known ground truth.

1. Survival: Kaplan-Meier time-to-churn by segment and by first-order experience (implemented from scratch)
2. Uplift targeting: who should a reactivation offer go to? T-learner vs churn-risk targeting vs random, scored by Qini
3. Multi-channel allocation: push / WhatsApp / SMS / call under a budget, by predicted uplift per rupee
4. Loyalty programme ROI: naive member vs non-member gap vs a propensity-score-matched estimate, against the truth
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

SEGMENTS = ["Kirana / retail", "D2C seller", "Furniture / home", "Individual mover"]
CHANNELS = {"Push": 0.5, "WhatsApp": 1.2, "SMS": 0.8, "Call": 25.0}       # cost per contact, INR
CH_STRENGTH = {"Push": 0.6, "WhatsApp": 1.0, "SMS": 0.5, "Call": 1.8}     # multiplies the true uplift
OFFER_COST = 60                                                           # INR discount when redeemed


def customers(seed=0, n=40000):
    r = np.random.default_rng(seed)
    seg = r.choice(SEGMENTS, n, p=[.35, .25, .15, .25])
    tenure = r.gamma(2, 10, n)                       # weeks since first order
    orders_90d = r.poisson(np.select([seg == SEGMENTS[0], seg == SEGMENTS[1], seg == SEGMENTS[2]], [8, 6, 2], 0.6))
    days_since = r.exponential(40, n).round()
    bad_first = r.random(n) < .12
    discount_share = r.beta(2, 5, n)                 # share of past orders that used a coupon
    # baseline probability of ordering in the next 30 days without any campaign
    z = -0.6 + .12 * np.log1p(orders_90d) * 3 - .035 * days_since - .9 * bad_first + .01 * tenure
    base = 1 / (1 + np.exp(-z))
    # TRUE uplift of a WhatsApp reactivation offer: largest for lapsing habitual users and coupon-lovers,
    # near zero for very active users (sure things), negative for movers who find it spammy (sleeping dogs)
    lapsing = (days_since > 20) & (days_since < 75)
    tau = (.01 + .11 * lapsing * (orders_90d >= 3) + .08 * discount_share * lapsing - .06 * (seg == "Individual mover") - .02 * (days_since > 120)
           - .03 * (days_since < 7))
    tau = np.clip(tau, -.1, .4)
    t = r.random(n) < .5                             # randomised campaign test, 50/50
    p = np.clip(base + t * tau, 0, 1)
    ordered = r.random(n) < p
    # churn time for survival: weeks until last order, censored at 26
    hazard = .03 + .08 * bad_first + .05 * (seg == "Individual mover") - .01 * np.log1p(orders_90d)
    churn_wk = r.exponential(1 / np.clip(hazard, .005, None))
    observed = churn_wk < 26
    churn_wk = np.minimum(churn_wk, 26)
    return pd.DataFrame(dict(segment=seg, tenure_wk=tenure.round(1), orders_90d=orders_90d, days_since_last=days_since,
                             bad_first_order=bad_first, discount_share=discount_share.round(3), treated=t,
                             ordered_30d=ordered, true_uplift=tau, base_p=base, churn_wk=churn_wk.round(2),
                             churned=observed))


FEATS = ["tenure_wk", "orders_90d", "days_since_last", "bad_first_order", "discount_share", "seg_code"]


def km(time, event):
    """Kaplan-Meier survival curve; returns DataFrame(t, S)."""
    d = pd.DataFrame(dict(t=time, e=event)).sort_values("t")
    out, S, at_risk = [(0.0, 1.0)], 1.0, len(d)
    for t, g in d.groupby("t"):
        deaths = g.e.sum()
        if deaths:
            S *= 1 - deaths / at_risk
            out.append((t, S))
        at_risk -= len(g)
    return pd.DataFrame(out, columns=["t", "S"])


def median_survival(curve):
    below = curve[curve.S <= .5]
    return float(below.t.iloc[0]) if len(below) else float("inf")


def survival(c):
    res = {}
    for name, mask in [("good first order", ~c.bad_first_order), ("bad first order", c.bad_first_order)] + \
                      [(s, c.segment == s) for s in SEGMENTS]:
        curve = km(c.churn_wk[mask], c.churned[mask])
        med = median_survival(curve)
        res[name] = dict(median_weeks_to_churn=round(med, 1) if np.isfinite(med) else "not reached in 26 wk",
                         retained_at_12wk=round(float(curve[curve.t <= 12].S.iloc[-1]) * 100, 1),
                         retained_at_26wk=round(float(curve.S.iloc[-1]) * 100, 1))
    return res


def qini(uplift_score, y, t, frac_grid=np.linspace(0, 1, 21)):
    """Incremental orders captured when targeting the top-k by score (Qini curve) and the area vs random."""
    order = np.argsort(-uplift_score)
    y, t = y[order], t[order]
    n = len(y)
    pts = []
    for f in frac_grid:
        k = max(int(f * n), 1)
        yt, nt = y[:k][t[:k]].sum(), t[:k].sum()
        yc, nc = y[:k][~t[:k]].sum(), (~t[:k]).sum()
        pts.append(yt - yc * (nt / nc if nc else 0))
    pts = np.array(pts)
    random_line = pts[-1] * frac_grid
    return pts, float(np.trapezoid(pts - random_line, frac_grid))


def uplift(c, seed=0):
    c = c.assign(seg_code=c.segment.map({s: i for i, s in enumerate(SEGMENTS)}))
    r = np.random.default_rng(seed)
    test = r.random(len(c)) < .4
    tr, te = c[~test], c[test]
    X = lambda d: d[FEATS].astype(float).values
    m1 = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=150).fit(X(tr[tr.treated]), tr[tr.treated].ordered_30d)
    m0 = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=150).fit(X(tr[~tr.treated]), tr[~tr.treated].ordered_30d)
    tau_hat = m1.predict_proba(X(te))[:, 1] - m0.predict_proba(X(te))[:, 1]
    risk = 1 - m0.predict_proba(X(te))[:, 1]                    # churn-risk targeting: contact the least likely to order
    y, t = te.ordered_30d.values, te.treated.values
    q_up, a_up = qini(tau_hat, y, t)
    q_risk, a_risk = qini(risk, y, t)
    q_rand, a_rand = qini(r.random(len(te)), y, t)
    # at a 30% contact budget, true incremental orders per 1,000 contacted (uses ground truth)
    k = int(.3 * len(te))
    def true_inc(score):
        return float(te.true_uplift.values[np.argsort(-score)[:k]].mean() * 1000)
    corr = float(np.corrcoef(tau_hat, te.true_uplift)[0, 1])
    return te.assign(tau_hat=tau_hat), dict(
        qini_area=dict(uplift_model=round(a_up, 1), churn_risk=round(a_risk, 1), random=round(a_rand, 1)),
        true_incremental_orders_per_1000_at_30pct=dict(uplift_model=round(true_inc(tau_hat), 1),
                                                       churn_risk=round(true_inc(risk), 1),
                                                       random=round(float(te.true_uplift.mean() * 1000), 1)),
        corr_predicted_vs_true_uplift=round(corr, 3),
        share_sleeping_dogs_in_risk_top30=round(float((te.true_uplift.values[np.argsort(-risk)[:k]] < 0).mean() * 100), 1),
        share_sleeping_dogs_in_uplift_top30=round(float((te.true_uplift.values[np.argsort(-tau_hat)[:k]] < 0).mean() * 100), 1),
    )


def allocate(te, budget=40000, value_per_order=180):
    """Greedy: each customer gets at most one channel; rank (customer, channel) pairs by expected profit per rupee."""
    rows = []
    for ch, cost in CHANNELS.items():
        inc = np.clip(te.tau_hat.values, 0, None) * CH_STRENGTH[ch]
        exp_cost = cost + (te.base_p.values + inc) * OFFER_COST * .5
        profit = inc * value_per_order - exp_cost
        rows.append(pd.DataFrame(dict(idx=te.index, ch=ch, inc=inc, cost=exp_cost, profit=profit,
                                      true_inc=te.true_uplift.values * CH_STRENGTH[ch])))
    pairs = pd.concat(rows)
    pairs = pairs[pairs.profit > 0].assign(roi=lambda d: d.profit / d.cost).sort_values("roi", ascending=False)
    chosen, used, spent = [], set(), 0.0
    for p in pairs.itertuples():
        if p.idx in used or spent + p.cost > budget:
            continue
        chosen.append(p); used.add(p.idx); spent += p.cost
    ch = pd.DataFrame(chosen)
    # baseline plan: same budget spent on WhatsApp to random customers
    per = CHANNELS["WhatsApp"] + te.base_p.mean() * OFFER_COST * .5
    n_rand = int(budget / per)
    rand_inc = te.true_uplift.sample(n_rand, random_state=1).sum() * CH_STRENGTH["WhatsApp"]
    return dict(budget=budget, contacted=len(ch), mix=ch.ch.value_counts().to_dict(),
                true_incremental_orders=round(float(ch.true_inc.sum()), 1),
                random_whatsapp_true_incremental=round(float(rand_inc), 1))


def loyalty(seed=3, n=30000):
    """Observational: heavier users self-select into a loyalty programme. True effect = +0.6 orders/month."""
    r = np.random.default_rng(seed)
    freq = r.gamma(2, 1.5, n)                       # pre-period orders per month
    tenure = r.gamma(2, 10, n)
    join = r.random(n) < 1 / (1 + np.exp(-(-2.2 + .45 * freq + .02 * tenure)))
    post = r.poisson(np.clip(freq * 1.05 + .6 * join + r.normal(0, .3, n), .05, None))
    d = pd.DataFrame(dict(freq=freq, tenure=tenure, member=join, post=post))
    naive = d[d.member].post.mean() - d[~d.member].post.mean()
    ps = LogisticRegression().fit(d[["freq", "tenure"]], d.member).predict_proba(d[["freq", "tenure"]])[:, 1]
    d["ps"] = ps
    m, c = d[d.member], d[~d.member]
    nn = NearestNeighbors(n_neighbors=1).fit(c[["ps"]])
    dist, idx = nn.kneighbors(m[["ps"]])
    ok = dist[:, 0] < .01                            # caliper
    att = (m.post.values[ok] - c.post.values[idx[ok, 0]]).mean()
    did = (m.post - m.freq * 1.05).mean() - (c.post - c.freq * 1.05).mean()
    return dict(members=int(d.member.sum()), true_effect=0.6, naive_gap=round(naive, 2),
                matched_estimate=round(float(att), 2), matched_pairs=int(ok.sum()))


def main():
    c = customers()
    surv = survival(c)
    te, up = uplift(c)
    alloc = allocate(te)
    loy = loyalty()
    res = dict(customers=len(c), survival=surv, uplift=up, channel_allocation=alloc, loyalty=loy)
    with open("results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
