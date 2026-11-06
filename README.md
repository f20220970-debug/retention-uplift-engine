# Retention & Reactivation Engine

[![tests](https://github.com/f20220970-debug/retention-uplift-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/retention-uplift-engine/actions/workflows/tests.yml)

This repo answers a growth team's core retention questions with methods that are checked against a **known ground truth**. The questions:

- When do customers churn?
- Who should a reactivation campaign actually go to?
- Through which channel?
- Does the loyalty programme really work?

> **Simulated data only:** 40,000 customers of a B2B/B2C logistics app (kirana and retail, D2C sellers, furniture, individual movers) with a randomised 50/50 reactivation campaign. The simulator stores each customer's **true** campaign effect, so every targeting rule is scored on real incremental orders, not on a proxy.

Run it: `pip install -r requirements.txt && python engine.py && pytest -q` (about 5 seconds).

## 1. When do customers churn? (Kaplan–Meier, written from scratch)

| Group | Retained at 12 weeks | Retained at 26 weeks | Median time to churn |
| --- | --- | --- | --- |
| Good first order | 75.2% | 58.7% | not reached |
| **Bad first order** (late or cancelled pickup) | **28.3%** | **7.3%** | **6.6 weeks** |
| Kirana / retail | 83.6% | 71.5% | not reached |
| Individual mover | 37.2% | 12.3% | 8.3 weeks |

**Takeaway:** the first order decides retention, so a bad first order should trigger a service-recovery flow within 24 hours. Individual movers are one-off by nature. Measure that segment on referrals, not on repeat orders.

## 2. Who should get the reactivation offer? (uplift modelling)

Targeting the customers *most likely to churn* is the usual default, and it's the wrong one. It spends money on customers who are already gone and on "sleeping dogs" whom a message pushes away. A **T-learner** (one gradient-boosting model per arm, uplift = difference) targets the customers whose behaviour the offer actually **changes**.

| Contact the top 30% by… | True incremental orders per 1,000 contacted | Sleeping dogs among those contacted | Qini area |
| --- | --- | --- | --- |
| **Predicted uplift (T-learner)** | **112.6** | **6.7%** | **80.9** |
| Churn risk | 69.0 | 16.8% | 48.3 |
| Random | 62.7 | n/a | −1.3 |

The uplift model delivers **63% more incremental orders** than churn-risk targeting from the same contact budget. Its predictions correlate 0.53 with the true uplift.

## 3. Which channel? (budget allocation)

Each customer gets at most one channel (push, WhatsApp, SMS or call), ranked by expected profit per rupee. That profit counts the cost per contact plus the discount cost, including discounts redeemed by people who would have ordered anyway.

- **Budget:** ₹40,000. The allocation contacts 3,406 customers, mostly on WhatsApp. Calls are never worth ₹25 at these uplift sizes.
- **Result: 452 true incremental orders, against 241 for the same budget spent on random WhatsApp blasts (+88%).**

## 4. Does the loyalty programme work? (propensity-score matching)

Heavy users join loyalty programmes more often, so comparing members with non-members flatters the programme.

| Estimate of the effect on orders per month | Value |
| --- | --- |
| Naive member vs non-member gap | +2.40 |
| **Propensity-matched estimate** (1:1, caliper 0.01, 11,858 pairs) | **+0.71** |
| True effect (from the simulator) | +0.60 |

**Takeaway:** the naive read overstates the programme's value **4×**. Reward budgets should be sized on the matched number, and the next version of the programme should be launched with a randomised holdout.

## Files

```
engine.py      simulator + four analyses (true uplift is stored per customer)
results.json   headline numbers
tests/         4 checks: Kaplan-Meier correctness, uplift beats churn-risk and random, allocation beats random spend, matching removes bias
```

Companion repos: [porter-category-growth-lab](https://github.com/f20220970-debug/porter-category-growth-lab) and [marketplace-pricing-experiments](https://github.com/f20220970-debug/marketplace-pricing-experiments).
