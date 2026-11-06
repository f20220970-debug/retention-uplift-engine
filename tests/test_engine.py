import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine as E

c = E.customers()


def test_km_matches_simple_case():
    curve = E.km(np.array([1, 2, 3, 4]), np.array([True, True, False, True]))
    assert np.allclose(curve.S.values, [1, .75, .5, 0])   # censored at t=3 drops from the risk set only


def test_uplift_beats_churn_risk_and_random():
    _, u = E.uplift(c)
    inc = u["true_incremental_orders_per_1000_at_30pct"]
    assert inc["uplift_model"] > inc["churn_risk"] > 0 and inc["uplift_model"] > inc["random"]


def test_allocation_beats_random_spend():
    te, _ = E.uplift(c)
    a = E.allocate(te)
    assert a["true_incremental_orders"] > a["random_whatsapp_true_incremental"]


def test_matching_removes_selection_bias():
    l = E.loyalty()
    assert abs(l["matched_estimate"] - l["true_effect"]) < abs(l["naive_gap"] - l["true_effect"]) / 3
