"""Pytest tests for the Avellaneda-Stoikov simulation and policy modules."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policies import (
    equation_faithful_spread,
    get_reservation_price,
    inventory_quotes_equation_faithful,
    inventory_quotes_table_faithful,
    symmetric_quotes_equation_faithful,
    symmetric_quotes_table_faithful,
    table_faithful_spread,
)
from src.simulation import SimulationParams, simulate_many, simulate_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_params(**overrides) -> SimulationParams:
    """Return a SimulationParams suitable for fast unit tests."""
    defaults = dict(
        S0=100.0, T=1.0, sigma=2.0, dt=0.005, A=140.0, k=1.5,
        q0=0, X0=0.0, n_paths=50, gamma=0.1, seed=42,
    )
    defaults.update(overrides)
    return SimulationParams(**defaults)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

def test_table_faithful_spread_values():
    """psi(0.1,1.5)≈1.29, psi(0.01,1.5)≈1.33, psi(0.5,1.5)≈1.15."""
    tol = 0.05
    assert abs(table_faithful_spread(0.1, 1.5) - 1.29) < tol
    assert abs(table_faithful_spread(0.01, 1.5) - 1.33) < tol
    assert abs(table_faithful_spread(0.5, 1.5) - 1.15) < tol


def test_inventory_quotes_shift_with_inventory():
    """Inventory quotes shift when q changes; both ask and bid move together."""
    params = _default_params()
    s, t = 100.0, 0.5
    pa0, pb0 = inventory_quotes_table_faithful(s, 0, t, params)
    pa1, pb1 = inventory_quotes_table_faithful(s, 1, t, params)
    pa_neg, pb_neg = inventory_quotes_table_faithful(s, -1, t, params)

    # Positive inventory shifts quotes down (market maker wants to sell less)
    assert pa1 < pa0
    assert pb1 < pb0

    # Negative inventory shifts quotes up
    assert pa_neg > pa0
    assert pb_neg > pb0


def test_symmetric_quotes_invariant_to_inventory():
    """Symmetric quotes do not depend on inventory q."""
    params = _default_params()
    s, t = 100.0, 0.3
    pa0, pb0 = symmetric_quotes_table_faithful(s, 0, t, params)
    pa5, pb5 = symmetric_quotes_table_faithful(s, 5, t, params)
    pa_neg, pb_neg = symmetric_quotes_table_faithful(s, -5, t, params)

    assert pa0 == pa5 == pa_neg
    assert pb0 == pb5 == pb_neg


def test_reservation_price_at_maturity():
    """r(t=T) equals the mid-price s regardless of inventory."""
    params = _default_params()
    for q in [-5, -1, 0, 1, 5]:
        r = get_reservation_price(100.0, q, params.T, params)
        assert abs(r - 100.0) < 1e-10, f"r≠s at T for q={q}: r={r}"


def test_equation_faithful_spread_formula():
    """delta_a + delta_b equals the full equation-faithful spread formula."""
    params = _default_params()
    for q in [0, 2, -3]:
        for t in [0.0, 0.5, 0.9]:
            pa, pb = inventory_quotes_equation_faithful(100.0, q, t, params)
            delta_a = pa - 100.0
            delta_b = 100.0 - pb
            total = delta_a + delta_b
            expected = equation_faithful_spread(100.0, q, t, params)
            assert abs(total - expected) < 1e-10, (
                f"Spread mismatch q={q} t={t}: {total} vs {expected}"
            )


def test_equation_faithful_midpoint_equals_reservation():
    """Midpoint of equation-faithful inventory quotes equals the reservation price."""
    params = _default_params()
    for q in [0, 1, -2]:
        for t in [0.0, 0.3, 0.99]:
            pa, pb = inventory_quotes_equation_faithful(100.0, q, t, params)
            mid = (pa + pb) / 2.0
            r = get_reservation_price(100.0, q, t, params)
            assert abs(mid - r) < 1e-10, (
                f"Midpoint≠r: q={q} t={t} mid={mid} r={r}"
            )


def test_symmetric_midpoint_equals_midprice():
    """Midpoint of symmetric (equation-faithful) quotes equals the mid-price s."""
    params = _default_params()
    s = 100.0
    for t in [0.0, 0.5, 1.0]:
        pa, pb = symmetric_quotes_equation_faithful(s, 0, t, params)
        mid = (pa + pb) / 2.0
        assert abs(mid - s) < 1e-10, f"Midpoint≠s at t={t}: mid={mid}"

    # Also verify for table-faithful
    pa_t, pb_t = symmetric_quotes_table_faithful(s, 0, 0.0, params)
    assert abs((pa_t + pb_t) / 2.0 - s) < 1e-10


def test_equation_faithful_spread_larger_than_table_faithful_early():
    """At t=0, equation-faithful spread > table-faithful spread for gamma=0.1."""
    params = _default_params(gamma=0.1)
    eq_spread = equation_faithful_spread(params.S0, 0, 0.0, params)
    tbl_spread = table_faithful_spread(params.gamma, params.k)
    # The extra term gamma * sigma^2 * T > 0
    assert eq_spread > tbl_spread, (
        f"eq_spread={eq_spread} should exceed tbl_spread={tbl_spread}"
    )


# ---------------------------------------------------------------------------
# simulate_path tests
# ---------------------------------------------------------------------------

def test_simulate_path_returns_correct_keys():
    """simulate_path returns a dict with all expected keys."""
    params = _default_params()
    path = simulate_path(params, inventory_quotes_table_faithful, _rng())
    expected_keys = {
        "S", "q", "X", "r", "p_ask", "p_bid",
        "delta_a", "delta_b", "lambda_a", "lambda_b",
        "terminal_profit", "terminal_inventory", "lambda_dt_violations",
    }
    assert expected_keys == set(path.keys())


def test_simulate_path_array_lengths():
    """Arrays have the correct lengths: n_steps+1 for state, n_steps for quotes."""
    params = _default_params(T=1.0, dt=0.005)
    path = simulate_path(params, inventory_quotes_table_faithful, _rng())
    n_steps = round(params.T / params.dt)  # 200
    assert len(path["S"]) == n_steps + 1
    assert len(path["q"]) == n_steps + 1
    assert len(path["X"]) == n_steps + 1
    assert len(path["r"]) == n_steps
    assert len(path["p_ask"]) == n_steps
    assert len(path["lambda_a"]) == n_steps


def test_simulate_path_terminal_profit():
    """terminal_profit == X_T + q_T * S_T."""
    params = _default_params()
    path = simulate_path(params, inventory_quotes_table_faithful, _rng())
    computed = path["X"][-1] + path["q"][-1] * path["S"][-1]
    assert abs(path["terminal_profit"] - computed) < 1e-10


def test_simulate_path_inventory_conservation():
    """Inventory can only change by -1, 0, or +1 per step."""
    params = _default_params()
    path = simulate_path(params, inventory_quotes_table_faithful, _rng())
    diffs = np.diff(path["q"].astype(float))
    assert np.all(np.isin(diffs, [-1, 0, 1])), (
        f"Unexpected inventory jumps: {np.unique(diffs)}"
    )


def test_simulate_path_initial_conditions():
    """Path starts from the correct initial state."""
    params = _default_params(S0=105.0, q0=2, X0=50.0)
    path = simulate_path(params, inventory_quotes_table_faithful, _rng())
    assert path["S"][0] == 105.0
    assert path["q"][0] == 2
    assert path["X"][0] == 50.0


def test_lambda_dt_violations_logged():
    """Violations are counted when a quote_fn posts delta <= 0 (large lambda)."""
    params = _default_params()

    # Force negative delta: ask price below mid, bid price above mid
    # => delta_a < 0, lambda_a = A*exp(-k*delta_a) >> 1/dt
    def aggressive_quotes(s, q, t, p):
        return s - 2.0, s + 2.0  # inverted quotes

    path = simulate_path(params, aggressive_quotes, _rng())
    assert path["lambda_dt_violations"] > 0, (
        "Expected violations with inverted (negative-spread) quotes"
    )


# ---------------------------------------------------------------------------
# simulate_many tests
# ---------------------------------------------------------------------------

def test_simulate_many_returns_correct_shape():
    """simulate_many returns n_paths terminal profits and inventories."""
    n = 30
    params = _default_params(n_paths=n)
    result = simulate_many(params, inventory_quotes_table_faithful, "test")
    assert result["terminal_profits"].shape == (n,)
    assert result["terminal_inventories"].shape == (n,)
    assert len(result["paths"]) == n


def test_simulate_many_reproducibility():
    """Two calls with the same params produce identical results."""
    params = _default_params(n_paths=20)
    r1 = simulate_many(params, inventory_quotes_table_faithful)
    r2 = simulate_many(params, inventory_quotes_table_faithful)
    np.testing.assert_array_equal(r1["terminal_profits"], r2["terminal_profits"])


def test_simulate_many_statistics_populated():
    """Summary statistics are finite floats."""
    params = _default_params(n_paths=20)
    result = simulate_many(params, inventory_quotes_table_faithful, "inv")
    for key in ("mean_profit", "std_profit", "mean_inventory", "std_inventory"):
        assert np.isfinite(result[key]), f"{key} is not finite"


def test_inventory_strategy_lower_std_than_symmetric():
    """Inventory strategy has lower std(profit) than symmetric over 500 paths.

    This is a statistical test that asserts the well-known result from
    Avellaneda & Stoikov (2008).  Using n_paths=500 for reliability; we
    require the inventory std to be strictly lower than symmetric's std.
    """
    params_inv = _default_params(gamma=0.1, n_paths=500, seed=42)
    params_sym = _default_params(gamma=0.1, n_paths=500, seed=142)

    inv_result = simulate_many(params_inv, inventory_quotes_table_faithful, "inventory")
    sym_result = simulate_many(params_sym, symmetric_quotes_table_faithful, "symmetric")

    # Allow a loose threshold: inventory std should be strictly smaller
    assert inv_result["std_profit"] < sym_result["std_profit"], (
        f"Expected inventory std ({inv_result['std_profit']:.4f}) < "
        f"symmetric std ({sym_result['std_profit']:.4f})"
    )
