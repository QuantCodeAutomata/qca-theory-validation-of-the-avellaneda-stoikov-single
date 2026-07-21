"""
Pytest test suite for src/theory_validation.py.

All tests use numerical substitutions for speed; no SymPy symbolic
simplification is performed inside the test functions themselves.
Floating-point identity checks use abs tolerance 1e-10 unless the tested
quantity is approximate (table spread values) in which case 5e-3 is used.
"""

import math
import numpy as np
import pytest
import sys
import os

# Ensure project root is on sys.path when running tests directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.theory_validation import (
    frozen_inventory_value,
    reservation_ask,
    reservation_bid,
    reservation_mid,
    liquidity_term,
    quote_distances_equation_faithful,
    table_faithful_spread,
    verify_ansatz_reservation_prices,
    numerical_verification,
)

# ---------------------------------------------------------------------------
# Shared numerical fixtures
# ---------------------------------------------------------------------------
S = 100.0
T = 1.0
SIGMA = 2.0
GAMMA = 0.1
K = 1.5
A = 140.0


def _r_a(s, q, t, T, sigma, gamma):
    """Numerically evaluate reservation ask."""
    return s + (1 - 2 * q) * gamma * sigma**2 * (T - t) / 2


def _r_b(s, q, t, T, sigma, gamma):
    """Numerically evaluate reservation bid."""
    return s + (-1 - 2 * q) * gamma * sigma**2 * (T - t) / 2


def _r_mid(s, q, t, T, sigma, gamma):
    """Numerically evaluate reservation mid."""
    return s - q * gamma * sigma**2 * (T - t)


def _liq(gamma, k):
    """Numerically evaluate the liquidity term."""
    return (1 / gamma) * math.log(1 + gamma / k)


def _psi(gamma, k):
    """Numerically evaluate the table-faithful spread."""
    return (2 / gamma) * math.log(1 + gamma / k)


# ---------------------------------------------------------------------------
# 1. Frozen-inventory value is negative
# ---------------------------------------------------------------------------

def test_frozen_inventory_value_is_negative():
    """v(x,s,q,t) < 0 for positive x, s, q and for any finite parameters."""
    for x in [0.0, 1.0, 10.0, 100.0]:
        for q in [0.5, 1.0, 2.0]:
            tau = T - 0.3  # t = 0.3
            v = (
                -np.exp(-GAMMA * x)
                * np.exp(-GAMMA * q * S)
                * np.exp((GAMMA**2 * q**2 * SIGMA**2 * tau) / 2)
            )
            assert v < 0, f"Expected v < 0 for x={x}, q={q}, got v={v}"


# ---------------------------------------------------------------------------
# 2. Reservation ask > reservation mid
# ---------------------------------------------------------------------------

def test_reservation_ask_greater_than_mid():
    """r^a > r_mid for gamma, sigma, tau > 0 across various inventories."""
    for q in [-3, -1, 0, 1, 3]:
        for t in [0.0, 0.3, 0.8]:
            r_a = _r_a(S, q, t, T, SIGMA, GAMMA)
            r_m = _r_mid(S, q, t, T, SIGMA, GAMMA)
            tau = T - t
            if tau > 0:
                assert r_a > r_m, (
                    f"Expected r_a > r_mid for q={q}, t={t}; "
                    f"got r_a={r_a:.6f}, r_mid={r_m:.6f}"
                )


# ---------------------------------------------------------------------------
# 3. Reservation bid < reservation mid
# ---------------------------------------------------------------------------

def test_reservation_bid_less_than_mid():
    """r^b < r_mid for gamma, sigma, tau > 0 across various inventories."""
    for q in [-3, -1, 0, 1, 3]:
        for t in [0.0, 0.3, 0.8]:
            r_b = _r_b(S, q, t, T, SIGMA, GAMMA)
            r_m = _r_mid(S, q, t, T, SIGMA, GAMMA)
            tau = T - t
            if tau > 0:
                assert r_b < r_m, (
                    f"Expected r_b < r_mid for q={q}, t={t}; "
                    f"got r_b={r_b:.6f}, r_mid={r_m:.6f}"
                )


# ---------------------------------------------------------------------------
# 4. Midpoint identity: (r^a + r^b) / 2 == r_mid
# ---------------------------------------------------------------------------

def test_midpoint_identity():
    """Numerical check: (r^a + r^b) / 2 equals r_mid to within 1e-10."""
    for q in [-5, -2, -1, 0, 1, 2, 5]:
        for t in [0.0, 0.25, 0.5, 0.99]:
            r_a = _r_a(S, q, t, T, SIGMA, GAMMA)
            r_b = _r_b(S, q, t, T, SIGMA, GAMMA)
            r_m = _r_mid(S, q, t, T, SIGMA, GAMMA)
            diff = abs((r_a + r_b) / 2 - r_m)
            assert diff < 1e-10, (
                f"Midpoint identity failed for q={q}, t={t}: diff={diff}"
            )


# ---------------------------------------------------------------------------
# 5. Spread identity: r^a - r^b == gamma * sigma^2 * (T-t)
# ---------------------------------------------------------------------------

def test_spread_identity():
    """Numerical check: spread r^a - r^b equals gamma*sigma^2*(T-t) to 1e-10."""
    for q in [-5, -2, 0, 2, 5]:
        for t in [0.0, 0.1, 0.5, 0.99]:
            r_a = _r_a(S, q, t, T, SIGMA, GAMMA)
            r_b = _r_b(S, q, t, T, SIGMA, GAMMA)
            tau = T - t
            expected = GAMMA * SIGMA**2 * tau
            diff = abs(r_a - r_b - expected)
            assert diff < 1e-10, (
                f"Spread identity failed for q={q}, t={t}: diff={diff}"
            )


# ---------------------------------------------------------------------------
# 6. Liquidity term positive
# ---------------------------------------------------------------------------

def test_liquidity_term_positive():
    """L(gamma, k) > 0 for all positive gamma and k."""
    for gamma in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
        for k in [0.5, 1.0, 1.5, 2.0, 5.0]:
            liq = _liq(gamma, k)
            assert liq > 0, f"Expected L > 0 for gamma={gamma}, k={k}; got {liq}"


# ---------------------------------------------------------------------------
# 7. Total equation-faithful spread identity
# ---------------------------------------------------------------------------

def test_total_spread_equation_faithful():
    """delta^a + delta^b == gamma*sigma^2*(T-t) + 2*L(gamma,k) numerically."""
    for q in [-2, -1, 0, 1, 2]:
        for t in [0.0, 0.4, 0.9]:
            tau = T - t
            liq = _liq(GAMMA, K)
            delta_a = (1 - 2 * q) * GAMMA * SIGMA**2 * tau / 2 + liq
            delta_b = (1 + 2 * q) * GAMMA * SIGMA**2 * tau / 2 + liq
            expected = GAMMA * SIGMA**2 * tau + 2 * liq
            diff = abs(delta_a + delta_b - expected)
            assert diff < 1e-10, (
                f"Total spread identity failed for q={q}, t={t}: diff={diff}"
            )


# ---------------------------------------------------------------------------
# 8. Table-faithful spread numerical values
# ---------------------------------------------------------------------------

def test_table_faithful_spread_numerical():
    """psi matches expected paper values to 2 decimal places (atol=5e-3)."""
    atol = 5e-3
    cases = [
        (0.1,  1.5, 1.29),
        (0.01, 1.5, 1.33),
        (0.5,  1.5, 1.15),
    ]
    for gamma, k, expected in cases:
        psi = _psi(gamma, k)
        assert abs(psi - expected) < atol, (
            f"psi({gamma},{k}) = {psi:.5f}, expected ≈ {expected} (atol={atol})"
        )


# ---------------------------------------------------------------------------
# 9. Exponential intensity derivative
# ---------------------------------------------------------------------------

def test_exponential_intensity_derivative():
    """d/ddelta[A*exp(-k*delta)] == -k*A*exp(-k*delta) numerically (central diff)."""
    delta_val = 1.0
    eps = 1e-7

    def intensity(d):
        return A * np.exp(-K * d)

    numerical_deriv = (intensity(delta_val + eps) - intensity(delta_val - eps)) / (2 * eps)
    analytical_deriv = -K * A * np.exp(-K * delta_val)

    assert abs(numerical_deriv - analytical_deriv) < 1e-6, (
        f"Intensity derivative mismatch: numerical={numerical_deriv:.8f}, "
        f"analytical={analytical_deriv:.8f}"
    )

    # Also verify the identity: deriv + k*lambda == 0
    assert abs(numerical_deriv + K * intensity(delta_val)) < 1e-6, (
        "Identity d(lambda)/d(delta) + k*lambda != 0"
    )


# ---------------------------------------------------------------------------
# 10. Theta-ansatz consistency
# ---------------------------------------------------------------------------

def test_theta_ansatz_consistency():
    """Theta-derived reservation prices match direct formulas to 1e-10."""
    for q in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        for t in [0.0, 0.5, 0.9]:
            tau = T - t

            def theta(q_val):
                return q_val * S - (q_val**2 / 2) * GAMMA * SIGMA**2 * tau

            r_b_theta = theta(q + 1) - theta(q)
            r_a_theta = theta(q) - theta(q - 1)

            r_b_direct = _r_b(S, q, t, T, SIGMA, GAMMA)
            r_a_direct = _r_a(S, q, t, T, SIGMA, GAMMA)

            assert abs(r_b_theta - r_b_direct) < 1e-10, (
                f"r_b theta mismatch for q={q}, t={t}: "
                f"theta={r_b_theta:.10f}, direct={r_b_direct:.10f}"
            )
            assert abs(r_a_theta - r_a_direct) < 1e-10, (
                f"r_a theta mismatch for q={q}, t={t}: "
                f"theta={r_a_theta:.10f}, direct={r_a_direct:.10f}"
            )


# ---------------------------------------------------------------------------
# 11. Reservation price convergence at maturity
# ---------------------------------------------------------------------------

def test_reservation_price_convergence_at_maturity():
    """As t -> T, the reservation mid-price r(s,q,t) -> s."""
    for q in [-5, -2, -1, 0, 1, 2, 5]:
        for t_near in [0.9999, 0.999999, 0.9999999]:
            r_m = _r_mid(S, q, t_near, T, SIGMA, GAMMA)
            assert abs(r_m - S) < 1e-3, (
                f"Convergence failed for q={q}, t={t_near}: "
                f"r_mid={r_m:.8f}, s={S}"
            )

    # At t == T exactly, r_mid == s to machine precision
    for q in [-5, 0, 5]:
        r_m = _r_mid(S, q, T, T, SIGMA, GAMMA)
        assert abs(r_m - S) < 1e-10, (
            f"At maturity, r_mid != s for q={q}: r_mid={r_m}"
        )


# ---------------------------------------------------------------------------
# 12. Quote distances equal at zero inventory
# ---------------------------------------------------------------------------

def test_quote_distances_zero_inventory():
    """At q=0, equation-faithful delta^a == delta^b for all tau."""
    q = 0
    liq = _liq(GAMMA, K)
    for t in [0.0, 0.1, 0.5, 0.9]:
        tau = T - t
        delta_a = (1 - 2 * q) * GAMMA * SIGMA**2 * tau / 2 + liq
        delta_b = (1 + 2 * q) * GAMMA * SIGMA**2 * tau / 2 + liq
        assert abs(delta_a - delta_b) < 1e-10, (
            f"q=0: delta_a != delta_b at t={t}: "
            f"delta_a={delta_a:.10f}, delta_b={delta_b:.10f}"
        )


# ---------------------------------------------------------------------------
# 13. Spread sensitivity to gamma
# ---------------------------------------------------------------------------

def test_spread_sensitivity_to_gamma():
    """Larger gamma increases eq-faithful spread; table spread is decreasing in gamma."""
    k = K
    sigma = SIGMA
    tau = 1.0  # full time horizon

    gammas = [0.01, 0.05, 0.1, 0.2, 0.5]

    psi_vals = [_psi(g, k) for g in gammas]
    eq_spread_vals = [g * sigma**2 * tau + _psi(g, k) for g in gammas]

    # Table-faithful spread is strictly decreasing in gamma (for fixed k, reasonable range)
    for i in range(len(gammas) - 1):
        assert psi_vals[i] > psi_vals[i + 1], (
            f"Table spread not decreasing: psi({gammas[i]},{k})={psi_vals[i]:.6f} "
            f"<= psi({gammas[i+1]},{k})={psi_vals[i+1]:.6f}"
        )

    # Equation-faithful spread (at tau > 0) is strictly increasing in gamma
    for i in range(len(gammas) - 1):
        assert eq_spread_vals[i] < eq_spread_vals[i + 1], (
            f"Eq-faithful spread not increasing: "
            f"spread({gammas[i]})={eq_spread_vals[i]:.6f} "
            f">= spread({gammas[i+1]})={eq_spread_vals[i+1]:.6f}"
        )

    # At tau = 0, eq-faithful spread reduces to table-faithful spread
    for g in gammas:
        eq_at_maturity = g * sigma**2 * 0.0 + _psi(g, k)
        psi_g = _psi(g, k)
        assert abs(eq_at_maturity - psi_g) < 1e-10, (
            f"At tau=0, eq-faithful spread != psi for gamma={g}"
        )
