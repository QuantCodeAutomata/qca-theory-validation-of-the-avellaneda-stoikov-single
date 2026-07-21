"""
Avellaneda-Stoikov (2008) theory validation module.

Provides symbolic (SymPy) implementations of the closed-form results from
"High-frequency trading in a limit order book" (Avellaneda & Stoikov, 2008)
together with numerical verification helpers and a derivation report generator.
"""

# All imports at top
from sympy import symbols, exp, sqrt, diff, simplify, log, Function, Eq, solve, latex, Rational, pi
from typing import Dict, Any, Tuple
import numpy as np
from itertools import product as iterproduct

# ---------------------------------------------------------------------------
# Module-level symbolic variables (reused across functions)
# ---------------------------------------------------------------------------
_s, _q, _t, _T = symbols("s q t T", real=True)
_sigma, _gamma, _k = symbols("sigma gamma k", positive=True)
_x = symbols("x", real=True)
_A, _delta = symbols("A delta", positive=True)


# ---------------------------------------------------------------------------
# Frozen-inventory value function
# ---------------------------------------------------------------------------

def frozen_inventory_value(x, s, q, t, T, sigma, gamma):
    """Return the frozen-inventory value function v(x, s, q, t).

    Closed-form solution to the HJB equation under the frozen-inventory
    assumption (q held constant), arithmetic Brownian motion for the mid-price,
    and exponential (CARA) utility.

    .. math::
        v(x,s,q,t) = -e^{-\\gamma x}\\,e^{-\\gamma q s}\\,
                      e^{\\frac{\\gamma^2 q^2 \\sigma^2 (T-t)}{2}}

    # Custom - Context7 found no library equivalent (Avellaneda-Stoikov 2008 Eq. frozen inventory)

    Parameters
    ----------
    x : sympy symbol or number
        Cash-account level.
    s : sympy symbol or number
        Current mid-price.
    q : sympy symbol or number
        Inventory (number of shares held).
    t : sympy symbol or number
        Current time.
    T : sympy symbol or number
        Terminal time (maturity).
    sigma : sympy symbol or number
        Volatility of the mid-price under arithmetic Brownian motion.
    gamma : sympy symbol or number
        Risk-aversion coefficient (CARA).

    Returns
    -------
    sympy.Expr
        The symbolic expression for v(x, s, q, t).
    """
    tau = T - t
    return -exp(-gamma * x) * exp(-gamma * q * s) * exp((gamma**2 * q**2 * sigma**2 * tau) / 2)


# ---------------------------------------------------------------------------
# Reservation prices
# ---------------------------------------------------------------------------

def reservation_ask(s, q, t, T, sigma, gamma):
    """Return the reservation ask price r^a(s, q, t).

    The price at which the market maker is indifferent between executing a sell
    (ask) order or not, given inventory q.

    .. math::
        r^a = s + \\frac{(1 - 2q)\\,\\gamma\\,\\sigma^2\\,(T-t)}{2}

    Parameters
    ----------
    s : sympy symbol or number
        Current mid-price.
    q : sympy symbol or number
        Current inventory.
    t : sympy symbol or number
        Current time.
    T : sympy symbol or number
        Terminal time.
    sigma : sympy symbol or number
        Mid-price volatility.
    gamma : sympy symbol or number
        Risk-aversion coefficient.

    Returns
    -------
    sympy.Expr
        Symbolic reservation ask price.
    """
    tau = T - t
    return s + (1 - 2 * q) * gamma * sigma**2 * tau / 2


def reservation_bid(s, q, t, T, sigma, gamma):
    """Return the reservation bid price r^b(s, q, t).

    The price at which the market maker is indifferent between executing a buy
    (bid) order or not, given inventory q.

    .. math::
        r^b = s + \\frac{(-1 - 2q)\\,\\gamma\\,\\sigma^2\\,(T-t)}{2}

    Parameters
    ----------
    s : sympy symbol or number
        Current mid-price.
    q : sympy symbol or number
        Current inventory.
    t : sympy symbol or number
        Current time.
    T : sympy symbol or number
        Terminal time.
    sigma : sympy symbol or number
        Mid-price volatility.
    gamma : sympy symbol or number
        Risk-aversion coefficient.

    Returns
    -------
    sympy.Expr
        Symbolic reservation bid price.
    """
    tau = T - t
    return s + (-1 - 2 * q) * gamma * sigma**2 * tau / 2


def reservation_mid(s, q, t, T, sigma, gamma):
    """Return the reservation mid-price r(s, q, t).

    Arithmetic mean of the reservation ask and bid, equal to the mid-price
    adjusted for inventory risk.

    .. math::
        r = s - q\\,\\gamma\\,\\sigma^2\\,(T-t)

    Parameters
    ----------
    s : sympy symbol or number
        Current mid-price.
    q : sympy symbol or number
        Current inventory.
    t : sympy symbol or number
        Current time.
    T : sympy symbol or number
        Terminal time.
    sigma : sympy symbol or number
        Mid-price volatility.
    gamma : sympy symbol or number
        Risk-aversion coefficient.

    Returns
    -------
    sympy.Expr
        Symbolic reservation mid-price.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------

def verify_midpoint_identity(s, q, t, T, sigma, gamma) -> bool:
    """Verify that (r^a + r^b) / 2 == r symbolically.

    Checks the algebraic identity

    .. math::
        \\frac{r^a + r^b}{2} = r

    by simplifying the difference to zero with SymPy.

    Parameters
    ----------
    s, q, t, T, sigma, gamma : sympy symbols
        Symbolic variables.

    Returns
    -------
    bool
        True if the identity holds exactly (simplifies to 0).
    """
    r_a = reservation_ask(s, q, t, T, sigma, gamma)
    r_b = reservation_bid(s, q, t, T, sigma, gamma)
    r_mid = reservation_mid(s, q, t, T, sigma, gamma)
    return simplify((r_a + r_b) / 2 - r_mid) == 0


def verify_spread_identity(s, q, t, T, sigma, gamma) -> bool:
    """Verify that r^a - r^b == gamma * sigma^2 * (T-t) symbolically.

    Confirms that the total bid-ask spread under the Avellaneda-Stoikov model
    is inventory-independent and equals

    .. math::
        r^a - r^b = \\gamma\\,\\sigma^2\\,(T-t)

    Parameters
    ----------
    s, q, t, T, sigma, gamma : sympy symbols
        Symbolic variables.

    Returns
    -------
    bool
        True if the identity holds exactly.
    """
    r_a = reservation_ask(s, q, t, T, sigma, gamma)
    r_b = reservation_bid(s, q, t, T, sigma, gamma)
    spread = gamma * sigma**2 * (T - t)
    return simplify(r_a - r_b - spread) == 0


# ---------------------------------------------------------------------------
# Liquidity term and quote distances
# ---------------------------------------------------------------------------

def liquidity_term(gamma, k):
    """Return the liquidity adjustment term L(gamma, k).

    Derived from the optimal control problem with exponential order-arrival
    intensity lambda = A * exp(-k * delta):

    .. math::
        L(\\gamma, k) = \\frac{1}{\\gamma}\\ln\\!\\left(1 + \\frac{\\gamma}{k}\\right)

    # Liquidity adjustment term under exponential intensity lambda=A*exp(-k*delta)

    Parameters
    ----------
    gamma : sympy symbol or number
        Risk-aversion coefficient.
    k : sympy symbol or number
        Sensitivity of order-arrival intensity to quote distance.

    Returns
    -------
    sympy.Expr
        Symbolic liquidity term.
    """
    return (1 / gamma) * log(1 + gamma / k)


def quote_distances_equation_faithful(s, q, t, T, sigma, gamma, k) -> Dict[str, Any]:
    """Return the equation-faithful optimal quote distances delta^a and delta^b.

    These are the distances from the reservation prices at which the market
    maker posts quotes, derived directly from the first-order optimality
    conditions (Avellaneda-Stoikov 2008, Eq. after (13)):

    .. math::
        \\delta^a = \\frac{(1 - 2q)\\,\\gamma\\,\\sigma^2\\,(T-t)}{2}
                   + \\frac{1}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

        \\delta^b = \\frac{(1 + 2q)\\,\\gamma\\,\\sigma^2\\,(T-t)}{2}
                   + \\frac{1}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

    Parameters
    ----------
    s, q, t, T, sigma, gamma, k : sympy symbols or numbers
        Model parameters.

    Returns
    -------
    dict
        Keys ``'delta_a'`` and ``'delta_b'`` with their symbolic expressions.
    """
    tau = T - t
    liq = liquidity_term(gamma, k)
    delta_a = (1 - 2 * q) * gamma * sigma**2 * tau / 2 + liq
    delta_b = (1 + 2 * q) * gamma * sigma**2 * tau / 2 + liq
    return {"delta_a": delta_a, "delta_b": delta_b}


def verify_total_spread_equation_faithful(s, q, t, T, sigma, gamma, k) -> bool:
    """Verify delta^a + delta^b == gamma*sigma^2*(T-t) + 2*L(gamma,k) symbolically.

    Confirms the additive decomposition of the total equation-faithful spread:

    .. math::
        \\delta^a + \\delta^b
        = \\gamma\\,\\sigma^2\\,(T-t)
          + \\frac{2}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

    Parameters
    ----------
    s, q, t, T, sigma, gamma, k : sympy symbols or numbers
        Model parameters.

    Returns
    -------
    bool
        True if the identity simplifies to zero.
    """
    distances = quote_distances_equation_faithful(s, q, t, T, sigma, gamma, k)
    total = distances["delta_a"] + distances["delta_b"]
    expected = gamma * sigma**2 * (T - t) + (2 / gamma) * log(1 + gamma / k)
    return simplify(total - expected) == 0


def table_faithful_spread(gamma, k):
    """Return the table-faithful total spread psi(gamma, k).

    This is the spread reported in the numerical tables of Avellaneda-Stoikov
    (2008).  It retains only the liquidity component of the total spread (i.e.
    it corresponds to the q=0, zero-inventory case for the mid-price component):

    .. math::
        \\psi(\\gamma, k) = \\frac{2}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

    Parameters
    ----------
    gamma : sympy symbol or number
        Risk-aversion coefficient.
    k : sympy symbol or number
        Order-arrival intensity sensitivity.

    Returns
    -------
    sympy.Expr
        Symbolic table-faithful spread.
    """
    return (2 / gamma) * log(1 + gamma / k)


# ---------------------------------------------------------------------------
# Exponential intensity derivative check
# ---------------------------------------------------------------------------

def verify_exponential_intensity_derivative(A, k, delta) -> bool:
    """Verify that d/d(delta) [A * exp(-k*delta)] + k*A*exp(-k*delta) == 0 symbolically.

    Confirms the derivative identity for the exponential order-arrival
    intensity lambda(delta) = A * exp(-k * delta):

    .. math::
        \\frac{\\partial \\lambda}{\\partial \\delta} + k\\,\\lambda = 0

    Parameters
    ----------
    A : sympy symbol or number
        Arrival-rate scale parameter.
    k : sympy symbol or number
        Sensitivity of arrival rate to quote distance.
    delta : sympy symbol or number
        Quote distance (half-spread).

    Returns
    -------
    bool
        True if the identity simplifies to zero.
    """
    intensity = A * exp(-k * delta)
    derivative = diff(intensity, delta)
    return simplify(derivative + k * A * exp(-k * delta)) == 0


# ---------------------------------------------------------------------------
# Ansatz consistency verification
# ---------------------------------------------------------------------------

def verify_ansatz_reservation_prices(s, q, t, T, sigma, gamma) -> dict:
    """Demonstrate that the theta-ansatz reproduces reservation prices symbolically.

    Under the ansatz u(x, s, q, t) = -exp(-gamma*x) * exp(-gamma*theta(s,q,t))
    the optimal reservation prices satisfy

    .. math::
        r^b = \\theta(s, q+1, t) - \\theta(s, q, t), \\qquad
        r^a = \\theta(s, q, t) - \\theta(s, q-1, t)

    Using the approximate closed-form

    .. math::
        \\theta_{\\text{approx}}(s, q, t)
        = q s - \\frac{q^2}{2}\\,\\gamma\\,\\sigma^2\\,(T-t)

    the bid and ask reservation prices are recovered exactly.

    Parameters
    ----------
    s, q, t, T, sigma, gamma : sympy symbols
        Symbolic variables.

    Returns
    -------
    dict
        Keys:
        - ``'r_b_from_theta'``: bid derived via theta difference.
        - ``'r_b_direct'``: bid from :func:`reservation_bid`.
        - ``'r_a_from_theta'``: ask derived via theta difference.
        - ``'r_a_direct'``: ask from :func:`reservation_ask`.
        - ``'r_b_consistent'``: True if both bid expressions agree.
        - ``'r_a_consistent'``: True if both ask expressions agree.
    """
    tau = T - t

    def theta_approx(q_val):
        """Approximate theta: theta(s,q,t) = q*s - (q^2/2)*gamma*sigma^2*(T-t)."""
        return q_val * s - (q_val**2 / 2) * gamma * sigma**2 * tau

    r_b_from_theta = simplify(theta_approx(q + 1) - theta_approx(q))
    r_a_from_theta = simplify(theta_approx(q) - theta_approx(q - 1))

    r_b_direct = reservation_bid(s, q, t, T, sigma, gamma)
    r_a_direct = reservation_ask(s, q, t, T, sigma, gamma)

    r_b_consistent = simplify(r_b_from_theta - r_b_direct) == 0
    r_a_consistent = simplify(r_a_from_theta - r_a_direct) == 0

    return {
        "r_b_from_theta": r_b_from_theta,
        "r_b_direct": r_b_direct,
        "r_a_from_theta": r_a_from_theta,
        "r_a_direct": r_a_direct,
        "r_b_consistent": r_b_consistent,
        "r_a_consistent": r_a_consistent,
    }


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------

def numerical_verification(params: dict) -> dict:
    """Numerically verify all key Avellaneda-Stoikov formulas at given parameters.

    For every combination of q in ``params['q_values']`` and t in
    ``params['t_values']``, evaluates the frozen-inventory value, reservation
    prices, spread identities, liquidity term and quote distances, and checks
    that all theoretical identities hold to machine precision.

    Parameters
    ----------
    params : dict
        Required keys:

        ``'s'``       – mid-price (float).
        ``'T'``       – terminal time (float).
        ``'sigma'``   – mid-price volatility (float).
        ``'gamma'``   – risk-aversion coefficient (float).
        ``'k'``       – intensity sensitivity (float).
        ``'A'``       – intensity scale (float).
        ``'q_values'``– list of inventory levels (list[float]).
        ``'t_values'``– list of current times (list[float]).

    Returns
    -------
    dict
        Nested dict keyed by ``(q, t)`` tuples.  Each entry contains:

        - ``'frozen_value'`` : float
        - ``'r_a'``, ``'r_b'``, ``'r_mid'`` : float
        - ``'midpoint_ok'`` : bool – |(r_a+r_b)/2 - r_mid| < 1e-12
        - ``'spread_ok'``   : bool – |r_a - r_b - gamma*sigma^2*(T-t)| < 1e-12
        - ``'liquidity_term'`` : float
        - ``'delta_a'``, ``'delta_b'`` : float
        - ``'total_spread_ok'`` : bool – total spread identity holds
    """
    s_val = params["s"]
    T_val = params["T"]
    sigma_val = params["sigma"]
    gamma_val = params["gamma"]
    k_val = params["k"]
    A_val = params["A"]
    q_values = params["q_values"]
    t_values = params["t_values"]

    results: dict = {}

    for q_val, t_val in iterproduct(q_values, t_values):
        tau = T_val - t_val

        # Frozen inventory value (use x = 0 as cash normalisation)
        frozen_val = (
            -np.exp(-gamma_val * 0)
            * np.exp(-gamma_val * q_val * s_val)
            * np.exp((gamma_val**2 * q_val**2 * sigma_val**2 * tau) / 2)
        )

        # Reservation prices
        r_a_val = s_val + (1 - 2 * q_val) * gamma_val * sigma_val**2 * tau / 2
        r_b_val = s_val + (-1 - 2 * q_val) * gamma_val * sigma_val**2 * tau / 2
        r_mid_val = s_val - q_val * gamma_val * sigma_val**2 * tau

        # Identity checks
        midpoint_ok = abs((r_a_val + r_b_val) / 2 - r_mid_val) < 1e-12
        spread_ok = abs(r_a_val - r_b_val - gamma_val * sigma_val**2 * tau) < 1e-12

        # Liquidity term
        liq_val = (1 / gamma_val) * np.log(1 + gamma_val / k_val)

        # Quote distances (equation-faithful)
        delta_a_val = (1 - 2 * q_val) * gamma_val * sigma_val**2 * tau / 2 + liq_val
        delta_b_val = (1 + 2 * q_val) * gamma_val * sigma_val**2 * tau / 2 + liq_val

        expected_total = gamma_val * sigma_val**2 * tau + (2 / gamma_val) * np.log(1 + gamma_val / k_val)
        total_spread_ok = abs(delta_a_val + delta_b_val - expected_total) < 1e-12

        results[(q_val, t_val)] = {
            "frozen_value": frozen_val,
            "r_a": r_a_val,
            "r_b": r_b_val,
            "r_mid": r_mid_val,
            "midpoint_ok": midpoint_ok,
            "spread_ok": spread_ok,
            "liquidity_term": liq_val,
            "delta_a": delta_a_val,
            "delta_b": delta_b_val,
            "total_spread_ok": total_spread_ok,
        }

    return results


# ---------------------------------------------------------------------------
# Derivation report
# ---------------------------------------------------------------------------

def generate_derivation_report() -> str:
    """Generate a comprehensive derivation report for the Avellaneda-Stoikov model.

    Covers:

    1. Frozen-inventory value function (HJB solution under CARA utility and
       arithmetic Brownian motion).
    2. Reservation ask and bid prices with the theta-ansatz derivation.
    3. Midpoint and spread identities (symbolic verification).
    4. Liquidity adjustment term from the exponential intensity model.
    5. Equation-faithful vs. table-faithful spread distinction.
    6. Numerical verification across default parameters.

    Returns
    -------
    str
        Multi-section plain-text / LaTeX-annotated report.
    """
    lines = []
    sep = "=" * 72

    def section(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append(sep)
    lines.append("  AVELLANEDA-STOIKOV (2008) THEORY VALIDATION REPORT")
    lines.append("  Experiment 1 – Symbolic & Numerical Derivation")
    lines.append(sep)

    # ------------------------------------------------------------------
    # 1. Frozen-inventory value function
    # ------------------------------------------------------------------
    section("1. FROZEN-INVENTORY VALUE FUNCTION")
    lines.append(
        "Under arithmetic Brownian motion  dS = sigma dW  and CARA utility\n"
        "  U(W) = -exp(-gamma * W),\n"
        "the frozen-inventory (q fixed) HJB equation admits the closed form:\n"
    )
    lines.append("  v(x,s,q,t) = -exp(-gamma*x) * exp(-gamma*q*s)")
    lines.append("               * exp( gamma^2 * q^2 * sigma^2 * (T-t) / 2 )\n")
    lines.append("LaTeX:")
    s_sym, q_sym, t_sym, T_sym = _s, _q, _t, _T
    sigma_sym, gamma_sym, k_sym = _sigma, _gamma, _k
    v_expr = frozen_inventory_value(_x, s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    lines.append(f"  v = {latex(v_expr)}\n")

    # ------------------------------------------------------------------
    # 2. Reservation prices
    # ------------------------------------------------------------------
    section("2. RESERVATION PRICES")
    lines.append("The optimal reservation prices satisfy:\n")
    r_a_expr = reservation_ask(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    r_b_expr = reservation_bid(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    r_mid_expr = reservation_mid(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    lines.append(f"  r^a = s + (1 - 2q)*gamma*sigma^2*(T-t)/2")
    lines.append(f"  LaTeX: r^a = {latex(r_a_expr)}\n")
    lines.append(f"  r^b = s + (-1 - 2q)*gamma*sigma^2*(T-t)/2")
    lines.append(f"  LaTeX: r^b = {latex(r_b_expr)}\n")
    lines.append(f"  r   = s - q*gamma*sigma^2*(T-t)  (reservation mid)")
    lines.append(f"  LaTeX: r = {latex(r_mid_expr)}\n")

    # ------------------------------------------------------------------
    # 3. Theta-ansatz derivation
    # ------------------------------------------------------------------
    section("3. THETA-ANSATZ DERIVATION OF RESERVATION PRICES")
    lines.append(
        "Substituting u = -exp(-gamma*x)*exp(-gamma*theta(s,q,t)) into the HJB\n"
        "equation and using the approximate solution\n"
        "  theta(s,q,t) = q*s - (q^2/2)*gamma*sigma^2*(T-t)\n"
        "yields:\n"
    )
    ansatz = verify_ansatz_reservation_prices(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    lines.append(f"  r^b from theta: {latex(ansatz['r_b_from_theta'])}")
    lines.append(f"  r^b direct:     {latex(ansatz['r_b_direct'])}")
    lines.append(f"  Consistent:     {ansatz['r_b_consistent']}\n")
    lines.append(f"  r^a from theta: {latex(ansatz['r_a_from_theta'])}")
    lines.append(f"  r^a direct:     {latex(ansatz['r_a_direct'])}")
    lines.append(f"  Consistent:     {ansatz['r_a_consistent']}\n")

    # ------------------------------------------------------------------
    # 4. Identity verification
    # ------------------------------------------------------------------
    section("4. SYMBOLIC IDENTITY VERIFICATION")
    mid_ok = verify_midpoint_identity(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    spread_ok = verify_spread_identity(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym)
    lines.append(f"  Midpoint identity  (r^a+r^b)/2 == r :  {mid_ok}")
    lines.append(f"  Spread identity    r^a - r^b == gamma*sigma^2*(T-t) :  {spread_ok}\n")

    # ------------------------------------------------------------------
    # 5. Liquidity term & quote distances
    # ------------------------------------------------------------------
    section("5. LIQUIDITY TERM AND QUOTE DISTANCES")
    liq_expr = liquidity_term(gamma_sym, k_sym)
    distances = quote_distances_equation_faithful(s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym, k_sym)
    total_spread_ok = verify_total_spread_equation_faithful(
        s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym, k_sym
    )
    lines.append(f"  L(gamma,k) = (1/gamma)*ln(1 + gamma/k)")
    lines.append(f"  LaTeX: L = {latex(liq_expr)}\n")
    lines.append(f"  delta^a = {latex(distances['delta_a'])}\n")
    lines.append(f"  delta^b = {latex(distances['delta_b'])}\n")
    lines.append(f"  Total spread identity delta^a+delta^b == gamma*sigma^2*(T-t)+2L:  {total_spread_ok}\n")

    # ------------------------------------------------------------------
    # 6. Equation-faithful vs table-faithful spread
    # ------------------------------------------------------------------
    section("6. EQUATION-FAITHFUL vs TABLE-FAITHFUL SPREAD")
    lines.append(
        "Equation-faithful total spread (from Eq. 13 in the paper):\n"
        "  spread_eq = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)\n\n"
        "  This includes the inventory-risk component gamma*sigma^2*(T-t)\n"
        "  which vanishes at t = T.\n\n"
        "Table-faithful spread (numerical tables in the paper use only the\n"
        "  liquidity component, evaluated at q=0 / t=T):\n"
        "  psi(gamma,k) = (2/gamma)*ln(1 + gamma/k)\n"
    )
    psi_expr = table_faithful_spread(gamma_sym, k_sym)
    lines.append(f"  LaTeX: psi = {latex(psi_expr)}\n")
    lines.append("  Numerical values of psi for default k=1.5:")
    for g in [0.01, 0.1, 0.5]:
        psi_val = (2 / g) * np.log(1 + g / 1.5)
        lines.append(f"    gamma={g:5.2f}  =>  psi = {psi_val:.4f}")
    lines.append("")

    # ------------------------------------------------------------------
    # 7. Exponential intensity derivative
    # ------------------------------------------------------------------
    section("7. EXPONENTIAL INTENSITY DERIVATIVE")
    A_sym = symbols("A", positive=True)
    delta_sym = symbols("delta", positive=True)
    intensity_ok = verify_exponential_intensity_derivative(A_sym, k_sym, delta_sym)
    lines.append(
        "  Intensity model: lambda(delta) = A * exp(-k*delta)\n"
        "  Derivative check: d(lambda)/d(delta) + k*lambda == 0\n"
        f"  Result: {intensity_ok}\n"
    )

    # ------------------------------------------------------------------
    # 8. Numerical verification for default parameters
    # ------------------------------------------------------------------
    section("8. NUMERICAL VERIFICATION – DEFAULT PARAMETERS")
    default_params = {
        "s": 100.0,
        "T": 1.0,
        "sigma": 2.0,
        "gamma": 0.1,
        "k": 1.5,
        "A": 140.0,
        "q_values": [-2, -1, 0, 1, 2],
        "t_values": [0.0, 0.5, 0.9],
    }
    lines.append(
        f"  s={default_params['s']}, T={default_params['T']}, "
        f"sigma={default_params['sigma']}, gamma={default_params['gamma']}, "
        f"k={default_params['k']}, A={default_params['A']}"
    )
    lines.append(f"  q_values = {default_params['q_values']}")
    lines.append(f"  t_values = {default_params['t_values']}\n")

    num_results = numerical_verification(default_params)

    all_midpoint_ok = all(v["midpoint_ok"] for v in num_results.values())
    all_spread_ok = all(v["spread_ok"] for v in num_results.values())
    all_total_ok = all(v["total_spread_ok"] for v in num_results.values())

    lines.append(f"  All midpoint identities hold:      {all_midpoint_ok}")
    lines.append(f"  All spread identities hold:        {all_spread_ok}")
    lines.append(f"  All total-spread identities hold:  {all_total_ok}\n")

    lines.append(
        f"  {'q':>4}  {'t':>5}  {'r_a':>10}  {'r_b':>10}  {'r_mid':>10}  "
        f"{'delta_a':>10}  {'delta_b':>10}"
    )
    lines.append("  " + "-" * 65)
    for (q_val, t_val), v in sorted(num_results.items()):
        lines.append(
            f"  {q_val:>4.0f}  {t_val:>5.2f}  {v['r_a']:>10.4f}  {v['r_b']:>10.4f}  "
            f"{v['r_mid']:>10.4f}  {v['delta_a']:>10.4f}  {v['delta_b']:>10.4f}"
        )

    lines.append("")
    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)
