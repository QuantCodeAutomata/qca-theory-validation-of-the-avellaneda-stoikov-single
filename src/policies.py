"""Quote-generation strategies for the Avellaneda-Stoikov model.

Two spread conventions are implemented:

* **Table-faithful** – constant spread ``psi = (2/gamma)*log(1 + gamma/k)``,
  matching the numerical tables in the paper (Avellaneda & Stoikov 2008).
* **Equation-faithful** – time-varying spread derived directly from the
  optimal-quote equations, which includes the additional term
  ``gamma * sigma^2 * (T - t)`` that decays to zero at the terminal time.
"""

import numpy as np
from typing import Tuple


def table_faithful_spread(gamma: float, k: float) -> float:
    """Compute the constant table-faithful total spread.

    The spread is the ``gamma → 0`` limiting form of the full spread
    formula, which is the value tabulated by Avellaneda & Stoikov (2008)
    in their numerical experiments:

    .. math::

        \\psi = \\frac{2}{\\gamma} \\ln\\!\\left(1 + \\frac{\\gamma}{k}\\right)

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    k : float
        Order-arrival intensity decay rate.

    Returns
    -------
    float
        Total (bid + ask) spread in price units.

    Examples
    --------
    >>> round(table_faithful_spread(0.1, 1.5), 2)
    1.29
    >>> round(table_faithful_spread(0.01, 1.5), 2)
    1.33
    >>> round(table_faithful_spread(0.5, 1.5), 2)
    1.15
    """
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


def inventory_quotes_table_faithful(
    s: float, q: int, t: float, params
) -> Tuple[float, float]:
    """Compute inventory-adjusted quotes using the table-faithful spread.

    The market maker centres quotes on the reservation price
    ``r = s - q * gamma * sigma^2 * (T - t)`` and uses the constant
    spread ``psi = table_faithful_spread(gamma, k)``.

    .. math::

        p^a = r + \\psi/2, \\quad p^b = r - \\psi/2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory.
    t : float
        Current time (in the same units as ``params.T``).
    params : SimulationParams
        Simulation parameters (uses ``gamma``, ``sigma``, ``T``, ``k``).

    Returns
    -------
    (p_ask, p_bid) : Tuple[float, float]
        Ask and bid quote prices.
    """
    r = get_reservation_price(s, q, t, params)
    psi = table_faithful_spread(params.gamma, params.k)
    p_ask = r + psi / 2.0
    p_bid = r - psi / 2.0
    return p_ask, p_bid


def symmetric_quotes_table_faithful(
    s: float, q: int, t: float, params
) -> Tuple[float, float]:
    """Compute inventory-agnostic symmetric quotes using the table-faithful spread.

    Quotes are centred on the mid-price *s* (ignoring inventory) with
    constant spread ``psi = table_faithful_spread(gamma, k)``.

    .. math::

        p^a = s + \\psi/2, \\quad p^b = s - \\psi/2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory (unused, present for API compatibility).
    t : float
        Current time (unused, present for API compatibility).
    params : SimulationParams
        Simulation parameters (uses ``gamma``, ``k``).

    Returns
    -------
    (p_ask, p_bid) : Tuple[float, float]
        Ask and bid quote prices.
    """
    psi = table_faithful_spread(params.gamma, params.k)
    p_ask = s + psi / 2.0
    p_bid = s - psi / 2.0
    return p_ask, p_bid


def equation_faithful_spread(s: float, q: int, t: float, params) -> float:
    """Compute the time-varying total spread from the full optimal-quote equations.

    The theoretical total spread at time *t* is:

    .. math::

        \\delta^a + \\delta^b = \\gamma \\sigma^2 (T - t)
                               + \\frac{2}{\\gamma} \\ln\\!\\left(1 +
                               \\frac{\\gamma}{k}\\right)

    The first term decays linearly to zero as ``t → T``; the second term
    equals the table-faithful spread.

    Parameters
    ----------
    s : float
        Current mid-price (unused, present for API compatibility).
    q : int
        Current inventory (unused, present for API compatibility).
    t : float
        Current time.
    params : SimulationParams
        Simulation parameters (uses ``gamma``, ``sigma``, ``T``, ``k``).

    Returns
    -------
    float
        Time-varying total spread at time *t*.
    """
    time_to_go = params.T - t
    return params.gamma * params.sigma ** 2 * time_to_go + table_faithful_spread(
        params.gamma, params.k
    )


def inventory_quotes_equation_faithful(
    s: float, q: int, t: float, params
) -> Tuple[float, float]:
    """Compute inventory-adjusted quotes using the equation-faithful spread.

    Quote distances from the mid-price are derived directly from the
    optimal-quote expressions (Avellaneda & Stoikov 2008, Eq. 14):

    .. math::

        \\delta^a = \\frac{(1 - 2q)\\,\\gamma\\sigma^2(T-t)}{2}
                   + \\frac{1}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

        \\delta^b = \\frac{(1 + 2q)\\,\\gamma\\sigma^2(T-t)}{2}
                   + \\frac{1}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

    so that:

    .. math::

        p^a = s + \\delta^a, \\quad p^b = s - \\delta^b

    The midpoint ``(p^a + p^b)/2`` equals the reservation price
    ``r = s - q * gamma * sigma^2 * (T - t)`` for all ``q``.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory.
    t : float
        Current time.
    params : SimulationParams
        Simulation parameters.

    Returns
    -------
    (p_ask, p_bid) : Tuple[float, float]
        Ask and bid quote prices.
    """
    time_to_go = params.T - t
    log_term = (1.0 / params.gamma) * np.log(1.0 + params.gamma / params.k)
    half_risk = params.gamma * params.sigma ** 2 * time_to_go / 2.0

    delta_a = (1.0 - 2.0 * q) * half_risk + log_term
    delta_b = (1.0 + 2.0 * q) * half_risk + log_term

    p_ask = s + delta_a
    p_bid = s - delta_b
    return p_ask, p_bid


def symmetric_quotes_equation_faithful(
    s: float, q: int, t: float, params
) -> Tuple[float, float]:
    """Compute inventory-agnostic symmetric quotes using the equation-faithful spread.

    Quotes are centred on the mid-price *s* using the time-varying total
    spread:

    .. math::

        \\text{spread}_t = \\gamma\\sigma^2(T-t)
                          + \\frac{2}{\\gamma}\\ln\\!\\left(1+\\frac{\\gamma}{k}\\right)

    .. math::

        p^a = s + \\text{spread}_t/2, \\quad p^b = s - \\text{spread}_t/2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory (unused, present for API compatibility).
    t : float
        Current time.
    params : SimulationParams
        Simulation parameters.

    Returns
    -------
    (p_ask, p_bid) : Tuple[float, float]
        Ask and bid quote prices.
    """
    spread_t = equation_faithful_spread(s, q, t, params)
    p_ask = s + spread_t / 2.0
    p_bid = s - spread_t / 2.0
    return p_ask, p_bid


def get_reservation_price(s: float, q: int, t: float, params) -> float:
    """Compute the Avellaneda-Stoikov reservation (certainty-equivalent) price.

    .. math::

        r = s - q \\cdot \\gamma \\cdot \\sigma^2 \\cdot (T - t)

    At ``t = T`` the reservation price equals the mid-price regardless of
    inventory.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory.
    t : float
        Current time.
    params : SimulationParams
        Simulation parameters (uses ``gamma``, ``sigma``, ``T``).

    Returns
    -------
    float
        Reservation price at time *t*.
    """
    return s - q * params.gamma * params.sigma ** 2 * (params.T - t)
