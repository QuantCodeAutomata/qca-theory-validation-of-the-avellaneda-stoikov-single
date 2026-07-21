# Custom - Context7 found no library equivalent (Avellaneda-Stoikov 2008 Section 3.3 MC)

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple, Optional


@dataclass
class SimulationParams:
    """Parameters for the Avellaneda-Stoikov market-making simulation.

    Attributes
    ----------
    S0 : float
        Initial mid-price.
    T : float
        Trading horizon in years.
    sigma : float
        Mid-price diffusion volatility (annualised).
    dt : float
        Discrete time step size.
    A : float
        Order-arrival intensity scale (Poisson baseline).
    k : float
        Order-arrival intensity decay rate with respect to quote distance.
    q0 : int
        Initial inventory (shares held).
    X0 : float
        Initial cash position.
    n_paths : int
        Number of independent Monte Carlo paths.
    gamma : float
        Risk-aversion coefficient of the market maker.
    seed : int
        Base random seed for reproducibility (per-path seeds are derived
        from this value).
    """

    S0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    A: float = 140.0
    k: float = 1.5
    q0: int = 0
    X0: float = 0.0
    n_paths: int = 1000
    gamma: float = 0.1
    seed: int = 42


def simulate_path(
    params: SimulationParams,
    quote_fn: Callable[[float, int, float, SimulationParams], Tuple[float, float]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Simulate one path of the Avellaneda-Stoikov market-making model.

    The mid-price follows a symmetric random walk with step size
    ``sigma * sqrt(dt)``.  At each step the market maker posts bid and ask
    quotes returned by ``quote_fn``; order arrivals are modelled as
    independent Bernoulli events with probability ``lambda * dt``.

    Parameters
    ----------
    params : SimulationParams
        Model and simulation parameters.
    quote_fn : Callable
        Function with signature ``(s, q, t, params) -> (p_ask, p_bid)``
        that returns the ask and bid quote *prices* given the current
        mid-price *s*, inventory *q*, time *t*, and parameters.
    rng : np.random.Generator
        NumPy random generator used for all stochastic draws, enabling
        per-path reproducibility.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``'S'``         – mid-price path, shape ``(n_steps+1,)``
        ``'q'``         – inventory path, shape ``(n_steps+1,)`` (int)
        ``'X'``         – cash path, shape ``(n_steps+1,)``
        ``'r'``         – reservation price path, shape ``(n_steps,)``;
                          filled with ``np.nan`` (recompute from S and q
                          if needed)
        ``'p_ask'``     – posted ask prices, shape ``(n_steps,)``
        ``'p_bid'``     – posted bid prices, shape ``(n_steps,)``
        ``'delta_a'``   – ask distance from mid (``p_ask - S``), shape
                          ``(n_steps,)``
        ``'delta_b'``   – bid distance from mid (``S - p_bid``), shape
                          ``(n_steps,)``
        ``'lambda_a'``  – ask execution intensity, shape ``(n_steps,)``
        ``'lambda_b'``  – bid execution intensity, shape ``(n_steps,)``
        ``'terminal_profit'``   – scalar ``X_T + q_T * S_T``
        ``'terminal_inventory'``– scalar ``q_T``
        ``'lambda_dt_violations'`` – int count of steps where
                                     ``lambda_a * dt > 1`` or
                                     ``lambda_b * dt > 1``
    """
    n_steps = round(params.T / params.dt)

    S = np.zeros(n_steps + 1)
    q = np.zeros(n_steps + 1, dtype=np.int64)
    X = np.zeros(n_steps + 1)

    S[0] = params.S0
    q[0] = params.q0
    X[0] = params.X0

    r = np.full(n_steps, np.nan)
    p_ask_arr = np.zeros(n_steps)
    p_bid_arr = np.zeros(n_steps)
    delta_a_arr = np.zeros(n_steps)
    delta_b_arr = np.zeros(n_steps)
    lambda_a_arr = np.zeros(n_steps)
    lambda_b_arr = np.zeros(n_steps)

    violations = 0
    sqrt_dt = np.sqrt(params.dt)

    for n in range(n_steps):
        t = n * params.dt
        s = S[n]
        q_curr = int(q[n])

        p_ask, p_bid = quote_fn(s, q_curr, t, params)

        delta_a = p_ask - s
        delta_b = s - p_bid

        lam_a = params.A * np.exp(-params.k * delta_a)
        lam_b = params.A * np.exp(-params.k * delta_b)

        if lam_a * params.dt > 1 or lam_b * params.dt > 1:
            violations += 1

        ask_fill = rng.random() < lam_a * params.dt
        bid_fill = rng.random() < lam_b * params.dt

        q[n + 1] = q_curr - int(ask_fill) + int(bid_fill)
        X[n + 1] = X[n] + int(ask_fill) * p_ask - int(bid_fill) * p_bid

        xi = rng.choice(np.array([-1, 1]))
        S[n + 1] = s + xi * params.sigma * sqrt_dt

        p_ask_arr[n] = p_ask
        p_bid_arr[n] = p_bid
        delta_a_arr[n] = delta_a
        delta_b_arr[n] = delta_b
        lambda_a_arr[n] = lam_a
        lambda_b_arr[n] = lam_b

    terminal_profit = float(X[n_steps] + q[n_steps] * S[n_steps])

    return {
        "S": S,
        "q": q,
        "X": X,
        "r": r,
        "p_ask": p_ask_arr,
        "p_bid": p_bid_arr,
        "delta_a": delta_a_arr,
        "delta_b": delta_b_arr,
        "lambda_a": lambda_a_arr,
        "lambda_b": lambda_b_arr,
        "terminal_profit": terminal_profit,
        "terminal_inventory": int(q[n_steps]),
        "lambda_dt_violations": violations,
    }


def simulate_many(
    params: SimulationParams,
    quote_fn: Callable[[float, int, float, SimulationParams], Tuple[float, float]],
    strategy_name: str = "",
) -> Dict[str, Any]:
    """Run ``params.n_paths`` independent Avellaneda-Stoikov simulations.

    Each path uses a deterministically derived seed
    ``[params.seed, path_idx]`` so that results are fully reproducible and
    independent across paths.

    Parameters
    ----------
    params : SimulationParams
        Model and simulation parameters (including ``n_paths`` and ``seed``).
    quote_fn : Callable
        Quote-generation function; same signature as for :func:`simulate_path`.
    strategy_name : str, optional
        Human-readable label stored in the returned dictionary.

    Returns
    -------
    dict
        ``'terminal_profits'``        – shape ``(n_paths,)``
        ``'terminal_inventories'``    – shape ``(n_paths,)``, int
        ``'mean_profit'``             – float
        ``'std_profit'``              – float
        ``'mean_inventory'``          – float
        ``'std_inventory'``           – float
        ``'mean_lambda_dt_violations'`` – float
        ``'strategy_name'``           – str
        ``'gamma'``                   – float
        ``'paths'``                   – list of path dicts (length n_paths)
    """
    terminal_profits = np.zeros(params.n_paths)
    terminal_inventories = np.zeros(params.n_paths, dtype=np.int64)
    all_paths: list = []
    total_violations = 0

    for path_idx in range(params.n_paths):
        path_rng = np.random.default_rng([params.seed, path_idx])
        path = simulate_path(params, quote_fn, path_rng)
        terminal_profits[path_idx] = path["terminal_profit"]
        terminal_inventories[path_idx] = path["terminal_inventory"]
        total_violations += path["lambda_dt_violations"]
        all_paths.append(path)

    return {
        "terminal_profits": terminal_profits,
        "terminal_inventories": terminal_inventories,
        "mean_profit": float(np.mean(terminal_profits)),
        "std_profit": float(np.std(terminal_profits)),
        "mean_inventory": float(np.mean(terminal_inventories)),
        "std_inventory": float(np.std(terminal_inventories)),
        "mean_lambda_dt_violations": total_violations / params.n_paths,
        "strategy_name": strategy_name,
        "gamma": params.gamma,
        "paths": all_paths,
    }


def compute_summary_table(
    results_by_gamma: Dict[float, Dict[str, Dict]],
    spread_fn: Callable[[float], float],
) -> pd.DataFrame:
    """Build a summary DataFrame from multi-gamma simulation results.

    Parameters
    ----------
    results_by_gamma : dict
        Nested dict ``{gamma: {'inventory': <simulate_many result>,
        'symmetric': <simulate_many result>}}``.
    spread_fn : Callable
        Function ``gamma -> spread_value`` used to populate the *spread*
        column (e.g. the table-faithful constant spread for that gamma).

    Returns
    -------
    pd.DataFrame
        Columns: ``gamma``, ``strategy``, ``spread``, ``mean_profit``,
        ``std_profit``, ``mean_inv``, ``std_inv``.
    """
    rows = []
    for gamma, strategies in results_by_gamma.items():
        spread = spread_fn(gamma)
        for strategy_label, result in strategies.items():
            rows.append(
                {
                    "gamma": gamma,
                    "strategy": strategy_label,
                    "spread": round(spread, 6),
                    "mean_profit": result["mean_profit"],
                    "std_profit": result["std_profit"],
                    "mean_inv": result["mean_inventory"],
                    "std_inv": result["std_inventory"],
                }
            )
    return pd.DataFrame(rows)
