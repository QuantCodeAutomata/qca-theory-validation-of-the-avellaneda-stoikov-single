"""Experiment 3 – Monte Carlo replication using the equation-faithful spread.

The equation-faithful spread is derived directly from the optimal-quote
distance expressions in Avellaneda & Stoikov (2008).  Unlike the
table-faithful spread (Experiment 2), it is time-varying:

    spread_t = gamma * sigma^2 * (T - t) + (2/gamma) * log(1 + gamma/k)

At ``t = T`` both conventions agree; earlier in the horizon the
equation-faithful spread is larger, penalising executions more heavily when
residual risk is high.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policies import (
    equation_faithful_spread,
    get_reservation_price,
    inventory_quotes_equation_faithful,
    symmetric_quotes_equation_faithful,
    table_faithful_spread,
)
from src.simulation import SimulationParams, compute_summary_table, simulate_many

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
GAMMAS = [0.1, 0.01, 0.5]

COMMON = dict(S0=100.0, T=1.0, sigma=2.0, dt=0.005, A=140.0, k=1.5, q0=0, X0=0.0, n_paths=1000)


def _inv_seed(gamma: float) -> int:
    return int(gamma * 1000) + 42


def _sym_seed(gamma: float) -> int:
    return int(gamma * 1000) + 142


# ---------------------------------------------------------------------------
# Pre-run validation checks
# ---------------------------------------------------------------------------
def validate_formulas(params: SimulationParams) -> None:
    """Verify equation-faithful quote properties at step n=0 (t=0, q=0).

    Three invariants are checked:
    1. ``delta_a + delta_b == gamma*sigma^2*T + (2/gamma)*log(1+gamma/k)``
    2. Midpoint of inventory quotes equals reservation price *r*.
    3. Midpoint of symmetric quotes equals mid-price *s*.
    """
    s = params.S0
    q = 0
    t = 0.0

    pa_inv, pb_inv = inventory_quotes_equation_faithful(s, q, t, params)
    pa_sym, pb_sym = symmetric_quotes_equation_faithful(s, q, t, params)

    delta_a = pa_inv - s
    delta_b = s - pb_inv
    total_spread = delta_a + delta_b
    expected_spread = equation_faithful_spread(s, q, t, params)

    mid_inv = (pa_inv + pb_inv) / 2.0
    r = get_reservation_price(s, q, t, params)
    mid_sym = (pa_sym + pb_sym) / 2.0

    tol = 1e-10
    ok1 = abs(total_spread - expected_spread) < tol
    ok2 = abs(mid_inv - r) < tol
    ok3 = abs(mid_sym - s) < tol

    gamma = params.gamma
    print(
        f"  [gamma={gamma}] delta_a+delta_b={total_spread:.6f}  "
        f"expected={expected_spread:.6f}  {'✓' if ok1 else '✗'}"
    )
    print(
        f"  [gamma={gamma}] mid_inv={mid_inv:.6f}  r={r:.6f}  {'✓' if ok2 else '✗'}"
    )
    print(
        f"  [gamma={gamma}] mid_sym={mid_sym:.6f}  s={s:.6f}  {'✓' if ok3 else '✗'}"
    )

    if not (ok1 and ok2 and ok3):
        raise AssertionError(
            f"Equation-faithful formula validation failed for gamma={gamma}."
        )


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
def run_all() -> dict:
    """Run inventory and symmetric strategies for all gammas.

    Returns
    -------
    dict
        ``{gamma: {'inventory': simulate_many_result,
                   'symmetric': simulate_many_result}}``
    """
    print("\n=== Formula Validation ===")
    for gamma in GAMMAS:
        p = SimulationParams(**COMMON, gamma=gamma)
        validate_formulas(p)
    print()

    results: dict = {}
    for gamma in GAMMAS:
        print(f"Running gamma={gamma} ...")

        inv_params = SimulationParams(**COMMON, gamma=gamma, seed=_inv_seed(gamma))
        sym_params = SimulationParams(**COMMON, gamma=gamma, seed=_sym_seed(gamma))

        inv_result = simulate_many(
            inv_params, inventory_quotes_equation_faithful, "inventory"
        )
        sym_result = simulate_many(
            sym_params, symmetric_quotes_equation_faithful, "symmetric"
        )

        print(
            f"  inventory : mean_profit={inv_result['mean_profit']:.4f}"
            f"  std_profit={inv_result['std_profit']:.4f}"
            f"  violations={inv_result['mean_lambda_dt_violations']:.3f}"
        )
        print(
            f"  symmetric : mean_profit={sym_result['mean_profit']:.4f}"
            f"  std_profit={sym_result['std_profit']:.4f}"
            f"  violations={sym_result['mean_lambda_dt_violations']:.3f}"
        )

        results[gamma] = {"inventory": inv_result, "symmetric": sym_result}

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def build_summary(results: dict) -> pd.DataFrame:
    """Construct and print the summary DataFrame.

    Uses the time-0 spread as the representative spread value for each gamma.
    """
    params_ref = SimulationParams(**COMMON)

    def spread_at_0(gamma: float) -> float:
        p = SimulationParams(**COMMON, gamma=gamma)
        return equation_faithful_spread(p.S0, 0, 0.0, p)

    df = compute_summary_table(results, spread_at_0)
    df.insert(df.columns.get_loc("spread") + 1, "spread_type", "equation-faithful (t=0)")

    print("\n=== Summary Table (Experiment 3 – Equation-Faithful) ===")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print()
    return df


# ---------------------------------------------------------------------------
# Comparison with table-faithful (informational note)
# ---------------------------------------------------------------------------
def compare_conventions() -> None:
    """Print the spread difference between conventions at t=0 for all gammas."""
    print("=== Convention Comparison (spread at t=0) ===")
    for gamma in GAMMAS:
        p = SimulationParams(**COMMON, gamma=gamma)
        eq_spread = equation_faithful_spread(p.S0, 0, 0.0, p)
        tbl_spread = table_faithful_spread(gamma, p.k)
        diff = eq_spread - tbl_spread
        print(
            f"  gamma={gamma}: eq={eq_spread:.4f}  table={tbl_spread:.4f}"
            f"  diff={diff:.4f}  (equation-faithful is {'larger' if diff > 0 else 'smaller'})"
        )
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _t_arrays(params: SimulationParams):
    n_steps = round(params.T / params.dt)
    t_full = np.linspace(0, params.T, n_steps + 1)
    t_steps = t_full[:-1]
    return t_full, t_steps


def plot_representative_path(results: dict) -> None:
    """Plot a representative path: S, r, bid, ask for inventory strategy (gamma=0.1, path 0)."""
    gamma = 0.1
    params = SimulationParams(**COMMON, gamma=gamma, seed=_inv_seed(gamma))
    path = results[gamma]["inventory"]["paths"][0]

    t_full, t_steps = _t_arrays(params)

    S = path["S"]
    q = path["q"]
    p_ask = path["p_ask"]
    p_bid = path["p_bid"]

    r = S[:-1] - q[:-1] * params.gamma * params.sigma ** 2 * (params.T - t_steps)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_full, S, label="Mid-price $S_t$", color="black", lw=1.2)
    ax.plot(t_steps, r, label="Reservation price $r_t$", color="blue", lw=1.0, linestyle="--")
    ax.plot(t_steps, p_ask, label="Ask quote $p^a_t$", color="red", lw=0.8, alpha=0.75)
    ax.plot(t_steps, p_bid, label="Bid quote $p^b_t$", color="green", lw=0.8, alpha=0.75)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Price")
    ax.set_title(
        f"Exp 3 – Representative Path (inventory strategy, $\\gamma={gamma}$, equation-faithful spread)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_png = os.path.join(RESULTS_DIR, "exp3_representative_path.png")
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    print(f"Saved: {path_png}")


def plot_profit_histograms(results: dict) -> None:
    """Overlaid terminal-profit histograms for each gamma."""
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(14, 4), sharey=False)

    for ax, gamma in zip(axes, GAMMAS):
        inv_profits = results[gamma]["inventory"]["terminal_profits"]
        sym_profits = results[gamma]["symmetric"]["terminal_profits"]

        bins = np.linspace(
            min(inv_profits.min(), sym_profits.min()),
            max(inv_profits.max(), sym_profits.max()),
            40,
        )
        ax.hist(inv_profits, bins=bins, alpha=0.6, label="Inventory", color="steelblue", density=True)
        ax.hist(sym_profits, bins=bins, alpha=0.6, label="Symmetric", color="tomato", density=True)
        ax.set_title(f"$\\gamma = {gamma}$")
        ax.set_xlabel("Terminal profit $X_T + q_T S_T$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 3 – Terminal Profit Distributions (Equation-Faithful Spread)", y=1.01)
    fig.tight_layout()
    hist_png = os.path.join(RESULTS_DIR, "exp3_profit_histograms.png")
    fig.savefig(hist_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_png}")


def plot_time_varying_spread(results: dict) -> None:
    """Plot the equation-faithful spread over time for each gamma."""
    params_ref = SimulationParams(**COMMON)
    n_steps = round(params_ref.T / params_ref.dt)
    t_steps = np.linspace(0, params_ref.T, n_steps + 1)[:-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["steelblue", "tomato", "seagreen"]
    for gamma, col in zip(GAMMAS, colours):
        p = SimulationParams(**COMMON, gamma=gamma)
        spread_t = np.array(
            [equation_faithful_spread(p.S0, 0, t, p) for t in t_steps]
        )
        tbl = table_faithful_spread(gamma, p.k)
        ax.plot(t_steps, spread_t, label=f"$\\gamma={gamma}$ (eq.)", color=col, lw=1.5)
        ax.axhline(tbl, color=col, lw=1.0, linestyle=":", alpha=0.7,
                   label=f"$\\gamma={gamma}$ (table)")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Total spread $\\delta^a + \\delta^b$")
    ax.set_title("Exp 3 – Time-Varying Spread (equation-faithful vs table-faithful)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    spread_png = os.path.join(RESULTS_DIR, "exp3_spread_over_time.png")
    fig.savefig(spread_png, dpi=120)
    plt.close(fig)
    print(f"Saved: {spread_png}")


def plot_comparison(results: dict, df3: pd.DataFrame) -> None:
    """Side-by-side comparison of equation-faithful vs table-faithful mean profit and std."""
    # Load table-faithful summary if CSV exists, else compute inline
    csv2 = os.path.join(RESULTS_DIR, "exp2_summary.csv")
    if os.path.exists(csv2):
        df2 = pd.read_csv(csv2)
        df2["exp"] = "table-faithful"
    else:
        # Build inline from results using same seeds
        results2: dict = {}
        from src.policies import inventory_quotes_table_faithful, symmetric_quotes_table_faithful
        for gamma in GAMMAS:
            inv_p = SimulationParams(**COMMON, gamma=gamma, seed=_inv_seed(gamma))
            sym_p = SimulationParams(**COMMON, gamma=gamma, seed=_sym_seed(gamma))
            results2[gamma] = {
                "inventory": simulate_many(inv_p, inventory_quotes_table_faithful, "inventory"),
                "symmetric": simulate_many(sym_p, symmetric_quotes_table_faithful, "symmetric"),
            }
        from src.policies import table_faithful_spread as tfs
        df2 = compute_summary_table(results2, lambda g: tfs(g, 1.5))
        df2["exp"] = "table-faithful"

    df3_copy = df3.copy()
    df3_copy["exp"] = "equation-faithful"
    # Align columns
    common_cols = ["gamma", "strategy", "mean_profit", "std_profit", "exp"]
    df3_common = df3_copy[[c for c in common_cols if c in df3_copy.columns]]
    df2_common = df2[[c for c in common_cols if c in df2.columns]]
    combined = pd.concat([df2_common, df3_common], ignore_index=True)

    strategies = ["inventory", "symmetric"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, ["mean_profit", "std_profit"]):
        for strat in strategies:
            sub = combined[combined["strategy"] == strat]
            for exp_label, ls in [("table-faithful", "-"), ("equation-faithful", "--")]:
                sub_exp = sub[sub["exp"] == exp_label].sort_values("gamma")
                if sub_exp.empty:
                    continue
                ax.plot(
                    sub_exp["gamma"].astype(str),
                    sub_exp[metric],
                    marker="o",
                    linestyle=ls,
                    label=f"{strat} ({exp_label})",
                )
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Exp 3 – {metric.replace('_', ' ')} comparison")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Equation-Faithful vs Table-Faithful Convention", y=1.01)
    fig.tight_layout()
    cmp_png = os.path.join(RESULTS_DIR, "exp3_comparison.png")
    fig.savefig(cmp_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cmp_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    compare_conventions()

    results = run_all()

    df = build_summary(results)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp3_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plots
    plot_representative_path(results)
    plot_profit_histograms(results)
    plot_time_varying_spread(results)
    plot_comparison(results, df)

    print("\nExp 3 complete.")


if __name__ == "__main__":
    main()
