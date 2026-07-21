"""Experiment 2 – Monte Carlo replication using the table-faithful spread.

The table-faithful spread ``psi = (2/gamma) * log(1 + gamma/k)`` is the
constant spread value that matches Avellaneda & Stoikov (2008) Table 1.
Both the inventory and symmetric strategies use this fixed spread, differing
only in whether they centre quotes on the reservation price *r* or the
mid-price *s*.
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
    get_reservation_price,
    inventory_quotes_table_faithful,
    symmetric_quotes_table_faithful,
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
# Spread verification
# ---------------------------------------------------------------------------
def verify_spreads() -> None:
    """Print and assert the table-faithful spread values for key gammas."""
    print("\n=== Spread Verification (table-faithful) ===")
    expected = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
    for gamma, approx in expected.items():
        psi = table_faithful_spread(gamma, 1.5)
        ok = abs(psi - approx) < 0.05
        print(f"  psi(gamma={gamma}, k=1.5) = {psi:.4f}  (expected ≈ {approx})  {'✓' if ok else '✗'}")
    print()


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
    results: dict = {}
    for gamma in GAMMAS:
        print(f"Running gamma={gamma} ...")

        inv_params = SimulationParams(**COMMON, gamma=gamma, seed=_inv_seed(gamma))
        sym_params = SimulationParams(**COMMON, gamma=gamma, seed=_sym_seed(gamma))

        inv_result = simulate_many(inv_params, inventory_quotes_table_faithful, "inventory")
        sym_result = simulate_many(sym_params, symmetric_quotes_table_faithful, "symmetric")

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
    """Construct and print the summary DataFrame."""
    df = compute_summary_table(results, lambda g: table_faithful_spread(g, 1.5))

    print("\n=== Summary Table (Experiment 2 – Table-Faithful) ===")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print()
    return df


# ---------------------------------------------------------------------------
# Statistical checks
# ---------------------------------------------------------------------------
def check_inventory_advantage(results: dict) -> None:
    """Verify inventory strategy has lower std_profit and std_final_q."""
    print("=== Inventory-vs-Symmetric Checks ===")
    for gamma in GAMMAS:
        inv = results[gamma]["inventory"]
        sym = results[gamma]["symmetric"]

        lower_std_profit = inv["std_profit"] < sym["std_profit"]
        lower_std_inv = inv["std_inventory"] < sym["std_inventory"]

        print(
            f"  gamma={gamma}:  "
            f"inv std_profit={inv['std_profit']:.4f} {'<' if lower_std_profit else '>='} "
            f"sym std_profit={sym['std_profit']:.4f}  {'✓' if lower_std_profit else '✗'}  |  "
            f"inv std_q={inv['std_inventory']:.4f} {'<' if lower_std_inv else '>='} "
            f"sym std_q={sym['std_inventory']:.4f}  {'✓' if lower_std_inv else '✗'}"
        )
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _t_arrays(params: SimulationParams):
    """Return time grids: t_full (n_steps+1), t_steps (n_steps)."""
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

    # Recompute reservation price from stored S and q
    r = S[:-1] - q[:-1] * params.gamma * params.sigma ** 2 * (params.T - t_steps)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_full, S, label="Mid-price $S_t$", color="black", lw=1.2)
    ax.plot(t_steps, r, label="Reservation price $r_t$", color="blue", lw=1.0, linestyle="--")
    ax.plot(t_steps, p_ask, label="Ask quote $p^a_t$", color="red", lw=0.8, alpha=0.75)
    ax.plot(t_steps, p_bid, label="Bid quote $p^b_t$", color="green", lw=0.8, alpha=0.75)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Price")
    ax.set_title(
        f"Exp 2 – Representative Path (inventory strategy, $\\gamma={gamma}$, table-faithful spread)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_png = os.path.join(RESULTS_DIR, "exp2_representative_path.png")
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

    fig.suptitle("Exp 2 – Terminal Profit Distributions (Table-Faithful Spread)", y=1.01)
    fig.tight_layout()
    hist_png = os.path.join(RESULTS_DIR, "exp2_profit_histograms.png")
    fig.savefig(hist_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_png}")


def plot_summary_table_figure(df: pd.DataFrame) -> None:
    """Render the summary DataFrame as a matplotlib table figure."""
    display_df = df.copy()
    for col in ["mean_profit", "std_profit", "mean_inv", "std_inv", "spread"]:
        display_df[col] = display_df[col].map("{:.4f}".format)

    fig, ax = plt.subplots(figsize=(11, 2.0 + 0.4 * len(display_df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    ax.set_title("Exp 2 – Summary Table (Table-Faithful Spread)", pad=10, fontsize=11)
    fig.tight_layout()
    tbl_png = os.path.join(RESULTS_DIR, "exp2_summary_table.png")
    fig.savefig(tbl_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tbl_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    verify_spreads()

    results = run_all()

    df = build_summary(results)

    check_inventory_advantage(results)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp2_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plots
    plot_representative_path(results)
    plot_profit_histograms(results)
    plot_summary_table_figure(df)

    print("\nExp 2 complete.")


if __name__ == "__main__":
    main()
