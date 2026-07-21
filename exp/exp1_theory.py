"""
Experiment 1 – Avellaneda-Stoikov theory validation.

Runs all symbolic and numerical verifications from
``src.theory_validation``, generates the derivation report, and
produces publication-quality figures saved under ``results/``.

Usage
-----
Run from the project root::

    python -m exp.exp1_theory
    # or
    python exp/exp1_theory.py
"""

import os
import sys

# Allow running directly as a script from any working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works in all environments
import matplotlib.pyplot as plt
from sympy import symbols

from src.theory_validation import (
    frozen_inventory_value,
    reservation_ask,
    reservation_bid,
    reservation_mid,
    verify_midpoint_identity,
    verify_spread_identity,
    liquidity_term,
    quote_distances_equation_faithful,
    verify_total_spread_equation_faithful,
    table_faithful_spread,
    verify_exponential_intensity_derivative,
    verify_ansatz_reservation_prices,
    numerical_verification,
    generate_derivation_report,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
S = 100.0
T = 1.0
SIGMA = 2.0
K = 1.5
A = 140.0
GAMMA_DEFAULT = 0.1
GAMMAS = [0.01, 0.1, 0.5]
Q_VALUES = [-2, -1, 0, 1, 2]
T_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]


def _ensure_results_dir() -> str:
    """Create the results/ directory if absent and return its path."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ---------------------------------------------------------------------------
# Symbolic verification helpers
# ---------------------------------------------------------------------------

def run_symbolic_verifications() -> dict:
    """Run all symbolic identity checks and return a summary dict.

    Returns
    -------
    dict
        Keys are short description strings; values are booleans.
    """
    s_sym, q_sym, t_sym, T_sym = symbols("s q t T", real=True)
    sigma_sym, gamma_sym, k_sym = symbols("sigma gamma k", positive=True)
    A_sym, delta_sym = symbols("A delta", positive=True)

    results = {}

    results["midpoint_identity"] = verify_midpoint_identity(
        s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym
    )
    results["spread_identity"] = verify_spread_identity(
        s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym
    )
    results["total_spread_eq_faithful"] = verify_total_spread_equation_faithful(
        s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym, k_sym
    )
    results["exponential_intensity_derivative"] = verify_exponential_intensity_derivative(
        A_sym, k_sym, delta_sym
    )

    ansatz = verify_ansatz_reservation_prices(
        s_sym, q_sym, t_sym, T_sym, sigma_sym, gamma_sym
    )
    results["ansatz_r_b_consistent"] = ansatz["r_b_consistent"]
    results["ansatz_r_a_consistent"] = ansatz["r_a_consistent"]

    return results


# ---------------------------------------------------------------------------
# Plot 1: Reservation mid-price vs time-to-maturity
# ---------------------------------------------------------------------------

def plot_reservation_price(results_dir: str) -> None:
    """Plot the reservation mid-price r vs time-to-maturity tau = T-t.

    One curve per q value in Q_VALUES, with gamma = GAMMA_DEFAULT.
    Saved to ``results/exp1_reservation_price.png``.

    Parameters
    ----------
    results_dir : str
        Directory in which to save the figure.
    """
    taus = np.linspace(0.0, T, 300)
    gamma = GAMMA_DEFAULT

    fig, ax = plt.subplots(figsize=(8, 5))

    for q in Q_VALUES:
        r_vals = S - q * gamma * SIGMA**2 * taus
        ax.plot(taus, r_vals, label=f"q = {q}")

    ax.set_xlabel("Time-to-maturity $\\tau = T - t$", fontsize=12)
    ax.set_ylabel("Reservation mid-price $r(s,q,t)$", fontsize=12)
    ax.set_title(
        f"Reservation Mid-Price vs Time-to-Maturity\n"
        f"$s={S},\\ \\sigma={SIGMA},\\ \\gamma={gamma},\\ k={K}$",
        fontsize=13,
    )
    ax.legend(title="Inventory $q$", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "exp1_reservation_price.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Quote distances vs time-to-maturity
# ---------------------------------------------------------------------------

def plot_quote_distances(results_dir: str) -> None:
    """Plot equation-faithful quote distances delta^a and delta^b vs tau.

    One pair of curves per q value in Q_VALUES, with gamma = GAMMA_DEFAULT.
    Saved to ``results/exp1_quote_distances.png``.

    Parameters
    ----------
    results_dir : str
        Directory in which to save the figure.
    """
    taus = np.linspace(0.0, T, 300)
    gamma = GAMMA_DEFAULT
    liq = (1 / gamma) * np.log(1 + gamma / K)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(Q_VALUES)))

    for ax, (label, sign) in zip(axes, [("Ask distance $\\delta^a$", 1), ("Bid distance $\\delta^b$", -1)]):
        for q, color in zip(Q_VALUES, colors):
            if sign == 1:  # delta_a = (1-2q)*gamma*sigma^2*tau/2 + liq
                d_vals = (1 - 2 * q) * gamma * SIGMA**2 * taus / 2 + liq
            else:           # delta_b = (1+2q)*gamma*sigma^2*tau/2 + liq
                d_vals = (1 + 2 * q) * gamma * SIGMA**2 * taus / 2 + liq
            ax.plot(taus, d_vals, label=f"q = {q}", color=color)

        ax.set_xlabel("Time-to-maturity $\\tau = T - t$", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend(title="Inventory $q$", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Equation-Faithful Quote Distances vs Time-to-Maturity\n"
        f"$s={S},\\ \\sigma={SIGMA},\\ \\gamma={gamma},\\ k={K}$",
        fontsize=13,
    )
    fig.tight_layout()

    path = os.path.join(results_dir, "exp1_quote_distances.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Table-faithful vs equation-faithful spread
# ---------------------------------------------------------------------------

def plot_spread_comparison(results_dir: str) -> None:
    """Compare table-faithful and equation-faithful spreads across gamma.

    For each gamma in GAMMAS, the equation-faithful spread at t=0 (tau=T) is
    plotted against k, alongside the table-faithful spread psi(gamma,k).
    Saved to ``results/exp1_spread_comparison.png``.

    Parameters
    ----------
    results_dir : str
        Directory in which to save the figure.
    """
    gammas_fine = np.linspace(0.005, 0.6, 300)
    k_fixed = K

    fig, ax = plt.subplots(figsize=(8, 5))

    # Table-faithful spread psi(gamma, k) – independent of tau
    psi_vals = (2 / gammas_fine) * np.log(1 + gammas_fine / k_fixed)
    ax.plot(gammas_fine, psi_vals, color="steelblue", linewidth=2, label="Table-faithful $\\psi(\\gamma,k)$")

    # Equation-faithful spread at tau = T (t=0): gamma*sigma^2*T + psi
    eq_spread_vals = gammas_fine * SIGMA**2 * T + psi_vals
    ax.plot(
        gammas_fine, eq_spread_vals,
        color="firebrick", linewidth=2, linestyle="--",
        label=f"Eq.-faithful $\\gamma\\sigma^2 T + \\psi$ ($\\tau={T}$)",
    )

    # Also show equation-faithful at tau = 0.5
    tau_half = 0.5
    eq_spread_half = gammas_fine * SIGMA**2 * tau_half + psi_vals
    ax.plot(
        gammas_fine, eq_spread_half,
        color="darkorange", linewidth=2, linestyle=":",
        label=f"Eq.-faithful $\\gamma\\sigma^2 T + \\psi$ ($\\tau={tau_half}$)",
    )

    # Mark the three default gamma values
    for g in GAMMAS:
        psi_g = (2 / g) * np.log(1 + g / k_fixed)
        eq_g = g * SIGMA**2 * T + psi_g
        ax.axvline(g, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.scatter([g], [psi_g], color="steelblue", zorder=5, s=50)
        ax.scatter([g], [eq_g], color="firebrick", zorder=5, s=50)
        ax.annotate(
            f"$\\gamma={g}$",
            xy=(g, psi_g),
            xytext=(g + 0.01, psi_g - 0.05),
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Risk-aversion $\\gamma$", fontsize=12)
    ax.set_ylabel("Spread", fontsize=12)
    ax.set_title(
        f"Table-Faithful vs Equation-Faithful Spread\n"
        f"$\\sigma={SIGMA},\\ k={K}$",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(results_dir, "exp1_spread_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full Experiment 1 theory-validation pipeline."""
    results_dir = _ensure_results_dir()

    # ------------------------------------------------------------------
    # Symbolic verifications
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SYMBOLIC VERIFICATIONS")
    print("=" * 60)
    sym_results = run_symbolic_verifications()
    for name, ok in sym_results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")

    all_sym_ok = all(sym_results.values())
    print(f"\n  All symbolic checks passed: {all_sym_ok}")

    # ------------------------------------------------------------------
    # Numerical verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  NUMERICAL VERIFICATION")
    print("=" * 60)
    default_params = {
        "s": S,
        "T": T,
        "sigma": SIGMA,
        "gamma": GAMMA_DEFAULT,
        "k": K,
        "A": A,
        "q_values": Q_VALUES,
        "t_values": T_VALUES,
    }
    num_results = numerical_verification(default_params)

    all_mid_ok = all(v["midpoint_ok"] for v in num_results.values())
    all_spr_ok = all(v["spread_ok"] for v in num_results.values())
    all_tot_ok = all(v["total_spread_ok"] for v in num_results.values())

    print(f"  All midpoint identities hold:     {all_mid_ok}")
    print(f"  All spread identities hold:       {all_spr_ok}")
    print(f"  All total-spread identities hold: {all_tot_ok}")

    print(
        f"\n  {'q':>4}  {'t':>5}  {'r_a':>10}  {'r_b':>10}  "
        f"{'r_mid':>10}  {'delta_a':>10}  {'delta_b':>10}"
    )
    print("  " + "-" * 65)
    for (q_val, t_val), v in sorted(num_results.items()):
        print(
            f"  {q_val:>4.0f}  {t_val:>5.2f}  {v['r_a']:>10.4f}  {v['r_b']:>10.4f}  "
            f"{v['r_mid']:>10.4f}  {v['delta_a']:>10.4f}  {v['delta_b']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Derivation report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GENERATING DERIVATION REPORT")
    print("=" * 60)
    report = generate_derivation_report()
    report_path = os.path.join(results_dir, "exp1_derivation_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"  Saved: {report_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GENERATING FIGURES")
    print("=" * 60)
    plot_reservation_price(results_dir)
    plot_quote_distances(results_dir)
    plot_spread_comparison(results_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_ok = all_sym_ok and all_mid_ok and all_spr_ok and all_tot_ok
    print(f"  All verifications passed: {all_ok}")
    print(f"  Report saved to:          {report_path}")
    print(f"  Figures saved to:         {results_dir}/")
    print("=" * 60 + "\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
