# Avellaneda-Stoikov Market-Making Model: Theory Validation and Monte Carlo Simulation

A rigorous Python implementation of the Avellaneda-Stoikov (2008) optimal market-making model,
comprising symbolic theory validation and two Monte Carlo replication experiments that reproduce
the paper's core results while carefully distinguishing between the paper's two spread conventions.

---

## Overview

The Avellaneda-Stoikov model derives closed-form expressions for the optimal bid and ask quotes
of a market maker who faces inventory risk and uncertain order arrival. The model produces:

- A **reservation price** that adjusts the mid-price for inventory exposure.
- An **optimal spread** that balances order-flow capture against adverse-selection risk.

This repository verifies the theoretical derivations symbolically with SymPy, validates them
numerically, and replicates the simulation results reported in the paper under both of the
paper's spread conventions.

---

## Experiments

### Experiment 1 — Theory Validation (`exp/exp1_theory.py`)

Performs a full symbolic and numerical audit of the Avellaneda-Stoikov closed-form solution:

- Verifies the reservation price formula `r(s, q, t) = s - q·γ·σ²·(T - t)`.
- Verifies the optimal full spread formula `δ* = γ·σ²·(T - t) + (2/γ)·ln(1 + γ/κ)`.
- Confirms that the optimal bid and ask quotes are `r ± δ*/2`.
- Checks boundary conditions at `t = T` and zero-inventory cases.
- Validates internal consistency: `(ask - bid) == δ*` and `(ask + bid)/2 == r`.

### Experiment 2 — Monte Carlo Replication, Table-Faithful Spread (`exp/exp2_table_faithful.py`)

Runs a Monte Carlo simulation of the market maker's strategy using the **table-faithful**
spread convention, which matches the numerical values reported in Table 1 of the paper.
Under this convention the half-spread placed around the reservation price equals `δ*/2`,
where `δ*` is the full optimal spread.

Reports terminal wealth statistics (mean P&L, standard deviation, Sharpe-equivalent) across
thousands of simulated trading days and compares them to the paper's tabulated results.

### Experiment 3 — Monte Carlo Replication, Equation-Faithful Spread (`exp/exp3_equation_faithful.py`)

Runs the same Monte Carlo simulation using the **equation-faithful** spread convention, which
follows the paper's equations literally: the quantity `δ*` from the formula is treated as the
*half*-spread, so the full quoted spread is `2·δ*`. This convention yields a wider spread and
different P&L profile than the table-faithful version.

Provides a side-by-side comparison with Experiment 2 to quantify the impact of the convention
choice on simulated outcomes.

---

## Repository Structure

```
.
├── README.md
├── AGENTS.md                        # Repository knowledge for AI agents
├── .gitignore
├── requirements.txt
│
├── src/                             # Core library modules
│   ├── theory_validation.py         # Symbolic & numerical formula verification
│   ├── simulation.py                # Monte Carlo simulation engine
│   └── policies.py                  # Optimal quoting policy implementations
│
├── exp/                             # Runnable experiment scripts
│   ├── exp1_theory.py               # Experiment 1: theory validation
│   ├── exp2_table_faithful.py       # Experiment 2: table-faithful spread
│   └── exp3_equation_faithful.py    # Experiment 3: equation-faithful spread
│
├── tests/                           # pytest test suite
│
├── data/                            # Placeholder — no external data required
│   └── .gitkeep
│
└── results/                         # Output tables and figures
    ├── .gitkeep
    └── RESULTS.md                   # Human-readable results template
```

---

## Installation

Python 3.10 or later is recommended.

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-directory>

# 2. (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Experiments

Each experiment script is self-contained and prints results to stdout. Run them from the
repository root so that relative imports resolve correctly.

```bash
# Experiment 1 — Theory Validation
python exp/exp1_theory.py

# Experiment 2 — Monte Carlo (Table-Faithful Spread)
python exp/exp2_table_faithful.py

# Experiment 3 — Monte Carlo (Equation-Faithful Spread)
python exp/exp3_equation_faithful.py
```

Generated figures and summary tables are written to `results/`.

---

## Running the Tests

```bash
pytest tests/
```

Add `-v` for verbose output or `-x` to stop on the first failure:

```bash
pytest tests/ -v
pytest tests/ -x
```

---

## Model Parameters

| Parameter | Symbol | Description | Typical value |
|-----------|--------|-------------|---------------|
| Risk aversion | `γ` (gamma) | Coefficient of absolute risk aversion | 0.1 |
| Volatility | `σ` (sigma) | Annualised mid-price volatility | 2.0 |
| Time horizon | `T` | Total trading horizon (in same units as σ) | 1.0 |
| Current time | `t` | Elapsed time | 0 … T |
| Inventory | `q` | Net signed inventory held by market maker | −∞ … +∞ |
| Order intensity | `A` | Baseline Poisson arrival rate of market orders | 140 |
| Fill decay | `κ` | Sensitivity of arrival rate to distance from mid | 1.5 |
| Initial price | `S₀` | Starting mid-price | 100 |
| Simulations | `N` | Number of Monte Carlo paths | 10 000 |
| Time steps | `n` | Discrete steps per simulation path | 5 000 |

---

## Key Design Decisions

### Spread Convention Distinction

The paper contains two apparently inconsistent usages of the spread symbol `δ`:

1. **Equation-faithful**: The formula `δ* = γσ²(T−t) + (2/γ)ln(1 + γ/κ)` is interpreted
   as the *half*-spread; quotes are placed at `r ± δ*`.
2. **Table-faithful**: The same symbol `δ*` is treated as the *full* spread; quotes are
   placed at `r ± δ*/2`.

Experiment 2 uses the table-faithful convention to reproduce the paper's numerical tables.
Experiment 3 uses the equation-faithful convention to follow the paper's equations literally.
`src/policies.py` implements both variants with an explicit `half_spread` flag.

---

## References

Avellaneda, M., & Stoikov, S. (2008). **High-frequency trading in a limit order book**.
*Quantitative Finance*, 8(3), 217–224. https://doi.org/10.1080/14697680701381228
