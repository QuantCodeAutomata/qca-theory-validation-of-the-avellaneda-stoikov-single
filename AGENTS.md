# AGENTS.md — Repository Knowledge for AI Agents

This file documents the structure, conventions, and design decisions of the
Avellaneda-Stoikov market-making repository so that AI agents can work efficiently
across sessions without re-discovering context.

---

## Repository Purpose

This repository implements, validates, and replicates the results of the
Avellaneda-Stoikov (2008) optimal market-making model. It is structured as a
research codebase with three runnable experiments, a reusable core library,
and a pytest test suite.

**All data is purely synthetic.** No external data sources, downloads, or API
keys are required at any point. The simulation engine generates price paths and
order arrivals internally using NumPy random number generation.

---

## Directory Layout

```
/workspace/project/
├── src/          # Core importable library (no __main__ blocks)
├── exp/          # Standalone experiment scripts (entry points)
├── tests/        # pytest test suite
├── data/         # Empty placeholder — kept in version control via .gitkeep
└── results/      # Output artefacts (figures, tables); RESULTS.md template
```

---

## Source Modules (`src/`)

### `src/theory_validation.py`

Symbolic and numerical verification of the Avellaneda-Stoikov closed-form solution.

**Key responsibilities:**
- Build SymPy expressions for the reservation price and optimal spread.
- Verify algebraic identities (e.g. `ask - bid == δ*`, `(ask + bid)/2 == r`).
- Check boundary conditions at `t = T` (zero time-to-go) and `q = 0`.
- Run numerical spot-checks with concrete parameter values.

**Exports:** verification functions that return structured result dicts; no
side-effects beyond return values (printing/saving is handled in `exp/exp1_theory.py`).

**Implemented functions (exp_1):**
- `frozen_inventory_value(x,s,q,t,T,sigma,gamma)` — HJB closed form under CARA + ABM.
- `reservation_ask/bid/mid(s,q,t,T,sigma,gamma)` — Eqs. 3-4 and mid-price.
- `verify_midpoint_identity / verify_spread_identity` — symbolic simplify checks.
- `liquidity_term(gamma,k)` — `(1/γ)·ln(1+γ/k)`.
- `quote_distances_equation_faithful(...)` — returns `{delta_a, delta_b}` dict.
- `verify_total_spread_equation_faithful(...)` — symbolic total-spread check.
- `table_faithful_spread(gamma,k)` — `(2/γ)·ln(1+γ/k)`.
- `verify_exponential_intensity_derivative(A,k,delta)` — symbolic derivative check.
- `verify_ansatz_reservation_prices(...)` — theta-ansatz derivation consistency dict.
- `numerical_verification(params)` — iterates over `(q,t)` product; returns nested dict.
- `generate_derivation_report()` — multi-section plain-text/LaTeX report string.

---

### `src/simulation.py`

Monte Carlo simulation engine for the Avellaneda-Stoikov trading environment.

**Exports:**
- `SimulationParams` (dataclass) — all model and MC parameters with defaults matching
  Avellaneda & Stoikov Table 1: `S0=100, T=1, sigma=2, dt=0.005, A=140, k=1.5,
  q0=0, X0=0, n_paths=1000, gamma=0.1, seed=42`.
- `simulate_path(params, quote_fn, rng) -> dict` — one MC path; returns arrays `S`,
  `q`, `X`, `r` (NaN; recompute externally), `p_ask`, `p_bid`, `delta_a`, `delta_b`,
  `lambda_a`, `lambda_b`, scalars `terminal_profit`, `terminal_inventory`,
  `lambda_dt_violations` (int).
- `simulate_many(params, quote_fn, strategy_name='') -> dict` — runs `n_paths`
  independent paths; per-path seed derived as `np.random.default_rng([seed, idx])`.
  Returns `terminal_profits`, `terminal_inventories`, summary stats, `paths` list.
- `compute_summary_table(results_by_gamma, spread_fn) -> pd.DataFrame` — builds a
  summary DataFrame from `{gamma: {strategy: simulate_many_result}}` dicts.

**Design notes:**
- Mid-price follows a symmetric ±1 random walk: `S[n+1] = S[n] ± sigma*sqrt(dt)`.
- Order fills are independent Bernoulli(lambda*dt) events; both ask and bid can fill
  simultaneously in one step.
- `lambda_dt_violations` counts steps where `lambda_a*dt > 1` or `lambda_b*dt > 1`
  (valid-approximation check for the Poisson/Bernoulli swap).
- `quote_fn` signature: `(s: float, q: int, t: float, params: SimulationParams) ->
  (p_ask: float, p_bid: float)`.

---

### `src/policies.py`

Optimal quoting policy implementations.

**Exports (table-faithful convention — constant spread):**
- `table_faithful_spread(gamma, k) -> float` — `(2/γ)·ln(1 + γ/k)`.
- `inventory_quotes_table_faithful(s, q, t, params) -> (p_ask, p_bid)` — centres on `r`.
- `symmetric_quotes_table_faithful(s, q, t, params) -> (p_ask, p_bid)` — centres on `s`.

**Exports (equation-faithful convention — time-varying spread):**
- `equation_faithful_spread(s, q, t, params) -> float` —
  `γσ²(T−t) + (2/γ)·ln(1+γ/k)`.
- `inventory_quotes_equation_faithful(s, q, t, params) -> (p_ask, p_bid)` — uses
  per-side delta formulas: `δ_a = ((1−2q)γσ²(T−t))/2 + (1/γ)ln(1+γ/k)`,
  `δ_b = ((1+2q)γσ²(T−t))/2 + (1/γ)ln(1+γ/k)`.
- `symmetric_quotes_equation_faithful(s, q, t, params) -> (p_ask, p_bid)` — centres
  on `s` with time-varying spread.
- `get_reservation_price(s, q, t, params) -> float` — `r = s - q·γ·σ²·(T−t)`.

**Invariants (tested):**
- For inventory strategies: `(p_ask + p_bid)/2 == r` exactly.
- For symmetric strategies: `(p_ask + p_bid)/2 == s` exactly.
- `delta_a + delta_b == equation_faithful_spread(...)` exactly.
- `table_faithful_spread` values: `psi(0.1,1.5)≈1.29`, `psi(0.01,1.5)≈1.33`,
  `psi(0.5,1.5)≈1.15`.

**Critical design decision — spread convention:** See the dedicated section below.

---

## Experiment Scripts (`exp/`)

Each script is a standalone entry point. Run from the repository root with
`python exp/<script>.py`.

### `exp/exp1_theory.py`

Imports functions from `src/theory_validation.py`, runs symbolic and numerical
checks, prints a structured report, and saves a summary to `results/`.

### `exp/exp2_table_faithful.py`

Runs Monte Carlo with `inventory_quotes_table_faithful` and `symmetric_quotes_table_faithful`.
Seeds: inventory `int(gamma*1000)+42`, symmetric `int(gamma*1000)+142`.
Gammas: `[0.1, 0.01, 0.5]`, 1000 paths each.
Saves: `results/exp2_summary.csv`, `exp2_representative_path.png`,
`exp2_profit_histograms.png`, `exp2_summary_table.png`.

### `exp/exp3_equation_faithful.py`

Runs Monte Carlo with `inventory_quotes_equation_faithful` and
`symmetric_quotes_equation_faithful` (same seeds and gammas as exp2).
Includes pre-run formula validation checks (delta_a+delta_b, midpoint invariants).
Saves: `results/exp3_summary.csv`, `exp3_representative_path.png`,
`exp3_profit_histograms.png`, `exp3_spread_over_time.png`, `exp3_comparison.png`.

---

## Testing (`tests/`)

Tests use **pytest** with no mocking. All tests exercise real code paths against
known analytical values or invariants.

**Implemented test files:**
- `tests/test_theory.py` — unit tests for `src/theory_validation.py`.
- `tests/test_simulation.py` — 18 tests covering policy functions and simulation
  engine: spread values, quote invariants, path array shapes, terminal profit
  identity, inventory conservation (diffs in {-1,0,1}), violation counting,
  reproducibility, and the statistical result that inventory std < symmetric std.

Run the full suite with:
```bash
pytest tests/
```

---

## Key Design Decisions

### Table-Faithful vs. Equation-Faithful Spread Convention

This is the single most important subtlety in the codebase.

Avellaneda & Stoikov (2008) define:

> δ* = γσ²(T − t) + (2/γ) ln(1 + γ/κ)

In the paper's **equations**, the bid and ask are written as:

> s^a = r + δ*/2    s^b = r − δ*/2

which implies `δ*` is the **full** spread. However, inspecting the paper's
**Table 1** numerical results, the tabulated spread values match the formula
output directly when `δ*` is treated as the *half*-spread (i.e. quotes at
`r ± δ*`, full spread = `2δ*`).

This repository resolves the ambiguity by implementing both:

| Convention | Quote placement | Full spread | Matches paper's… |
|------------|-----------------|-------------|------------------|
| Table-faithful | `r ± δ*/2` | `δ*` | Table 1 numbers |
| Equation-faithful | `r ± δ*` | `2δ*` | Equations 3–4 |

Always be explicit about which convention is in use when discussing results.

### No External Data

The simulation is entirely self-contained. `data/` is kept as an empty placeholder
(tracked via `.gitkeep`) in case future work requires storing market data, but
currently nothing reads from it.

### Reproducibility

All stochastic components accept an integer `seed` parameter. Experiments use
a fixed default seed (e.g. `seed=42`) so results are reproducible across runs.

### Parameter Defaults

Default parameter values match the paper's Table 1 setup:

| Parameter | Default |
|-----------|---------|
| γ (gamma) | 0.1 |
| σ (sigma) | 2.0 |
| T | 1.0 |
| A | 140 |
| κ | 1.5 |
| S₀ | 100 |
| N (paths) | 10 000 |
| n (steps) | 5 000 |

---

## Common Pitfalls

- **Import paths**: Run experiment scripts from the repository root so that
  `import src.xxx` resolves correctly. Alternatively, install the package in
  editable mode if a `pyproject.toml` is added in future.
- **Spread convention bugs**: When adding new policy variants, always document
  which convention they follow and add a corresponding test in `tests/test_policies.py`.
- **NumPy seeding**: Use `np.random.default_rng(seed)` (the new Generator API)
  rather than the legacy `np.random.seed()` global state.

---

## References

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
*Quantitative Finance*, 8(3), 217–224. https://doi.org/10.1080/14697680701381228
