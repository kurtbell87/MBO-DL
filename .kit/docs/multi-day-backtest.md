# Phase 3: Multi-Day Oracle Backtest [Engineering]

**Spec**: TRAJECTORY.md §9 (oracle replay backtest), §5.3 (comparison framework)
**Depends on**: Phase 2 (oracle-replay) — needs `OracleReplay`, `BacktestRunner`, `ExecutionCosts`.
**Unlocks**: Phase 6 (synthesis) — provides oracle expectancy data.

---

## Objective

Run the oracle replay across all 2022 trading days for each bar type × oracle config × labeling method combination. Aggregate results. Compute regime stratification. Produce a go/no-go assessment.

---

## Scope

### Configurations to Test

```
Bar types:
  volume V ∈ {50, 100, 200}
  tick   K ∈ {25, 50, 100}
  time   ∈ {1s, 5s, 60s}

Oracle configs:
  Default:  target_ticks=10, stop_ticks=5, volume_horizon=500, take_profit_ticks=20
  Sweep (if default fails §9.4 criteria):
    target_ticks ∈ {5, 8, 10, 15, 20}
    stop_ticks   ∈ {3, 5, 8, 10}
    volume_horizon ∈ {200, 500, 1000, 2000}
    take_profit_ticks = 2 × target_ticks

Labeling methods:
  FIRST_TO_HIT
  TRIPLE_BARRIER
```

### Data Range (§9.3)

```
Dataset: DATA/GLBX-20260207-L953CAPU5B/ (312 daily .dbn.zst files, MES MBO 2022)

In-sample:      Jan–Jun 2022
Out-of-sample:  Jul–Dec 2022

Instrument rollover (quarterly):
  MESH2 (13614) → MESM2 (13615): ~March 18, 2022
  MESM2 (13615) → MESU2 (10039): ~June 17, 2022
  MESU2 (10039) → MESZ2 (????):  ~September 16, 2022

Process each quarterly contract independently. Do not trade across rollovers.
Exclude final 3 trading days before each rollover date (anomalous volume profiles).
Log excluded dates.
```

### Success Criteria (§9.4)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Net expectancy | > $0.50 per trade | Must clear costs with margin |
| Profit factor | > 1.3 | Gross wins exceed gross losses |
| Win rate | > 45% | Breakeven with 2:1 reward:risk after costs |
| OOS net PnL | > 0 | In-sample could be overfit |
| Max drawdown | < 50 × expectancy | Recoverable within ~50 trades |
| Trade count | > 10 per day avg | Statistical validity |

At least one labeling method must pass. If both pass, prefer higher expectancy × trade_count.

---

## Regime Stratification (§9.6)

Partition results along 4 dimensions:

1. **Realized volatility quartiles** — 20-bar realized vol, Q1-Q4. If expectancy concentrates in Q4, strategy is fragile.
2. **Time-of-day** — Open (09:30–10:30), Mid (10:30–14:00), Close (14:00–16:00).
3. **Volume regimes** — Daily total volume quartiles. High-volume days may capture event-driven moves.
4. **Trend vs. mean-reversion** — Days classified by |OTC return|: strong trend (>1%), range-bound (<0.3%), moderate.

**Cross-regime stability score**: `stability = min(regime_expectancy) / max(regime_expectancy)` across volatility quartiles.
- > 0.5 → robust
- 0.2–0.5 → regime-dependent
- < 0.2 → fragile

---

## Oracle Comparison (§5.3)

Run both labeling methods on identical bar sequences. Compare:
- Label distribution (class frequencies)
- Label stability (consecutive label agreement rate)
- Label-return correlation
- Expectancy after costs
- Conditional entropy of labels given time-of-day
- Regime dependence

---

## Outputs

```
results/backtest/
  Per-config JSON: bar_type, bar_param, oracle_config, label_method → BacktestResult + all TradeRecords

Summary tables (printed):
  labeling_method × bar_type × oracle_config → (expectancy, win_rate, profit_factor, sharpe, trades/day)
  Regime stratification per §9.6 with stability scores
  Oracle comparison: first-to-hit vs triple barrier
  Go/no-go assessment per §9.4

If oracle fails default config:
  Pareto frontier of (expectancy, trade_count, max_drawdown) across parameter sweep
```

---

## Oracle Failure Diagnosis (§9.5)

If no parameterization produces positive expectancy:
1. Costs too high for scale → try larger targets (20, 40 ticks)
2. MES microstructure too noisy → filter: only label when spread < 2 ticks
3. Oracle threshold logic too naive → proceed to feature discovery on returns
4. Feature discovery may reveal better labels → continue to Phase 4 anyway

---

## Validation Gate

```
Assert: Backtest covers all trading days in 2022 (excluding holidays, no-data days,
        and rollover transition days — log excluded dates)
Assert: Instrument rollover handled correctly (no trades on excluded rollover days)
Assert: Results written to JSON with full trade-level detail including exit_reason
Assert: Both labeling methods (first-to-hit, triple barrier) run on identical bar sequences
Assert: Safety cap trigger rate < 1% during RTH (if not, volume_horizon needs recalibration)
Print:  Summary table:
        labeling_method × bar_type × oracle_config → (expectancy, win_rate, profit_factor, sharpe, trades/day)
Print:  Regime stratification table per §9.6 with stability scores
Print:  Oracle comparison: first-to-hit vs triple barrier expectancy, label-return correlation
Print:  Go/no-go assessment per §9.4 criteria
```
