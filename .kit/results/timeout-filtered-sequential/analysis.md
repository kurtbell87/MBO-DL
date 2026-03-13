# Analysis: Timeout-Filtered Sequential Execution

## Verdict: REFUTED

SC-2 (exp >= $3.50) and SC-3 (min_acct <= $30K) both clearly fail. The hypothesis is rejected. Additionally, SC-S4 (monotonicity of expectancy) fails, revealing that the underlying premise — timeout trades are the primary dilution mechanism — is **factually wrong**. The experiment produces useful information about risk reduction, but the mechanism it was designed to exploit (timeout elimination) accounts for a negligible fraction of the observed effect.

---

## Results vs. Success Criteria

- [x] SC-1: **PASS** — 315 simulations (7 cutoffs x 45 splits) completed
- [ ] SC-2: **FAIL** — optimal exp = $3.02 vs threshold $3.50 (86.2% of target)
- [ ] SC-3: **FAIL** — optimal min_acct_all = $34,000 vs threshold $30,000 (113% of target)
- [x] SC-4: **PASS** — 7-row sweep table fully populated
- [x] SC-5: **PASS** — comparison table populated
- [x] SC-6: **PASS** — account sizing curve produced ($500-$50K)
- [x] SC-7: **PASS** — all 12 deliverable files written to results directory
- [x] SC-8: **PASS** — bar-level split 0 exp = $1.0652 vs PR #38 reference $1.065186 (delta < $0.001)
- [x] SC-S1: **PASS** — bar_level_exp_split0 = $1.0652 vs PR #38 $1.065186 (within $0.01)
- [x] SC-S2: **PASS** — cutoff_390 exp = $2.5008 vs PR #39 $2.50 (within $0.10)
- [x] SC-S3: **PASS** — cutoff_390 trades/day = 162.21 vs PR #39 162.2 (within 5)
- [ ] SC-S4: **FAIL** — monotonicity violated: expectancy DECREASES from $2.50 (390) to $2.27 (330) before rising to $3.02 (270). U-shaped, not monotonic.
- [x] SC-S5: **PASS** — trades/day strictly non-increasing (162.2 → 116.8)

**Outcome: B** — Filtering helps but neither primary target met. Selected via fallback rule (max daily PnL under $35K min_acct).

---

## Critical Finding: The Timeout Premise Is Wrong

The hypothesis rationale stated: "PR #39 showed sequential expectancy of $2.50/trade vs ~$5.00 for pure barrier-hit trades — a 50% dilution from timeout trades."

**The data refutes this mechanism.** Timeout fraction at baseline (cutoff=390) is **0.40%** — only 4 out of every 1,000 trades are timeouts. For timeouts to explain the $5.00 → $2.50 dilution:

```
0.996 × $5.00 + 0.004 × X = $2.50
X = -$620 per timeout trade
```

This is physically impossible (maximum loss on a 7-tick stop is $11.24). **The $2.50 gap is NOT from timeouts.** The dilution comes from the sequential selection mechanism itself: the hold-skip dynamics (66.1% of entry opportunities are skipped because a position is already held) create a non-random sample of the available signals. The executed trades are those occurring immediately after barrier exits during volatile periods — their $2.50 mean reflects the quality of this selected subset, not timeout contamination.

This means the entire time-of-day filter approach is attacking the wrong target. The expectancy improvement at cutoff=270 ($3.02 vs $2.50) comes not from timeout elimination (which changed by only 0.18pp) but from removing lower-quality late-day entries where the model has less signal or market conditions are less favorable for the strategy.

---

## Metric-by-Metric Breakdown

### Primary Metrics

**1. Cutoff Sweep Table (7 rows x 11 columns)**

| Cutoff | Trades/Day | Exp/Trade | Daily PnL | DD Worst | DD Median | Min Acct All | Min Acct 95% | Win Rate | Time Skip% | Hold Skip% | Timeout% |
|--------|-----------|-----------|-----------|----------|-----------|-------------|-------------|---------|------------|------------|----------|
| 390 | 162.2 | **$2.50** | **$412.77** | $47,894 | $12,917 | $48,000 | $26,500 | 49.93% | 0.0% | 66.1% | 0.40% |
| 375 | 158.9 | $2.49 | $400.34 | $47,947 | $12,917 | $48,000 | $27,000 | 49.92% | 6.7% | 63.4% | 0.39% |
| 360 | 155.8 | $2.46 | $384.03 | $47,989 | $12,922 | $48,000 | $26,500 | 49.89% | 12.9% | 60.8% | 0.38% |
| 345 | 153.1 | $2.28 | $346.49 | $47,923 | $12,906 | $48,000 | $30,000 | 49.84% | 17.0% | 60.3% | 0.35% |
| 330 | 148.6 | **$2.27** | $331.22 | $48,089 | $13,331 | **$48,500** | $32,000 | 49.94% | 22.9% | 59.2% | 0.25% |
| 300 | 131.3 | $2.48 | $317.71 | $43,202 | $12,596 | $43,500 | $30,500 | 50.03% | 44.9% | 44.1% | 0.22% |
| 270 | **116.8** | **$3.02** | $336.77 | **$33,984** | **$8,687** | **$34,000** | **$25,500** | **50.27%** | 58.0% | 34.4% | 0.22% |

Notable patterns:
- **Expectancy is U-shaped**, not monotonic. It dips from $2.50 to $2.27 (390→330) then rises to $3.02 (270). The intermediate cutoffs (375-330) remove trades that are *better* than what remains.
- **Daily PnL is monotonically decreasing** from $412.77 (390) to $317.71 (300), with a partial recovery at 270 ($336.77). Unfiltered maximizes daily income.
- **Drawdown only improves at aggressive cutoffs** (300 and 270). Cutoffs 375-330 have essentially the same worst-case DD ($47.9K-$48.1K) as unfiltered.
- **Hold-skip and time-skip are partially substitutive:** hold_skip drops from 66.1% to 34.4% as time_skip rises to 58.0%. Total skip rate rises (66.1% → 92.4%), meaning far fewer entries are considered.

**2. Optimal Cutoff Expectancy:** $3.02/trade at cutoff=270 (FAIL vs $3.50 target, 86.2% achieved)

**3. Optimal Cutoff Min Account All:** $34,000 at cutoff=270 (FAIL vs $30,000 target, 113%)

### Secondary Metrics

**4. Optimal Daily PnL:** $336.77/day — 18.4% lower than unfiltered ($412.77). The risk-adjusted trade is: accept $76/day less income for $14K less capital requirement. At the margin, the Calmar ratio favors filtering (2.49 vs 2.16).

**5. Optimal Calmar:** 2.49 — 15.0% improvement over unfiltered (2.16). This is the strongest argument for filtering: risk-adjusted returns improve even though absolute returns decline.

**6. Optimal Sharpe:** 20.19 (annualized from daily) — 38.7% improvement over unfiltered (14.56). The daily Sharpe is $336.77 / $264.28 = 1.27, annualized by √251. This is a high-frequency strategy Sharpe; the high number reflects intraday diversification across ~117 trades/day.

**7. Optimal Trades/Day:** 116.8 — 28.0% reduction from unfiltered 162.2. The strategy still executes ~117 trades/day, which is substantial.

**8. Timeout Fraction by Cutoff:**

| Cutoff | Timeout % | Barrier Hit % | Delta from 390 |
|--------|----------|--------------|----------------|
| 390 | 0.40% | 99.60% | — |
| 375 | 0.39% | 99.61% | -0.01pp |
| 360 | 0.38% | 99.62% | -0.02pp |
| 345 | 0.35% | 99.65% | -0.05pp |
| 330 | 0.25% | 99.75% | -0.15pp |
| 300 | 0.22% | 99.78% | -0.18pp |
| 270 | 0.22% | 99.78% | -0.18pp |

Timeouts decline from 0.40% to 0.22% — a reduction of 0.18pp. This is economically trivial. At $2.50/trade on 162 trades/day, 0.18pp of timeout elimination affects $0.07/day. The actual $76/day change in daily PnL is driven by trade selection, not timeout elimination.

**9. Barrier Hit Fraction:** See above. Barrier hit rate is already 99.6% at baseline. The time filter cannot materially improve this.

**10. Splits 18 & 32 Comparison:**

Split 18 (group-4 worst path):
| Cutoff | Exp/Trade | Max DD | Daily PnL |
|--------|----------|--------|-----------|
| 390 | -$2.28 | $47,894 | -$482 |
| 270 | **-$1.08** | **$33,984** | **-$171** |
| Delta | +$1.20 | -$13,910 | +$311 |

Split 32 (2nd worst path):
| Cutoff | Exp/Trade | Max DD | Daily PnL |
|--------|----------|--------|-----------|
| 390 | -$5.71 | $46,888 | -$965 |
| 270 | **-$4.77** | **$32,662** | **-$615** |
| Delta | +$0.94 | -$14,226 | +$350 |

Both outlier paths improve substantially but **remain deeply negative**. Split 18 loses $171/day instead of $482/day; split 32 loses $615/day instead of $965/day. The worst-case drawdown drops by ~$14K on both. These paths represent structural model failures on temporal group 4, not timeout-driven losses. Filtering reduces the magnitude but cannot flip the sign.

**11. Daily PnL Percentiles (cutoff=270):**

| Percentile | Value | Interpretation |
|-----------|-------|---------------|
| p5 | -$3,171 | 1-in-20 worst day |
| p25 | -$418 | Typical losing day |
| p50 | **+$219** | Median day is profitable |
| p75 | +$924 | Typical winning day |
| p95 | +$3,969 | 1-in-20 best day |

The distribution is roughly symmetric around a positive median. The p5/p95 tails are roughly equal in magnitude (~$3.1K vs $4.0K), suggesting no extreme left tail. The positive median (+$219) with a mean of +$337 indicates mild positive skew in the filtered distribution.

**12. Annual Expectancy:** $84,530 (= $336.77/day x 251 days). This compares to $103,605 unfiltered — a $19,075 annual reduction (-18.4%). The question is whether the $14K capital savings offsets this: $84.5K on $34K capital = 249% annual return; $103.6K on $48K capital = 216% annual return. **Risk-adjusted, filtering wins on ROC.**

### Sanity Checks

**SC-S1 (Bar-level match):** PASS. Split 0 bar-level expectancy $1.0652 matches PR #38's $1.065186 to 4 decimal places. Training is identical — confirmed.

**SC-S2 (Cutoff 390 exp):** PASS. Unfiltered exp $2.5008 vs PR #39's $2.50 — delta $0.0008 (within $0.10 tolerance). Simulation logic unchanged.

**SC-S3 (Cutoff 390 trades/day):** PASS. 162.21 vs PR #39's 162.2 — delta 0.01 (within 5 tolerance). Identical simulation.

**SC-S4 (Monotonicity of expectancy):** FAIL. This is the most informative sanity check failure. The expectancy path: $2.50 → $2.49 → $2.46 → $2.28 → $2.27 → $2.48 → $3.02. The dip to $2.27 at cutoff=330 (removing last hour) followed by recovery at 270 (removing last 2 hours) means:
- Minutes 330-389 (last hour) contain above-average quality trades
- Minutes 270-329 (hour before that) contain below-average quality trades

This U-shape undermines the simple "late = bad" narrative. The quality distribution across the RTH session is non-monotonic.

**SC-S5 (Monotonicity of trades):** PASS. Trades/day is strictly non-increasing: 162.2 → 158.9 → 155.8 → 153.1 → 148.6 → 131.3 → 116.8. This is mechanically guaranteed by the filter logic.

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| Wall-clock | 15 min | 2.64 min | Under budget (17.6%) |
| Training runs | 90 | 90 | On budget |
| Simulations | 315 | 315 | On budget |
| GPU-hours | 0 | 0 | As expected |

Budget was appropriate. 3.5x safety margin was generous for this workload. Local Apple Silicon execution was the correct choice.

---

## Confounds and Alternative Explanations

### 1. The Timeout Premise Is Falsified

The experiment was motivated by "50% dilution from timeout trades." The data shows timeouts are 0.40% of baseline trades. The actual dilution mechanism is sequential selection bias (hold-skip dynamics), not timeouts. The time-of-day filter improves results at cutoff=270, but the mechanism is trade quality selection (removing lower-quality late-day signals), not timeout elimination.

### 2. The U-Shape Could Be Noise

Expectancy's U-shape (declining 390→330, rising 330→270) could reflect: (a) genuine time-of-day quality variation in model signals, (b) interaction between cutoff and the sequential entry selection dynamics, or (c) statistical noise from the 45-split aggregation. With 45 CPCV splits and 7 cutoffs, we have 315 data points but high cross-split correlation. The U-shape should be interpreted with caution.

### 3. The Cutoff Interacts With Hold-Skip Dynamics

At cutoff=270, time_skip=58.0% and hold_skip=34.4%. At cutoff=390, hold_skip=66.1%. The total "unavailable" fraction rises from 66.1% to 92.4%. This means the filtered strategy is far more selective about which signals it can access. The improvement may come from this increased selectivity rather than time-of-day quality per se. The strategy effectively becomes "only trade the first 4.5 hours AND only when no position is held" — a double filter.

### 4. Drawdown Improvement May Be Mechanical

Fewer trades mechanically compress the equity curve's variance. If you trade 28% fewer times, worst-case drawdowns decline roughly in proportion. The 29% drawdown reduction at 270 is almost exactly proportional to the 28% trade reduction — suggesting the improvement is mechanical (fewer opportunities for consecutive losses), not from removing specifically bad trades.

### 5. Outlier Paths (18, 32) Are Structural

Both outlier splits remain deeply negative at all cutoffs. Split 32 loses $4.77/trade even at cutoff=270. These paths represent fundamental model failure on certain temporal groups, not late-day effects. Time-of-day filtering cannot fix split-level model inadequacy.

### 6. Calmar and Sharpe Improvement May Be Misleading

Calmar improves 15% and Sharpe improves 39%, but these are both driven by the ~29% drawdown reduction while daily PnL only drops 18%. This is a favorable ratio, but it's partially mechanical (fewer trades = less variance). The Sharpe of 20.19 is high in absolute terms but standard for intraday strategies with 100+ trades/day.

---

## What This Changes About Our Understanding

1. **Timeouts are negligible.** Only 0.40% of sequential trades are timeouts with 1-hour time horizon. The $5.00→$2.50 dilution identified in PR #39 is entirely from sequential selection dynamics (hold-skip filtering), not timeout contamination. Future optimization should target the selection mechanism, not timeouts.

2. **Time-of-day filtering works, but modestly and non-monotonically.** Aggressive filtering (cutoff=270, removing last 2 hours) improves per-trade quality by +$0.52 (+21%) and reduces worst-case drawdown by $14K (-29%). But it reduces daily income by $76 (-18%). Intermediate cutoffs (last 30-60 min) hurt, not help — the relationship between time-of-day and trade quality is complex.

3. **Risk-adjusted returns favor filtering.** ROC at cutoff=270 is 249% vs 216% unfiltered. Calmar +15%. If the binding constraint is capital (not income), filtering is preferable.

4. **The $34K account minimum is a structural floor for this model.** Splits 18 and 32 drive the worst-case drawdown regardless of filtering. The min_acct can only drop further by (a) fixing the model's failure on temporal group 4, or (b) accepting less than 100% path survival (min_acct_95 = $25.5K).

5. **The hold-skip mechanism deserves more investigation.** Hold-skip rate is the dominant factor in sequential execution economics. At 66%, two-thirds of signals are wasted. Understanding which skipped signals would have been profitable vs unprofitable is more likely to unlock value than time-of-day filtering.

---

## Proposed Next Experiments

1. **Volatility-conditional entry filtering** — The spec's Outcome B prescription. Instead of time-of-day (a proxy), filter on `volatility_50` directly. The strategy already self-selects for volatile moments (PR #39 finding); making this explicit via a volatility threshold may be more effective than the blunt time cutoff. Test: skip entries when volatility_50 < p25 or p50 of training distribution.

2. **Hold-skip analysis** — NOT a full experiment, but a diagnostic pass: for each skipped entry opportunity, record what the trade would have been (direction, barrier outcome, PnL). Compute the expectancy of skipped trades vs executed trades. This answers: is the sequential selection bias helping or hurting? If skipped trades have lower expectancy, the hold-skip is beneficial and the strategy is already self-optimizing. If skipped trades have similar expectancy, there may be a multi-contract opportunity.

3. **Multi-contract scaling (N=2,5,10)** — If the hold-skip rate (66%) means 2/3 of signals are wasted, running N=2 contracts halves the wasted fraction (roughly). The question is whether marginal signals (2nd-best available) maintain the same quality as primary signals. Test at N=2 first as smallest increment.

4. **Regime-conditional trading (Q1-Q2 only)** — GBT was marginally profitable in H1 2022 (+$0.003 Q1, +$0.029 Q2 at bar-level). Sequential execution amplifies the edge (bar-level $1.81 → sequential $2.50). A Q1-Q2-only strategy with 126 trading days may produce a viable year-round schedule if H2 is flat (not negative enough to overcome H1 gains). Limited by single-year data.

---

## Program Status

- Questions answered this cycle: 1 (timeout-filtered sequential execution)
- New questions added this cycle: 0 (volatility-conditional and hold-skip analysis are refinements of existing questions, not new)
- Questions remaining (open, not blocked): 5 (long-perspective labels P0, message regime P2, cost sensitivity P2, regime-conditional P3, multi-contract P2)
- Handoff required: NO
