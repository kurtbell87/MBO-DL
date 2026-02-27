# Analysis: Daily Stop Loss — Sequential Execution

## Verdict: CONFIRMED (Outcome A)

All 9 success criteria pass. All 5 sanity checks pass. DSL=$2,000 is the recommended threshold — the loosest achieving all four primary targets. Daily stop loss is a highly effective variance truncation mechanism for this strategy.

**Executive summary:** Adding a $2,000 daily stop loss to the sequential 19:7 pipeline compresses worst-case drawdown from $34K to $18K (-47%) while *increasing* annual PnL from $85K to $114K (+35%). This counterintuitive result — better returns AND lower risk — occurs because the trades removed by DSL (those executed after intra-day losses exceed $2K) are net-negative in aggregate: 92.7% of stopped days end negative without DSL. Recovery sacrifice is only 7.3%. The Calmar ratio improves from 2.49 to 6.37 (+156%).

---

## Results vs. Success Criteria

- [x] **SC-1: PASS** — min_account_survive_all = **$18,000** vs threshold $20,000
- [x] **SC-2: PASS** — annual PnL = **$114,352** vs threshold $50,000
- [x] **SC-3: PASS** — Calmar = **6.37** vs threshold 2.0
- [x] **SC-4: PASS** — recovery sacrifice = **7.3%** vs threshold 20%
- [x] **SC-5: PASS** — 45 splits x 9 DSL levels = **405 simulations** completed
- [x] **SC-6: PASS** — sweep table fully populated (9 rows x 16+ columns)
- [x] **SC-7: PASS** — recovery sacrifice for all 8 non-baseline DSL levels reported
- [x] **SC-8: PASS** — all output files written to results directory
- [x] **SC-9: PASS** — bar-level split 0 = **$1.0652** vs PR #38 reference $1.065186 (delta < $0.01)

### Sanity Checks

- [x] **SC-S1: PASS** — bar-level split 0 = $1.0652 vs reference $1.065186 (within $0.01). Training unchanged.
- [x] **SC-S2: PASS** — DSL=None exp = $3.016 vs PR #40 reference $3.016 (essentially identical).
- [x] **SC-S3: PASS** — DSL=None trades/day = 116.77 vs PR #40 reference 116.8 (within 5).
- [x] **SC-S4: PASS** — DSL=None min_acct_all = $34,000 vs PR #40 reference $34,000 (exact match).
- [x] **SC-S5: PASS** — Trades/day strictly non-increasing: 116.8 → 115.1 → 113.7 → 111.1 → 109.6 → 106.5 → 102.0 → 94.0 → 78.2. Monotonicity holds.

**No abort criteria triggered.** Wall-clock = 2.75 min (budget: 15 min). No NaN values. No zero-trade splits. No monotonicity violations.

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. DSL Sweep Table (9 levels x 16 metrics)

| DSL | Trades/Day | Exp/Trade | Daily Mean | Daily Std | Skew | Kurt | DD Worst | DD Med | MinAcct All | MinAcct 95 | Calmar | Sharpe | Annual PnL | WR | Trigger% | Sacrifice% |
|-----|-----------|-----------|------------|-----------|------|------|----------|--------|-------------|------------|--------|--------|-----------|------|----------|------------|
| None | 116.8 | $3.02 | $337 | $2,229 | 0.90 | 19.5 | $33,984 | $8,687 | $34,000 | $25,500 | 2.49 | 2.20 | $84,530 | 50.3% | 0.0% | 0.0% |
| $5000 | 115.1 | $3.16 | $354 | $2,137 | 2.12 | 15.3 | $27,189 | $8,684 | $27,500 | $17,500 | 3.27 | 2.45 | $88,852 | 50.3% | 2.5% | 0.0% |
| $4000 | 113.7 | $3.39 | $378 | $2,088 | 2.42 | 16.6 | $26,156 | $8,227 | $26,500 | $16,500 | 3.63 | 2.68 | $94,927 | 50.4% | 4.2% | 0.0% |
| $3000 | 111.1 | $3.76 | $413 | $2,022 | 2.81 | 18.6 | $22,776 | $6,979 | $23,000 | $15,500 | 4.55 | 3.02 | $103,701 | 50.6% | 6.9% | 0.0% |
| $2500 | 109.6 | $4.04 | $439 | $1,975 | 3.06 | 20.2 | $20,437 | $6,506 | $20,500 | $15,000 | 5.39 | 3.28 | $110,108 | 50.8% | 7.8% | 2.1% |
| **$2000** | **106.5** | **$4.32** | **$456** | **$1,932** | **3.32** | **21.9** | **$17,945** | **$6,389** | **$18,000** | **$13,000** | **6.37** | **3.48** | **$114,352** | **50.9%** | **10.6%** | **7.3%** |
| $1500 | 102.0 | $4.77 | $483 | $1,875 | 3.63 | 24.3 | $14,653 | $5,954 | $15,000 | $10,500 | 8.27 | 3.78 | $121,140 | 51.2% | 14.3% | 9.7% |
| $1000 | 94.0 | $5.04 | $468 | $1,761 | 4.17 | 29.8 | $10,955 | $5,084 | $11,000 | $8,500 | 10.73 | 3.85 | $117,519 | 51.4% | 22.6% | 15.9% |
| $500 | 78.2 | $5.72 | $444 | $1,508 | 4.89 | 41.0 | $8,199 | $3,986 | $8,500 | $6,500 | 13.60 | 4.16 | $111,543 | 51.9% | 38.8% | 24.1% |

**Key observation:** Every DSL level from $500 to $5,000 produces *higher* annual PnL than the DSL=None baseline. Annual PnL peaks at DSL=$1,500 ($121,140 — 43% above baseline) before declining at tighter levels. The trades removed by DSL are net-negative in aggregate across all threshold levels.

#### 2. Optimal DSL: min_account_survive_all = $18,000

At DSL=$2,000. This is a $16,000 reduction from baseline ($34,000), a 47% decrease. The worst-case split (split 32, test groups 4,7) had max drawdown $17,945 with 30-day drawdown duration and 25% DSL trigger rate — the single binding constraint.

#### 3. Optimal DSL: annual PnL = $114,352

At DSL=$2,000. This is $29,822 *higher* than baseline ($84,530), a 35% increase.

### Secondary Metrics

#### 4. Recovery Sacrifice Table

| DSL | Total Days | Dipped Days | Dipped % | Recovered Days | Sacrifice % |
|-----|-----------|-------------|----------|----------------|-------------|
| $5,000 | 1,809 | 46 | 2.5% | 0 | **0.0%** |
| $4,000 | 1,809 | 76 | 4.2% | 0 | **0.0%** |
| $3,000 | 1,809 | 124 | 6.9% | 0 | **0.0%** |
| $2,500 | 1,809 | 142 | 7.8% | 3 | **2.1%** |
| $2,000 | 1,809 | 192 | 10.6% | 14 | **7.3%** |
| $1,500 | 1,809 | 259 | 14.3% | 25 | **9.7%** |
| $1,000 | 1,809 | 409 | 22.6% | 65 | **15.9%** |
| $500 | 1,809 | 702 | 38.8% | 169 | **24.1%** |

**Critical finding:** At DSL=$2,000, of 192 days that trigger the stop, only 14 (7.3%) would have recovered to positive PnL without DSL. The other 178 days ended negative regardless — DSL is cutting genuinely bad days, not recovery days. The sacrifice rate is zero through $3,000, meaning no day that dipped to -$3K or deeper ever fully recovered within that session.

The inflection point is around $1,000 where sacrifice crosses 15%, and $500 breaches the 20% threshold (24.1%) — this is where DSL starts cutting into days with meaningful recovery potential.

#### 5. Intra-Day PnL Path Statistics (Baseline, DSL=None)

| Statistic | Mean | P5 | P25 | P50 | P75 | P95 | Extreme |
|-----------|------|-----|-----|-----|-----|-----|---------|
| Intra-day min PnL | -$790 | -$3,625 | -$912 | -$282 | -$34 | +$42 | -$21,395 |
| Intra-day max PnL | +$1,138 | — | — | +$551 | — | — | +$22,430 |
| Final PnL | +$337 | — | — | +$219 | — | — | — |
| Trades/day | 116.9 | — | — | 98 | — | — | — |
| Max consec. losses | 8.5 | — | — | 8 | — | — | 27 |

The daily min PnL distribution is heavily left-skewed: median -$282, but P5 = -$3,625 and worst = -$21,395. This fat left tail is precisely what DSL addresses. At $2,000 DSL, roughly 10.6% of days are stopped (those dipping below -$2,000), truncating the P5 and beyond territory.

Maximum consecutive losses per day averages 8 (range up to 27). At $11.24/loss, 8 consecutive losses = -$89.92 — far below even $500 DSL. The streaks that trigger DSL are longer sequences mixed with occasional wins that still net below the threshold.

#### 6. Optimal DSL: Calmar = 6.37

A 156% improvement over baseline (2.49). Driven by both higher numerator (+35% PnL) and lower denominator (-47% drawdown).

#### 7. Optimal DSL: Sharpe = 3.48

A 58% improvement over baseline (2.20). Driven by +35% increase in mean daily PnL and -13% decrease in daily PnL std.

#### 8. Optimal DSL: trades per day = 106.5

An 8.8% decrease from baseline (116.8). On average, 10.3 fewer trades per day — the trades removed on stopped days.

#### 9. Daily PnL Percentiles at Optimal DSL ($2,000)

| Percentile | Value |
|------------|-------|
| p1 | -$2,195 |
| p5 | -$2,050 |
| p10 | -$2,003 |
| p25 | -$449 |
| p50 | +$203 |
| p75 | +$898 |
| p90 | +$2,245 |
| p95 | +$3,969 |
| p99 | +$9,381 |

The left tail is truncated near -$2,000 (p10 = -$2,003, consistent with the DSL mechanism — the triggering trade completes, so final daily PnL can slightly exceed the stop). The distribution is strongly right-skewed (skew=3.32): bounded left at ~-$2,200, unbounded right to +$9,381+ at p99. This is the ideal shape for risk management.

#### 10. DSL Trigger Rate by Level

| DSL | Trigger Rate |
|-----|-------------|
| None | 0.0% |
| $5,000 | 2.5% |
| $4,000 | 4.2% |
| $3,000 | 6.9% |
| $2,500 | 7.8% |
| $2,000 | 10.6% |
| $1,500 | 14.3% |
| $1,000 | 22.6% |
| $500 | 38.8% |

At DSL=$2,000, roughly 1 in 9.4 trading days triggers the stop. Per-split trigger rates range from 0% (splits 27, 29, 35) to 40% (split 11) — highly split-dependent, reflecting regime variation in test periods.

---

## DSL=None vs Recommended DSL ($2,000) Comparison

| Metric | DSL=None | DSL=$2,000 | Delta | % Change |
|--------|----------|------------|-------|----------|
| Trades/day | 116.77 | 106.50 | -10.27 | -8.8% |
| Exp/trade | $3.02 | $4.32 | +$1.30 | +43.2% |
| Daily PnL mean | $337 | $456 | +$119 | +35.3% |
| Daily PnL std | $2,229 | $1,932 | -$298 | -13.4% |
| Skew | 0.90 | 3.32 | +2.42 | +269% |
| Kurtosis | 19.5 | 21.9 | +2.4 | +12.1% |
| DD worst | $33,984 | $17,945 | -$16,038 | -47.2% |
| DD median | $8,687 | $6,389 | -$2,297 | -26.5% |
| min_acct_all | $34,000 | $18,000 | -$16,000 | -47.1% |
| min_acct_95 | $25,500 | $13,000 | -$12,500 | -49.0% |
| Calmar | 2.49 | 6.37 | +3.88 | +156% |
| Sharpe | 2.20 | 3.48 | +1.27 | +57.7% |
| Annual PnL | $84,530 | $114,352 | +$29,822 | +35.3% |
| Win rate | 50.27% | 50.93% | +0.66pp | +1.3% |
| DSL trigger rate | 0.0% | 10.6% | +10.6pp | — |

---

## Account Sizing at Recommended DSL ($2,000)

| Account Size | Splits Surviving | Survival Rate |
|-------------|-----------------|---------------|
| $3,000 | 3/45 | 6.7% |
| $6,000 | 21/45 | 46.7% |
| $8,500 | 33/45 | 73.3% |
| $10,000 | 37/45 | 82.2% |
| $11,000 | 40/45 | 88.9% |
| $13,000 | 44/45 | 97.8% |
| $15,000 | 44/45 | 97.8% |
| **$18,000** | **45/45** | **100%** |

**min_account_survive_all = $18,000** (100% survival across all 45 CPCV splits).
**min_account_survive_95pct = $13,000** (97.8% survival — 44/45 splits).

The survival curve has a sharp jump at $13K→$18K: one outlier split (split 32, DD=$17,945) forces a $5K gap between 95th and 100th percentile survival. Without split 32, 100% survival would be at $13K.

---

## Optimal DSL Selection Rationale

Per the spec's selection rule: the recommended DSL is the **loosest (highest dollar) threshold** achieving all four primary criteria.

| DSL | SC-1 (≤$20K) | SC-2 (≥$50K) | SC-3 (Cal≥2.0) | SC-4 (Sac≤20%) | All Pass? |
|-----|-------------|-------------|---------------|----------------|-----------|
| None | $34,000 FAIL | $84,530 PASS | 2.49 PASS | 0.0% PASS | **NO** |
| $5,000 | $27,500 FAIL | $88,852 PASS | 3.27 PASS | 0.0% PASS | **NO** |
| $4,000 | $26,500 FAIL | $94,927 PASS | 3.63 PASS | 0.0% PASS | **NO** |
| $3,000 | $23,000 FAIL | $103,701 PASS | 4.55 PASS | 0.0% PASS | **NO** |
| $2,500 | $20,500 FAIL | $110,108 PASS | 5.39 PASS | 2.1% PASS | **NO** (by $500) |
| **$2,000** | **$18,000 PASS** | **$114,352 PASS** | **6.37 PASS** | **7.3% PASS** | **YES — SELECTED** |
| $1,500 | $15,000 PASS | $121,140 PASS | 8.27 PASS | 9.7% PASS | YES |
| $1,000 | $11,000 PASS | $117,519 PASS | 10.73 PASS | 15.9% PASS | YES |
| $500 | $8,500 PASS | $111,543 PASS | 13.60 PASS | 24.1% FAIL | **NO** |

$2,000 is the loosest DSL passing all four criteria. $2,500 fails SC-1 by $500 (min_acct=$20,500). $500 fails SC-4 (sacrifice=24.1% > 20%).

**Note:** DSL=$1,500 achieves strictly better metrics on all four primary criteria. If operational preference favors lower account requirements ($15K vs $18K) and higher Calmar (8.27 vs 6.37), $1,500 is viable at the cost of a modestly higher trigger rate (14.3% vs 10.6%) and sacrifice rate (9.7% vs 7.3%).

---

## Resource Usage

| Resource | Actual | Budget | Status |
|----------|--------|--------|--------|
| Wall-clock | 2.75 min | 15 min | 18% of budget |
| Training runs | 90 | 90 | On budget |
| Simulations | 405 | 405 | On budget |
| GPU-hours | 0 | 0 | On budget |

Budget was conservatively set. Efficient local Apple Silicon execution.

---

## Confounds and Alternative Explanations

### 1. The PnL increase demands scrutiny

DSL should trade off PnL for lower drawdown, yet annual PnL *increases* 35%. This is not an artifact — it's explained by the extreme left-tail kurtosis of the baseline daily PnL distribution (kurtosis=19.5). The worst days (intra-day min down to -$21,395) are so catastrophically negative that truncating them raises the daily mean. The recovery sacrifice rate (7.3%) confirms: 92.7% of stopped days end negative even without DSL.

**Mechanistic explanation:** After cumulative daily PnL drops below -$2,000, the remaining trades on that day are executed during a losing regime (trending against the model, high volatility, or regime shift). These late-drawdown trades have systematically negative expected value. Removing them improves the aggregate. The per-trade expectancy increase ($3.02 → $4.32) and win rate increase (50.27% → 50.93%) confirm that the removed trades are below-average quality.

**Skeptical counter:** This mechanism assumes the model's losing regimes are persistent within a day. If intra-day loss clustering is random (i.i.d. trade outcomes), DSL should reduce PnL, not increase it. The positive PnL effect is evidence against i.i.d. trade outcomes — the model's edge is regime-dependent within the trading day. This is plausible but not independently verified.

### 2. In-sample threshold optimization

The $2,000 DSL threshold was selected based on 2022 CPCV results. A different market regime could shift the optimal threshold. However, the broad effectiveness across ALL levels ($500-$5,000 all beat baseline PnL) provides structural robustness — DSL works across a wide range, not just at one magic number. The $2,000 vs $2,500 boundary is sensitive ($500 margin on min_acct) but the qualitative finding (DSL compresses drawdowns substantially with low sacrifice) is robust.

### 3. Single worst-split sensitivity

min_acct_all = $18,000 is determined entirely by split 32 (test groups 4,7, DD=$17,945, 30-day drawdown duration, 25% DSL trigger rate). This represents the most adverse test period across 45 CPCV combinations. The 95th percentile metric ($13,000) is more robust — 44/45 splits survive at $13K. If split 32's test period coincides with an unusually volatile regime, the $18K figure may overstate typical capital requirements. The gap between $13K (95th pct) and $18K (100th pct) is large relative to the median DD ($6,389).

### 4. Seed variance

Not applicable — same seed (42 + split_idx) for all DSL levels within each split. Training is identical. DSL is a pure post-training simulation filter with no stochastic element.

### 5. DSL as real-time implementable mechanism

DSL uses only cumulative realized intra-day PnL — information available in real-time. No lookahead, no test-set labels, no future information. This is a valid live-trading risk control, not a backtest artifact. The only implementation detail: the triggering trade completes before stopping (per spec), so the actual daily loss may slightly exceed the threshold.

### 6. Could the improvement be due to removing correlated losses across splits?

Each CPCV split has different test days, so the 192 "dipped days" at DSL=$2,000 are not the same days across splits. However, truly catastrophic market days (e.g., FOMC surprises, flash crashes) would appear in multiple splits, amplifying DSL's cross-split benefit. This is a feature, not a confound — DSL is designed to protect against exactly such events.

### 7. Annual PnL peaking at DSL=$1,500 — not at the selection point

PnL peaks at $1,500 ($121K) before declining to $118K at $1,000 and $112K at $500. This means at thresholds below $1,500, DSL starts cutting profitable trade volume faster than it removes losing trades. The optimal PnL point and the optimal risk-return point differ — the selection rule correctly optimizes for the risk-return composite (min_acct + Calmar + sacrifice), not raw PnL.

---

## What This Changes About Our Understanding

1. **Daily drawdown is addressable, and dramatically so.** The prior assumption in QUESTIONS.md (Decision Gate for P1 position-level stop-loss) stated "Daily cumulative stop failed structurally (2.6% DD reduction)." This experiment refutes that claim: DD reduction is 47%, not 2.6%. The previous figure likely came from an informal or different analysis. Daily cumulative stop loss is the single most impactful risk intervention discovered in this research program.

2. **The model's worst trades cluster during intra-day drawdowns.** Post-threshold trades are not random draws from the trade distribution — they have systematically negative expected value. This implies the model's edge is regime-dependent within the trading day. When the model is losing, it continues to lose — autocorrelated trade outcomes within days. DSL exploits this autocorrelation.

3. **Account sizing drops to retail-accessible levels.** $18K (all-paths) and $13K (95%-paths) for 1 /MES makes this strategy accessible to the retail MES trading segment. Previously at $34K/$25.5K, many retail accounts were excluded.

4. **DSL stacks with time-cutoff.** The baseline already includes cutoff=270 from PR #40. DSL addresses a different dimension (intra-day loss magnitude vs time-of-day). The interventions are complementary, not redundant.

5. **Per-trade expectancy improves as a bonus.** $3.02 → $4.32 (+43%) means DSL is not merely a risk control — it's an edge quality filter. The removed trades dilute per-trade expectancy. This has implications for Kelly sizing: higher per-trade expectancy with lower variance supports more aggressive position sizing.

---

## Proposed Next Experiments

1. **Multi-year DSL robustness** — Apply DSL=$2,000 to 2021/2023 data (if available) or walk-forward out-of-sample periods. Key question: does the $2K threshold remain effective in different volatility regimes? The broad effectiveness across $500-$5K provides structural confidence, but the specific optimal threshold may shift.

2. **Position-level stop-loss (intra-trade)** — Now that daily cumulative stop is confirmed highly effective, the complementary question is whether intra-trade unrealized loss limits provide additional compression. Given the 47% DD reduction already achieved, the marginal benefit of position-level stops may be small. This remains blocked on exit_bar column in Parquet.

3. **Multi-contract scaling with DSL** — The $18K min account at 1 /MES could support 2+ contracts at higher account sizes. DSL threshold should scale linearly with contract count. How does the risk profile evolve with scaling?

4. **Regime-conditional DSL** — Instead of fixed $2K, adapt the threshold to intra-day realized volatility (e.g., 2x session ATR). Fixed thresholds may be too aggressive on volatile days and too loose on calm ones.

5. **Paper trading with DSL** — The strategy profile (Calmar 6.37, Sharpe 3.48, $114K/yr on 1 MES, $18K min account) is the strongest produced by this research program. The natural next step is forward validation via paper trading with R|API+.

---

## Program Status

- Questions answered this cycle: 0 directly (no §4 question maps exactly, but strongly informs P1 position-level stop-loss Decision Gate)
- New questions added this cycle: 1 (multi-year DSL robustness)
- Questions remaining (open, not blocked): 5
- Handoff required: NO
