# Analysis: Timeout-Filtered Sequential Execution

## Verdict: REFUTED

Both primary success criteria fail (SC-2: $3.02 < $3.50 threshold; SC-3: $34K > $30K threshold), and the hypothesis's core mechanism is empirically falsified: **timeout fraction is invariant at ~41.3% across all 7 cutoff levels** (range 0.4119–0.4140, total variation 0.21pp). Time-of-day filtering does NOT reduce timeouts. The expectancy improvement at the aggressive cutoff=270 (+$0.52/trade) is driven by a trade selection effect unrelated to timeout avoidance.

---

## Results vs. Success Criteria

- [x] **SC-1: PASS** — All 7 cutoffs x 45 splits = 315 simulations completed
- [ ] **SC-2: FAIL** — Optimal cutoff exp $3.016 < $3.50 threshold (deficit: $0.48/trade, 14% short)
- [ ] **SC-3: FAIL** — Optimal cutoff min_acct $34,000 > $30,000 threshold (excess: $4,000, 13% over)
- [x] **SC-4: PASS** — Sweep table fully populated (7 rows x 14 columns)
- [x] **SC-5: PASS** — Unfiltered vs filtered comparison populated
- [x] **SC-6: PASS** — Account sizing curve at recommended cutoff ($500–$50K)
- [x] **SC-7: PASS** — All 12 output files written to `.kit/results/timeout-filtered-sequential/`
- [x] **SC-8: PASS** — Bar-level split 0 exp $1.0652 matches PR #38 reference $1.065186 (delta < $0.01)
- [ ] **SC-S1: PASS** — $1.0652 vs $1.065186 (delta $0.00)
- [ ] **SC-S2: PASS** — Cutoff=390 exp $2.5008 vs PR #39 reference $2.50 (delta $0.01)
- [ ] **SC-S3: PASS** — Cutoff=390 trades/day 162.21 vs PR #39 reference 162.2 (delta 0.01)
- [ ] **SC-S4: FAIL** — Expectancy NOT monotonically non-decreasing. U-shaped: drops $2.50→$2.27 (cutoffs 390→330), then rises $2.48→$3.02 (cutoffs 300→270)
- [ ] **SC-S5: PASS** — Trades/day strictly non-increasing (162.2 → 116.8)

**Summary: 6/8 SC pass, 4/5 sanity checks pass. SC-2, SC-3 fail. SC-S4 fails (non-monotonic expectancy). Outcome B per spec decision rules.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. Cutoff Sweep Table (7 rows x 14 columns)

| Cutoff | RTM Rem | Trades/Day | Exp/Trade | Daily PnL | DD Worst | DD Median | Min Acct All | Min Acct 95% | Win Rate | Time Skip% | Hold Skip% | Timeout Frac | Calmar |
|--------|---------|------------|-----------|-----------|----------|-----------|-------------|-------------|----------|------------|------------|-------------|--------|
| 390 | 0 | 162.2 | $2.50 | $413 | $47,894 | $12,917 | $48,000 | $26,500 | 49.93% | 0.0% | 66.1% | 41.33% | 2.16 |
| 375 | 15 | 158.9 | $2.49 | $400 | $47,947 | $12,917 | $48,000 | $27,000 | 49.92% | 6.7% | 63.4% | 41.35% | 2.10 |
| 360 | 30 | 155.8 | $2.46 | $384 | $47,989 | $12,922 | $48,000 | $26,500 | 49.89% | 12.9% | 60.8% | 41.40% | 2.01 |
| 345 | 45 | 153.1 | $2.28 | $346 | $47,923 | $12,906 | $48,000 | $30,000 | 49.84% | 17.0% | 60.3% | 41.40% | 1.81 |
| 330 | 60 | 148.6 | $2.27 | $331 | $48,089 | $13,331 | $48,500 | $32,000 | 49.94% | 22.9% | 59.2% | 41.29% | 1.73 |
| 300 | 90 | 131.3 | $2.48 | $318 | $43,202 | $12,596 | $43,500 | $30,500 | 50.03% | 44.9% | 44.1% | 41.19% | 1.85 |
| **270** | **120** | **116.8** | **$3.02** | **$337** | **$33,984** | **$8,687** | **$34,000** | **$25,500** | **50.27%** | **58.0%** | **34.4%** | **41.30%** | **2.49** |

#### 2. Optimal Cutoff Expectancy: $3.016/trade

Selected by Rule 2 (no cutoff achieves both SC-2 and SC-3 simultaneously; maximize daily PnL subject to min_acct <= $35K). Cutoff=270 is the only level meeting the relaxed $35K constraint.

**Note on selection rule ambiguity:** The spec states Rule 2 as "maximize daily PnL subject to min_acct <= $35K." Cutoff=270 daily PnL ($337) is actually LOWER than cutoff=300 ($318). Wait — cutoff=270 ($337) > cutoff=300 ($318), so 270 does maximize daily PnL among levels meeting $35K. The selection is correct.

#### 3. Optimal Cutoff Min Account: $34,000

Misses $30K target by $4,000 (13%). The $34K is driven by split 18's max drawdown of $33,984.

### Secondary Metrics

#### 4. Optimal Daily PnL: $336.77/day
Down 18.4% from baseline $412.77. The trade-off is 28% fewer trades producing 20.6% higher per-trade expectancy — the per-trade improvement doesn't fully compensate for lost volume.

#### 5. Optimal Calmar: 2.49
Best among all cutoffs. Up 15% from baseline 2.16. Drawdown compression (-29%) exceeds PnL reduction (-18%), producing the best risk-adjusted daily returns.

#### 6. Optimal Sharpe: 2.20
Up 4.9% from baseline 2.10. Modest improvement — Sharpe is less sensitive to tail compression than Calmar.

#### 7. Optimal Trades/Day: 116.8
Down 28% from 162.2. Still substantial volume — 117 trades/day at $3.02/trade is economically meaningful.

#### 8–9. Timeout and Barrier Hit Fractions by Cutoff

**This is the single most important finding of the experiment.**

| Cutoff | Timeout Fraction | Delta vs. 390 |
|--------|-----------------|---------------|
| 390 | 0.41329 | — |
| 375 | 0.41350 | +0.00021 |
| 360 | 0.41398 | +0.00069 |
| 345 | 0.41405 | +0.00076 |
| 330 | 0.41286 | -0.00043 |
| 300 | 0.41191 | -0.00138 |
| 270 | 0.41298 | -0.00031 |

**Total range: 0.21 percentage points (0.4119 to 0.4140).** This is noise. Timeouts are uniformly distributed across the RTH session. The hypothesis's core premise — that timeouts cluster in late-day entries where insufficient time remains for barrier resolution — is empirically falsified.

**Why timeouts are time-invariant:** The average barrier race at 19:7 geometry resolves in ~28 bars (2.3 minutes). Even at cutoff=270 (entering up to 4.5h into the 6.5h session), there are still ~2 hours remaining — far more than the typical 2.3-minute race duration. Timeouts are caused by the *volume horizon* (50,000 contracts) expiring, not by running out of clock time. Since the volume horizon is activity-driven rather than clock-driven, filtering by time-of-day cannot reduce timeout incidence.

#### 10. Splits 18 & 32 Comparison

| Cutoff | Split 18 Exp | Split 18 DD | Split 32 Exp | Split 32 DD |
|--------|-------------|-------------|-------------|-------------|
| 390 | -$2.28 | $47,894 | -$5.71 | $46,888 |
| 330 | -$3.07 | $48,089 | -$5.87 | $46,964 |
| 270 | -$1.08 | $33,984 | -$4.77 | $32,662 |

Both outlier splits improve substantially at cutoff=270:
- Split 18: DD drops $13,910 (-29%), exp improves by $1.20
- Split 32: DD drops $14,226 (-30%), exp improves by $0.94
- Both remain deeply negative-expectancy (group 4 problem persists)
- Timeout fraction on outlier splits also invariant: 0.396–0.407 at cutoff=270 vs 0.407–0.409 at cutoff=390

The drawdown improvement on outlier paths is what drives min_acct from $48K to $34K. The improvement is from removing low-quality trades (reducing cumulative loss), not from reducing timeouts.

#### 11. Daily PnL Percentiles at Cutoff=270

| Percentile | Value |
|------------|-------|
| p5 | -$3,171 |
| p25 | -$418 |
| p50 | +$219 |
| p75 | +$924 |
| p95 | +$3,969 |

Positive median ($219) and positive-skewed upper tail. The p5 left tail (-$3,171) at ~117 trades/day implies worst-day loss of ~$27/trade — within the $8.75 stop loss × loss-clustering factor.

#### 12. Annual Expectancy at Optimal: $84,530

Down from $103,605 (baseline) — a $19,075/year reduction (-18.4%) from filtering out 28% of trades.

### Sanity Checks

- **SC-S1: PASS** — Bar-level split 0 exp $1.0652 vs PR #38 $1.065186. Delta < $0.001. Training code is identical.
- **SC-S2: PASS** — Cutoff=390 exp $2.5008 vs PR #39 $2.50. Delta $0.01. Unfiltered simulation reproduces exactly.
- **SC-S3: PASS** — Cutoff=390 trades/day 162.21 vs PR #39 162.2. Delta 0.01. Simulation logic unchanged.
- **SC-S4: FAIL** — Expectancy is NOT monotonically non-decreasing. Pattern is U-shaped: $2.50 → $2.49 → $2.46 → $2.28 → $2.27 → $2.48 → $3.02. Expectancy DECREASES for mild cutoffs (375–330), then recovers aggressively at 300–270. This means intermediate-late-day trades (entered 5–6h into session) have BELOW-average expectancy, while very-late-day trades (5.5–6.5h) have above-average expectancy. The non-monotonicity reveals a more complex time-of-day structure than "late = bad."
- **SC-S5: PASS** — Trades/day strictly non-increasing: 162.2 → 158.9 → 155.8 → 153.1 → 148.6 → 131.3 → 116.8. Monotonically decreasing as expected (tighter filter can only remove trades).

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| Wall clock | 15 min | 2.66 min | Well under (18% of budget) |
| Training runs | 90 | 90 | As planned |
| Simulations | 315 | 315 | As planned |
| GPU hours | 0 | 0 | CPU-only as expected |

Budget was appropriate. Apple Silicon handled 45 splits × 2-stage XGBoost + 315 sequential simulations in under 3 minutes.

---

## Confounds and Alternative Explanations

### 1. Timeout fraction invariance needs an explanation

The invariant timeout fraction (~41.3% ± 0.1pp) is the central surprise. Two competing explanations:

**Explanation A (volume horizon dominance):** Timeouts are triggered by the volume horizon (50,000 contracts), not by running out of clock time. Since daily MES volume is ~150K–250K contracts and the horizon is 50,000, even entries at minute 270 have sufficient volume remaining for barrier resolution. The volume horizon expires stochastically based on market activity patterns, not clock position.

**Explanation B (replacement effect):** The sequential simulator enters the FIRST available signal after exit. When a late-day entry is filtered out, the simulator doesn't enter; but it also doesn't enter at the *next* available bar either (because the filter blocks all bars after the cutoff). The trades that survive are the SAME early-day trades regardless of cutoff — the only difference is where the sequence truncates. Trades 1–N are identical; cutoff only determines whether trades N+1, N+2, etc. occur. Since the timeout rate of the first N trades is the same as the population rate, the fraction is invariant.

Explanation B is more likely given the sequential execution model. The time filter is a truncation of the sequence's tail, not a random sampling of non-timeout trades.

### 2. The expectancy U-shape is a composition effect, not a quality gradient

The U-shape (exp drops 390→330, then rises 300→270) does NOT mean "very late trades are bad, extremely late trades are good." It means:

- **Mild cutoffs (375–345):** Remove a few late-day trades. The removed trades had slightly above-average expectancy (hence average drops). This contradicts the "late = bad" premise.
- **Aggressive cutoffs (300–270):** Remove many late-day trades AND trigger a cascade effect. With fewer entry opportunities, the sequential simulator is more selective — hold-skip rate drops from 66.1% to 34.4% (the simulator is "less picky" because there are fewer chances). The surviving trades trade during the higher-activity first 4.5 hours of the session, when volatility and barrier resolution rates may differ.

The composition changes. It's not the same population minus bad trades — it's a different population of trades entirely.

### 3. Cutoff=270 improvement may be driven by the hold-skip rate change

At cutoff=390, hold-skip rate = 66.1%. At cutoff=270, hold-skip rate = 34.4%. This 32pp change in hold-skip behavior means the simulator at cutoff=270 is accepting trades it would have skipped at cutoff=390 (because it enters at bars where it's currently flat but wouldn't have been flat if the late-day continuation had happened). The expectancy improvement may reflect the quality of these "new" entry points rather than the removal of late-day entries.

### 4. Seed variance at aggressive cutoffs

Cutoff=270 removes 28% of trades. With fewer trades per split, per-split metrics have higher variance. The $3.02 per-trade expectancy is an average across 45 splits — some splits may be substantially negative. The report doesn't provide per-split variance at cutoff=270. The improvement from $2.50 to $3.02 (+$0.52) should be evaluated against the cross-split standard deviation.

### 5. min_account is dominated by 2 paths

min_account_all = $34K is determined by split 18 alone (DD = $33,984). If split 18 had DD of $30K instead of $34K, SC-3 would pass. This is a 1-split sensitivity. min_account_95% = $25,500 (already well below $30K) — 95% of paths need only $25.5K.

### 6. Annual PnL trade-off may not favor filtering

Baseline: $103,605/year at $48K account = 216% annual return on capital.
Filtered: $84,530/year at $34K account = 249% annual return on capital.

ROC is HIGHER with filtering (+33pp) despite lower absolute PnL. The filtering improves capital efficiency even though it reduces gross revenue.

---

## What This Changes About Our Understanding

### 1. Timeouts are NOT a time-of-day phenomenon

The hypothesis assumed timeouts cluster in late-day entries. They don't. Timeout fraction is invariant at ~41.3% regardless of when the trade is entered (within RTH). This means the volume horizon — not the clock — determines timeout incidence. The 50,000-contract volume horizon is the binding constraint, and it operates independently of time-of-day.

**Implication:** Any future timeout reduction strategy must target the volume horizon mechanism, not the clock. Options include: (a) increasing the volume horizon (but this extends barrier races), (b) conditioning entry on recent volume flow (enter when volume is high, so the horizon is consumed faster), (c) switching to a pure time horizon (1 hour max hold).

### 2. The sequential model's expectancy is driven by volatility-timing, and the time cutoff changes the timing

The expectancy improvement at cutoff=270 (+$0.52) doesn't come from avoiding timeouts. It comes from concentrating trading in the first 4.5 hours of the session, when the sequential simulator's selection patterns interact differently with market microstructure. The hold-skip rate drops from 66.1% to 34.4%, meaning the simulator takes a higher fraction of available signals during the morning session.

### 3. Account sizing is path-dependent, not expectancy-dependent

SC-3 failed despite the 29% drawdown reduction because the outlier paths (splits 18, 32) remain deeply negative. Split 18 still loses $1.08/trade at cutoff=270 (improved from -$2.28 but still negative). The $34K minimum is driven by the cumulative loss on these 2 paths, not by per-trade expectancy on the average path. Reducing min_account below $30K requires either (a) making splits 18/32 less negative, or (b) accepting that some CPCV paths will blow up and using 95% survival ($25.5K) as the sizing target.

### 4. Return on capital is better with the filter

Despite losing $19K/year in absolute PnL, the filtered strategy at cutoff=270 is more capital-efficient: 249% ROC vs 216% ROC. For a single MES contract, the $14K lower capital requirement more than compensates for the $19K lower PnL. This trade-off favors filtering for small accounts but disfavors it for accounts where capital is not the binding constraint.

---

## Proposed Next Experiments

### 1. Volume-flow conditioned entry (highest priority)

Since timeouts are driven by the volume horizon (not clock time), condition entry on recent volume flow. Hypothesis: entries during high-volume periods have lower timeout rates because the 50,000-contract horizon is consumed faster, reaching barrier resolution before stochastic expiry. IV: rolling N-bar volume quantile at entry time. This is a direct attack on the timeout mechanism, unlike the time-of-day proxy which this experiment proved ineffective.

### 2. Pure time horizon analysis at $34K account size

PR #39 showed avg_bars_held = 28 (2.3 min). The time-horizon CLI is now available (--max-time-horizon). Re-export with a 30-minute or 1-hour max time horizon instead of the volume horizon. If clock-based timeouts concentrate differently than volume-based ones, this may be more filterable.

### 3. 95th-percentile account sizing as the operational target

min_acct_95% = $25.5K (already below $30K target) vs min_acct_all = $34K. The $8.5K gap is driven by 2-3 outlier CPCV paths. Evaluate whether 95% survival is an acceptable operational standard (the 5% failure paths may correspond to extreme regime conditions that would trigger manual risk management).

### 4. Volatility-conditional entry filter

Instead of time-of-day, condition on volatility_50 at entry time. Hypothesis: low-volatility entries have longer barrier races and higher timeout rates. volatility_50 is the dominant XGBoost feature (49.7% gain share) and directly relates to barrier reachability.

---

## Cutoff Sweep Details

### Timeout Fraction Analysis

The timeout fraction across all 7 cutoff levels:

| Cutoff | Timeout Frac | Barrier Hit Frac | Delta vs 390 |
|--------|-------------|------------------|-------------|
| 390 | 0.41329 | 0.58671 | — |
| 375 | 0.41350 | 0.58650 | +0.00021 |
| 360 | 0.41398 | 0.58602 | +0.00069 |
| 345 | 0.41405 | 0.58595 | +0.00076 |
| 330 | 0.41286 | 0.58714 | -0.00043 |
| 300 | 0.41191 | 0.58809 | -0.00138 |
| 270 | 0.41298 | 0.58702 | -0.00031 |

**Conclusion: Time-of-day cutoff does NOT reduce timeouts.** The 0.21pp total range is noise. This directly invalidates the hypothesis's mechanism. The expectancy improvement at cutoff=270 is a side effect of trade composition changes, not timeout avoidance.

### Unfiltered vs. Filtered Comparison (cutoff=390 vs cutoff=270)

| Metric | Unfiltered (390) | Filtered (270) | Delta | % Change |
|--------|-----------------|----------------|-------|----------|
| Trades/day | 162.2 | 116.8 | -45.4 | -28.0% |
| Exp/trade | $2.50 | $3.02 | +$0.52 | +20.6% |
| Daily PnL | $412.77 | $336.77 | -$76.00 | -18.4% |
| Annual PnL | $103,605 | $84,530 | -$19,075 | -18.4% |
| DD worst | $47,894 | $33,984 | -$13,910 | -29.0% |
| DD median | $12,917 | $8,687 | -$4,230 | -32.7% |
| Min acct (all) | $48,000 | $34,000 | -$14,000 | -29.2% |
| Min acct (95%) | $26,500 | $25,500 | -$1,000 | -3.8% |
| Win rate | 49.93% | 50.27% | +0.34pp | +0.7% |
| Timeout frac | 41.33% | 41.30% | -0.03pp | -0.1% |
| Calmar | 2.16 | 2.49 | +0.32 | +15.0% |
| Sharpe | 2.10 | 2.20 | +0.10 | +4.9% |
| Hold-skip rate | 66.1% | 34.4% | -31.7pp | -48.0% |
| Avg bars held | 28.0 | 29.0 | +1.0 | +3.7% |

### Account Sizing at Cutoff=270

| Threshold | Account Size |
|-----------|-------------|
| All 45 paths survive | $34,000 |
| 95% paths survive | $25,500 |
| ROC (all-path basis) | 249% annual |
| ROC (95%-path basis) | 331% annual |

### Outlier Paths (Splits 18 & 32) at Selected Cutoffs

| Split | Cutoff=390 Exp | Cutoff=390 DD | Cutoff=270 Exp | Cutoff=270 DD | DD Change |
|-------|---------------|---------------|---------------|---------------|-----------|
| 18 | -$2.28 | $47,894 | -$1.08 | $33,984 | -29.1% |
| 32 | -$5.71 | $46,888 | -$4.77 | $32,662 | -30.3% |

Both outlier paths remain deeply negative-expectancy. The drawdown reduction is proportional (~29-30%), suggesting the filtering uniformly compresses the equity curve rather than selectively removing the worst trades. Split 32's timeout fraction at cutoff=270 (0.396) is almost identical to cutoff=390 (0.409) — confirming the timeout invariance holds even on the worst paths.

### Daily PnL Distribution at Cutoff=270

| Percentile | Value |
|------------|-------|
| p5 | -$3,171 |
| p25 | -$418 |
| p50 | +$219 |
| p75 | +$924 |
| p95 | +$3,969 |

Positive-skewed distribution. Median positive ($219). IQR = $1,342. The p95/p5 ratio is 1.25 (roughly symmetric tails). 75%+ of days are positive (p25 is -$418, well within one-day recovery).

---

## SC-S4 Monotonicity Violation — Detailed Interpretation

The expectancy pattern across cutoffs is U-shaped:

```
$2.50 → $2.49 → $2.46 → $2.28 → $2.27 → $2.48 → $3.02
 390     375     360     345     330     300     270
```

This has three regimes:
1. **Mild filter (375–360):** Marginal expectancy decline. The 15–30 minutes of removed late-day trades had slightly above-average quality.
2. **Moderate filter (345–330):** Accelerated decline. The 45–60 minutes of removed trades were materially above-average. This is the "sweet spot" of late-day trading where barrier races benefit from end-of-day directional moves.
3. **Aggressive filter (300–270):** Sharp recovery. Removing 90–120 minutes restructures the trading population — hold-skip drops from 66%→34%, the simulator enters different (and apparently higher-quality) trades during the morning session.

The SC-S4 failure confirms: **late-day trades are NOT uniformly worse.** The intermediate period (5–5.75h into session) appears to be BETTER than average for this strategy. The improvement at cutoff=270 is a population restructuring effect, not a quality gradient.

---

## Outcome Verdict

**Outcome B — SC-1/4/5/6/7/8 pass, SC-2 and SC-3 fail.**

With critical Outcome C mechanistic overlay: **the timeout fraction is invariant, invalidating the hypothesis's causal mechanism.** The improvement at cutoff=270 is real (Calmar +15%, drawdown -29%) but operates through trade composition changes (hold-skip restructuring), not timeout avoidance.

The spec's Outcome B decision rule states:
> "If SC-2 passes but SC-3 fails: expectancy improves but drawdown is structural."
> "If SC-3 passes but SC-2 fails: drawdown compresses but per-trade edge doesn't improve."

Neither subcase applies — both SC-2 AND SC-3 fail. The result is: time-of-day filtering provides meaningful risk-adjusted improvement (ROC 216%→249%) but cannot reach either absolute target. The improvement is a windfall from the composition change, not the intended timeout reduction.

**Recommended next steps per spec:** Volatility-conditional filtering (entry-time volatility_50 threshold) or accept the $34K/$25.5K account requirement as-is.

---

## Program Status

- Questions answered this cycle: 1 (timeout-filtered sequential — REFUTED)
- New questions added this cycle: 1 (volume-flow conditioned entry — replaces time-of-day with the correct mechanism)
- Questions remaining (open, not blocked): 5 (long-perspective labels P0, message regime-stratified P2, cost sensitivity P2, regime-conditional P3, multi-contract P2, volume-flow conditioned entry)
- Handoff required: NO
