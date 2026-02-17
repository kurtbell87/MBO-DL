# Kenoma Labs — MES Backtest & Feature Discovery Spec
## Oracle Expectancy, Event-Driven Bars, Representation Theory

**Version**: 0.3 (Draft)
**Date**: 2026-02-16
**Author**: Brandon Bell / Kenoma Labs LLC
**Predecessor**: Model Orchestrator Spec v0.6 (retired to `past_specs/`). The overfit harness validated pipeline correctness (all 30/30 exit criteria met, 2026-02-16). This spec builds on that validated C++ infrastructure.
**Changes from v0.2**: Replaced fixed ε thresholds with relative + significance-tested thresholds (§2.2, §R2). Added triple barrier oracle as co-primary labeling method (§5). Revised volume horizon safety cap logic (§5.1). Added raw message sequence model to R2 protocol (§R2). Moved Kyle's lambda to rolling multi-bar computation (§8.3). Added explicit warm-up and lookahead bias policy (§8.6). Added regime stratification to backtest analysis (§9.6). Added power analysis and multiple comparison framework (§8.7). Specified per-day memory model for MBO events (§10.3). Refactored Bar struct to separate data from encoder input format (§4.1). Added rollover volume migration handling (§9.3). Noted VPIN circularity in volume-bar context (§8.3 Category 6).

---

## 1. Purpose

Before training models to replicate the oracle, determine whether the oracle's signals are worth replicating. Before designing model architectures, determine what representational structure the data demands.

1. **Oracle expectancy**: Does the oracle produce positive expectancy after realistic MES execution costs across out-of-sample days?
2. **Bar construction**: Replace fixed-time (100ms) sampling with event-driven bars (volume, tick, dollar) and measure the effect on signal quality.
3. **Feature discovery**: Identify which features are predictive of direction or returns, at what event-denominated scales, and with what decay characteristics.
4. **Representation architecture validation**: Quantify the information content of raw book snapshots, intra-bar message sequences, and hand-crafted features — to determine how much each encoder stage (spatial, message, temporal) contributes to predictive power.

This spec produces the empirical evidence needed to make principled design decisions for the eventual model architecture.

---

## 2. Theoretical Foundations

This section formalizes three ideas that constrain the system design. The math isn't decorative — each result maps to a concrete design decision.

### 2.1 Subordinated Processes and Event-Driven Sampling

**Claim**: Price returns sampled at event-driven boundaries (volume, tick) are closer to IID Gaussian than returns sampled at fixed time intervals.

**Setup**: Let $P(t)$ be the mid-price process in calendar time $t$. Let $N(t)$ be a counting process representing cumulative market activity (e.g., cumulative volume traded). Define *business time* $\tau$ as the inverse of $N$: the $k$-th bar boundary occurs at calendar time $t_k = \inf\{t : N(t) \geq kV\}$ for volume bars with threshold $V$.

The price process in business time is the *subordinated process*:

$$
\tilde{P}(k) = P(t_k)
$$

Returns in business time:

$$
\tilde{r}_k = \tilde{P}(k+1) - \tilde{P}(k) = P(t_{k+1}) - P(t_k)
$$

**Why this helps** (Clark 1973, Ané & Geman 2000): If we model $P(t) = W(N(t))$ where $W$ is a Wiener process in business time and $N(t)$ is a stochastic time change, then the calendar-time returns $r_{\Delta t} = P(t + \Delta t) - P(t)$ are a *mixture of normals* — Gaussian conditional on the realized activity $N(t+\Delta t) - N(t)$, but unconditionally fat-tailed and heteroskedastic because $N$ varies. Sampling at fixed increments of $N$ (i.e., volume bars) conditions out the stochastic time change:

$$
\tilde{r}_k = W(kV + V) - W(kV) \sim \mathcal{N}(0, \sigma^2 V)
$$

This is exact if the subordination model holds, and approximately true to the extent that volume is a good proxy for information arrival rate.

**Testable predictions** (implemented in §8.5):
- Jarque-Bera statistic on $\{\tilde{r}_k\}$ (volume bars) < Jarque-Bera on $\{r_{\Delta t}\}$ (time bars) for matched sample sizes
- ARCH(1) coefficient on $\{\tilde{r}_k\}$ < ARCH(1) coefficient on $\{r_{\Delta t}\}$
- Autocorrelation of $|\tilde{r}_k|$ decays faster than autocorrelation of $|r_{\Delta t}|$ (volatility clustering is reduced)

**Design decision**: If volume bars produce significantly more Gaussian, less heteroskedastic returns than time bars on MES data, this validates event-driven sampling as the primary bar type. If the effect is weak, the subordination model is a poor fit for MES microstructure (possible — MES is a derivative, and its information arrival may be driven by ES, not its own order flow).

### 2.2 Information Decomposition Across Encoder Stages

**Claim**: The predictive information about future returns decomposes across three data sources (book state, message sequence, temporal history), and this decomposition maps directly to the three encoder stages of the target architecture.

**Setup**: Let $r_{t+h}$ denote the forward return $h$ bars ahead. At bar $t$, we observe:

- $\mathbf{B}_t$ — the order book state at bar boundary (10 bid levels, 10 ask levels with prices and sizes)
- $\mathbf{M}_t$ — the sequence of MBO messages within bar $t$ (Adds, Cancels, Modifies, Trades, Fills)
- $\mathbf{H}_t = (\mathbf{B}_{t-1}, \mathbf{M}_{t-1}, \mathbf{B}_{t-2}, \mathbf{M}_{t-2}, \ldots)$ — the history of previous bars

The total predictive information is:

$$
I(\mathbf{B}_t, \mathbf{M}_t, \mathbf{H}_t\,;\, r_{t+h})
$$

By the chain rule for mutual information:

$$
I(\mathbf{B}_t, \mathbf{M}_t, \mathbf{H}_t\,;\, r_{t+h}) = \underbrace{I(\mathbf{B}_t\,;\, r_{t+h})}_{\text{spatial (CNN)}} + \underbrace{I(\mathbf{M}_t\,;\, r_{t+h} \mid \mathbf{B}_t)}_{\text{message encoder}} + \underbrace{I(\mathbf{H}_t\,;\, r_{t+h} \mid \mathbf{B}_t, \mathbf{M}_t)}_{\text{temporal (SSM)}}
$$

Each term maps to an encoder:

| Term | Meaning | Encoder | Architecture |
|------|---------|---------|-------------|
| $I(\mathbf{B}_t\,;\, r_{t+h})$ | How much does the current book shape predict? | Spatial encoder | CNN on price ladder |
| $I(\mathbf{M}_t\,;\, r_{t+h} \mid \mathbf{B}_t)$ | What do intra-bar messages add beyond the resulting book state? | Message encoder | Small transformer/SSM on MBO events |
| $I(\mathbf{H}_t\,;\, r_{t+h} \mid \mathbf{B}_t, \mathbf{M}_t)$ | What does temporal context add beyond current state? | Temporal encoder | SSM/Transformer on bar embedding sequence |

**Key insight**: If $I(\mathbf{M}_t\,;\, r_{t+h} \mid \mathbf{B}_t) \approx 0$, the message encoder adds no value — the book state is a *sufficient statistic* for the message sequence with respect to future returns. In that case, the two-level architecture (CNN + SSM) is optimal and the message encoder is wasted computation.

This is an empirical question. Intuitively, the book state at bar close is the *result* of the message sequence, but it discards the *path* — a bar where liquidity was pulled and replaced 5 times looks identical at close to a bar where the book was stable. Whether the path carries predictive information beyond the endpoint is testable.

**Estimation strategy** (§8.4 Research Cycle R2):

Direct MI estimation between $\mathbf{M}_t$ and $r_{t+h}$ is intractable (message sequences are variable-length, high-dimensional). Instead, use a **two-tier proxy** — hand-crafted message summaries for the cheap test, and a learned sequence model for the definitive test:

**Tier 1 (cheap, may underestimate):**
1. Train a simple model (linear or shallow MLP) to predict $r_{t+h}$ from hand-crafted bar features $\mathbf{f}_t^{\text{bar}}$ (Category 1-5 features from §8.3). Record $R^2_{\text{bar}}$.
2. Train the same model to predict $r_{t+h}$ from the raw book snapshot $\mathbf{B}_t$ (flattened 20×2 = 40 features, or CNN embedding). Record $R^2_{\text{book}}$.
3. Train the same model with additional hand-crafted intra-bar message summary features (Category 6). Record $R^2_{\text{book+msg\_summary}}$.
4. Train with lookback window of previous bars' features. Record $R^2_{\text{full\_summary}}$.

**Tier 2 (expensive, definitive):**
5. Train a small LSTM (1 layer, 32 hidden units) or 1-layer transformer (2 heads, d=32) on the raw intra-bar MBO event sequence, conditioned on the book snapshot embedding. Record $R^2_{\text{book+msg\_learned}}$.
6. If $R^2_{\text{book+msg\_learned}} - R^2_{\text{book}} > R^2_{\text{book+msg\_summary}} - R^2_{\text{book}}$ by a meaningful margin, the hand-crafted summaries are failing to capture message-level signal, and the message encoder is justified even if Tier 1 suggested otherwise.

The gaps $R^2_{\text{book}} - R^2_{\text{bar}}$ quantify information lost by hand-crafting. The Tier 2 gap $R^2_{\text{book+msg\_learned}} - R^2_{\text{book}}$ is the definitive test of the message encoder's potential contribution.

**Threshold policy** (revised from v0.2): Fixed absolute ε thresholds are unreliable in noisy microstructure data where baseline R² may itself be small. Instead:

- **Relative threshold**: An encoder stage is justified if the R² gap exceeds 20% of the baseline R². For example, if $R^2_{\text{book}} = 0.005$, the message encoder is justified if it adds $> 0.001$ (not a fixed $0.01$).
- **Statistical threshold**: The R² gap must be statistically significant across CV folds. Use a paired t-test (or Wilcoxon signed-rank if non-normal) on the per-fold R² values, requiring $p < 0.05$ after Holm-Bonferroni correction for the number of encoder comparisons tested.
- **Both conditions must hold**: a gap that is large but inconsistent across folds (high variance) doesn't justify architectural complexity. A gap that is tiny but statistically significant doesn't justify the compute cost.

### 2.3 Sufficient Statistics and the CNN Embedding

**Claim**: For the spatial encoder to be well-designed, its embedding should be approximately sufficient for the book state with respect to future returns.

**Formal statement**: A function $f : \mathcal{B} \to \mathbb{R}^d$ (the CNN embedding) is a *sufficient statistic* for $\mathbf{B}_t$ with respect to $r_{t+h}$ if:

$$
r_{t+h} \perp \mathbf{B}_t \mid f(\mathbf{B}_t)
$$

That is, once you know the embedding, the raw book provides no additional predictive information. By the data processing inequality:

$$
I(f(\mathbf{B}_t)\,;\, r_{t+h}) \leq I(\mathbf{B}_t\,;\, r_{t+h})
$$

Equality holds iff $f$ is sufficient. In practice, we want to get close to equality with a low-dimensional $f$ — this is a compression problem.

**Testable implication**: After training the CNN, compare $R^2$ of predicting $r_{t+h}$ from the CNN embedding vs. from the raw 40-dimensional book vector. If the CNN embedding achieves comparable $R^2$ with $d \ll 40$ dimensions, it's compressing without losing predictive information. If $R^2$ drops substantially, the CNN architecture is discarding signal (wrong inductive bias, insufficient capacity, or training issue).

**Design implication for the price ladder**: The v0.6 spec's CNN constructs a price ladder:

```
[bid[9], bid[8], ..., bid[0], ask[0], ..., ask[9]]
```

This ordering imposes a spatial inductive bias: adjacent elements in the convolution kernel correspond to adjacent price levels. The Conv1d kernel learns local patterns (e.g., "thin level flanked by thick levels"). This is a reasonable prior if predictive book patterns are spatially local (they often are — BBO dynamics, level clustering). But it cannot capture long-range spatial correlations (e.g., a large hidden order 8 levels deep predicting a move) without deep stacking or large kernels.

**Alternative to validate**: Compare the price ladder CNN against a simple attention-based book encoder (each level attends to all other levels). If the attention model achieves higher $R^2_{\text{book}}$ with the same parameter count, the local spatial prior is suboptimal and the book has long-range structure.

### 2.4 Entropy Rate and Bar Type Selection

**Claim**: The optimal bar type maximizes the entropy rate of the return process, conditional on fixed bar count.

**Setup**: For a bar type $\beta$ with parameter $\theta$ (e.g., $V$ for volume bars), let $\{r^{(\beta, \theta)}_k\}_{k=1}^{N_\beta}$ be the return series for one trading day, where $N_\beta$ is the (random) bar count.

The *entropy rate* of the return process is:

$$
h(\beta, \theta) = \lim_{n \to \infty} \frac{1}{n} H(r_1, r_2, \ldots, r_n)
$$

For IID returns, $h = H(r_k)$ (single-bar entropy). For dependent returns, $h < H(r_k)$ because temporal structure makes the sequence partially predictable — which is exactly what we want to exploit.

**Practical metric**: Since we want to *find* and exploit temporal structure, we actually want bar types that *minimize* the conditional entropy:

$$
H(r_{k+1} \mid r_k, r_{k-1}, \ldots, r_{k-p})
$$

for some lag $p$. Equivalently, we want to *maximize*:

$$
I(r_{k+1}\,;\, r_k, r_{k-1}, \ldots, r_{k-p})
$$

A bar type where returns are more predictable given recent history is better for our purposes — it means the temporal encoder has more signal to learn from.

**Practical estimation**: Estimate via $k$-nearest-neighbor MI estimator (Kraskov et al.) or via the predictive $R^2$ of a simple autoregressive model on the return series. Compare across bar types at matched daily bar counts.

**Subtlety**: A bar type that produces highly autocorrelated returns might just be introducing spurious serial dependence (e.g., time bars during lunch produce many identical bars). Control for this by also measuring the *useful* predictability — the predictive $R^2$ for predicting returns $h$ bars ahead (for $h > 1$), which separates genuine temporal structure from trivial 1-lag autocorrelation.

---

## 3. Prior: The Market Is a Process in Order Flow, Not in Time

Fixed-time bars impose uniform sampling on a non-uniform process. A 15-minute bar at 9:31 AM on FOMC day and a 15-minute bar at 1:15 PM on a quiet Thursday contain wildly different information content. Returns sampled at fixed time intervals are heteroskedastic almost by definition because activity regimes are mixed.

Price changes are caused by orders. Time doesn't move price — order flow does. The natural unit of analysis should be denominated in order flow.

**Implications for this spec:**

- Bar boundaries are defined by cumulative order flow events, not clock time.
- Oracle lookahead horizons are denominated in events (e.g., "next 500 contracts traded"), not seconds.
- Time remains a *feature* (time-of-day effects are real), not the *indexing dimension*.
- The framework supports pluggable bar types so we can empirically compare signal quality across bar constructions.

The formal justification for this prior is in §2.1 (subordinated processes).

---

## 4. Bar Construction

### 4.1 Bar Types

All bar types consume the same raw input: the validated `BookSnapshot` stream from the book builder (§2.1.1 of the predecessor spec). Each bar type defines a boundary condition that triggers aggregation of the accumulated snapshots into a single bar.

The `Bar` struct is a pure data container. Encoder-specific input formats (price ladders, message sequence tensors) are constructed downstream by adapter classes, not stored in the bar itself. This separation ensures that changes to encoder architecture (e.g., switching from CNN to attention in §R3) don't require modifying the core data pipeline.

```cpp
// ── Core bar data (pure data container) ──

struct Bar {
    // ── Temporal ──
    int64_t     open_ts;          // exchange timestamp of first snapshot in bar
    int64_t     close_ts;         // exchange timestamp of last snapshot in bar
    float       time_of_day;      // fractional hours ET at bar close
    float       bar_duration_s;   // wall-clock seconds elapsed during bar

    // ── OHLCV ──
    float       open_mid;         // mid_price at bar open
    float       close_mid;        // mid_price at bar close
    float       high_mid;         // max mid_price during bar
    float       low_mid;          // min mid_price during bar
    float       vwap;             // volume-weighted average price (trade prices)
    uint64_t    volume;           // total contracts traded during bar
    uint32_t    tick_count;       // number of trades during bar
    float       buy_volume;       // contracts where aggressor was buyer
    float       sell_volume;      // contracts where aggressor was seller

    // ── Book state at bar close ──
    float       bids[10][2];      // top 10 bid levels (price, size)
    float       asks[10][2];      // top 10 ask levels (price, size)
    float       spread;           // spread at bar close

    // ── Intra-bar spread dynamics ──
    float       max_spread;       // widest spread observed during bar
    float       min_spread;       // tightest spread observed during bar
    uint32_t    snapshot_count;   // number of 100ms snapshots in this bar

    // ── MBO event reference ──
    // Points into the day's MBO event buffer (managed by DayEventBuffer, §10.3).
    // The Bar does NOT own this memory. The DayEventBuffer is valid for the
    // lifetime of the current day's processing and is released at day boundary.
    uint64_t    mbo_event_begin;  // index into day's MBO event buffer
    uint64_t    mbo_event_end;    // exclusive end index

    // ── Message summary statistics (computed during bar construction) ──
    uint32_t    add_count;        // number of Add events in bar
    uint32_t    cancel_count;     // number of Cancel events
    uint32_t    modify_count;     // number of Modify events
    uint32_t    trade_event_count;// number of Trade events (= tick_count, but explicit)
    float       cancel_add_ratio; // cancel_count / (add_count + eps)
    float       message_rate;     // total messages / bar_duration_s (messages per second)
};

// ── Encoder input adapters (constructed from Bar, not stored in Bar) ──

struct PriceLadderInput {
    // CNN spatial encoder input: (20, 2) tensor
    // [bid[9], bid[8], ..., bid[0], ask[0], ..., ask[9]]
    // Channels: (price_delta_from_mid, normalized_size)
    float data[20][2];

    static PriceLadderInput from_bar(const Bar& bar, float mid_price);
};

struct MessageSequenceInput {
    // Message encoder input: variable-length sequence of MBO events
    // Each event: (action_type, price, size, side, time_offset_ns)
    // Retrieved from DayEventBuffer using bar.mbo_event_begin/end
    std::vector<std::array<float, 5>> events;

    static MessageSequenceInput from_bar(const Bar& bar, const DayEventBuffer& buf);
};

// ── Bar builder interface ──

class BarBuilder {
public:
    virtual ~BarBuilder() = default;

    // Feed snapshots one at a time. Returns a Bar when boundary is hit.
    // Returns std::nullopt when accumulating (boundary not yet reached).
    virtual std::optional<Bar> on_snapshot(const BookSnapshot& snap) = 0;

    // Flush any partial bar at end of session. Returns nullopt if empty.
    virtual std::optional<Bar> flush() = 0;
};
```

### 4.2 Volume Bars

Boundary condition: cumulative volume (contracts traded) reaches threshold $V$.

```
Parameters:
  V = 100          # contracts per bar (tunable)

Boundary rule:
  Accumulate trade volume from each snapshot's trade buffer (deduplicated per §2.7).
  When cumulative volume >= V, emit bar and reset accumulator.

  If a single trade crosses the boundary (e.g., 80 accumulated + 40 trade = 120),
  the trade belongs to the current bar (bar gets 120 volume). The next bar starts
  fresh. Do NOT split trades across bars.
```

**Expected properties** (per §2.1): Returns across volume bars should be closer to IID Gaussian than time bars. Volume bars naturally compress low-activity periods (fewer bars during lunch) and expand high-activity periods (more bars at the open). Each bar represents approximately the same amount of market participation.

**Calibration**: Choose $V$ such that a typical RTH session (09:30–16:00) produces 200–1000 bars. For MES, daily volume varies widely (50k–200k+ contracts depending on regime). Start with $V = 100$ and adjust based on empirical bar counts per day. Log the bar count per session.

### 4.3 Tick Bars

Boundary condition: cumulative number of trades reaches threshold $K$.

```
Parameters:
  K = 50           # trades per bar (tunable)

Boundary rule:
  Count deduplicated trades (action='T', size > 0).
  When cumulative trade count >= K, emit bar and reset.
```

**Difference from volume bars**: A single large block trade increments volume substantially but only counts as one tick. Tick bars normalize by *transaction count* rather than *contract volume*, which can better capture institutional vs. retail flow dynamics. If large trades carry more information than small trades, volume bars are more appropriate. If the *act of trading* matters regardless of size, tick bars are more appropriate. This is an empirical question this spec helps answer.

### 4.4 Dollar Bars (Index-Point Bars)

Boundary condition: cumulative dollar volume (price × size × multiplier) reaches threshold $D$.

```
Parameters:
  D = 50000.0      # dollar volume per bar (tunable)
  multiplier = 5.0 # MES contract multiplier ($5 per index point)

Boundary rule:
  For each deduplicated trade: dollar_volume += price * size * multiplier
  When cumulative dollar_volume >= D, emit bar and reset.
```

**Rationale**: Dollar bars normalize for price level. If MES is at 4000 vs. 5000, the same number of contracts represents different economic exposure. Dollar bars keep each bar's economic significance roughly constant. This matters less for short time horizons (price doesn't move much intraday) but becomes relevant for multi-day or multi-instrument analysis.

### 4.5 Time Bars (Control)

Boundary condition: wall-clock time reaches next interval boundary.

```
Parameters:
  interval_s = 60  # seconds per bar (tunable: 1, 5, 15, 60, 300)

Boundary rule:
  Emit bar at each interval boundary (aligned to session clock).
  This is the traditional approach and serves as the control group.
```

### 4.6 Information-Driven Bars (Stretch Goal)

Boundary condition: cumulative signed order flow imbalance exceeds a dynamic threshold.

```
Parameters:
  theta_init = 100.0    # initial imbalance threshold (contracts)
  ewma_span = 100       # bars for EWMA of expected imbalance

Boundary rule (simplified imbalance bar):
  Track cumulative net signed volume: imbalance += aggressor_side * size
  When |imbalance| >= theta, emit bar and reset.
  Update theta via EWMA of |imbalance| at bar boundaries.
```

**Rationale**: Information-driven bars (per López de Prado) fire when the market has absorbed an unusual amount of one-sided flow — i.e., when something *informative* has likely happened. These should produce the best returns distributions for prediction but are the most complex to implement and calibrate. Defer to after volume/tick bars are validated.

---

## 5. Event-Denominated Oracle

### 5.1 Revised Oracle Design

The predecessor spec's oracle uses a fixed time horizon (100 snapshots = 10 seconds). This conflates signal strength with activity regime. The revised oracle denominates its lookahead in *events*, not time.

**Two labeling methods run in parallel** — first-to-hit (directional threshold) and triple barrier (symmetric with time expiry). Both produce labels on the same bar sequence. Comparing their expectancy characteristics is part of the analysis, not a fallback.

```cpp
// Event-denominated oracle
//
// Instead of "look forward 100 snapshots", the oracle now asks:
// "look forward until V_horizon contracts have traded"

struct OracleConfig {
    // Event-denominated horizons
    uint64_t    volume_horizon = 500;    // look forward until this many contracts trade

    // Safety cap: prevents unbounded lookahead during extended zero-volume
    // periods (e.g., halt, data gap). NOT intended to clip normal low-volume
    // regimes like lunch — see rationale below.
    uint32_t    max_time_horizon_s = 300; // 5 minutes. Only triggers during abnormal
                                          // inactivity. If this fires during normal
                                          // RTH, the volume_horizon is set too high.

    // Thresholds (in ticks — price-denominated, not time-denominated)
    int         target_ticks = 10;       // 2.50 points
    int         stop_ticks = 5;          // 1.25 points
    int         take_profit_ticks = 20;  // 5.00 points

    float       tick_size = 0.25f;

    // Labeling method
    enum class LabelMethod { FIRST_TO_HIT, TRIPLE_BARRIER };
    LabelMethod label_method = LabelMethod::FIRST_TO_HIT;
};
```

**Safety cap rationale** (revised from v0.2): The previous 60-second cap was too aggressive. During lunch on a quiet day, MES might trade 500 contracts in 3-5 minutes — well within normal market behavior, but the 60s cap would truncate the oracle's lookahead and force HOLD labels precisely when the event-driven paradigm should be operating. The 300s cap is a circuit breaker for abnormal conditions only. If telemetry shows it firing more than 1% of bars during RTH, either (a) volume_horizon is miscalibrated for the regime, or (b) there's a data gap. Log every cap-triggered HOLD with timestamp for post-hoc inspection.

### 5.2 Triple Barrier Labeling

The first-to-hit oracle has a known bias in mean-reverting microstructure: price oscillations through both target and stop levels cause systematic mislabeling when the eventual profitable direction is correct but the path hits the stop first.

The triple barrier method (López de Prado) addresses this by adding a time/volume expiry barrier:

```cpp
// Triple barrier oracle
//
// Three barriers:
//   Upper: entry + target_ticks × tick_size     → label = +1 (long profitable)
//   Lower: entry - stop_ticks × tick_size        → label = -1 (long stopped)
//   Expiry: volume_horizon contracts traded       → label = sign(return at expiry)
//                                                  or HOLD if |return| < min_return_ticks
//
// For short signals, barriers are mirrored.

struct TripleBarrierConfig {
    int         target_ticks = 10;
    int         stop_ticks = 5;
    uint64_t    volume_horizon = 500;    // expiry in volume, not time
    int         min_return_ticks = 2;    // at expiry, |return| must exceed this
                                          // to label as directional (otherwise HOLD)
    uint32_t    max_time_horizon_s = 300; // same safety cap as first-to-hit
};

int triple_barrier_label(
    const std::vector<Bar>& bars,
    int t,
    int position_state,
    float entry_price,
    const TripleBarrierConfig& config
);

// Returns:
//   +1: upper barrier hit first (long signal)
//   -1: lower barrier hit first (short signal)
//    0: HOLD (expiry hit with insufficient return, or safety cap triggered)
```

**Analysis requirement**: Run both labeling methods on the same data. Compare:
- Label distribution (first-to-hit will produce more extreme labels; triple barrier more balanced)
- Label-return correlation (which labeling method better predicts actual forward returns?)
- Expectancy after costs (which produces more profitable replay?)
- Label stability (consecutive label agreement rate)

If triple barrier dominates on expectancy and label-return correlation, use it as the primary label for supervised training. First-to-hit may still be useful as a secondary target for the model to learn threshold-crossing dynamics.

### 5.3 Comparison Framework

Run the oracle in all modes on the same data to quantify differences:

```
For each trading day in the test set:
  1. Generate trajectory with time-denominated oracle (v0.6 config: horizon=100 snapshots)
  2. Generate trajectory with first-to-hit event-denominated oracle (volume_horizon=500)
  3. Generate trajectory with triple barrier event-denominated oracle (volume_horizon=500)
  4. Compare:
     - Label distribution (class frequencies)
     - Label stability (how often consecutive labels differ)
     - Conditional entropy of labels given time-of-day
     - Correlation between label and subsequent realized PnL
     - Regime dependence (do labels degrade in specific volatility regimes?)
```

The event-denominated oracle should produce more uniform label distributions across the session (less time-of-day confounding) and higher correlation between labels and realized outcomes.

---

## 6. Execution Cost Model

### 6.1 MES Cost Structure

```
Per round-trip (entry + exit):
  Commission:     $0.62 per side × 2 = $1.24 per round-trip
                  (NinjaTrader / common retail rate; adjust for actual broker)

  Spread cost:    0.5 × spread × $5.00 per index point
                  (assume crossing the spread on entry; may improve with limit orders)
                  MES typical spread = 0.25 (1 tick) → $0.625 per side = $1.25 per RT

  Slippage:       0.0–0.25 points per side (0–1 tick)
                  Model as configurable parameter. Start with 0 for oracle replay
                  (oracle uses mid_price, not executable prices).

  Total per RT:   $1.24 + $1.25 + slippage = ~$2.49 minimum per round-trip
                  At 10 ticks target ($12.50 gross), cost is ~20% of gross profit.
                  At 5 ticks stop ($6.25 gross loss), cost adds ~40% to loss.

Cost parameters (configurable):
  commission_per_side = 0.62       # USD
  spread_model = "fixed"           # or "empirical" (use actual spread from snapshot)
  fixed_spread_ticks = 1           # used when spread_model = "fixed"
  slippage_ticks = 0               # start conservative
  contract_multiplier = 5.0        # MES: $5 per index point
  tick_size = 0.25                 # MES: 0.25 index points per tick
  tick_value = 1.25                # MES: $1.25 per tick ($5.00 × 0.25)
```

### 6.2 Execution Assumptions

For oracle replay (Phase 1 of this spec), assume:

- **Entry**: Executed at mid_price at the bar where the oracle signals ENTER LONG/SHORT. This is optimistic — real fills are at best bid/ask, not mid. The spread cost (§6.1) accounts for this statistically.
- **Exit**: Executed at mid_price at the bar where the oracle signals EXIT.
- **No partial fills**: 1 contract, fully filled (MES is liquid enough for this to be realistic for 1-lot).
- **No market impact**: 1 MES contract does not move the market.

For model-based backtesting (future), tighten these:
- Entry/exit at best bid/ask (not mid), with spread as a feature, not a cost assumption.
- Slippage model calibrated from actual fill data.

---

## 7. Representation Architecture (Forward-Looking)

This section defines the target model architecture that feature discovery informs. Nothing here is implemented in this spec — it exists to ensure the data pipeline and feature analysis produce the right inputs for the eventual build.

### 7.1 Three-Level Encoder

The architecture is a composition of three encoders, each corresponding to one term in the information decomposition (§2.2):

```
Level 1 — Message Encoder (intra-bar):
  Input:  Sequence of MBO events within bar t
          Variable length: could be 50 events or 5000
  Output: Fixed-dimension message embedding m_t ∈ ℝ^d_msg
  Architecture: Small transformer or SSM over the event sequence
  Purpose: Learn the "grammar" of order flow — sweep patterns, quote
           stuffing, iceberg detection, cancellation cascades

Level 2 — Spatial Encoder (per bar):
  Input:  Book snapshot at bar t close: (20 levels × 2 channels)
          + message embedding m_t
          + scalar features (spread, time, position_state)
  Output: Bar embedding e_t ∈ ℝ^d_bar
  Architecture: CNN on price ladder (as in v0.6 §5.3) with m_t concatenated
                after spatial pooling
  Purpose: Learn spatial structure of the book — shape of liquidity,
           asymmetries, depth patterns — augmented by intra-bar dynamics

Level 3 — Temporal Encoder (across bars):
  Input:  Sequence of bar embeddings (e_{t-W}, ..., e_{t-1}, e_t)
          W = lookback window in bars (not time!)
  Output: Action logits ∈ ℝ^5
  Architecture: SSM (Mamba) or Transformer over the bar embedding sequence
  Purpose: Learn temporal dynamics — momentum, mean-reversion, regime
           shifts, sweep-then-reload patterns across event-driven bars
```

The full forward pass:

```
For each bar t in the lookback window:
  m_t = MessageEncoder(mbo_events[bar_t])          # Level 1
  e_t = SpatialEncoder(book_snapshot_t, m_t)        # Level 2

logits = TemporalEncoder(e_{t-W}, ..., e_t)         # Level 3
```

### 7.2 Simplification Cascade

The information decomposition (§2.2) determines which levels are needed. Decisions use the revised threshold policy (§2.2): relative R² gap > 20% of baseline AND statistically significant across CV folds (paired test, p < 0.05 after Holm-Bonferroni correction).

| Research Finding | Architecture Decision |
|-----------------|----------------------|
| $I(\mathbf{M}_t ; r_{t+h} \mid \mathbf{B}_t) \approx 0$ (Tier 2 test) | Drop Level 1 (message encoder). Book state is sufficient. Architecture becomes: CNN + SSM. |
| $I(\mathbf{H}_t ; r_{t+h} \mid \mathbf{B}_t, \mathbf{M}_t) \approx 0$ | Drop Level 3 (temporal encoder). Current state is sufficient. Architecture becomes: CNN (+ message encoder if needed). This would mean the market is memoryless at the bar scale — unlikely but testable. |
| $R^2_{\text{book}} \approx R^2_{\text{bar features}}$ | Hand-crafted features capture most spatial information. CNN unnecessary — use GBT or MLP on engineered features. This would be surprising but would massively simplify the system. |
| All terms contribute (Tier 2 confirms message value) | Full three-level architecture. Most complex but most powerful. |

The feature discovery phases (§8, §9) produce the numbers to fill this table.

**Critical note on the message encoder decision**: This decision is gated on Tier 2 (learned sequence model), not Tier 1 (hand-crafted summaries). If Tier 1 shows no gap but Tier 2 shows a significant gap, the message encoder is justified — it means the signal exists in message sequences but is not captured by the summary statistics in Category 6. If both tiers show no gap, the message encoder is definitively not worth the complexity.

### 7.3 Why Not End-to-End on Raw Messages?

An alternative to the three-level design: feed the raw MBO message stream directly into a single large transformer, letting it learn everything from scratch.

Arguments against (for now):
- **Sample efficiency**: The compositional architecture encodes structural priors (spatial locality in the book, event semantics in messages, temporal dynamics across bars). A monolithic model must learn these from data alone, requiring far more samples.
- **Interpretability**: The three-level design lets us inspect each stage — which book patterns activate, which message sequences are encoded similarly, which temporal motifs drive decisions.
- **Computational cost**: MBO event rates for MES can exceed 10,000 messages/second. A full-day transformer over raw messages is infeasible. Bar construction is a principled compression step.
- **Testability**: Each encoder can be validated independently (cf. the overfit harness philosophy from v0.6).

If the three-level architecture fails to capture signal that a larger model could find, revisit this. But start compositional.

---

## 8. Feature Discovery

### 8.1 Objective

Two complementary goals:

1. **Diagnostic**: Identify which hand-crafted features are predictive of future returns, at what scales, and with what decay. This informs the GBT interpretable baseline and validates that the data contains learnable signal.
2. **Architectural**: Quantify how much predictive information lives in raw representations (book snapshots, message sequences, temporal history) vs. hand-crafted summaries. This determines which encoder stages are needed (§7.2).

### 8.2 Return Definitions

Returns are computed over bars (not clock time). Each bar type produces its own return series.

```
For bar i in a bar sequence:
  return_1   = (bar[i+1].close_mid - bar[i].close_mid) / tick_size     # 1-bar return in ticks
  return_5   = (bar[i+5].close_mid - bar[i].close_mid) / tick_size     # 5-bar return
  return_20  = (bar[i+20].close_mid - bar[i].close_mid) / tick_size    # 20-bar return
  return_100 = (bar[i+100].close_mid - bar[i].close_mid) / tick_size   # 100-bar return

For volume bars with V=100:
  return_1  ≈ return over ~100 contracts
  return_5  ≈ return over ~500 contracts
  return_20 ≈ return over ~2000 contracts
  return_100 ≈ return over ~10000 contracts
```

### 8.3 Feature Taxonomy

Features are organized by source, computation scale, and representation track.

**Track A: Hand-Crafted Features (GBT / interpretable baseline)**

```
Category 1: Book Shape (static per bar — snapshot of order book at bar close)

  1.1  book_imbalance:       (bid_vol - ask_vol) / (bid_vol + ask_vol + eps)
                             Levels: top-1, top-3, top-5, top-10
  1.2  weighted_imbalance:   same as 1.1 but weighted by inverse distance from mid
                             (closer levels weighted more heavily)
  1.3  spread:               in ticks
  1.4  bid_depth_profile:    [size_L0, size_L1, ..., size_L9] (raw sizes, 10 features)
  1.5  ask_depth_profile:    [size_L0, size_L1, ..., size_L9]
  1.6  depth_concentration:  HHI of bid sizes across levels (low = dispersed, high = concentrated)
                             Separate for bid and ask sides.
  1.7  book_slope:           linear regression slope of log(size) vs level index
                             Captures whether the book is "thick" near BBO or deep in the stack.
                             Separate for bid and ask.
  1.8  level_count:          number of non-empty levels on each side (0–10)


Category 2: Order Flow (aggregated within bar — event-denominated by construction)

  2.1  net_volume:           buy_volume - sell_volume (signed contracts)
  2.2  volume_imbalance:     net_volume / (total_volume + eps)
  2.3  trade_count:          number of trades in bar
  2.4  avg_trade_size:       volume / trade_count
  2.5  large_trade_count:    trades with size > 2 × rolling median trade size (20-bar window)
  2.6  vwap_distance:        (close_mid - vwap) / tick_size
  2.7  kyle_lambda:          Regression of price_change on signed_volume, computed over a
                             rolling window of the last 20 bars (not within a single bar).

                             Rationale: Within-bar Kyle's lambda requires regressing on
                             sub-bar observations. With volume bars of V=100, a single bar
                             may contain only 10-50 trades — far too few for a reliable
                             slope estimate. A 20-bar rolling window provides 200-1000
                             data points, yielding a statistically meaningful estimate of
                             permanent price impact per unit flow.

                             Computation:
                               For bars (t-19, ..., t), regress:
                                 Δmid_i = α + λ × net_volume_i + ε_i
                               kyle_lambda_t = λ (the slope coefficient)

                             NaN policy: set to NaN for the first 20 bars of each session
                             (insufficient lookback). Downstream consumers handle NaN
                             via imputation or exclusion.


Category 3: Price Dynamics (across bars — lookback windows)

  3.1  return_n:             mid_price return over last N bars (N = 1, 5, 20)
  3.2  volatility_n:         std(1-bar returns) over last N bars (N = 20, 50)
  3.3  momentum:             sum of signed 1-bar returns over last N bars (vs. just endpoint return)
                             Captures path dependence — same return with consistent vs. choppy path.
  3.4  high_low_range_n:     (max(high_mid) - min(low_mid)) over last N bars / tick_size
  3.5  close_position:       (close_mid - low_mid_N) / (high_mid_N - low_mid_N + eps)
                             Where in its recent range is the current price? (0 = at low, 1 = at high)


Category 4: Cross-Scale Dynamics (interaction between bar scales)

  4.1  volume_surprise:      current bar volume / EWMA(volume, span=20)
                             >1 means unusually active bar.
  4.2  duration_surprise:    current bar duration_s / EWMA(duration_s, span=20)
                             <1 means bar completed faster than usual (activity spike).
                             Only meaningful for volume/tick/dollar bars (time bars have fixed duration).
  4.3  acceleration:         return_1 - return_1[lag=1]
                             Is momentum increasing or decreasing?
  4.4  vol_price_corr:       rolling correlation(volume, |return_1|) over 20 bars
                             Are large moves accompanied by large volume?


Category 5: Time Context (features, not indexing)

  5.1  time_sin, time_cos:   sinusoidal encoding of time-of-day at bar close
  5.2  minutes_since_open:   continuous, from 09:30
  5.3  minutes_to_close:     continuous, to 16:00
  5.4  session_volume_frac:  cumulative volume so far / historical average daily volume
                             How "done" is the session by volume? Normalizes cross-day comparison.

                             Lookahead policy: "historical average daily volume" is computed
                             from the TRAINING SET ONLY (expanding window of prior days).
                             For the first day in the dataset, use the full-day volume of that
                             day itself (introduces mild lookahead for day 1 only; acceptable
                             since day 1 is always in-sample warmup). For subsequent days,
                             use the running mean of all prior completed days' total volumes.
                             This value is computed once per day at session open and held constant.


Category 6: Message Microstructure (intra-bar MBO summary — bridge to message encoder)

  6.1  cancel_add_ratio:     cancel_count / (add_count + eps)
                             High ratio = aggressive liquidity withdrawal. Proxy for informed trading.
  6.2  message_rate:         total MBO messages / bar_duration_s
                             Raw activity intensity. Complements volume_surprise (6.2 measures
                             message activity, 4.1 measures trade activity).
  6.3  modify_fraction:      modify_count / (add_count + cancel_count + modify_count + eps)
                             High modify rate may indicate algorithmic participation (HFT adjusting quotes).
  6.4  order_flow_toxicity:  fraction of trades that move the mid_price within the bar
                             Proxy for VPIN (Volume-synchronized Probability of Informed Trading).
                             High values indicate trades are adversely selecting the passive side.

                             Circularity note: VPIN is designed to be computed in volume time
                             (López de Prado). When using volume bars, this feature is somewhat
                             circular — each bar already represents a fixed volume quantum, so
                             the denominator of the VPIN-like computation is trivially constant.
                             The feature retains value as a within-bar measure (what fraction of
                             trades moved the mid?), but its cross-bar dynamics lose the
                             volume-normalization benefit that motivates VPIN. When comparing
                             this feature's predictive power across bar types, interpret with
                             caution: higher MI on time bars may reflect the volume-normalization
                             effect, not superior signal quality.

  6.5  cancel_concentration: are cancels clustered at specific levels or spread across the book?
                             Compute HHI of cancel counts per price level.
                             High = targeted pulling (informed). Low = broad adjustment (market-making).
```

**Track B: Raw Representations (encoder inputs)**

These are not features in the traditional sense — they are the raw data structures that the learned encoders will consume. They are exported alongside Track A features so that the $R^2$ comparison (§2.2) can quantify the information gap.

```
B.1  Book snapshot:          (20, 2) tensor per bar — price ladder with (price_delta, size_norm)
                             Constructed via PriceLadderInput::from_bar() adapter (§4.1).
                             Identical to v0.6 CNN input, sampled at bar close.
                             This is the CNN spatial encoder input.

B.2  Message sequence:       Variable-length sequence of MBO events within bar.
                             Constructed via MessageSequenceInput::from_bar() adapter (§4.1).
                             Each event encoded as: (action_type, price, size, side, time_offset)
                             where time_offset = ts_event - bar_open_ts (nanoseconds from bar open).

                             For R2 Tier 1 (cheap test): compress to fixed-length summary via:
                               - Binned action counts per time decile within bar
                               - Cancel/add/modify rates in first half vs. second half of bar
                               - Max instantaneous message rate (rolling 100ms window)

                             For R2 Tier 2 (definitive test): feed the full variable-length
                             sequence into a small LSTM or 1-layer transformer. This requires
                             retrieving raw events from DayEventBuffer (§10.3).

                             Sequence length management for Tier 2:
                               - Truncate to max 500 events per bar (keep most recent if exceeded)
                               - Pad shorter sequences to batch dimension
                               - Log truncation rate per bar type/parameter for calibration

B.3  Lookback book sequence: Sequence of (20, 2) book snapshots at the close of the previous W bars.
                             This is the temporal encoder's input (the sequence of CNN embeddings,
                             or raw book states as a proxy before the CNN is trained).
```

### 8.4 Feature Predictiveness Analysis

For each feature × return horizon × bar type combination, compute:

```
Predictiveness metrics:

  1. Mutual Information (MI):
     MI(feature, return_sign) — non-parametric measure of dependence.
     Discretize feature into quantiles (5 or 10 bins). Compute MI in bits.
     Advantage: captures nonlinear relationships.
     Baseline: MI ≈ 0 for independent variables. Report as excess MI over
     bootstrapped null (shuffle feature, recompute MI, take 95th percentile).

  2. Spearman Rank Correlation:
     corr(feature, return_n) — linear monotonic relationship.
     Report with p-value. Apply Holm-Bonferroni correction for multiple
     comparisons (see §8.7).

  3. Feature Importance (GBT):
     Train XGBoost regressor: features → return_n (regression, not classification).
     Extract gain-based feature importance. This captures nonlinear and interaction effects.
     Use 5-fold time-series cross-validation (expanding window, no future leakage).

  4. Conditional Returns:
     Bucket feature into quintiles. Compute mean return_n per quintile.
     Report monotonicity: is there a consistent relationship between feature
     quantile and subsequent return? Compute the spread (Q5 mean - Q1 mean)
     and its t-statistic.

  5. Decay Analysis:
     For each predictive feature, compute correlation with return_n for
     n = 1, 2, 5, 10, 20, 50, 100 bars. Plot the decay curve.
     Features with sharp decay are short-horizon signals. Features with
     slow decay are regime indicators.
```

### 8.5 Bar Type Comparison

For each bar type, compute aggregate signal quality metrics. These tests directly validate the predictions from §2.1 and §2.4.

```
Per bar type (volume, tick, dollar, time at multiple intervals):

  1. Returns normality (tests §2.1 prediction):
     Jarque-Bera statistic on 1-bar returns.
     Lower = more Gaussian = subordination model fits better.

  2. Heteroskedasticity (tests §2.1 prediction):
     ARCH LM test on 1-bar returns.
     Lower test statistic = more homoskedastic = the bar type is
     better at conditioning out the stochastic time change.

  3. Volatility clustering (tests §2.1 prediction):
     Autocorrelation of |return_1| at lags 1, 5, 10.
     Faster decay = bar type better normalizes activity regimes.

  4. Returns autocorrelation:
     Ljung-Box test on 1-bar returns at lags 1, 5, 10.
     Significant positive autocorrelation = momentum signal exploitable.
     Significant negative = mean-reversion signal.
     Neither = efficient at that scale.

  5. Predictive information (tests §2.4):
     Autoregressive R² for predicting return_{h} from (return_{-1}, ..., return_{-p})
     for h = 1, 5, 10 and p = 10. Higher = more temporal structure for the SSM.

  6. Aggregate feature MI:
     Sum of excess MI across all Track A features for return_1.
     Higher = more total predictive information available.

  7. Bar count stability:
     Coefficient of variation of daily bar counts.
     Lower = more stable (easier to set consistent lookback windows).
```

### 8.6 Warm-Up and Lookahead Bias Policy

All features with lookback dependencies require explicit warm-up handling. This section defines the policy to prevent subtle lookahead leakage.

```
Warm-up rules:

  1. EWMA-based features (volume_surprise, duration_surprise):
     - Initialize EWMA state at bar 0 of each session using the first bar's value.
     - Mark the first `ewma_span` bars (default 20) as WARMUP.
     - WARMUP bars are included in feature export but flagged. Downstream
       analysis excludes them from MI/correlation computations.
     - EWMA state does NOT carry across sessions (each day starts fresh).
       Rationale: overnight gaps can cause discontinuities. If cross-session
       EWMA proves valuable, add it as a separate feature.

  2. Rolling window features (volatility_n, momentum, close_position, kyle_lambda,
     vol_price_corr, large_trade_count):
     - Set to NaN for the first N bars of each session where N is the
       window length (e.g., first 20 bars for volatility_20).
     - NaN propagates to downstream consumers. GBT handles NaN natively.
       For MLP/linear models, impute NaN with feature median computed
       from TRAINING SET ONLY (not the current fold's test set).

  3. session_volume_frac (Category 5.4):
     - Uses expanding-window mean of prior days' total volumes (see §8.3).
     - First day in dataset: use that day's actual total volume.
       This is mild lookahead for day 1 only, acceptable since day 1
       is always in-sample.

  4. Forward returns:
     - return_n at bar i requires bar[i+n] to exist. The last n bars
       of each session have undefined forward returns. Exclude from analysis.
     - CRITICAL: Forward returns are NEVER used as features. They are
       TARGETS only. This is obvious but worth stating explicitly
       because the export format places features and targets side by side.

  5. Cross-validation splits:
     - All splits use expanding window (train on days 1..k, test on day k+1..k+m).
     - No shuffling. No random splits. Time ordering is sacred.
     - Feature normalization (if applied) uses statistics from training
       days only. Test-set features are normalized using training-set
       mean/std, NOT their own statistics.

Bar-level warm-up flag in export:
  Each exported row includes a boolean `is_warmup` field. A bar is marked
  warmup if ANY of its features are in warmup state. Analysis scripts filter
  on `is_warmup == false` by default.
```

### 8.7 Power Analysis and Multiple Comparisons

The feature analysis involves a large number of simultaneous statistical tests. Without correction, false discovery is near-certain.

```
Dimensionality of the test space:

  Features (Track A):           ~45 features (Categories 1-6)
  Return horizons:              4 (return_1, _5, _20, _100)
  Bar types:                    ~10 configurations (4 types × 2-3 params each)
  Predictiveness metrics:       5 (MI, Spearman, GBT, conditional returns, decay)

  Total pairwise tests:         45 × 4 × 10 = 1,800 feature-horizon-bartype combinations
                                (per metric)

Multiple comparison correction strategy:

  1. Primary analysis: Use Holm-Bonferroni correction within each metric family.
     For Spearman correlations: 1,800 tests → adjusted p-values.
     For MI excess tests: 1,800 bootstrap comparisons → adjusted significance.

  2. Feature importance (GBT): Not a p-value-based test. Use stability
     selection instead: run GBT 20 times with different random seeds and
     80% subsamples. Report features that appear in the top-20 in >60%
     of runs. This is more robust than single-run importance.

  3. Bar type comparison (§8.5): 10 bar configurations × 7 metrics = 70 tests.
     Apply Holm-Bonferroni within each metric (10 tests per metric family).

  4. Encoder comparison (§R2): 4 representation tracks × 4 horizons × 2 model types
     = 32 R² comparisons. Use paired t-test on per-fold R² values with
     Holm-Bonferroni correction across the 32 tests.

Minimum sample size guidance:

  For Spearman correlation to detect r = 0.05 (small effect) at α = 0.05
  with 80% power: n ≈ 2,500 bars.

  For a typical MES session producing 200-1000 volume bars per day across
  ~250 trading days, total bars ≈ 50,000-250,000. This provides adequate
  power for detecting small effects on the full dataset.

  Per-fold sample sizes in 5-fold CV will be smaller (~40,000-200,000 train,
  ~10,000-50,000 test). Still adequate for the effect sizes we care about.

  For per-day or per-regime stratified analysis (§9.6), subsets may be as
  small as 5,000-20,000 bars. This limits the detectable effect size to
  r ≈ 0.03-0.04. Report per-stratum power alongside results.

Reporting standard:
  All tables report: point estimate, 95% CI, raw p-value, corrected p-value,
  and whether the result survives correction. Results that are significant
  before but not after correction are flagged as "suggestive, insufficient
  evidence after correction."
```

---

## 9. Oracle Replay Backtest

### 9.1 Design

The oracle has perfect future knowledge. The backtest replays the oracle's own trading decisions against the execution cost model to determine whether the *strategy itself* has positive expectancy, independent of any model's ability to predict the labels.

This is the cheapest possible validation: if the oracle loses money after costs, no model can profitably replicate it.

```cpp
struct BacktestConfig {
    OracleConfig oracle;
    ExecutionCosts costs;

    // Bar configuration
    std::string bar_type;         // "volume", "tick", "dollar", "time"
    float bar_param;              // V, K, D, or interval_s depending on bar_type

    // Labeling method
    OracleConfig::LabelMethod label_method;  // FIRST_TO_HIT or TRIPLE_BARRIER

    // Data range
    std::string data_path;        // path to .dbn.zst files
    uint32_t instrument_id;
    std::vector<std::string> dates; // list of trading days to replay
};

struct TradeRecord {
    int64_t     entry_ts;
    int64_t     exit_ts;
    float       entry_price;
    float       exit_price;
    int         direction;        // +1 long, -1 short
    float       gross_pnl;        // in dollars, before costs
    float       net_pnl;          // in dollars, after costs
    int         entry_bar_idx;
    int         exit_bar_idx;
    int         bars_held;        // number of bars in position
    float       duration_s;       // wall-clock seconds in position
    int         exit_reason;      // 0=target, 1=stop, 2=take_profit,
                                  // 3=expiry (triple barrier), 4=session_end,
                                  // 5=safety_cap
};

struct BacktestResult {
    std::vector<TradeRecord> trades;

    // Summary statistics
    int         total_trades;
    int         winning_trades;
    int         losing_trades;
    float       win_rate;
    float       gross_pnl;
    float       net_pnl;
    float       avg_win;          // dollars
    float       avg_loss;         // dollars
    float       profit_factor;    // gross_wins / gross_losses
    float       expectancy;       // net_pnl / total_trades (avg dollars per trade)
    float       sharpe;           // annualized, computed from per-trade returns
    float       max_drawdown;     // peak-to-trough in dollars
    float       avg_bars_held;
    float       avg_duration_s;

    // Per-hour breakdown
    std::map<int, float> pnl_by_hour;  // hour (9-15) → net PnL

    // Label distribution stats
    std::map<int, int> label_counts;   // action → count
    float       hold_fraction;         // fraction of bars labeled HOLD

    // Exit reason distribution
    std::map<int, int> exit_reason_counts;

    // Safety cap telemetry
    int         safety_cap_triggered_count;
    float       safety_cap_fraction;   // safety_cap_triggered / total_labels
};
```

### 9.2 Backtest Procedure

```
For each trading day in the test set:
  1. Run book builder on the day's .dbn.zst file → BookSnapshot stream
  2. Run bar builder on the snapshot stream → Bar sequence
  3. Run trajectory builder with BOTH oracle types on the bar sequence:
     a. First-to-hit event-denominated oracle
     b. Triple barrier event-denominated oracle
  4. For each ENTER/EXIT pair in each trajectory:
     - Record TradeRecord with gross and net PnL, exit reason
  5. Aggregate into BacktestResult per oracle type
  6. Log safety cap triggers with timestamps

Repeat for each bar type × oracle config combination.
```

### 9.3 Test Data

Use the validated 2022 MES.FUT dataset (`DATA/GLBX-20260207-L953CAPU5B/`).

```
In-sample:   Jan–Jun 2022 (instrument_id 13615 = MESM2 for Q1-Q2, 10039 = MESU2 for Q3)
Out-of-sample: Jul–Dec 2022

Note: Instrument rollover occurs quarterly. The trajectory builder must handle
instrument ID changes at rollover dates. For simplicity, process each quarterly
contract independently and concatenate results. Do not trade across rollovers.

Rollover schedule (approximate for MES 2022):
  MESH2 (13614) → MESM2 (13615): ~March 18, 2022
  MESM2 (13615) → MESU2 (10039): ~June 17, 2022
  MESU2 (10039) → MESZ2 (????): ~September 16, 2022

Use the front-month contract. Transition on the standard rollover date
(Thursday before expiration). Verify instrument IDs against symbology.json.

Rollover volume migration handling:
  In the 3-5 trading days before each rollover, front-month volume declines
  as participants migrate to the next contract. This creates a volume regime
  shift that affects volume bar calibration.

  Policy:
  - Exclude the final 3 trading days before each rollover date from analysis.
    These days have anomalous volume profiles that would distort bar type
    comparison and feature statistics.
  - Log the excluded dates and their volume characteristics for transparency.
  - If volume bars produce significantly fewer bars on these days (< 50% of
    the 20-day rolling median bar count), this confirms the exclusion is correct.
  - Do NOT attempt to adjust bar parameters dynamically near rollover —
    this introduces complexity with little benefit for a 3-day window.
```

### 9.4 Success Criteria

The oracle replay backtest PASSES (i.e., it's worth training models) if:

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Net expectancy | > $0.50 per trade | Must clear costs with margin |
| Profit factor | > 1.3 | Gross wins meaningfully exceed gross losses |
| Win rate | > 45% | With 2:1 reward:risk (10 tick target vs 5 tick stop), 45% is breakeven after costs |
| Out-of-sample net PnL | > 0 | In-sample could be overfit to 2022H1 market regime |
| Max drawdown | < 50 × expectancy | Drawdown should be recoverable within ~50 trades |
| Trade count | > 10 per day avg | Sufficient sample size for statistical validity |

These criteria must be met by at least one of the two labeling methods (first-to-hit or triple barrier). If triple barrier passes and first-to-hit fails, use triple barrier labels for supervised training. If both pass, prefer whichever has higher expectancy × trade_count (total net PnL).

If the oracle fails these criteria with the default config (`target_ticks=10`, `stop_ticks=5`, `volume_horizon=500`), iterate on oracle parameters before concluding no signal exists:

- Sweep `target_ticks` ∈ {5, 8, 10, 15, 20}
- Sweep `stop_ticks` ∈ {3, 5, 8, 10}
- Sweep `volume_horizon` ∈ {200, 500, 1000, 2000}
- Fix `take_profit_ticks = 2 × target_ticks`
- Run both labeling methods at each parameterization

Report the Pareto frontier of (expectancy, trade_count, max_drawdown) across the sweep.

### 9.5 Oracle Failure Diagnosis

If the oracle doesn't produce positive expectancy across any reasonable parameterization:

1. **Costs too high for the scale**: The 10-tick target may be too small relative to MES costs (~2 ticks per round-trip). Try larger targets (20, 40 ticks) with proportionally larger stops and horizons.
2. **MES microstructure is too noisy at this scale**: Mid-price oscillations in a 1-tick spread market create false signals. The oracle's directional threshold may be triggering on noise. Filter: only label when spread < 2 ticks (tight book).
3. **The oracle's threshold logic is too naive**: First-to-hit labeling may systematically mislabel in mean-reverting vs. trending regimes. If triple barrier also fails, the issue is more fundamental than labeling method. Consider adding a regime filter or using a different approach (e.g., fixed-horizon sign of return with magnitude threshold).
4. **Feature discovery may reveal better labels**: Proceed to §8 anyway — feature analysis on *returns* (not oracle labels) can reveal predictive structure that a different oracle could exploit.

### 9.6 Regime Stratification

2022 was a bear market with elevated volatility (Fed hiking cycle, inflation). Oracle results from this year may not generalize. While the spec cannot control what data is available, it can control how results are analyzed.

```
Regime stratification protocol:

  1. Realized volatility regimes:
     Compute 20-bar realized volatility (std of 1-bar returns) at each bar.
     Partition bars into quartiles (Q1 = lowest vol, Q4 = highest vol).
     Report backtest metrics separately per volatility quartile.

     Expected finding: Oracle expectancy may concentrate in Q3-Q4 (high vol
     periods have larger moves relative to costs). If expectancy is ONLY
     positive in Q4, the strategy is fragile — it depends on extreme regimes.

  2. Time-of-day regimes:
     Partition by session phase:
       Open:  09:30–10:30 (high activity, large spreads early)
       Mid:   10:30–14:00 (variable)
       Close: 14:00–16:00 (often active, MOC flows)
     Report metrics per phase.

  3. Volume regimes:
     Partition days by total daily volume quartile.
     Report metrics separately. If the strategy only works on high-volume
     days, it may be capturing event-driven moves (FOMC, CPI, etc.) that
     are inherently easier to predict.

  4. Trend vs. mean-reversion days:
     Classify each day by the sign and magnitude of open-to-close return.
     Partition into: strong trend up (>1%), strong trend down (<-1%),
     range-bound (|OTC return| < 0.3%), moderate.
     Report metrics per day type.

  5. Cross-regime stability score:
     Define: stability = min(regime_expectancy) / max(regime_expectancy)
     across the primary stratification (volatility quartiles).
     stability > 0.5 → robust strategy
     stability 0.2-0.5 → regime-dependent but potentially viable
     stability < 0.2 → fragile, do not proceed without regime filter

Report format:
  For each stratification × each oracle config, produce a table:
    regime | trades | win_rate | expectancy | profit_factor | sharpe
  Plus the cross-regime stability score.
```

---

## 10. Infrastructure

### 10.1 Dependencies (additions to predecessor spec)

```
C++ (additions):
  Eigen3               # matrix operations for correlation, regression
                        # (or use libtorch tensors if sufficient)

Python (analysis & visualization):
  polars                # fast DataFrame operations for feature analysis
  scipy                 # MI estimation, statistical tests
  scikit-learn          # mutual_info_regression, cross-validation
  matplotlib            # decay curves, feature importance plots
  xgboost               # feature importance via Python API (analysis only)
  seaborn               # heatmaps for feature × horizon × bar_type
  torch                 # Tier 2 message encoder (small LSTM/transformer)

All C++ infrastructure from the predecessor spec remains:
  databento-cpp, libtorch, xgboost (C API), Catch2, nlohmann_json, zstd
```

### 10.2 Project Structure (additions)

```
kenoma-models/
├── past_specs/
│   └── model_orchestrator_v0.6.md    # Retired overfit harness spec
├── include/kenoma/
│   ├── bars/
│   │   ├── bar.hpp                   # Bar struct (pure data), BarBuilder interface,
│   │   │                             # PriceLadderInput, MessageSequenceInput adapters
│   │   ├── volume_bar_builder.hpp
│   │   ├── tick_bar_builder.hpp
│   │   ├── dollar_bar_builder.hpp
│   │   ├── time_bar_builder.hpp
│   │   └── bar_factory.hpp           # Create builder from config string
│   ├── backtest/
│   │   ├── execution_costs.hpp       # Cost model (commission, spread, slippage)
│   │   ├── oracle_replay.hpp         # Replay oracle signals with cost accounting
│   │   ├── triple_barrier.hpp        # Triple barrier labeling implementation
│   │   ├── trade_record.hpp          # TradeRecord, BacktestResult structs
│   │   └── backtest_runner.hpp       # Orchestrates multi-day, multi-config runs
│   ├── features/
│   │   ├── bar_features.hpp          # Track A: hand-crafted features (Categories 1-6)
│   │   ├── raw_representations.hpp   # Track B: PriceLadderInput export, message summaries
│   │   ├── feature_export.hpp        # Export both tracks + returns to Parquet/CSV
│   │   └── warmup.hpp               # Warm-up state tracking per feature, is_warmup flag
│   └── data/
│       ├── day_event_buffer.hpp      # DayEventBuffer: per-day MBO event storage (§10.3)
│       └── (existing: book_builder, feature_encoder, etc.)
├── src/
│   ├── bars/
│   ├── backtest/
│   ├── features/
│   └── data/
├── tests/
│   ├── test_bar_builders.cpp         # Bar boundary correctness, volume accounting,
│   │                                 # MBO event index tracking, message summary computation
│   ├── test_execution_costs.cpp      # Cost model arithmetic
│   ├── test_oracle_replay.cpp        # PnL accounting, trade record accuracy
│   ├── test_triple_barrier.cpp       # Triple barrier labeling, expiry behavior
│   ├── test_bar_features.cpp         # Feature computation, NaN handling, warm-up flags
│   ├── test_raw_representations.cpp  # PriceLadderInput shape, MessageSequenceInput correctness
│   ├── test_warmup.cpp               # Warm-up policy: correct NaN placement, flag propagation
│   └── test_day_event_buffer.cpp     # Memory lifecycle, index validity across bars
├── analysis/
│   ├── oracle_expectancy.py          # Read backtest results, produce summary tables
│   ├── regime_analysis.py            # Regime stratification (§9.6), stability scores
│   ├── feature_analysis.py           # MI, correlation, GBT importance, decay curves
│   │                                 # with Holm-Bonferroni correction (§8.7)
│   ├── bar_comparison.py             # Normality, autocorrelation, heteroskedasticity tests
│   │                                 # Tests predictions from §2.1 and §2.4
│   ├── representation_gap.py         # R² comparison: hand-crafted vs. raw book vs. book+msg
│   │                                 # Tests information decomposition from §2.2
│   │                                 # Includes Tier 2 learned message encoder test
│   └── notebooks/
│       ├── exploration.ipynb         # Interactive analysis
│       └── encoder_decision.ipynb    # Synthesize R² findings → architecture decision (§7.2)
├── research/
│   ├── R1_subordination_test.py      # Test §2.1 predictions on MES data
│   │                                 # Jarque-Bera, ARCH, vol-clustering across bar types
│   ├── R2_information_decomposition.py  # Estimate information decomposition §2.2
│   │                                    # Tier 1: R² with hand-crafted message summaries
│   │                                    # Tier 2: R² with learned LSTM/transformer on raw messages
│   ├── R3_book_sufficiency.py        # Test §2.3: CNN embedding vs. raw book R²
│   │                                 # + attention baseline comparison
│   └── R4_entropy_rate.py            # Test §2.4: autoregressive R² across bar types
├── scripts/
│   ├── run_oracle_backtest.sh        # Multi-day oracle replay across bar types
│   │                                 # Runs both first-to-hit and triple barrier
│   └── run_feature_analysis.sh       # Feature export + analysis pipeline
└── results/
    ├── backtest/                     # Per-config backtest results (JSON)
    ├── features/                     # Feature analysis outputs (plots, tables)
    ├── regime/                       # Regime stratification results
    └── research/                     # Research cycle outputs (R1-R4)
```

### 10.3 Memory Model for MBO Events

```cpp
// DayEventBuffer — manages the lifecycle of raw MBO events for a single trading day.
//
// Design rationale: Bars reference MBO events via index ranges (mbo_event_begin/end).
// The raw events must remain in memory while any bar referencing them is in use.
// The simplest correct approach: load all MBO events for one day, process all bars,
// run all analyses that need raw events, then release.
//
// Memory sizing:
//   MES MBO event rate: ~5,000-50,000 messages/second during active periods.
//   RTH session (6.5 hours): ~100M-500M messages on extreme days.
//   Per event storage: ~32 bytes (action, price, size, side, timestamp).
//   Worst case: 500M × 32B = 16GB. This is too large for a single buffer.
//
//   Mitigation:
//   - Filter to relevant instrument_id during loading (reduces by ~10-50x for
//     MES vs. full GLBX feed).
//   - After filtering, typical MES-only event count: 2M-10M per day.
//   - Memory: 10M × 32B = 320MB per day. Acceptable.
//   - If memory is still tight, implement a two-pass approach: first pass
//     builds bars (only needs running counts, not raw events), second pass
//     loads raw events for specific bars needed by R2 Tier 2 analysis.

class DayEventBuffer {
public:
    // Load MBO events for a single day from .dbn.zst file.
    // Filters to the specified instrument_id.
    void load(const std::string& dbn_path, uint32_t instrument_id);

    // Access events by index range (as stored in Bar.mbo_event_begin/end)
    std::span<const MBOEvent> get_events(uint64_t begin, uint64_t end) const;

    // Total event count
    size_t size() const;

    // Release memory. Called at day boundary.
    void clear();

private:
    std::vector<MBOEvent> events_;
};

// Lifecycle:
//   for each trading_day:
//     buffer.load(day_file, instrument_id);     // allocate
//     bars = build_bars(snapshots, buffer);      // bars reference buffer indices
//     analyze(bars, buffer);                     // R2 Tier 2 reads raw events
//     buffer.clear();                            // release before next day
```

---

## 11. Orchestrator Agent Instructions

Phases alternate between engineering (TDD, implementation, validation gates) and research (analysis, empirical tests of theoretical predictions). Engineering phases produce tested code. Research phases produce empirical findings that inform design decisions. Both are required.

### Phase 1: Bar Construction [Engineering]

1. Implement `Bar` struct (pure data container with MBO event indices + message summary fields), `PriceLadderInput` and `MessageSequenceInput` adapters, and `BarBuilder` interface.
2. Implement `DayEventBuffer` for per-day MBO event storage and lifecycle management.
3. Implement `VolumeBarBuilder`, `TickBarBuilder`, `DollarBarBuilder`, `TimeBarBuilder`.
4. Implement `BarFactory` for config-driven instantiation.
5. Implement message summary computation (add_count, cancel_count, modify_count, cancel_add_ratio, message_rate) during bar construction.
6. Implement warm-up state tracking (`warmup.hpp`): per-feature warm-up flag, composite `is_warmup` bar flag.
7. Unit tests: verify bar boundary conditions, volume/tick/trade accounting, flush behavior, message summary correctness, DayEventBuffer lifecycle, adapter output shapes, edge cases (no trades in a period, single large trade exceeding boundary, session boundaries).

**Phase 1 Validation Gate:**
```
Assert: VolumeBarBuilder with V=100 produces bars where each bar.volume >= V
        (except possibly the last flushed bar)
Assert: TickBarBuilder with K=50 produces bars where each bar.tick_count >= K
Assert: TimeBarBuilder produces bars aligned to interval boundaries
Assert: All bar types produce identical total volume across a session
        (sum of bar volumes = total trades in snapshot stream)
Assert: Bar OHLC is consistent: low_mid <= open_mid, close_mid <= high_mid
Assert: Bars are non-overlapping and contiguous (no gaps in timestamp coverage)
Assert: flush() returns partial bar at session end
Assert: Message summary fields are consistent:
        add_count + cancel_count + modify_count + trade_event_count = total MBO events in bar
Assert: cancel_add_ratio computed with epsilon guard
Assert: mbo_event_begin < mbo_event_end for all non-empty bars
Assert: DayEventBuffer.get_events() returns correct events for bar's index range
Assert: DayEventBuffer.clear() releases memory (check with allocation counter or similar)
Assert: PriceLadderInput::from_bar() produces (20, 2) output
Assert: MessageSequenceInput::from_bar() produces correct event sequence from buffer
```

### Phase 2: Execution Cost Model & Oracle Replay [Engineering]

8. Implement `ExecutionCosts` — configurable per §6.1.
9. Implement event-denominated oracle (§5.1) — both first-to-hit and triple barrier labeling methods on bar sequences with volume-based lookahead.
10. Implement `OracleReplay` — runs the oracle on a bar sequence, tracks positions, computes PnL per trade, records exit reason and safety cap telemetry.
11. Implement `BacktestRunner` — orchestrates multi-day runs, aggregates results.

**Phase 2 Validation Gate:**
```
Assert: OracleReplay PnL accounting is correct:
        sum(trade.net_pnl) == result.net_pnl
Assert: Every ENTER has a matching EXIT (no open positions at session end)
        (Force-close at session end if oracle hasn't exited; log warning, exit_reason=4)
Assert: Commission is applied per-side (2× per round-trip)
Assert: Spread cost uses actual spread from bar data when spread_model="empirical"
Assert: Trade direction matches oracle label (+1 for ENTER LONG, -1 for ENTER SHORT)
Assert: No trades during first W bars (observation window warmup)
Assert: Triple barrier correctly handles all three exit conditions (target, stop, expiry)
Assert: Triple barrier expiry labels: sign(return) when |return| >= min_return_ticks, else HOLD
Assert: Safety cap triggers are logged with timestamps and counted in BacktestResult
Assert: exit_reason is correctly recorded for every trade
```

### Phase 3: Multi-Day Oracle Backtest [Engineering]

12. Run oracle replay across all 2022 trading days, for each bar type × oracle config × labeling method. Exclude rollover transition days (final 3 trading days before each quarterly rollover).
13. Aggregate results. Compute summary statistics per §9.4.
14. Generate oracle expectancy report with regime stratification per §9.6.

**Phase 3 Validation Gate:**
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

### Phase R1: Subordination Hypothesis Test [Research]

Test the predictions from §2.1 on actual MES data. This runs after Phase 1 (needs bar builders) but can run in parallel with Phases 2–3 (doesn't need the oracle).

15. For each bar type (volume V∈{50,100,200}, tick K∈{25,50,100}, time ∈{1s,5s,60s}), compute the 1-bar return series across 10+ trading days.
16. For each return series, compute:
    - Jarque-Bera statistic (normality)
    - ARCH(1) coefficient (conditional heteroskedasticity)
    - Autocorrelation of |returns| at lags 1, 5, 10 (volatility clustering)
17. Rank bar types by each metric. Apply Holm-Bonferroni correction for bar type comparisons (10 configs per metric). Determine whether volume/tick bars significantly outperform time bars.
18. Compute bar count statistics (mean, std, CV per day) for each bar type and parameter.

**Phase R1 Deliverable:**
```
Table: bar_type × param → (JB_stat, ARCH_coeff, |r| autocorrelation, bar_count_mean, bar_count_CV)
        with p-values and Holm-Bonferroni corrected significance
Finding: Does the subordination model hold for MES? Which bar type best conditions out the
         stochastic time change?
Decision: Recommended primary bar type and parameter for subsequent phases.
```

### Phase 4: Feature Computation & Export [Engineering]

19. Implement Track A (hand-crafted) bar-level feature computation per §8.3 Categories 1–6, including revised Kyle's lambda (rolling 20-bar window).
20. Implement Track B (raw representations) export: PriceLadderInput at bar close, message sequence summaries. Prepare raw event export capability for R2 Tier 2.
21. Implement warm-up flag propagation per §8.6: mark bars where any feature is in warm-up state.
22. Export both tracks + forward returns to Parquet (or CSV) for Python analysis, including `is_warmup` flag.
23. Handle NaN/edge cases (empty bars, insufficient lookback, division by zero) per §8.6.

**Phase 4 Validation Gate:**
```
Assert: No unexpected NaN in Track A features (NaN only where documented:
        kyle_lambda first 20 bars, volatility/momentum during lookback warmup)
Assert: Feature count matches taxonomy (verify exact count per category)
Assert: Forward returns computed correctly (no lookahead leakage in feature computation)
Assert: is_warmup flag is set for bars where ANY feature has insufficient lookback
Assert: session_volume_frac uses expanding-window prior-day mean (not future data)
Assert: EWMA state resets at session boundaries
Assert: Track B PriceLadderInput has shape (20, 2) per bar
Assert: Track B message summaries are consistent with Track A Category 6 fields
Assert: Export includes bar metadata (timestamp, bar_type, bar_param, day, is_warmup)
Assert: Rollover transition days excluded from export
```

### Phase 5: Feature Analysis [Engineering + Research]

24. Run MI analysis per §8.4 with bootstrapped null and Holm-Bonferroni correction (§8.7).
25. Run GBT feature importance with stability selection (20 runs, 80% subsamples, report features appearing in top-20 in >60% of runs).
26. Run decay analysis.
27. Run bar type signal quality comparison per §8.5.
28. Produce Track A summary report: which hand-crafted features, at which scales, on which bar types, are most predictive. Report raw p-values, corrected p-values, and power analysis per §8.7.

**Phase 5 Outputs:**
```
- Feature × horizon heatmap (MI or correlation) per bar type
  with significance markers (★ = survives correction, ○ = suggestive)
- Top-20 features ranked by excess MI for each bar type (stability-selected)
- Decay curves for top features
- Bar type comparison table (normality, autocorrelation, heteroskedasticity, aggregate MI)
  with Holm-Bonferroni corrected p-values
- Per-stratum power analysis (detectable effect size at α=0.05, power=0.80)
- Recommendations for GBT model input features and bar type selection
```

### Phase R2: Information Decomposition [Research]

This is the key research phase. It tests §2.2 empirically and determines the target architecture (§7.2). Uses the revised two-tier proxy and threshold policy.

**Tier 1 (hand-crafted proxies):**
29. Using the exported data from Phase 4, train a linear model and a shallow MLP (2 hidden layers, 64 units) to predict `return_5` from:
    - (a) Track A hand-crafted features only → $R^2_{\text{bar}}$
    - (b) Track B raw book snapshot (flattened 40 features) only → $R^2_{\text{book}}$
    - (c) Track B raw book + Category 6 hand-crafted message summary features → $R^2_{\text{book+msg\_summary}}$
    - (d) (c) + lookback window of 20 previous bars' book snapshots → $R^2_{\text{full\_summary}}$
30. Use 5-fold expanding-window time-series cross-validation. Report mean and std of $R^2$ across folds.

**Tier 2 (learned message encoder):**
31. Train a small LSTM (1 layer, 32 hidden units) on the raw intra-bar MBO event sequence from `DayEventBuffer`, producing a fixed-size message embedding. Concatenate with flattened book snapshot. Predict `return_5`. Record $R^2_{\text{book+msg\_learned}}$.
32. If compute allows, also train a 1-layer transformer (2 heads, d_model=32) as an alternative sequence model. Record $R^2_{\text{book+msg\_attn}}$.
33. Truncate message sequences to max 500 events per bar. Log truncation rate.

**Analysis:**
34. Compute the gaps:
    - $R^2_{\text{book}} - R^2_{\text{bar}}$ (information lost by hand-crafting)
    - $R^2_{\text{book+msg\_summary}} - R^2_{\text{book}}$ (Tier 1: do summaries add value?)
    - $R^2_{\text{book+msg\_learned}} - R^2_{\text{book}}$ (Tier 2: do raw messages add value?)
    - $R^2_{\text{full\_summary}} - R^2_{\text{book+msg\_summary}}$ (does temporal history add value?)
35. Apply threshold policy from §2.2: relative gap > 20% of baseline R² AND paired test p < 0.05 (Holm-Bonferroni corrected across comparisons).
36. Repeat for `return_1`, `return_20`, `return_100` to see if the gaps vary by horizon.

**Phase R2 Deliverable:**
```
Table: return_horizon × representation × tier → (mean_R², std_R², per-fold R² values)
Table: Information gaps with relative magnitude, paired test p-values, corrected p-values

Critical comparison:
  If Tier 2 gap > Tier 1 gap by a significant margin:
    → Category 6 summaries are insufficient. Message encoder is justified.
    → The hand-crafted features miss sequential patterns in the message stream.
  If Tier 2 gap ≈ Tier 1 gap:
    → Category 6 summaries capture the message information adequately.
    → Message encoder adds complexity without benefit.
  If both gaps ≈ 0:
    → Messages carry no incremental predictive information beyond the book state.
    → Drop message encoder entirely.

Decision matrix (filling in §7.2):
  If R²_book - R²_bar > relative ε AND significant:     Include spatial encoder
  If R²_book+msg_learned - R²_book > relative ε AND significant: Include message encoder
  If R²_full - R²_book+msg > relative ε AND significant: Include temporal encoder

Output: Recommended architecture from the simplification cascade (§7.2)
```

### Phase R3: Book Encoder Inductive Bias [Research]

Test §2.3 — does the CNN's spatial prior fit the data?

37. Train a small CNN (v0.6 architecture, scaled down) on raw book snapshots → `return_5`. Record $R^2_{\text{CNN}}$.
38. Train a small attention-based encoder (each level attends to all 20 levels, 2 heads, 1 layer) on raw book snapshots → `return_5`. Record $R^2_{\text{attn}}$.
39. Train a simple MLP on the flattened 40-feature book vector → `return_5`. Record $R^2_{\text{MLP}}$.
40. Compare: $R^2_{\text{CNN}}$ vs $R^2_{\text{attn}}$ vs $R^2_{\text{MLP}}$. Use paired t-test on per-fold R² with Holm-Bonferroni correction (3 pairwise comparisons).

**Phase R3 Deliverable:**
```
If R²_CNN ≈ R²_attn >> R²_MLP:  Spatial structure matters, local prior is fine → use CNN
If R²_attn >> R²_CNN >> R²_MLP: Spatial structure matters, but long-range → consider attention
If R²_MLP ≈ R²_CNN ≈ R²_attn:  No spatial structure to exploit → MLP on flattened book is sufficient
(All comparisons tested for statistical significance per §8.7)
```

### Phase R4: Entropy Rate and Temporal Predictability [Research]

Test §2.4 — which bar type maximizes exploitable temporal structure?

41. For each bar type (using the recommended parameters from R1), compute autoregressive $R^2$ for predicting `return_h` from the previous 10 returns, for $h \in \{1, 5, 10, 20\}$.
42. Compute using both linear AR and GBT AR (captures nonlinear temporal structure).
43. Compare across bar types at matched daily bar counts. Apply Holm-Bonferroni correction for bar type comparisons within each horizon × model combination.

**Phase R4 Deliverable:**
```
Table: bar_type × horizon × model_type → AR_R² (with corrected p-values)
Finding: Which bar type produces the most temporally predictable returns?
         Is the temporal structure linear (AR sufficient) or nonlinear (GBT > AR)?
Decision: Final bar type recommendation for the temporal encoder.
```

### Phase 6: Synthesis [Research]

44. Combine findings from Phases 3 (oracle expectancy), 5 (feature predictiveness), R1–R4 (theoretical validation) into a single decision document.
45. The document answers:
    - Does the oracle have positive expectancy? (go/no-go for supervised training)
    - Which labeling method is preferred? (first-to-hit vs. triple barrier)
    - Which bar type is optimal? (for all downstream work)
    - Which encoder stages are needed? (architecture decision for model build spec)
    - Which features are most predictive at which horizons? (input selection for GBT baseline and encoder design)
    - Does the subordination model hold for MES? (theoretical validation)
    - Is the book state sufficient or do messages add value? (message encoder go/no-go, per Tier 2)
    - Is the oracle robust across regimes? (regime stratification stability score)
    - What are the statistical limitations? (power analysis, comparisons that failed correction)

---

## 12. What This Spec Does NOT Cover

- Model training (that follows from feature discovery and architecture decision)
- RL reward shaping (depends on which features are predictive)
- Live execution / order management
- Multi-instrument support
- The 9:25 AM supervised strategy (parallel spec)
- Information-driven bars (deferred to after volume/tick validation)
- Limit order fill modeling (future spec — oracle uses mid-price)
- Regime detection / classification (may emerge from feature analysis; §9.6 provides descriptive stratification, not a predictive regime model)
- Transaction cost optimization (market vs. limit orders)
- Full message encoder implementation (only summary features + small Tier 2 model in this spec; full sequence processing is in the model build spec)
- Multi-year regime robustness (requires data beyond 2022; noted as a limitation in §9.6)

---

## 13. Exit Criteria

This spec is COMPLETE when:

**Engineering:**
- [ ] Bar builders (volume, tick, dollar, time) produce correct bars from MBO snapshot stream
- [ ] Bar builders compute message summary fields (add/cancel/modify counts, cancel_add_ratio, message_rate)
- [ ] Bar builders track MBO event indices for future message encoder access
- [ ] Bar builders account for trade deduplication (inherited from predecessor §2.7)
- [ ] DayEventBuffer correctly manages per-day MBO event lifecycle (load, access, clear)
- [ ] PriceLadderInput and MessageSequenceInput adapters produce correct shapes from Bar data
- [ ] Event-denominated oracle produces labels with volume-based lookahead (both first-to-hit and triple barrier)
- [ ] Triple barrier oracle correctly handles target, stop, and expiry exits
- [ ] Safety cap telemetry logged and reported; cap trigger rate < 1% during RTH
- [ ] Execution cost model correctly computes per-trade net PnL
- [ ] Oracle replay backtest runs across all 2022 trading days (excluding rollover transition days) with correct PnL accounting
- [ ] Oracle expectancy is computed for at least 3 bar types × 3 oracle configs × 2 labeling methods
- [ ] Go/no-go decision is documented with supporting data
- [ ] Regime stratification (§9.6) completed with stability scores
- [ ] Feature taxonomy (§8.3) Track A is fully implemented with per-bar computation
- [ ] Feature taxonomy (§8.3) Track B raw representations exported (book snapshots + message summaries)
- [ ] Kyle's lambda computed over rolling 20-bar window (not single-bar)
- [ ] Warm-up and lookahead bias policy (§8.6) fully implemented with is_warmup flag
- [ ] Features exported to analyzable format (Parquet/CSV) with no unexpected NaN
- [ ] All C++ unit tests pass
- [ ] Results are reproducible (same data + config → same results)

**Research:**
- [ ] R1: Subordination hypothesis tested — Jarque-Bera, ARCH, volatility clustering compared across bar types with Holm-Bonferroni corrected p-values
- [ ] R1: Primary bar type recommended with empirical justification
- [ ] R2 Tier 1: Information decomposition estimated with hand-crafted message summaries — $R^2$ gaps computed for all representation tracks × return horizons
- [ ] R2 Tier 2: Learned message encoder (LSTM/transformer) tested on raw event sequences — definitive message encoder go/no-go
- [ ] R2: Architecture decision (§7.2 simplification cascade) filled in using Tier 2 results and revised threshold policy (relative gap > 20% AND significant)
- [ ] R3: Book encoder inductive bias tested — CNN vs. attention vs. MLP on raw book snapshots with significance tests
- [ ] R3: Spatial encoder architecture recommended
- [ ] R4: Entropy rate / temporal predictability compared across bar types with corrected p-values
- [ ] R4: Final bar type recommendation for temporal encoder
- [ ] MI analysis, GBT importance (stability-selected), and decay analysis completed for all feature × horizon × bar_type combinations
- [ ] All statistical tests report raw p-values, corrected p-values, and power analysis
- [ ] Bar type comparison completed with statistical tests per §8.5
- [ ] Synthesis document produced with go/no-go decisions, architecture recommendations, regime robustness assessment, and documented statistical limitations
