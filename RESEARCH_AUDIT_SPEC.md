# Research Audit: MES Backtest Program — State of Knowledge

**Date:** 2026-02-18
**Purpose:** Comprehensive audit of all completed research before proceeding to model architecture build.

---

## Why This Audit Exists

The MES backtest research program has completed 10+ experiments across multiple lines of inquiry (R1–R6, R4a–R4d, oracle backtest, feature discovery). Findings are spread across multiple result files, analysis documents, and spec revisions. A fresh agent cannot reliably reconstruct the full picture from any single document.

This audit produces a single source of truth: what we know, what we don't know, what's been decided, and what remains open. No new analysis. No new code. Just read, reconcile, and summarize.

---

## Task

Read every result file and analysis document listed below. For each, extract the core finding and its implications. Then produce the synthesis document described in §Output.

### Files to Read (in order)

The research agent should locate these under `.kit/` (results, experiments, docs). Paths may vary slightly — use `find .kit -name "*.md" -o -name "*.csv" -o -name "*.json"` to inventory all available files first.

**Spec documents:**
- Main backtest spec (likely `.kit/docs/` — the v0.2/v0.3 document covering bar construction, oracle design, feature taxonomy, research cycles R1–R4)
- Hybrid model spec (`.kit/docs/hybrid-model.md`)

**Research results (in execution order):**
- R1: Subordination / bar type comparison
- R2: Message encoder / information decomposition
- R3: Spatial encoder (CNN vs. attention vs. engineered features)
- R4: Temporal predictability on time_5s
- R4b: Temporal predictability on volume_100, dollar_25k
- R4c: Temporal predictability completion (tick bars, extended horizons, event bar calibration)
- R5: (if exists) Any additional research cycle
- R6: Synthesis / architecture decision
- Oracle backtest results (GO/NO-GO gate)
- Feature discovery / Tier 1 / Tier 1.5 results

**Phase B artifacts:**
- Any exported data files (e.g., `time_5s.csv`)
- Any partial model training results

If any listed experiment has no corresponding result file, note it as "NO RESULT FOUND" — do not speculate about what the result might have been.

---

## Output

Produce a single markdown file: `.kit/results/research-audit/audit.md`

Structure it as follows:

### Section 1: Experiment Registry

A table of every experiment, its status, and one-line verdict:

```
| ID | Name | Status | Verdict |
|----|------|--------|---------|
| R1 | Subordination / bar type | COMPLETE | [one line] |
| R2 | Message encoder | COMPLETE/PARTIAL/NOT FOUND | [one line] |
| ...| ... | ... | ... |
```

### Section 2: Core Findings (one paragraph each)

For each completed experiment, state:
1. What question it answered
2. What the answer was (with key numbers)
3. What architectural/design decision it drove

Be precise. Include R² values, p-values, and effect sizes where available. Do not editorialize or reinterpret — just report what the analysis documents say.

### Section 3: The Two Types of Predictability

This is the most important section. The research program has tested two fundamentally different questions that are easy to conflate:

**Temporal predictability:** Do past returns / past bar features predict future returns? (R4 chain)
**Spatial predictability:** Does the current book state predict near-future returns? (R3, feature discovery)

For each, summarize:
- Which experiments tested it
- What the finding was
- At what bar types and timescales
- With what statistical confidence

Make the distinction crystal clear. A reader should walk away understanding that "MES returns are martingale" (temporal) and "current book state has predictive power" (spatial) are not contradictory claims.

### Section 4: Bar Type Decision Audit

Trace the bar type decision from R1 through R6 to the current time_5s choice. For each step:
- What evidence supported the decision
- What alternatives were considered
- What gaps remain

Specifically address:
- R4b found dollar_25k static R² = 0.080 at h=1 (vs. ~0.003 for time_5s). Why wasn't this sufficient to switch to dollar bars?
- R4c/R4d calibrated dollar bars to actionable timescales. What was the static-feature R² at those timescales? (If this was never tested — i.e., R4c/R4d only tested temporal features at actionable thresholds but not spatial features — flag this as an open gap.)
- Is the time_5s choice justified by the evidence, or is it a default that was never properly challenged?

### Section 5: Open Questions and Gaps

List every unresolved question or untested condition you identify, categorized as:

**Blocking (must resolve before model build):**
- Questions whose answers could change the model architecture, bar type, or feature set

**Non-blocking (can defer):**
- Questions that would refine but not fundamentally change the approach

**Closed (no further investigation needed):**
- Questions that have been definitively answered with sufficient evidence

### Section 6: Readiness Assessment

A direct answer: **Is the research program ready to proceed to model architecture build?**

If yes: state what's been validated and what the build should use (bar type, feature set, architecture, labels).

If no: state what must be resolved first, in priority order.

If conditional: state the conditions and what changes if they're not met.

---

## Rules for the Research Agent

1. **Read first, write second.** Inventory all files before writing anything. The file list above is a guide, not exhaustive — there may be additional results, logs, or specs.

2. **Report, don't reinterpret.** If an analysis document says R² = 0.080, report 0.080. Do not adjust, re-derive, or second-guess the number. If you believe a number is wrong, flag it as "POSSIBLE DISCREPANCY" with your reasoning, but still report the original.

3. **Flag gaps explicitly.** If a question was raised in a spec but no corresponding result exists, say so. If an experiment tested condition A but the decision framework also required condition B, and B was never tested, say so.

4. **No new analysis.** Do not run any code, compute any statistics, or produce any new results. This is a documentation audit, not a research experiment.

5. **No architecture recommendations.** Section 6 assesses readiness. It does not prescribe what to build. That's the next step after the audit.

---

## Compute Budget

Zero. This is a read-only task. If you find yourself wanting to run code, stop and flag what you wanted to compute as a gap instead.

---

## Exit Criteria

- [x] All result files inventoried and read
- [x] Experiment registry table complete
- [x] Section 3 (two types of predictability) clearly distinguishes temporal vs. spatial
- [x] Section 4 (bar type audit) identifies whether static-feature R² was tested at actionable event-bar timescales
- [x] Section 5 gaps are categorized as blocking/non-blocking/closed
- [x] Section 6 gives a clear readiness assessment
- [x] No code was executed
- [x] Output written to `.kit/results/research-audit/audit.md`
