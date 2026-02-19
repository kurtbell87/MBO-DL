# Comparison Table: Tick Bars vs Time_5s Baseline

| Threshold | Bars/Day | CV | Duration (med) | Mean R2 | Std | Fold 3 R2 | Delta vs 0.089 | Verdict |
|-----------|----------|------|----------------|---------|------|-----------|----------------|---------|
| time_5s (baseline) | 4,630 | 0.000 | 5.0s | 0.0890 | 0.074 | -0.049 | -- | BASELINE |
| tick_25 | 16,836 | 0.188 | 1.4s | 0.0636 | 0.0537 | 0.0037 | -0.0254 | WORSE |
| tick_100 | 4,171 | 0.190 | 5.7s | 0.1244 | 0.1074 | -0.0583 | +0.0354 | BETTER |
| tick_500 | 794 | 0.200 | 28.5s | 0.0499 | 0.0546 | -0.0043 | -0.0391 | WORSE |