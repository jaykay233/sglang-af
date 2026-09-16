# Starvation ablation numbers

| Mode | file | quiet mean | flood mean | flood p99 | flood max |
|---|---|---:|---:|---:|---:|
| Baseline (prefill-first) | starvation_baseline_heavy.json | 1.8 | 13.4 | 173.6 | 473.6 |
| Mixed chunk only | starvation_mixed_heavy.json | 1.9 | 14.5 | 59.8 | 379.5 |
| Mixed + stall guard | starvation_budget_heavy.json | 1.8 | 14.6 | 59.4 | 378.6 |
