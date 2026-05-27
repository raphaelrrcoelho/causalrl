# Reproducible Benchmarks

The maintained benchmark runner reports per-seed values rather than only a favorable single
seed. Reports include the raw tail-reward observations, mean, sample standard deviation, and
a normal-approximation 95% interval.

## Run A Report

```bash
uv run --extra dev python benchmarks/scbandit_report.py confounded-chain \
  --seeds 0,1,2,3,4 --steps 8000 --tail-window 2000 --n-mc 2000
```

For the non-manipulable front-door demonstration:

```bash
uv run --extra dev python benchmarks/scbandit_report.py frontdoor \
  --seeds 0,1,2,3,4 --steps 30000 --tail-window 10000 --n-mc 20000
```

The commands emit JSON suitable for saving beside experiment configurations or consuming in
notebooks.

## Interpret Reports Conservatively

- `confounded-chain` compares a POMIS-restricted agent to brute-force and fixed-set
  Thompson-sampling baselines in the maintained chain SCM.
- `frontdoor` compares a manipulability-aware POMIS agent with the naive filter baseline in
  the maintained R-40-inspired SCM.
- Performance ordering in these environments validates those demonstrations; it is not
  evidence of universal RL superiority.

## Programmatic Use

```python
from causalrl import report_to_dict, run_confounded_chain_benchmark

report = run_confounded_chain_benchmark(seeds=(0, 1, 2))
print(report_to_dict(report))
```
