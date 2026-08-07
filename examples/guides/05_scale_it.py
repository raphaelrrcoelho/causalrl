"""Task guide 5: scale the same certificate to a large log.

The streaming kernels consume a TrajectoryLog one batch at a time and never materialise it, so the
same importance-sampling off-policy certificate you compute on a toy log runs over a log far larger
than memory. (Swap the in-memory log for a Parquet path — `log.to_parquet(path)` then
`stream_policy_value(path)` — and it streams row-groups from disk unchanged.)

Run: python examples/guides/05_scale_it.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl import TrajectoryLog
from causalrl.ope.ipw import stream_policy_value


def build_large_log(n: int, seed: int) -> TrajectoryLog:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    behaviour = 1.0 / (1.0 + np.exp(-0.8 * z))
    action = rng.binomial(1, behaviour).astype(float)
    reward = 1.5 * action + z + 0.5 * rng.standard_normal(n)
    target = 1.0 / (1.0 + np.exp(-0.4 * z))
    pb = action * behaviour + (1.0 - action) * (1.0 - behaviour)
    pt = action * target + (1.0 - action) * (1.0 - target)
    weight = pt / pb
    rows: list[dict[str, Any]] = []
    for i in range(n):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "w", "name": "weight", "value": float(weight[i])})
        rows.append({**base, "kind": "r", "name": "reward", "value": float(reward[i])})
    return TrajectoryLog.from_rows(rows).sorted_by_key()


def main() -> None:
    log = build_large_log(300_000, seed=0)  # 600k rows, streamed in 100k-row batches
    cert = stream_policy_value(log, weight="weight", reward="reward", batch_size=100_000)
    print(cert)
    assert cert.value is not None and cert.ci is not None
    ess = cert.assumptions[0].diagnostic["ess"] if cert.assumptions[0].diagnostic else None
    print(f"streamed {len(log):,} rows -> V(pi) = {cert.value:.4f}  (ESS ~ {ess:.0f})")
    print("OK — the same certificate at scale, without materialising the log")


if __name__ == "__main__":
    main()
