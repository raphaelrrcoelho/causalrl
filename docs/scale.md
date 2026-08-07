# Scale: confounded offline RL with d3rlpy

causalrl is not a trainer. At real scale, **d3rlpy trains and causalrl certifies**: learn a policy
with any d3rlpy offline algorithm on your confounded logs, then bound whether its value improvement
over the behaviour policy survives hidden confounding.

Install the optional dependency with `pip install causalrl[scale]`.

## The division of labour

| Step | Tool |
|---|---|
| Bring confounded logs into a trainer | `causalrl.scale.d3rlpy.to_mdp_dataset` |
| Learn a policy at scale | d3rlpy (CQL, BCQ, …) |
| Certify the learned policy vs. behaviour | `causalrl.certify_policy` |

```python
import numpy as np
from d3rlpy.algos import DiscreteCQLConfig

from causalrl import certify_policy
from causalrl.scale.d3rlpy import to_mdp_dataset

# `dataset` is a causalrl ConfoundedTrajectoryDataset (e.g. from causalrl.generate_logs)
algo = DiscreteCQLConfig().create()
algo.fit(to_mdp_dataset(dataset), n_steps=1000)

obs = np.eye(dataset.n_states, dtype=np.float32)[[tr.state for tr in dataset.transitions]]
target_actions = algo.predict(obs).tolist()

cert = certify_policy(dataset, target_actions)
print(cert.recommendation)   # "act" (ship the learned policy) or "abstain"
print(cert)                  # tipping-Γ: how much hidden confounding would erase the improvement

# Safe policy improvement: also gate on a calibrated downside (finite-sample, distribution-free).
gated = certify_policy(dataset, target_actions, alpha=0.1)
print(gated.conformal_lcb)   # one decision under π returns at least this, w.p. ≥ 0.95
print(gated.recommendation)  # "abstain" unless that beats the logging policy's own downside
```

## Honest scope

`certify_policy` bounds the off-policy value contrast `V(π) − V(behaviour)` under Tan's marginal
sensitivity model, with the nominal logging propensities taken from the dataset's empirical
`behavior_propensity`. The behaviour value is the logged empirical return (on-policy for the
behaviour policy, hence unconfounded). This is the one-step / terminal-return contrast; the per-step
cumulative-reward extension uses `causalrl.msm_per_step_bounds`. d3rlpy owns training; causalrl adds
only the bridge and the certificate.

`alpha` adds an orthogonal, finite-sample layer: `causalrl.conformal_action_value` calibrates the
band of **one decision's return** under π by feeding the propensity ratio `π/π_behaviour` into the
weighted conformal path. It is not a confidence interval for `V(π)` — a policy with the better mean
and a heavier downside fails this gate on purpose — and it refuses (rather than passes) when the
effectively-weighted sample is too small to calibrate the level at all.

Runnable end-to-end: [`examples/scale_d3rlpy_certify.py`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/scale_d3rlpy_certify.py).
