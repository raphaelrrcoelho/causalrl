# Tutorials

End-to-end notebooks that go beyond the single-call examples in the
[Tour by Task](tour.md) and walk through a full problem across every layer of the library
(SCM → environment → agents → evaluation). They live in
[`examples/`](https://github.com/raphaelrrcoelho/causalrl/tree/main/examples) and render
directly on GitHub; run them locally after `uv sync --extra dev`.

## MABUC vertical slice

[`examples/mabuc_vertical_slice.ipynb`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/mabuc_vertical_slice.ipynb)

The end-to-end walkthrough of the Multi-Armed Bandit with Unobserved Confounders: build the
structural causal model, wrap it as a Gymnasium environment, run the causal and naive Thompson
agents, and evaluate why conditioning on intent beats a confounding-naive policy even when the
interventional arm means are identical.

## Causal offline-to-online

[`examples/offline_to_online.ipynb`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/offline_to_online.ipynb)

The three-way comparison on a confounded dynamic treatment regime: a naive offline learner that
trusts the logs, against UC-DTR / DOVI / DeepDeconfoundedQ agents that read the same logs through
Manski causal bounds and recover the optimal policy.

## Where to intervene (POMIS)

[`examples/where_to_intervene.ipynb`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/where_to_intervene.ipynb)

Possibly-Optimal Minimal Intervention Sets on a confounded chain: how POMIS prunes the
exponential intervention space to the few sets that could be optimal, and why a POMIS-restricted
agent learns far faster than brute force over all interventions.

---

To run a notebook end to end:

```bash
uv sync --extra dev
uv run jupyter lab examples/
```
