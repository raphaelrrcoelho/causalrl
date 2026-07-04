# Interop: certify a DoWhy / EconML estimate against confounding

DoWhy and EconML answer *"what is the effect?"* causalrl answers a different question —
*"does this ship / abstain decision survive hidden confounding?"* — and returns a **certificate**.

| Library | Answers | Output |
|---|---|---|
| DoWhy / EconML | What is the effect? | point estimate / CATE |
| **causalrl** | Does the decision survive hidden confounding? | ship / abstain **certificate** (tipping-Γ) |

The bridge is one typed object, `PolicyValueContrast`: any estimator that can name, per logged unit,
the outcome, the logging propensity, and the two target policies' action probabilities can be
certified by `certify_estimate`.

Install the optional dependencies with `pip install causalrl[interop]`.

## DoWhy (propensity weighting)

```python
from causalrl import certify_estimate
from causalrl.interop.dowhy import from_dowhy_estimate

# `estimate` is a fitted DoWhy propensity-based CausalEstimate; y / f / z are the logged
# outcome, binary treatment, and (optionally) a measured confounder.
contrast = from_dowhy_estimate(estimate, outcomes=y, treated=f, confounder_bins=z)
cert = certify_estimate(contrast, labels=("treated", "control"))
print(cert.recommendation)   # "act" or "abstain" — read the verdict, not cert.decision
print(cert)                  # human-readable summary with the tipping-Γ
```

The fitted propensity scores become the logging propensities, so both the marginal-sensitivity-model
(MSM) layer and — with a measured `confounder_bins` or a structural `mi_cap` — the pivotality layer
apply. **Honest scope:** the certified sensitivity is on the propensity weights, not on DoWhy's point
estimator.

## EconML (CATE-induced policy)

```python
from causalrl import certify_estimate
from causalrl.interop.econml import from_econml_cate

# `cate` is a fitted EconML CATE estimator; e0 are the logging/behaviour propensities.
contrast = from_econml_cate(cate, X, outcomes=y, treated=f, logging_propensities=e0)
print(certify_estimate(contrast))
```

This certifies the **value of the induced treatment policy** (`treat iff τ̂(x) > 0`) against its
complement under the MSM. **MSM layer only** — the induced policy varies per unit, so it is not a
fixed binary arm and the structural pivotality layer does not apply. The object certified is the
induced policy's value, not the CATE point estimate itself.

## Runnable examples

- [`examples/interop_dowhy_certify.py`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/interop_dowhy_certify.py)
- [`examples/interop_econml_certify.py`](https://github.com/raphaelrrcoelho/causalrl/blob/main/examples/interop_econml_certify.py)

Each runs the full estimate → certificate flow and prints a `DecisionCertificate`; both exit cleanly
with a skip message if the optional dependency is absent.
