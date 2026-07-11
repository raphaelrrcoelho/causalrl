# Migrating to causalrl 2.0

causalrl 2.0 is the first release with a **breaking change**. It is small, mechanical, and was
announced with a `FutureWarning` since 1.5. This page tells you exactly what changed and how to
update.

## The one breaking change: certificates by default

Three routines that had both a legacy return type and a certificate-returning variant now return a
[`Certificate`](guarantees.md) **by default**:

| Routine | 1.x default | 2.0 default |
| --- | --- | --- |
| `identify_effect` | `Estimand` | `Certificate` (`IDENTIFIED`) |
| `ipw_sensitivity_bounds` | `Interval` | `Certificate` (`BOUNDED`) |
| `msm_policy_value_bounds` | `Interval` | `Certificate` (`BOUNDED`) |

Everything else in the public API is unchanged: no other function changed its return type, and no
name was removed.

### If you want the certificate (recommended)

Do nothing — the default now gives you a unified, serializable `Certificate` that carries the claim,
the epistemic `kind`, the numeric value, the assumptions consumed, and provenance:

```python
from causalrl import ipw_sensitivity_bounds

cert = ipw_sensitivity_bounds(outcomes, propensities, gamma=1.5)  # -> BOUNDED Certificate
print(cert.kind, cert.value)          # Kind.BOUNDED  Interval(lower=..., upper=...)
print(cert.to_json())                 # serialize the whole claim
```

### If you want the legacy value

Pass `return_certificate=False`. This is the exact 1.x behaviour and it is **not** deprecated — it
is a supported escape hatch that returns the raw `Estimand` / `Interval`:

```python
interval = ipw_sensitivity_bounds(outcomes, propensities, gamma=1.5, return_certificate=False)
lo, hi = interval                     # unchanged 1.x Interval
```

A mechanical migration is: find bare calls whose result you index, unpack, or compare as an
`Interval` / `Estimand`, and add `return_certificate=False`. The certificate wraps *exactly* the
same numerics (a byte-stability test pins this), so nothing else needs to change.

### Removed

- The pre-2.0 `FutureWarning` and the private `causalrl._deprecation` helper are gone (the flip they
  announced has happened).

Error behaviour is unchanged: invalid inputs still raise the same exceptions on the default path.

## What else is new in 2.0

2.0 is mostly **additive** — the breaking change above is the only removal. Highlights across the
1.3 → 2.0 line:

- **Unified certificates** (`Certificate`, `Kind` ∈ `IDENTIFIED` / `BOUNDED` / `EMPIRICAL`) that the
  shipped `DecisionCertificate` / `PivotalityCertificate` / `TransportRegretCertificate` conform to
  via `as_certificate`.
- **Continuous causal core**: doubly-robust `certify_effect`, sharp estimated-propensity and
  heavy-tailed bounds, conformal wrappers, neural / flow mechanisms with exact abduction, sequential
  DR, and transport *estimation*.
- **Multi-agent core** (`causalrl.magames`): typed populations, `certify_equilibrium`, per-agent
  `CausalEnv` views, and a duck-typed PettingZoo adapter.
- **Scale & data plane**: streaming certificate kernels (`stream_policy_value`, `stream_msm_bounds`,
  `stream_quantile_certificate`) over a columnar `TrajectoryLog` — in memory or streamed from
  Parquet without materialising it — plus an optional JAX backend.
- **Interop & scale**: an SBI/NumPyro posterior → `Regime` bridge (`regimes_from_posterior`,
  `across_regimes`), a generic columnar-simulator adapter (`ColumnarSimulator`), and a deepened
  d3rlpy path (`TrajectoryLog` ↔ `MDPDataset` both ways, `certify_fqe`, `policy_actions`).

See the [task guides](guides.md) for runnable, end-to-end walkthroughs, and the
[assumption glossary](assumptions.md) for what each certificate's assumptions mean.
