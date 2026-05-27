# Reproducing the Literature

A gallery of classic causal-inference and causal-RL results, each reproduced end-to-end with
`causalrl`. Every case below is a passing test in `tests/test_literature_classics.py` that asserts
the textbook result — run them with `pytest tests/test_literature_classics.py`.

## Causal-inference classics

| Case | What it shows | In causalrl |
|---|---|---|
| **Simpson's paradox** (kidney stones) | Adjusting for the confounder reverses the naive sign | `estimate_effect` (back-door) vs the raw association |
| **Front-door** (smoking → tar → cancer) | A confounded effect is still identifiable through a mediator | `identify_effect` / `estimate_effect` |
| **Pearl's napkin** | Identifiable despite latent confounding | `is_identifiable_effect` → `True` |
| **Instrumental variable** | Not point-identified, but boundable | `is_identifiable_effect` → `False`; `manski_bounds` |
| **Bow arc** | The simplest non-identifiable confounded effect | `is_identifiable_effect` → `False` |
| **Transportability** (LA → NYC covariate shift) | Re-weight a source effect to a new population | `identify_transport` |

## Difficult RL problems where causal beats associational

These are classic decision problems where a confounding-aware (causal) agent provably outperforms an
associational one on the same data:

| Problem | Associational RL | Causal RL |
|---|---|---|
| **MABUC** — bandit with unobserved confounders | `NaiveThompsonSampling` (ignores the context) | `CausalThompsonSampling` conditions on the "intuition" and earns more |
| **Greedy Casino** — counterfactual decision-making | best fixed arm `do(X=a)` ≈ 0.37 | acting on the counterfactual (intuition) ≈ 0.80 |
| **Hard exploration** — sparse far goal | flat Q-learning rarely reaches it | a causal prerequisite curriculum (`curriculum_q_learning`) does |

The point each makes: associational policies condition on what they *see*, while causal policies
reason about what they *do* (Layer 2) and what they *would have done* (Layer 3). Under unobserved
confounding the gap is not subtle — the counterfactual "follow your intuition" policy more than
doubles the best interventional arm's reward.
