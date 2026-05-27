# Transportability

Transportability decides whether a causal effect learned in one or more *source* domains transfers
to a *target* domain that differs in some mechanisms, and produces the formula to compute it. Domains
are related by a **selection diagram** — a causal graph whose selection-marked variables have a
mechanism that differs from the target.

`causalrl` implements this over the ID algorithm: the target effect is decomposed into c-factors and
each is taken from whichever domain can supply it. With a single observational source this is the
complete single-domain sID; surrogate experiments (mz) and multiple source domains (meta) extend it.
See [Guarantees And Scope](guarantees.md) for the precise contract.

Faithful to E. Bareinboim & J. Pearl, *Transportability of Causal Effects: Completeness Results*
(AAAI 2012), *A General Algorithm for Deciding Transportability of Experimental Results* (Journal of
Causal Inference 2013), and *Meta-Transportability of Causal Effects* (AISTATS 2013).

## Covariate shift (single source)

The running example: a treatment `X`, an outcome `Y`, and a covariate `Z` whose distribution differs
between source and target (`Z → X`, `Z → Y`, `X → Y`). The effect transports by re-weighting the
source's invariant `Y` mechanism with the target's `P*(Z)`:

```python
from causalrl import CausalGraph, identify_transport

graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])
estimand = identify_transport(graph, ["X"], ["Y"], selection=["Z"])
print(estimand.render())  # mixes P_target(Z) with the invariant P_source(...)
```

The shifted `Z` is taken from the target; the invariant `Y` mechanism from the source.

## Surrogate experiments (mz)

When a c-factor is a hedge that no observational distribution identifies, an experiment in a source
domain can supply it. For a bow arc (`X ↔ Y`, `X → Y`) the effect is not transportable from
observation alone, but a source `do(X)` experiment identifies it:

```python
from causalrl import CausalGraph, Domain, is_transportable_general

graph = CausalGraph(directed_edges=[("X", "Y")], bidirected_edges=[("X", "Y")])
assert not is_transportable_general(graph, ["X"], ["Y"], [Domain("source")])

source = Domain("source", experiments=frozenset({frozenset({"X"})}))
assert is_transportable_general(graph, ["X"], ["Y"], [source])
```

## Multiple domains (meta)

With several source domains marked differently, each invariant c-factor is contributed by a domain
where it is invariant. With `Z1` shifted in domain `a` and `Z2` shifted in domain `b`, the target
effect is assembled from both:

```python
from causalrl import CausalGraph, Domain, identify_transport_general

graph = CausalGraph(
    directed_edges=[("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")]
)
domains = [Domain("a", frozenset({"Z1"})), Domain("b", frozenset({"Z2"}))]
print(identify_transport_general(graph, ["X"], ["Y"], domains).render())  # references P_a and P_b
```

Evaluate any of these on data with `estimate_transport_general` — one observational dataset per
domain name (including `"target"`), plus a randomized dataset per `(domain, intervention)` pair for
experiments. When no domain can supply a needed c-factor the call raises `NotIdentifiableError` with
the witnessing transport-hedge.
