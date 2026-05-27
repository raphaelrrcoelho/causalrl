# Causal Discovery

`causalrl` learns causal structure from discrete observational data by constraint-based search,
testing conditional independence with conditional mutual information. See
[Guarantees And Scope](guarantees.md) for the precise contract (faithfulness, the thresholded test).

## PC — assuming no latent confounders

`discover` runs the PC algorithm and returns a `CPDAG` (a Markov-equivalence class of DAGs):
unshielded colliders are oriented, then Meek's rules propagate. It assumes *causal sufficiency*
(every common cause is observed) and faithfulness.

```python
from causalrl import discover

cpdag = discover(data, ["X", "Y", "Z"])
```

`discover_interventional` refines the CPDAG toward the true DAG using experimental (L2) data via the
invariance principle.

## FCI — allowing latent confounders

When causal sufficiency cannot be assumed, `discover_latent` runs the FCI algorithm and returns a
`PAG` (partial ancestral graph). FCI adds a Possible-D-SEP skeleton refinement (sound under latents)
and the complete orientation rules R1–R10 (Zhang 2008, including selection bias). In a PAG:

- `a -> b` — `a` is a cause of `b`;
- `a <-> b` — a **latent confounder** of `a` and `b` (neither causes the other);
- a circle endpoint `o` — undetermined by the equivalence class.

```python
from causalrl import discover_latent

pag = discover_latent(data, ["X", "Y", "Z"])
print(pag.render())
```

### Detecting a latent confounder

With `C -> A`, `D -> B`, and an unobserved `L` confounding `A` and `B`, the two unshielded colliders
force arrowheads at both ends of `A-B`, so FCI reports the confounder as `A <-> B`:

```python
pag = discover_latent(data, ["A", "B", "C", "D"])
assert pag.is_bidirected("A", "B")
```

### M-bias — do not condition on the collider

The classic M-bias structure `X <- L1 -> Z <- L2 -> Y` (with `L1`, `L2` latent) makes `Z` a collider.
FCI marks arrowheads at `Z` (`X o-> Z <-o Y`) and finds no `X-Y` edge — a warning that conditioning
on `Z` would open a spurious `X-Y` association.

Faithful to Spirtes, Glymour & Scheines (PC, FCI) and Zhang (2008) (the complete orientation rules).
