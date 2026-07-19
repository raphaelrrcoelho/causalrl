# Anonymized code artifact

Instruments and experiment scripts for the submission
"When Can You Trust an Equilibrium Counterfactual?".

`eqcert/` is the minimal, anonymized subset of a larger open-source library
(name and links redacted for double-blind review) containing the instruments
the paper uses: the equilibrium-vs-unrolling comparator and spectral
diagnostics (`eqcert/experimental/cyclic/`), and the CCE polytope constructor,
LP bounds, measured realized regret, and game certifier (`eqcert/magames/`),
including the dependency-free two-phase simplex with dual multipliers
(`eqcert/magames/_lp.py`).

## Requirements

Python >= 3.11, numpy. (`paper_figs.py` additionally needs matplotlib.)

## Reproducing the paper's numbers

From this directory:

```
PYTHONPATH=. python experiments/e1_cobweb.py     # Table 2 (cobweb ladder)
PYTHONPATH=. python experiments/e2_saf_chaos.py  # Section 6.2 (Bounded rung)
PYTHONPATH=. python experiments/e4_macro_loop.py # Figure 1 numbers (sign flip)
PYTHONPATH=. python experiments/e7_basins.py     # Figure 2 numbers (selection)
PYTHONPATH=. python experiments/paper_figs.py    # regenerates both figures into figs/
```

Seeds are fixed in-script; every number in the paper is a direct output of
these runs. LP values can be re-verified independently of the shipped simplex
via the weak-duality identity stated in Appendix C of the paper.
