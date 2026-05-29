# Citing causalrl

If you use `causalrl` in research, please cite the software. The canonical metadata is in
[`CITATION.cff`](https://github.com/raphaelrrcoelho/causalrl/blob/main/CITATION.cff); a BibTeX
entry:

```bibtex
@software{coelho_causalrl,
  author  = {Coelho, Raphael},
  title   = {{causalrl}: Causal intervention-selection and causal-RL research tools},
  year    = {2026},
  version = {0.99.0},
  license = {MIT},
  url     = {https://github.com/raphaelrrcoelho/causalrl}
}
```

## Please also cite the primary sources

`causalrl` implements published methods and is faithful to their originals. When you use a
specific capability, cite the paper it comes from — each is attributed inline in the
[Tour by Task](tour.md) and in the relevant source module. The load-bearing references include:

- **Causal RL taxonomy** — Bareinboim, *Causal Reinforcement Learning* (the 9-task framing).
- **do-calculus / identification** — Pearl, *Causality*; Shpitser & Pearl; Bareinboim & Pearl
  (gID, JMLR 2015).
- **Transportability** — Bareinboim & Pearl (AAAI 2012; NeurIPS 2014).
- **Structural causal bandits / POMIS** — Lee & Bareinboim (NeurIPS 2018; AAAI 2019). The POMIS
  engine is adapted from the MIT-licensed
  [`sanghack81/SCMMAB-NIPS2018`](https://github.com/sanghack81/SCMMAB-NIPS2018).
- **Bandits with unobserved confounders** — Bareinboim, Forney & Pearl (NeurIPS 2015).
- **Causal imitation** — Zhang, Kumor & Bareinboim (NeurIPS 2020).
- **Discovery** — Spirtes, Glymour & Scheines; Meek (UAI 1995); Zhang (2008, FCI).
- **Partial identification & sensitivity** — Manski (1990); Tan (2006).
