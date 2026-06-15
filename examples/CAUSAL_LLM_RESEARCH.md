# Building a small causal LLM — grounded research synthesis

> Local research note (not part of the published library docs). Method caveat: the underlying
> literature sweep hit HTTP 403 on direct full-text fetches, so citations were reconstructed from
> abstracts/snippets and cross-checked against known primary sources. Pre-2025 papers are
> high-confidence and mutually corroborating; arXiv IDs with late-2025/2026 numbering
> (`2602.*`, `2512.*`, `2601.*`) are **provisional**.

This note records the theory behind, and a recommended architecture for, a **small language
model with genuine causal reasoning** (Pearl/Bareinboim sense: it must distinguish and correctly
compute L1 *seeing*, L2 *doing*, and L3 *counterfactual* queries — not just correlation). It is
the research backing for the `examples/causal_*` prototypes.

## 1. The theoretical wall that defines the problem

**Causal Hierarchy Theorem (CHT)** — Bareinboim, Correa, Ibeling & Icard, *On Pearl's Hierarchy
and the Foundations of Causal Inference* (2022), [ACM/Pearl book](https://dl.acm.org/doi/10.1145/3501714.3501743):
the set of SCMs in which a higher layer (L2/L3) is determined by a lower layer (L1) has
**measure zero** (and is topologically *meager*). You cannot infer intervention/counterfactual
from observation alone without structural assumptions — "no causes in, no causes out."

**The Causal-Neural Connection** — Xia, Lee, Bengio, Bareinboim, NeurIPS 2021,
[arXiv:2107.00793](https://arxiv.org/abs/2107.00793):

- **Expressiveness ≠ identifiability** (Corollary 1): an NCM that fits `P(V)` perfectly does *not*
  in general recover `P(V | do(x))`. Network capacity does not buy causality.
- Fix: the **G-constrained NCM (G-NCM)** — bake the causal diagram's constraints into the
  architecture as an inductive bias. *(This is exactly the graph-surgery forward pass in
  `examples/causal_ncm_reasoning.py`.)*
- **Duality (Theorem 4):** a query is classically identifiable **iff** it is neural-identifiable;
  the arbiter is do-calculus, which is **sound and complete** (Shpitser & Pearl,
  [JMLR 2008](https://www.jmlr.org/papers/volume9/shpitser08a/shpitser08a.pdf); Huang & Valtorta 2006).
- **L3 needs more:** counterfactuals are point-identified only under extra structural assumptions
  such as **monotonicity** (Tian & Pearl, *Probabilities of Causation*); otherwise only bounds.
  Xia, Pan & Bareinboim, *Neural Causal Models for Counterfactual ID*, ICLR 2023,
  [arXiv:2210.00035](https://arxiv.org/abs/2210.00035). *(This is the threshold-monotone mechanism
  assumption encoded in the NCM prototype's L3 path.)*

**Architectural consequence (hard):** any "causal LLM" needs one of three things — there is no way
around the theorem: (i) be given/learn the graph and respect it structurally; (ii) be trained on
real interventional/counterfactual data; or (iii) call an engine that runs do-calculus. A decoder
trained only on observational text *cannot* become causal on its own.

## 2. Empirical reality: LLMs are "causal parrots" where it counts

- **Causal Parrots** — Zečević et al., TMLR 2023, [arXiv:2308.13067](https://arxiv.org/abs/2308.13067):
  LLMs recite correlations among *causal facts* (a "meta-SCM"), they do not infer.
- **Corr2Cause** — Jin et al., ICLR 2024, [arXiv:2306.05836](https://arxiv.org/abs/2306.05836):
  17 LLMs (incl. GPT-4) ≈ random on formal causal inference; fine-tuning does **not** generalize OOD
  (GPT-4 ≈ 29 macro-F1 zero-shot).
- **CLadder** — NeurIPS 2023, [arXiv:2312.04350](https://arxiv.org/abs/2312.04350): GPT-4 + CausalCoT
  ≈ 70%; counterfactual tier stays weakest (~62%).
- **Contested tension:** Kıcıman et al., [arXiv:2305.00050](https://arxiv.org/abs/2305.00050), report
  ~97% on Tübingen cause-effect pairs — but a critical review,
  [arXiv:2407.08029](https://arxiv.org/abs/2407.08029), attributes this to *memorized knowledge
  retrieval* and benchmark leakage, not causal computation.

**The encouraging result for "small + best theory":**
**Axiomatic Training** — Vashishtha et al., NeurIPS 2024,
[arXiv:2407.07612](https://arxiv.org/abs/2407.07612): a **67M-parameter transformer trained from
scratch** on demonstrations of causal axioms (transitivity, d-separation) **generalizes** to longer,
reversed, and branching graphs; axiomatic fine-tuning of Llama-3-8B beats GPT-4 on Corr2Cause. This
is the proof that genuine *structural* causal generalization is learnable at small scale.

## 3. Architecture space, mapped to the prototypes

| Approach | What it is | Theory | Honest limit |
|---|---|---|---|
| (a) Data-conditioned LM (`causal_lm_real_from_scratch.py`) | `<see>`/`<do>` tokens + both regimes in data | none; learns the distribution it saw | memorises regimes; CHT ⇒ no generalisation to unseen queries |
| (b) Differentiable NCM (`causal_ncm_reasoning.py`) | graph surgery + abduction in the forward pass | G-NCM + ID duality (Xia 2021); L3 via monotonicity | needs known graph + identifiability; not an LM |
| (c) Hybrid LLM + engine | LLM parses NL→(graph, query); engine runs do-calculus; LLM verbalises | do-calculus completeness; symbolic verifier | accuracy bounded by **graph elicitation** |
| (d) Axiomatic / structural training (`causal_reasoner_prototype.py`) | transformer learns causal *rules* from SCM-generated traces | Axiomatic Training (67M works) | narrow scope; general is open frontier |

Hybrid systems cover only *parts* of the chain: **Causal Agent**
([arXiv:2408.06849](https://arxiv.org/abs/2408.06849)), **Causal-Copilot**
([arXiv:2504.13263](https://arxiv.org/abs/2504.13263)), **Long et al. "imperfect experts"**
([arXiv:2307.02390](https://arxiv.org/abs/2307.02390)). The decisive caveat: **CausalGraph2LLM**
([arXiv:2410.15939](https://arxiv.org/html/2410.15939v1)) — GPT-4/Gemini vary **~60%** by graph
text-encoding alone. Elicitation, not the engine, is the bottleneck.

## 4. Recommended architecture — "small Causal Reasoner"

Do **not** rely on (a) alone. Combine **(d) axiomatic training** as the spine + **(b) the NCM** as a
numeric reasoning head + `causalrl` as **generator AND verifier**, honouring the CHT by construction.

1. **Backbone:** small from-scratch transformer, TinyStories/nanoGPT scale (≤100M, single GPU —
   feasible: TinyStories, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759), shows <10M speaks
   coherent English; phi, [arXiv:2306.11644](https://arxiv.org/abs/2306.11644), shows curated data >
   scale for a target skill).
2. **Objective = causal traces, not text:** `causalrl` generates **derivation demonstrations**
   (graph → query → d-separation / do-calculus steps → answer). Teaches the *rule*, not the *fact* —
   what generalises. Pieces already in the lib: `identify_effect` / `is_identifiable` (complete ID),
   `_separation` (d-separation), `transport_formula`, and `StructuralCausalModel.see/do/counterfactual`
   for labelled L1/L2/L3 traces.
3. **Reasoning head = the differentiable NCM** as a callable module for numeric L2/L3 answers
   **when identifiable**. The transformer parses NL→(graph, estimand)→calls the head→verbalises.
   **Gated by `is_identifiable`:** if non-identifiable, **abstain or return bounds**
   (`manski_bounds`, `tipping_gamma`) — honouring the CHT instead of hallucinating.
4. **Post-training with RLVR + causal verifier:** DeepSeek-R1 template
   ([Nature 2025](https://www.nature.com/articles/s41586-025-09422-z)), with `causalrl` as the
   reward oracle (compare against the SCM's `see/do/counterfactual` or the ID estimand). *Honest flag:
   RLVR with a causal verifier is a principled extrapolation, not yet published — the frontier part.*

**Why this combination:** axiomatic training is the only evidence a *small, from-scratch* model learns
*structural* causal reasoning that generalises; the NCM gives correct L2/L3 numbers by construction
(ID duality); the identifiability gate respects the theorem that says the rest is unanswerable; and
`causalrl` is uniquely suited to serve both roles — trace **generator** and **verifier/reward** —
because it has complete ID + an executable SCM + partial-ID bounds.

**Trade-offs (honest):**

- ✅ Feasible on one GPU: a *narrow, well-scoped* causal skill (axiomatic / TinyStories / phi confirm).
- ❌ A *general* from-scratch causal LLM at Qwen scale does not exist and is open research.
- ⚠️ Naive counterfactual data augmentation (CAD, [arXiv:1909.12434](https://arxiv.org/abs/1909.12434))
  has **contested** robustness — fails OOD for lack of diversity
  ([EMNLP 2021](https://aclanthology.org/2021.emnlp-main.28.pdf)). Prefer *derivation traces* (rules)
  over *edited examples* (facts).
- ⚠️ Pure hybrid (c) is capped by graph elicitation (~60% variance) — the graph must come from a
  trusted source / SCM, not be guessed by the LLM.

## 5. Prototype ladder in this repo

1. `examples/causal_lm_from_scratch.py` — 6-token toy: proves do≠see is learnable. *(approach a, minimal)*
2. `examples/causal_lm_real_from_scratch.py` — real from-scratch GPT-2 learning do≠see in natural
   language. *(approach a)*
3. `examples/causal_ncm_reasoning.py` — differentiable NCM: do() by graph surgery, counterfactual by
   abduction-action-prediction, trained obs-only, generalises to unseen identifiable queries.
   *(approach b)*
4. `examples/causal_reasoner_prototype.py` — small transformer trained on **identifiability-decision
   traces** generated and verified by `causalrl`; tests generalisation to unseen graph sizes. The
   first executable slice of the recommended architecture. *(approach d + identifiability gating)*
