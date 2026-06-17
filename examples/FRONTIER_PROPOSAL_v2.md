# Frontier proposal v2 — Causal Grounding of LLM Cognition

> Overcome the root cognitive limitation of LLMs — **reasoning over surface correlations instead of a
> grounded causal model** — by *identifying* the latent causal variables inside the network (via
> interchange interventions) and *installing/strengthening* them (interchange-intervention training),
> so the model's output is causally mediated by a faithful internal causal model of the task. Prove it
> by intervention: turn the installed mechanism off and the cognitive gain disappears. `causalrl` is
> the ground-truth world that makes faithfulness exactly measurable.

This replaces the earlier "external symbolic engine" proposal. The target is **causality inside the
model's mechanisms**, not causal inference as an external task.

## 1. Decision and thesis

- **Most cognitively powerful target:** *causal grounding* — make the model's cognition causally
  mediated by a faithful internal causal model of the situation rather than by surface statistics.
- **Primary limitation:** *correlational reasoning* (the "causal parrot"). It is the **root**;
  hallucination and OOD brittleness are downstream symptoms of pattern-matching instead of reasoning
  over a grounded causal model. Fixing the root subsumes the others.
- **Mechanism:** identify-and-install — locate the latent causal variable, then shape the model so a
  designated internal subspace *causally implements* it. This is "repair via internal" taken to its
  limit, where it becomes "install causal structure".
- **Explicitly NOT relied on:** the model's own introspection / chain-of-thought self-report — these
  are documented to be unfaithful (Turpin et al. 2023, *Language Models Don't Always Say What They
  Think*). We use **external interchange interventions**, which are faithful by construction.

**The disease.** An LLM's output is `P(next token)`. There is no internal mechanism that separates
"this follows from a causal/structural model of the situation" from "this is associatively likely."
The causal parrot is the symptom; the disease is the absence of a *grounded causal latent* the model
reasons over. The leap is to install one and route cognition through it.

## 2. Approach — locate → diagnose → install → attribute

The methods exist (this is *causal interpretability*); the leap is going from **describing** circuits
to **installing grounded cognition with causal attribution**.

1. **Locate the latent causal model.** **Distributed Alignment Search (DAS)** (Geiger, Wu, Potts,
   Icard, Goodman) + **interchange interventions** (activation patching = `do()` on activations): find
   the subspace(s) in which the model represents the *task's causal variables* (e.g. the see/do regime,
   a latent confounder, the entities/relations of a reasoning problem).
2. **Diagnose the failure causally.** **Causal mediation analysis** (Vig et al. 2020; ROME / causal
   tracing, Meng et al. 2022): quantify how much the output is mediated by the *structural latent* vs by
   *surface features*. Hypothesis: on causal-parrot failures the output is mediated by the surface path
   — the model either lacks the latent or fails to route through it.
3. **Install the causal latent (the powerful step).** **Interchange Intervention Training (IIT)**
   (Geiger et al. 2022, *Inducing Causal Structure for Interpretable Neural Networks*): train so a
   designated subspace *causally implements* the target causal variable — i.e. patching that subspace
   between examples changes the output exactly as the high-level causal variable would. This shapes the
   model's internal causal structure directly; it is not fine-tuning on more data.
4. **Attribute by intervention (what makes it mechanistic control, not another fine-tune).** Show the
   cognitive gain (causal-reasoning accuracy, OOD robustness, reduced confabulation) **and** that it
   *vanishes when the installed mechanism is ablated*. Causal attribution to a mechanism you can turn
   on/off is the proof.

## 3. Why `causalrl` is the uniquely right testbed

The Achilles heel of causal interpretability: to ask "is the output mediated by the *true* causal
variable?", you must **know the true causal variable**. In natural tasks you do not. `causalrl`
*generates tasks from an SCM*, so the target causal structure is **known exactly**, giving:

- a **known alignment target** for DAS/IIT → faithfulness is measured exactly, not guessed;
- **ground-truth counterfactuals** to quantify mediation;
- a **leakage-free** environment that separates "installed a causal latent" from "memorised surface".

`causalrl` is the controlled world with **known internal-target causal structure** where the method is
validated and calibrated *before* transfer to a natural-language task. It is the test bench, not the
activation tooling.

## 4. First experiment — end to end (the flag-planting result)

**Domain.** A small decoder trained on a `causalrl` see/do task (cf. `causal_lm_real_from_scratch.py`):
serialized scenarios where outcome `Y` depends on treatment `X` and a **latent confounder `Z`**, under
two regimes — `see` (observational, confounded) and `do` (interventional). The known causal variable of
interest: **the regime + the confounder's role**. The model must produce the interventional vs
observational outcome distribution.

**Steps.**
1. **Base model.** Train on a mix of `see`/`do`. Measure see/do accuracy in-distribution and OOD
   (unseen graph sizes; perturbed variable names) — expect OOD collapse (the parrot failure).
2. **Locate (DAS).** Learn an orthonormal subspace at a chosen layer aligned to the binary `regime`
   variable. Validation = **interchange accuracy**: take a `do` example, patch the aligned subspace from
   a `see` example, and check the output flips to the `see`-consistent prediction. High interchange
   accuracy ⇒ the subspace *causally implements* the regime.
3. **Diagnose (mediation).** Show that on OOD failures, the regime distinction is *not* mediated by this
   subspace but by surface tokens (low interchange effect, high surface effect).
4. **Install (IIT).** Fine-tune with interchange-intervention training so the subspace robustly
   implements `regime` (and a second subspace the confounder `Z`).
5. **Demonstrate the cognitive gain + attribution.**
   - **Gain:** the IIT-grounded model now holds the see/do distinction **OOD** (unseen sizes, perturbed
     names) where base and a plain-SFT baseline fail — because behaviour is grounded in the latent, not
     the surface form.
   - **Attribution:** ablate/corrupt the aligned subspace → the see/do distinction collapses → the gain
     is *causally* due to the installed mechanism.
   - **Transfer (stretch):** a minimal hand-off to a natural-language causal-reasoning probe
     (Corr2Cause-style) to test whether the grounding transfers beyond synthetic.

**Frontier signal:** a cognitive improvement (OOD causal reasoning) that is **causally attributable to
an internal mechanism we installed and can switch on/off** — not a black-box fine-tune.

## 5. Phased plan (de-risked)

- **Phase 0 — controlled validation.** DAS + mediation on a frozen base model over `causalrl` see/do
  tasks; establish that the regime variable *can* be located and that failures are surface-mediated.
- **Phase 1 — flag-planting.** IIT to install the latent; show OOD gain + causal attribution
  (ablation kills it). The end-to-end §4 result.
- **Phase 2 — generality.** Multiple causal variables (confounder, mediator, collider), multiple
  task families; test whether grounding a *small set* of causal primitives yields broad gains.
- **Phase 3 — transfer & scale.** Move from from-scratch small models to a small open LLM, and from
  synthetic to natural causal-reasoning benchmarks; measure how much synthetic-grounded structure
  transfers. This is the central scientific question and the central risk.

## 6. Evaluation — metrics, baselines, ablations

- **Metrics.** Interchange accuracy (does the subspace implement the variable?); causal-mediation
  fraction (output mediated by latent vs surface); see/do accuracy in-dist and **OOD** (unseen sizes,
  perturbed names); calibration/abstention on non-identifiable cases; transfer accuracy.
- **Baselines.** Base model; plain SFT (more data, no IIT); activation steering without DAS-located
  subspace; data-augmentation-only (the causal-as-add-in approach this repo already showed is brittle).
- **Ablations.** Remove IIT (DAS-only); ablate the installed subspace (attribution); vary which
  layer/subspace dimensionality; grounding one causal variable vs several.

## 7. The leap vs existing causal interpretability

Current causal interp (DAS, causal scrubbing, IIT, causal tracing) is largely **descriptive** ("does the
model implement variable X?") on **small circuits/toy tasks**. The leap: **from diagnosis to grounded-
cognition repair with causal attribution** — using interchange-intervention training to *install/
strengthen a faithful causal latent* and **demonstrably overcome** a cognitive limitation (correlational
reasoning), proving by intervention that the named mechanism is responsible. "Make the model reason over
a grounded causal model of the task, verified by intervening on that model."

## 8. Honest risks and mitigations

- **Synthetic→natural gap (central risk).** A latent grounded on `causalrl` tasks may not transfer to
  open-ended natural cognition. Mitigation: Phase 3 measures transfer explicitly; report it honestly
  even if small. The synthetic result is valuable on its own as a controlled existence proof.
- **You must know the target causal variables for IIT.** Clean in synthetic (`causalrl`); open in the
  wild. Mitigation: start with a small set of universal causal primitives (regime/confounder/mediator/
  collider) that recur across tasks.
- **Scale.** DAS/IIT are heaviest on larger models; current strong results are small-model/synthetic.
  Mitigation: phased scale-up; the contribution is the *method + controlled demonstration*, not a
  frontier-scale model.
- **Introspection unfaithfulness.** Avoided by construction — we use external interchange
  interventions, never the model's self-report.

## 9. Anthropic fit

This sits exactly at Anthropic's core — **mechanistic interpretability + alignment + honesty** — with
the differentiator of moving from *understanding* mechanisms to *grounding cognition* in them with
causal attribution. A model whose reasoning is provably mediated by a faithful internal causal model,
and which can therefore know (via the Causal Hierarchy Theorem boundary) what it cannot ground, is the
causal-inference instantiation of honest, oversight-friendly cognition.

---

**One line:** identify, via interchange interventions, the latent causal model governing the LLM's
reasoning, and install/strengthen it (IIT) so cognition is causally grounded — overcoming correlational
reasoning at the root, with `causalrl` as the ground-truth world that makes faithfulness measurable and
the gain causally attributable.
