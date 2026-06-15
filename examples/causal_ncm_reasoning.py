"""Causal reasoning *inside the architecture*: a Neural Causal Model (NCM) head.

Examples 1-2 made a vanilla decoder *look* causal by feeding it ``<see>`` / ``<do>`` data. That
is conditioning, not reasoning: the model could only answer interventions it had seen, because
the do-distribution was in its training set. Nothing in the network computes a causal query.

This file changes the **architecture**. Instead of an unstructured transformer, the model is a
differentiable Neural Causal Model (Xia et al. 2021) built on a causalrl
:class:`~causalrl.CausalGraph`: one small neural *mechanism* per variable, evaluated in
topological order. Two operators are then implemented as computation in the forward pass:

* **do(X=x)  (Layer 2)** — *graph surgery*: we cut X's incoming edges and clamp it, then
  marginalise the **prior** P(Z) over the confounder. A merely correlational model would instead
  weight Z by the **posterior** P(Z | X=x). Same learned mechanisms; the only difference is the
  structural edge-cut. That single architectural difference is the causal reasoning.

* **counterfactual (Layer 3)** — *abduction-action-prediction*: infer the exogenous noise that
  is consistent with a specific unit's evidence, then re-run the mechanisms under a different
  action holding that noise fixed. An unstructured LM cannot do this for a named individual.

The headline test of *reasoning* (vs memorisation): we train the NCM on **observational data
only** — it never sees a single ``do`` sample — and it still recovers P(Y | do X) correctly,
because the answer is *computed* from structure. causalrl certifies the query is identifiable
(back-door on the observed confounder Z) and provides the ground truth.

Run::

    uv run --extra torch python examples/causal_ncm_reasoning.py

How this becomes an "LLM with reasoning inside": this NCM is a differentiable *causal reasoning
head*. The language model's job is to parse a natural-language question into a (graph, query)
and call this head, then verbalise the answer — tool-use, but the tool is differentiable and
trained jointly. The head is where the causality lives; the transformer is the interface.
Didactic demonstration, not a performance claim.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Uniform

from causalrl import (
    CausalGraph,
    FunctionalMechanism,
    StructuralCausalModel,
    backdoor_adjustment_set,
    is_identifiable,
)

# --------------------------------------------------------------------------------------------
# Ground-truth SCM (causalrl). Z is the confounder and is now OBSERVED, so the effect of X on Y
# is back-door identifiable from observational data — which is what lets the NCM *reason* it out.
#     Z -> X, Z -> Y, X -> Y      (Z = severity, X = drug, Y = recovery; all binary)
# Same numbers as the earlier examples: P(Y|do X=1)=0.65, but confounded P(Y|X=1)=0.86.
# --------------------------------------------------------------------------------------------


def build_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")])

    def z_mech(_p: dict[str, Tensor], noise: Tensor) -> Tensor:
        return (noise < 0.5).float()

    def x_mech(p: dict[str, Tensor], noise: Tensor) -> Tensor:
        return (noise < (0.2 + 0.6 * p["Z"])).float()

    def y_mech(p: dict[str, Tensor], noise: Tensor) -> Tensor:
        prob = 0.5 + 0.15 * (2 * p["X"] - 1) + 0.35 * (2 * p["Z"] - 1)
        return (noise < prob.clamp(0.0, 1.0)).float()

    mechanisms = {
        "Z": FunctionalMechanism([], z_mech),
        "X": FunctionalMechanism(["Z"], x_mech),
        "Y": FunctionalMechanism(["X", "Z"], y_mech),
    }
    exogenous = {n: Uniform(0.0, 1.0) for n in ("Z", "X", "Y")}
    return StructuralCausalModel(graph, mechanisms, exogenous)


# --------------------------------------------------------------------------------------------
# The architecture: one neural mechanism per node. Each is monotone in its exogenous noise
# (V = 1[u < sigmoid(net(parents))]), which mirrors the SCM's threshold form and is what makes
# counterfactual abduction well defined. The structure (which mechanism feeds which) IS the model.
# --------------------------------------------------------------------------------------------


class _MechanismNet(nn.Module):
    def __init__(self, n_parents: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1)) if n_parents == 0 else None
        self.net = (
            None
            if n_parents == 0
            else nn.Sequential(nn.Linear(n_parents, 16), nn.ReLU(), nn.Linear(16, 1))
        )

    def logit(self, parents: Tensor | None, n: int) -> Tensor:
        if self.net is None:
            assert self.bias is not None
            return self.bias.expand(n)
        assert parents is not None
        return self.net(parents).squeeze(-1)


class NeuralCausalModel(nn.Module):
    """A differentiable SCM over a fixed causal graph, with do() and counterfactual operators."""

    def __init__(self, graph: CausalGraph) -> None:
        super().__init__()
        self.graph = graph
        self.order = graph.topological_order()
        self.parents = {v: graph.parents(v) for v in self.order}
        self.mech = nn.ModuleDict(
            {v: _MechanismNet(len(self.parents[v])) for v in self.order}
        )

    def _parent_tensor(self, v: str, values: dict[str, Tensor]) -> Tensor | None:
        pa = self.parents[v]
        if not pa:
            return None
        return torch.stack([values[p] for p in pa], dim=1)

    def prob(self, v: str, values: dict[str, Tensor]) -> Tensor:
        """P(v=1 | its parents), from the learned mechanism."""
        n = next(iter(values.values())).shape[0]
        return torch.sigmoid(self.mech[v].logit(self._parent_tensor(v, values), n))

    def observational_nll(self, data: dict[str, Tensor]) -> Tensor:
        """Negative log-likelihood of observed (Z,X,Y) — the only thing we train on."""
        loss = data["Z"].new_zeros(())
        for v in self.order:
            p = self.prob(v, data)
            loss = loss + nn.functional.binary_cross_entropy(p, data[v])
        return loss

    @torch.no_grad()
    def do(self, intervention: dict[str, float], target: str) -> float:
        """Layer 2 by graph surgery: cut the intervened node's parents, marginalise the PRIOR.

        Enumerates the binary confounder exactly. Note we weight Z by its learned *prior* P(Z),
        never by P(Z | X) — that edge no longer exists once X is intervened. That is the do.
        """
        total = 0.0
        for z in (0.0, 1.0):
            values: dict[str, Tensor] = {"Z": torch.tensor([z])}
            p_z = self.prob("Z", values).item()
            p_z = p_z if z == 1.0 else 1.0 - p_z
            for node, val in intervention.items():
                values[node] = torch.tensor([val])  # clamp; parents are simply ignored
            # fill any remaining non-target, non-intervened node from its mechanism mean
            for v in self.order:
                if v not in values and v != target:
                    values[v] = (self.prob(v, values) > 0.5).float()
            total += p_z * self.prob(target, values).item()
        return total

    @torch.no_grad()
    def see(self, condition: dict[str, float], target: str) -> float:
        """The *correlational* answer P(target=1 | condition): weight Z by the POSTERIOR.

        Same learned mechanisms as do(); the only change is that here X keeps its edge from Z,
        so seeing X=x updates our belief about Z. This is what a non-causal model computes.
        """
        x_node, x_val = next(iter(condition.items()))
        num, den = 0.0, 0.0
        post: dict[float, float] = {}
        for z in (0.0, 1.0):
            values = {"Z": torch.tensor([z])}
            p_z = self.prob("Z", values).item()
            p_z = p_z if z == 1.0 else 1.0 - p_z
            p_x = self.prob(x_node, values).item()
            p_x = p_x if x_val == 1.0 else 1.0 - p_x
            post[z] = p_z * p_x
            den += post[z]
        for z in (0.0, 1.0):
            values = {"Z": torch.tensor([z]), x_node: torch.tensor([x_val])}
            num += (post[z] / den) * self.prob(target, values).item()
        return num

    @torch.no_grad()
    def counterfactual(self, evidence: dict[str, float], action: dict[str, float], target: str
                       ) -> float:
        """Layer 3 abduction-action-prediction for a single unit, monotone-noise closed form.

        Abduct: the observed target pins its exogenous noise to an interval. Act: change the
        action. Predict: probability the (fixed-noise) mechanism flips under the new action.
        """
        ev = {k: torch.tensor([v]) for k, v in evidence.items()}
        p_factual = self.prob(target, ev).item()
        cf_parents = dict(ev)
        cf_parents.update({k: torch.tensor([v]) for k, v in action.items()})
        p_cf = self.prob(target, cf_parents).item()
        y_obs = evidence[target]
        if y_obs == 1.0:  # noise u ~ U(0, p_factual); P(u < p_cf)
            return min(p_cf, p_factual) / p_factual if p_factual > 0 else 0.0
        # y_obs == 0: noise u ~ U(p_factual, 1); P(u < p_cf)
        return max(0.0, p_cf - p_factual) / (1.0 - p_factual) if p_factual < 1 else 0.0


# --------------------------------------------------------------------------------------------
# Train on observational data ONLY, then ask interventional / counterfactual questions.
# --------------------------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)
    scm = build_scm()

    # causalrl certifies the query is answerable from observation, and how (adjustment set).
    g = scm.graph
    print(f"is P(Y | do X) identifiable from observation?  {is_identifiable(g, 'X', 'Y')}")
    print(f"back-door adjustment set for X -> Y:            {backdoor_adjustment_set(g, 'X', 'Y')}")

    # Observational training data — note: NO do() samples anywhere.
    obs = scm.see(40_000, seed=0)
    data = {k: obs[k] for k in ("Z", "X", "Y")}

    ncm = NeuralCausalModel(scm.graph)
    opt = torch.optim.AdamW(ncm.parameters(), lr=5e-3)
    print("\ntraining the NCM on observational data only ...")
    for step in range(800):
        loss = ncm.observational_nll(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"  step {step + 1}  nll {loss.item():.3f}")

    # Ground truth from causalrl.
    truth_do1 = float(scm.do({"X": 1.0}).see(200_000, seed=8)["Y"].mean())
    truth_do0 = float(scm.do({"X": 0.0}).see(200_000, seed=9)["Y"].mean())
    yx = obs["Y"][obs["X"] > 0.5].mean().item()

    print("\nLayer 2 — interventions the NCM was NEVER trained on:")
    print("                                  truth    NCM(reason)   correlational")
    print(f"  P(Y=1 | do X=1)               {truth_do1:.3f}     {ncm.do({'X': 1.0}, 'Y'):.3f}"
          f"        {ncm.see({'X': 1.0}, 'Y'):.3f}")
    print(f"  P(Y=1 | do X=0)               {truth_do0:.3f}     {ncm.do({'X': 0.0}, 'Y'):.3f}"
          f"        {ncm.see({'X': 0.0}, 'Y'):.3f}")
    print(f"  (observational P(Y|X=1) in data = {yx:.3f}; the correlational column reproduces "
          "the confounded value, the NCM column reasons past it.)")

    # Layer 3 — a counterfactual for one specific patient.
    evidence = {"Z": 1.0, "X": 1.0, "Y": 1.0}  # severe patient, took drug, recovered
    cf_truth = float(
        scm.counterfactual(evidence, {"X": 0.0}, n=200_000, seed=3)["Y"].mean()
    )
    cf_ncm = ncm.counterfactual(evidence, {"X": 0.0}, "Y")
    print("\nLayer 3 — counterfactual for one severe patient who took the drug and recovered:")
    print(f"  P(would have recovered ANYWAY, without the drug)   truth {cf_truth:.3f}   "
          f"NCM {cf_ncm:.3f}")

    print("\nThe NCM never saw a do-sample, yet recovers the interventional and counterfactual "
          "answers — because the architecture *computes* them (edge-cut + abduction), it does "
          "not recall them. That is causal reasoning inside the model.")


if __name__ == "__main__":
    main()
