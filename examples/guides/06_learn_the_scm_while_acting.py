"""Task guide 6: train an agent that learns its causal model while acting.

The other five guides score a policy someone else trained. This one trains one. `OnlineCausalMBRL`
holds a *belief* — the interventional Markov equivalence class (I-MEC), one fitted SCM per DAG still
consistent with everything it has seen — refits it from its own experiments, and plans inside it by
Thompson sampling over structure.

The world is binary `Z -> A -> Y` with `Z -> Y`, parameterised so the observational contrast
*reverses* the causal one and the observational skeleton is a triangle with no v-structure anywhere.
No quantity of logged rows resolves it: the six DAGs in that equivalence class induce the same
observational law and disagree about which action is best, and five of the six deploy the wrong one.
Experiments do resolve it — which is the point. Regret falls here because the agent *acted*, not
because it read more rows (Sun et al., `Learning by Doing`, arXiv:2402.04869; the fusion of logged
and experimental data is Bareinboim & Forney).

Run: python examples/guides/06_learn_the_scm_while_acting.py
"""

from __future__ import annotations

import numpy as np

from causalrl import OnlineCausalMBRL

VARIABLES = ("Z", "A", "Y")
ACTIONS = (0.0, 1.0)
N_ROLLOUT = 256

P_Z1 = 0.20
P_A1_GIVEN_Z = {0: 0.95, 1: 0.10}  # the logging policy takes A=1 almost only where Y is worst
P_Y1_GIVEN_A_Z = {(0, 0): 0.05, (1, 0): 0.45, (0, 1): 0.95, (1, 1): 0.45}


def step(action: float | None, rng: np.random.Generator) -> dict[str, float]:
    """One transition of the true world. An ``action`` given means a perfect intervention on A."""
    z = float(rng.random() < P_Z1)
    a = float(rng.random() < P_A1_GIVEN_Z[int(z)]) if action is None else float(action)
    y = float(rng.random() < P_Y1_GIVEN_A_Z[(int(a), int(z))])
    return {"Z": z, "A": a, "Y": y}


def log(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """``n`` rows of off-policy log, columnar — what the agent starts from."""
    rows = [step(None, rng) for _ in range(n)]
    return {v: np.array([row[v] for row in rows], dtype=float) for v in VARIABLES}


def true_value(action: float) -> float:
    """``E[Y | do(A = action)]`` exactly, by back-door adjustment over Z on the true law."""
    a = int(action)
    return (1.0 - P_Z1) * P_Y1_GIVEN_A_Z[(a, 0)] + P_Z1 * P_Y1_GIVEN_A_Z[(a, 1)]


OPTIMAL = max(true_value(a) for a in ACTIONS)


def regret(action: float) -> float:
    """What deploying ``action`` costs against the true optimum — exact, never estimated."""
    return OPTIMAL - true_value(action)


def observational_contrast() -> tuple[float, float]:
    """``(E[Y | A=0], E[Y | A=1])`` on the true law — the trap the log sets, in closed form."""
    out: list[float] = []
    for a in (0, 1):
        w = [
            (1.0 - P_Z1) * (P_A1_GIVEN_Z[0] if a == 1 else 1.0 - P_A1_GIVEN_Z[0]),
            P_Z1 * (P_A1_GIVEN_Z[1] if a == 1 else 1.0 - P_A1_GIVEN_Z[1]),
        ]
        out.append(sum(wi * P_Y1_GIVEN_A_Z[(a, z)] for z, wi in enumerate(w)) / sum(w))
    return out[0], out[1]


def preferred_actions(agent: OnlineCausalMBRL) -> list[float]:
    """What each member of the belief would deploy, through the public SCM surface.

    Members are scored the way ``act`` scores them: ``do(A=a)``, ``see(n_rollout)``, mean outcome —
    with one seed per member so its two actions share exogenous draws and the gap between them is
    not partly the gap between two samples. A member in which A is not an ancestor of Y scores both
    actions identically and so falls through to the tie-break, the first action.
    """
    chosen: list[float] = []
    for index, member in enumerate(agent.belief()):
        values = [
            float(member.do({"A": a}).see(N_ROLLOUT, seed=index)["Y"].detach().numpy().mean())
            for a in ACTIONS
        ]
        chosen.append(ACTIONS[max(range(len(ACTIONS)), key=lambda k: values[k])])
    return chosen


def main() -> None:
    rng = np.random.default_rng(0)
    obs0, obs1 = observational_contrast()
    print(
        f"truth:   E[Y|do(A=0)]={true_value(0.0):.3f}  E[Y|do(A=1)]={true_value(1.0):.3f}"
        f"  -> A=1 is optimal, and A=0 costs {regret(0.0):.3f} a round"
    )
    print(
        f"the log: E[Y|A=0]   ={obs0:.3f}  E[Y|A=1]   ={obs1:.3f}"
        "  -> reversed; a correlational read picks A=0\n"
    )

    agent = OnlineCausalMBRL(
        VARIABLES, treatment="A", outcome="Y", actions=ACTIONS, n_rollout=N_ROLLOUT, seed=0
    )
    agent.ingest(log(3_000, rng))  # the off-policy log it starts from
    agent.refit()

    chosen = preferred_actions(agent)
    wrong = sum(action != 1.0 for action in chosen)
    print(
        f"on the log alone the belief holds {agent.belief_size()} DAGs and {wrong} of them would "
        f"deploy A=0.\nThey induce the same observational law, so no further logged row separates "
        "them. act() draws one\n(Thompson sampling over structure), which is why round 0 below is "
        "a lottery over models.\n"
    )

    print("  round     rows    belief   action   regret   probe")
    first = agent.belief_size()
    rounds = 4
    for index in range(rounds):
        # The documented loop: experiment while the structure is undetermined, otherwise deploy.
        probe = agent.probe() if agent.structure_uncertain() else "-"
        action = agent.act()
        print(
            f"  {index:>5}  {agent.steps:>7}  {agent.belief_size():>8}  "
            f"{action:>6.0f}   {regret(action):>6.3f}   {probe:>5}"
        )
        if index == rounds - 1:
            break
        # Its only lever is the treatment, so every experiment is do(A = Bernoulli(1/2)) — and the
        # rows come back one at a time through observe(), the on-policy path.
        for _ in range(200):
            a = float(rng.random() < 0.5)
            agent.observe(step(a, rng), intervention={"A": a})
        agent.refit()

    final = agent.act()
    print(
        "\nThe experiments read A -> Y off Y's marginal moving under do(A) and Z -> A off Z's not"
        f"\nmoving; Meek's rules propagate the rest, and the belief collapses {first} -> "
        f"{agent.belief_size()}."
        "\nThe probe column names Z, the confounder: the most informative experiment available is"
        "\none this agent cannot run, since nothing it controls sets Z. Printed, not hidden."
    )
    assert agent.belief_size() <= first, "experimenting must not widen the equivalence class"
    assert regret(final) == 0.0, "the collapsed belief should plan to the optimal action"
    print(f"OK — trained online; deploying A={final:.0f} at regret {regret(final):.3f}")


if __name__ == "__main__":
    main()
