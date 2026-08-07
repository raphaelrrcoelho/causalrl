"""Learn the SCM while acting: three data diets on a world observation alone cannot solve.

    uv run python examples/online_causal_mbrl.py

``examples/learned_scm_policy.py`` fits a world model from a *fixed* table and then plans in it.
This example runs the online loop instead: :class:`~causalrl.agents.online_causal_mbrl.
OnlineCausalMBRL` refits its belief -- the **interventional Markov equivalence class** (I-MEC), one
fitted SCM per DAG still consistent with the data -- as its own experiments arrive, and replans
each round. What is demonstrated is not that a causal model is more accurate. It is that
observational data alone leaves the *decision* undetermined at every sample size, and that
experiments resolve it.

**The world (synthetic, exact ground truth).** Binary ``Z -> A -> Y`` with ``Z -> Y``: a confounder
``Z``, a treatment ``A`` the agent sets, and an outcome ``Y`` it is scored by. The parameters are
chosen so that the observational contrast **reverses** the causal one, and so that the
observational skeleton is the complete triangle with no v-structure anywhere. PC therefore returns
a fully undirected CPDAG, its equivalence class has six members, and those six do not agree about
which action is best:

* the member matching the truth (``Z -> A``, ``Z -> Y``, ``A -> Y``) back-door-adjusts and prefers
  ``A=1``;
* the two in which ``A`` is a source read ``do(A=a)`` straight off ``E[Y|A=a]`` and prefer ``A=0``;
* the three in which ``A`` is not an ancestor of ``Y`` see no effect at all -- both actions score
  identically, to the last bit, because the rollouts share their exogenous draws -- so ``act()``
  falls through to its documented tie-break and takes the first action, ``A=0``.

Five of the six therefore choose the wrong action, and no quantity of observational rows separates
them: they induce the *same* observational law. That is Bareinboim & Forney's point -- under
structure the data cannot orient, an agent needs interventional data as well, and averaging the
ambiguity away can cost unbounded regret -- and it is the reason this agent exists. The three arms
differ only in what they are fed:

* **observational only** -- a growing log, no experiments;
* **interventional only** -- a small observational seed (:meth:`~causalrl.agents.
  online_causal_mbrl.OnlineCausalMBRL.refit` runs PC on the observational sample for the skeleton,
  so this arm cannot be given *none*) plus one small randomized experiment per round;
* **both** -- the same log as the first arm and the same experiments as the second.

**What the agent may intervene on.** Its one lever is the treatment, so every experiment is a
randomized perfect intervention on ``A``. :meth:`~causalrl.agents.online_causal_mbrl.
OnlineCausalMBRL.probe` -- which returns the target whose effect the belief's members disagree
about most -- is called and reported each round, but on this world it almost always names ``Z``:
the most informative experiment is one this agent cannot run, because nothing in it can set a
confounder. That is worth printing rather than hiding. The ``probe`` column is a diagnostic here,
not the next action, and the experiment goes where the lever is.

**What is reported.** Per round, per arm: transitions buffered, the belief size across seeds, the
share of seeds whose belief still *contains* the true DAG, and the mean regret of the policy
:meth:`~causalrl.agents.online_causal_mbrl.OnlineCausalMBRL.act` would deploy at that round --
``max_a E[Y|do(A=a)] - E[Y|do(A=chosen)]``, computed exactly from the world's own parameters, never
estimated. Belief size sits beside regret so the I-MEC collapse is visible rather than asserted,
and it is printed as the spread across seeds because the collapse is **not** guaranteed monotone:
PC and the marginal-shift invariance test are finite-sample procedures, and a later CPDAG can be
less oriented than an earlier one. The truth column is there because belief size alone would
flatter the agent: a belief of one member is not thereby a belief in the *right* member, and on
this world a single mis-read marginal can orient the graph confidently and wrongly.

**Citations.** The alternating explore-structure / exploit-structure loop is Sun et al., *Learning
by Doing*, Science China Information Sciences (2024), `arXiv:2402.04869
<https://arxiv.org/abs/2402.04869>`_; the observational-plus-experimental fusion is Bareinboim &
Forney (MABUC) and Forney, Pearl & Bareinboim, *Counterfactual Data-Fusion for Online
Reinforcement Learners*; acting by drawing one member of the belief is Thompson sampling over
structure, Ortega & Braun, `arXiv:1303.4431 <https://arxiv.org/abs/1303.4431>`_; scoring candidate
experiments by predictive disagreement is standard active intervention design (Scherrer et al.,
`arXiv:2109.02429 <https://arxiv.org/abs/2109.02429>`_). This example implements those methods; it
contributes none of them.

This is a synthetic world with known ground truth, built to make one published failure mode
visible. It is not a benchmark result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from causalrl import OnlineCausalMBRL

VARIABLES = ("Z", "A", "Y")
TREATMENT, OUTCOME = "A", "Y"
ACTIONS = (0.0, 1.0)

P_Z1 = 0.20
"""``P(Z = 1)``."""

P_A1_GIVEN_Z = {0: 0.95, 1: 0.10}
"""``P(A = 1 | Z = z)`` -- the logging policy takes ``A=1`` almost only where ``Y`` is worst."""

P_Y1_GIVEN_A_Z = {(0, 0): 0.05, (1, 0): 0.45, (0, 1): 0.95, (1, 1): 0.45}
"""``P(Y = 1 | A = a, Z = z)``: ``A=1`` helps a lot at ``Z=0`` and hurts a lot at ``Z=1``."""

TRUE_EDGES = (("A", "Y"), ("Z", "A"), ("Z", "Y"))
"""The DAG the world is generated from -- sorted, so a member's edges compare to it directly."""


def true_action_value(action: float) -> float:
    """``E[Y | do(A = action)]`` exactly, by back-door adjustment over ``Z`` on the true law."""
    a = int(action)
    return (1.0 - P_Z1) * P_Y1_GIVEN_A_Z[(a, 0)] + P_Z1 * P_Y1_GIVEN_A_Z[(a, 1)]


TRUE_ACTION_VALUES = tuple(true_action_value(a) for a in ACTIONS)
OPTIMAL_VALUE = max(TRUE_ACTION_VALUES)


def regret(action: float) -> float:
    """What deploying ``action`` costs against the true optimum -- exact, not estimated."""
    return OPTIMAL_VALUE - true_action_value(action)


def observational_contrast() -> tuple[float, float]:
    """``(E[Y | A=0], E[Y | A=1])`` on the true law -- the trap, in closed form."""
    out: list[float] = []
    for a in (0, 1):
        weights = [
            (1.0 - P_Z1) * (P_A1_GIVEN_Z[0] if a == 1 else 1.0 - P_A1_GIVEN_Z[0]),
            P_Z1 * (P_A1_GIVEN_Z[1] if a == 1 else 1.0 - P_A1_GIVEN_Z[1]),
        ]
        out.append(sum(w * P_Y1_GIVEN_A_Z[(a, z)] for z, w in enumerate(weights)) / sum(weights))
    return out[0], out[1]


def sample(
    n: int, *, seed: int, do: Mapping[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """``n`` rows of the true world, with each node in ``do`` set by a perfect intervention.

    ``do`` maps a variable to a per-row column of set values, so a *randomized* experiment is one
    call with a random column: the intervened node's own mechanism is replaced and every
    downstream node is generated from the value that was set.
    """
    rng = np.random.default_rng(seed)
    do = dict(do or {})
    z = do["Z"] if "Z" in do else (rng.random(n) < P_Z1).astype(float)
    p_a = np.where(z == 1.0, P_A1_GIVEN_Z[1], P_A1_GIVEN_Z[0])
    a = do["A"] if "A" in do else (rng.random(n) < p_a).astype(float)
    p_y = np.where(
        z == 1.0,
        np.where(a == 1.0, P_Y1_GIVEN_A_Z[(1, 1)], P_Y1_GIVEN_A_Z[(0, 1)]),
        np.where(a == 1.0, P_Y1_GIVEN_A_Z[(1, 0)], P_Y1_GIVEN_A_Z[(0, 0)]),
    )
    y = do["Y"] if "Y" in do else (rng.random(n) < p_y).astype(float)
    return {
        "Z": np.asarray(z, dtype=float),
        "A": np.asarray(a, dtype=float),
        "Y": np.asarray(y, dtype=float),
    }


def randomized_experiment(target: str, n: int, *, seed: int) -> dict[str, np.ndarray]:
    """``n`` rows under ``do(target = Bernoulli(1/2))`` -- a balanced perfect intervention.

    Balanced rather than one-sided for two reasons. Every downstream parent configuration gets
    rows, so the experiment can be *fitted* from and not only oriented from; and a one-sided design
    would leave one arm of the treatment unsampled, which is the arm the agent is deciding about.
    """
    rng = np.random.default_rng(seed)
    return sample(n, seed=seed + 1, do={target: (rng.random(n) < 0.5).astype(float)})


class MemberVerdict(NamedTuple):
    """One I-MEC member: its DAG, what it thinks each action is worth, and what it would pick."""

    edges: tuple[str, ...]
    values: tuple[float, ...]
    preferred: float | None
    """``None`` when the treatment is not an ancestor of the outcome: this member says the action
    cannot matter, and it is a structural fact about the DAG rather than a Monte-Carlo near-tie."""

    def row(self, index: int) -> str:
        edges = ", ".join(self.edges)
        values = ", ".join(f"{v:.3f}" for v in self.values)
        verdict = (
            "A is not an ancestor of Y: indifferent, so act() takes the first action, A=0"
            if self.preferred is None
            else f"prefers A={self.preferred:.0f}"
        )
        return f"  member {index}: {edges:<20} E[Y|do(A=0,1)]=({values})  {verdict}"


class ArmSpec(NamedTuple):
    """One data diet: what the agent starts with and what each round adds."""

    name: str
    initial_observational: int
    observational_per_round: int
    interventional_per_round: int


class RoundSummary(NamedTuple):
    """One round of one arm, pooled across seeds."""

    index: int
    mean_steps: float
    belief_sizes: tuple[int, ...]
    truth_share: float
    """Share of seeds whose belief still contains the true DAG. A belief that has collapsed to a
    single member is not thereby a *correct* one, and this is the column that separates the two."""
    mean_regret: float
    optimal_share: float
    probes: tuple[str, ...]

    def row(self) -> str:
        low, high = min(self.belief_sizes), max(self.belief_sizes)
        sizes = str(low) if low == high else f"{low}-{high}"
        probe = ",".join(sorted(set(self.probes)))
        return (
            f"  {self.index:>5}  {self.mean_steps:>7.0f}  {sizes:>8}  {self.truth_share:>9.0%}  "
            f"{self.mean_regret:>11.3f}  {self.optimal_share:>9.0%}  {probe:>7}"
        )


class ArmResult(NamedTuple):
    """What one arm did over the whole run."""

    name: str
    rounds: tuple[RoundSummary, ...]

    @property
    def final(self) -> RoundSummary:
        return self.rounds[-1]

    def summary(self) -> str:
        header = (
            f"=== {self.name} ===\n"
            "  round    steps    belief   truth in   mean regret   optimal    probe\n"
            "                                belief"
        )
        return "\n".join([header, *(r.row() for r in self.rounds)])


ROWS_PER_ROUND = 200
"""Rows each arm buys per round -- observational for the first arm, experimental for the others.

The same number for all three so the arms differ in the *kind* of data, not the quantity. Its size
is set by the weaker of the two invariance decisions ``do(A)`` has to make, and both can fail:
deciding ``A -> Y`` means detecting a marginal shift of 0.184 against a 0.05 threshold, and deciding
``Z -> A`` means detecting that ``Z``'s marginal did *not* move. The second is the hard one, because
``shift_threshold`` is an absolute cutoff that does not know the sample size: the empirical total
variation between two Bernoulli(0.2) samples exceeds 0.05 by chance 44% of the time at 40 rows
against a 3,000-row reference, 9% at 200 rows, and 19% at 200 rows against a 250-row reference.
Those false alarms are what a flickering belief and a lost truth are made of here; the do-buffer
accumulates across rounds, so the rate falls after the first experiment. Both errors are properties
of a fixed-threshold test at a finite sample rather than of this world.
"""

ARMS = (
    ArmSpec(
        "observational only",
        initial_observational=3_000,
        observational_per_round=ROWS_PER_ROUND,
        interventional_per_round=0,
    ),
    ArmSpec(
        "interventional only",
        initial_observational=250,
        observational_per_round=0,
        interventional_per_round=ROWS_PER_ROUND,
    ),
    ArmSpec(
        "both",
        initial_observational=3_000,
        observational_per_round=0,
        interventional_per_round=ROWS_PER_ROUND,
    ),
)
"""The three diets. The seed of the interventional-only arm is 250 rows because that is about the
smallest observational sample PC recovers this world's complete skeleton from reliably; below it
the arm's failures are missing *edges*, which is a different story from the one being told."""


def _agent(*, n_rollout: int, seed: int) -> OnlineCausalMBRL:
    return OnlineCausalMBRL(
        VARIABLES,
        treatment=TREATMENT,
        outcome=OUTCOME,
        actions=ACTIONS,
        n_rollout=n_rollout,
        seed=seed,
    )


def member_verdicts(
    log: Mapping[str, np.ndarray], *, n_rollout: int, seed: int
) -> tuple[MemberVerdict, ...]:
    """Fit the I-MEC on ``log`` alone and report what each member would do.

    Goes through the public SCM surface (``member.graph``, ``member.do(...).see(...)``) rather than
    the agent's internals, so what is printed is the same quantity ``act()`` ranks.
    """
    agent = _agent(n_rollout=n_rollout, seed=seed)
    agent.ingest(dict(log))
    verdicts: list[MemberVerdict] = []
    for index, member in enumerate(agent.refit()):
        values = tuple(
            float(
                member.do({TREATMENT: action})
                .see(n_rollout, seed=seed + index)[OUTCOME]
                .detach()
                .numpy()
                .mean()
            )
            for action in ACTIONS
        )
        is_ancestor = TREATMENT in member.graph.ancestors(OUTCOME)
        best = ACTIONS[max(range(len(ACTIONS)), key=lambda i: values[i])]
        verdicts.append(
            MemberVerdict(
                edges=tuple(f"{u}->{v}" for u, v in sorted(member.graph.directed_edges)),
                values=values,
                preferred=best if is_ancestor else None,
            )
        )
    return tuple(verdicts)


class SeedRun(NamedTuple):
    """One agent's whole run, round by round."""

    steps: tuple[int, ...]
    belief_sizes: tuple[int, ...]
    truth_kept: tuple[bool, ...]
    actions: tuple[float, ...]
    probes: tuple[str, ...]


def belief_contains_truth(agent: OnlineCausalMBRL) -> bool:
    """Whether the true DAG is still one of the belief's members."""
    return any(
        tuple(sorted(member.graph.directed_edges)) == TRUE_EDGES for member in agent.belief()
    )


def run_seed(spec: ArmSpec, *, seed: int, rounds: int, n_rollout: int) -> SeedRun:
    """Run one agent on ``spec``'s diet: ingest, refit, act, probe, repeat."""
    agent = _agent(n_rollout=n_rollout, seed=seed)
    stream = seed * 10_000
    agent.ingest(sample(spec.initial_observational, seed=stream))
    actions: list[float] = []
    probes: list[str] = []
    truth_kept: list[bool] = []
    for index in range(rounds):
        if index:
            if spec.observational_per_round:
                agent.ingest(sample(spec.observational_per_round, seed=stream + index))
            if spec.interventional_per_round:
                # The agent's one lever is the treatment: its experiments are its own actions.
                agent.ingest(
                    randomized_experiment(
                        TREATMENT, spec.interventional_per_round, seed=stream + 500 + index
                    ),
                    source="interventional",
                    target=TREATMENT,
                )
        agent.refit()
        truth_kept.append(belief_contains_truth(agent))
        actions.append(agent.act())
        probes.append(agent.probe() if agent.structure_uncertain() else "-")
    history = agent.history()
    return SeedRun(
        steps=tuple(record.step for record in history),
        belief_sizes=tuple(record.belief_size for record in history),
        truth_kept=tuple(truth_kept),
        actions=tuple(actions),
        probes=tuple(probes),
    )


def run_arm(spec: ArmSpec, *, seeds: Sequence[int], rounds: int, n_rollout: int) -> ArmResult:
    """Run ``spec`` once per seed and pool the rounds. Regret is exact, never estimated."""
    runs = [run_seed(spec, seed=seed, rounds=rounds, n_rollout=n_rollout) for seed in seeds]
    summaries = [
        RoundSummary(
            index=index,
            mean_steps=float(np.mean([run.steps[index] for run in runs])),
            belief_sizes=tuple(run.belief_sizes[index] for run in runs),
            truth_share=float(np.mean([run.truth_kept[index] for run in runs])),
            mean_regret=float(np.mean([regret(run.actions[index]) for run in runs])),
            optimal_share=float(np.mean([regret(run.actions[index]) == 0.0 for run in runs])),
            probes=tuple(run.probes[index] for run in runs),
        )
        for index in range(rounds)
    ]
    return ArmResult(name=spec.name, rounds=tuple(summaries))


class Demonstration(NamedTuple):
    """Everything the run produced: the ambiguity, then the three arms."""

    verdicts: tuple[MemberVerdict, ...]
    arms: tuple[ArmResult, ...]


def run_online_causal_mbrl(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    rounds: int = 8,
    n_rollout: int = 512,
    n_ambiguity_log: int = 3_000,
) -> Demonstration:
    """The whole demonstration: the six-member ambiguity, then the three data diets."""
    return Demonstration(
        verdicts=member_verdicts(sample(n_ambiguity_log, seed=99), n_rollout=n_rollout, seed=0),
        arms=tuple(run_arm(spec, seeds=seeds, rounds=rounds, n_rollout=n_rollout) for spec in ARMS),
    )


def main(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    rounds: int = 8,
    n_rollout: int = 512,
    n_ambiguity_log: int = 3_000,
) -> Demonstration:
    result = run_online_causal_mbrl(
        seeds=seeds, rounds=rounds, n_rollout=n_rollout, n_ambiguity_log=n_ambiguity_log
    )
    obs0, obs1 = observational_contrast()
    print(
        "Synthetic world with exact ground truth: binary Z -> A -> Y, plus Z -> Y.\n"
        f"  E[Y|do(A=0)]={TRUE_ACTION_VALUES[0]:.3f}   E[Y|do(A=1)]={TRUE_ACTION_VALUES[1]:.3f}"
        f"   -> A=1 is optimal; deploying A=0 costs {regret(0.0):.3f} a round.\n"
        f"  E[Y|A=0]   ={obs0:.3f}   E[Y|A=1]   ={obs1:.3f}"
        "   -> the log reverses that ordering, so a correlational\n"
        "                                        reading picks A=0.\n"
    )
    print("The ambiguity observation cannot resolve -- the I-MEC fitted on the log alone:")
    for index, verdict in enumerate(result.verdicts):
        print(verdict.row(index))
    print(
        "  Every member reproduces the same observational law, so no number of observational\n"
        "  rows chooses between them; only an intervention can. Five of the six deploy the\n"
        "  wrong action.\n"
    )
    for arm in result.arms:
        print(arm.summary())
        print()
    print(
        "Read it this way. All three arms start from the same six-member ambiguity, five of whose\n"
        "members point at the wrong action. The observational-only arm keeps all six for the\n"
        "whole run and so deploys the wrong action about five times in six, whatever its regret\n"
        "happens to be on a given round: its log grows every round and buys nothing, because the\n"
        "six members are observationally indistinguishable. The two arms that experiment read\n"
        "A -> Y off Y's marginal moving under do(A) and Z -> A off Z's not moving, Meek's second\n"
        "rule propagates Z -> Y, the belief collapses to one member, and regret goes to zero --\n"
        "within a round or two of the first experiment. The log is not thereby useless: it is\n"
        "what the interventional-only arm has to buy with experiments before its mechanisms are\n"
        "worth planning in, and it is what the invariance test compares its do-samples *against*,\n"
        "so an arm working from a 250-row reference is likelier both to read an unmoved marginal\n"
        "as moved and to miss one that moved.\n"
        "Watch the two structure columns together rather than either alone. Belief size is the\n"
        "spread across seeds, and it is not guaranteed to fall monotonically -- PC and the\n"
        "marginal-shift test are finite-sample procedures, so a belief that has shrunk can widen\n"
        "again. And a belief of size one is not thereby the right one: where 'truth in belief'\n"
        "drops below 100% the agent has oriented an edge from a marginal it misread, excluded the\n"
        "true DAG, and gone on planning confidently inside a wrong model. Reporting belief size\n"
        "without it would flatter exactly the runs that deserve it least.\n"
        "This is a synthetic world with known ground truth, built to make one published failure\n"
        "mode visible. It is not a benchmark result."
    )
    return result


if __name__ == "__main__":
    main()
