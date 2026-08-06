"""OnlineCausalMBRL: the buffers, ``refit``, the I-MEC belief, and the acting layer.

The world is a noisy chain ``X -> Y -> Z``. Observationally it is Markov-equivalent to
``X <- Y -> Z`` and ``X <- Y <- Z`` (no v-structure to orient), so PC leaves both edges undirected
and the belief holds three members. A perfect ``do(X)`` sample shifts ``Y``'s marginal but leaves
nothing upstream to be invariant, orienting ``X -> Y``; Meek's R1 then propagates ``Y -> Z`` and the
belief collapses to one member. That collapse is what the task-1 tests measure.

The task-2 tests (``act`` / ``probe``) do not fit anything. They install a hand-built belief whose
members are *deterministic* -- every mechanism ignores its noise -- so each rollout is a point mass
and the expected maximin, mean and total-variation answers are exact rather than statistical. What
is under test there is how the members' verdicts are combined, not how they were fitted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from causalrl.agents.online_causal_mbrl import OnlineCausalMBRL
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

VARIABLES = ("X", "Y", "Z")


def _chain(n: int, seed: int, *, x: float | None = None) -> dict[str, np.ndarray]:
    """Sample the noisy chain ``X -> Y -> Z`` (each child copies its parent, flipped w.p. 0.1).

    ``x`` pins the treatment column, which makes the sample a draw from ``do(X = x)``.
    """
    rng = np.random.default_rng(seed)
    xs = np.full(n, x) if x is not None else (rng.random(n) < 0.5).astype(float)
    ys = np.where(rng.random(n) < 0.1, 1.0 - xs, xs)
    zs = np.where(rng.random(n) < 0.1, 1.0 - ys, ys)
    return {"X": xs, "Y": ys, "Z": zs}


def _agent(*, max_members: int = 32) -> OnlineCausalMBRL:
    return OnlineCausalMBRL(
        VARIABLES, treatment="X", outcome="Z", actions=(0.0, 1.0), max_members=max_members
    )


def _constant(value: float) -> FunctionalMechanism:
    """A parentless mechanism pinned to ``value``: it ignores its noise, so rollouts do not vary."""
    return FunctionalMechanism([], lambda parents, noise: torch.full_like(noise, value))


def _affine(parent: str, weight: float, intercept: float) -> FunctionalMechanism:
    """``intercept + weight * parent``, noiseless for the same reason as :func:`_constant`."""
    return FunctionalMechanism(
        [parent], lambda parents, noise: intercept + weight * parents[parent]
    )


def _member(
    edges: list[tuple[str, str]], mechanisms: dict[str, Mechanism]
) -> StructuralCausalModel:
    """A deterministic SCM over ``VARIABLES``; the exogenous distributions are never read."""
    return StructuralCausalModel(
        CausalGraph(directed_edges=edges, nodes=list(VARIABLES)),
        mechanisms,
        {variable: Normal(0.0, 1.0) for variable in VARIABLES},
    )


def _disagreeing_belief() -> tuple[StructuralCausalModel, StructuralCausalModel]:
    """Two members that disagree about the treatment: one wants ``X = 1``, the other ``X = 0``.

    ``Z | do(X = a)`` is ``6a`` under the first (the chain, with a strong positive effect) and
    ``3 - 4a`` under the second (a direct edge with a negative one). So per action the values are
    ``a = 0 -> (0, 3)`` and ``a = 1 -> (6, -1)``: means ``1.5`` and ``2.5``, minima ``0`` and
    ``-1``. Averaging picks ``1.0``; maximin picks ``0.0``; and a thompson draw reveals which
    member it drew, since the two argmaxes differ.
    """
    chain = _member(
        [("X", "Y"), ("Y", "Z")],
        {"X": _constant(0.0), "Y": _affine("X", 6.0, 0.0), "Z": _affine("Y", 1.0, 0.0)},
    )
    direct = _member(
        [("X", "Z")],
        {"X": _constant(0.0), "Y": _constant(0.0), "Z": _affine("X", -4.0, 3.0)},
    )
    return chain, direct


def test_observational_data_leaves_the_i_mec_ambiguous_and_do_data_collapses_it() -> None:
    agent = _agent()
    agent.ingest(_chain(800, 0))
    agent.refit()
    # Three DAGs generate this distribution: X->Y->Z, X<-Y->Z, X<-Y<-Z. No amount of observational
    # data separates them, which is exactly why the agent has to act.
    assert agent.belief_size() == 3
    assert agent.structure_uncertain()

    agent.ingest(_chain(200, 1, x=1.0), source="interventional", target="X")
    agent.refit()
    assert agent.belief_size() == 1
    assert not agent.structure_uncertain()
    assert sorted(agent.belief()[0].graph.directed_edges) == [("X", "Y"), ("Y", "Z")]


def test_observe_routes_a_do_row_to_its_target_buffer() -> None:
    # Same collapse, but the do-rows arrive one at a time through observe(intervention=...) --
    # the acting path. The row omits X: a perfect intervention SET it, so the do-value is authority.
    agent = _agent()
    agent.ingest(_chain(800, 0))
    do_sample = _chain(200, 1, x=1.0)
    for i in range(200):
        agent.observe(
            {"Y": float(do_sample["Y"][i]), "Z": float(do_sample["Z"][i])},
            intervention={"X": 1.0},
        )
    agent.refit()
    assert agent.belief_size() == 1
    assert agent.steps == 1000


def test_observe_without_an_intervention_fills_the_observational_buffer() -> None:
    # The acting loop's off-policy path: rows arriving one at a time with nothing set. They are
    # the rows PC runs on, so a refit right after must find the chain's unoriented CPDAG.
    agent = _agent()
    sample = _chain(300, 0)
    for i in range(300):
        agent.observe({variable: float(sample[variable][i]) for variable in VARIABLES})
    agent.refit()
    assert agent.steps == 300
    assert agent.belief_size() == 3


def test_observe_requires_a_full_row() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="missing variable"):
        agent.observe({"X": 1.0, "Y": 0.0})


def test_interventional_ingest_without_a_target_raises_naming_the_argument() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="requires target="):
        agent.ingest(_chain(10, 0, x=1.0), source="interventional")


def test_observational_ingest_with_a_target_raises_naming_the_argument() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="forbids target="):
        agent.ingest(_chain(10, 0), source="observational", target="X")


def test_unknown_source_raises_naming_the_argument() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="source="):
        agent.ingest(_chain(10, 0), source="counterfactual")  # type: ignore[arg-type]


def test_a_multi_node_intervention_raises_naming_the_single_target_limit() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="exactly one target"):
        agent.observe({"Z": 1.0}, intervention={"X": 1.0, "Y": 1.0})


def test_history_records_one_entry_per_refit_in_order() -> None:
    agent = _agent()
    agent.ingest(_chain(400, 0))
    agent.refit()
    after_first = agent.history()

    agent.ingest(_chain(200, 1, x=1.0), source="interventional", target="X")
    agent.refit()
    after_second = agent.history()

    # One entry per call, appended -- and the snapshot handed out earlier does not grow.
    assert len(after_first) == 1
    assert len(after_second) == 2
    assert after_first == after_second[:1]
    assert [record.step for record in after_second] == [400, 600]
    assert [record.belief_size for record in after_second] == [3, 1]


def test_refit_without_observational_data_raises_before_pc_sees_it() -> None:
    agent = _agent()
    agent.ingest(_chain(200, 0, x=1.0), source="interventional", target="X")
    with pytest.raises(ValueError, match="observational"):
        agent.refit()


def test_ingest_requires_a_full_row_per_sample() -> None:
    # discover_interventional needs every variable in every dataset it is handed, so a partial
    # column set is refused at the buffer rather than deep inside PC.
    agent = _agent()
    full = _chain(50, 0)
    with pytest.raises(ValueError, match="'Z'"):
        agent.ingest({"X": full["X"], "Y": full["Y"]})


def test_ingest_refuses_ragged_columns() -> None:
    agent = _agent()
    full = _chain(50, 0)
    with pytest.raises(ValueError, match="unequal length"):
        agent.ingest({"X": full["X"], "Y": full["Y"], "Z": full["Z"][:10]})


def test_an_empty_do_buffer_orients_nothing() -> None:
    # A target's buffer can exist and hold nothing -- here because the ingest that would have
    # filled it was rejected. discover_interventional decides orientation by comparing marginals,
    # and an EMPTY marginal differs maximally from any observational one, so an empty do-sample
    # reads as a shift and would orient every edge incident to that target from no data at all.
    # refit() drops empty buffers, so the belief must stay where the observational data leaves it.
    agent = _agent()
    agent.ingest(_chain(800, 0))
    partial = _chain(50, 1, x=1.0)
    with pytest.raises(ValueError, match="'Z'"):
        agent.ingest({"X": partial["X"], "Y": partial["Y"]}, source="interventional", target="X")
    agent.refit()
    assert agent.belief_size() == 3


def test_the_belief_is_empty_before_the_first_refit() -> None:
    agent = _agent()
    agent.ingest(_chain(50, 0))
    assert agent.belief() == ()
    assert agent.belief_size() == 0
    assert agent.history() == ()
    assert agent.cpdag is None


def test_the_over_size_refusal_propagates() -> None:
    # fit_scm_mec refuses an equivalence class above max_members rather than truncating the belief;
    # the agent must not swallow that -- a silently truncated belief is a misreported one.
    agent = _agent(max_members=2)
    agent.ingest(_chain(800, 0))
    with pytest.raises(ValueError, match="3 members"):
        agent.refit()


def test_constructor_rejects_an_ill_formed_problem() -> None:
    with pytest.raises(ValueError, match="treatment="):
        OnlineCausalMBRL(VARIABLES, treatment="A", outcome="Z", actions=(0.0, 1.0))
    with pytest.raises(ValueError, match="outcome="):
        OnlineCausalMBRL(VARIABLES, treatment="X", outcome="R", actions=(0.0, 1.0))
    with pytest.raises(ValueError, match="actions"):
        OnlineCausalMBRL(VARIABLES, treatment="X", outcome="Z", actions=())
    with pytest.raises(ValueError, match="policy="):
        OnlineCausalMBRL(
            VARIABLES,
            treatment="X",
            outcome="Z",
            actions=(0.0, 1.0),
            policy="greedy",  # type: ignore[arg-type]
        )


def test_observe_rejects_an_intervention_on_an_unknown_variable() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="not a variable"):
        agent.observe({"X": 1.0, "Y": 1.0, "Z": 1.0}, intervention={"Q": 1.0})


# -- task 2: act, probe, and the empty-belief guard ----------------------------------------------
def _thompson_sequence(seed: int, *, calls: int = 12) -> list[float]:
    """The first ``calls`` actions a thompson agent takes on the two-member disagreeing belief."""
    agent = OnlineCausalMBRL(
        VARIABLES, treatment="X", outcome="Z", actions=(0.0, 1.0), n_rollout=8, seed=seed
    )
    agent._belief = _disagreeing_belief()
    return [agent.act() for _ in range(calls)]


def test_thompson_replays_under_a_fixed_seed_and_diverges_across_seeds() -> None:
    # The two members' argmaxes differ, so the action sequence IS the sequence of members drawn.
    # A fixed seed must replay it exactly; different seeds must not all produce the same one; and
    # a single seed must reach both members, which is the exploration the draw exists to provide.
    assert _thompson_sequence(0) == _thompson_sequence(0)
    sequences = {tuple(_thompson_sequence(seed)) for seed in range(4)}
    assert len(sequences) > 1
    assert set(_thompson_sequence(0)) == {0.0, 1.0}


def test_robust_takes_the_maximin_and_average_the_higher_mean() -> None:
    # Exact, not statistical: the members are deterministic, so a = 0 is worth (0, 3) and a = 1 is
    # worth (6, -1). Means 1.5 vs 2.5 -> average picks 1.0. Minima 0 vs -1 -> maximin picks 0.0.
    # The two policies must therefore disagree here, which is the point of the construction.
    for policy, expected in (("average", 1.0), ("robust", 0.0)):
        agent = OnlineCausalMBRL(
            VARIABLES,
            treatment="X",
            outcome="Z",
            actions=(0.0, 1.0),
            policy=policy,  # type: ignore[arg-type]
            n_rollout=8,
        )
        agent._belief = _disagreeing_belief()
        assert agent.act() == expected


def test_probe_returns_the_target_the_members_disagree_about() -> None:
    # Y -> Z against Z -> Y, with X disconnected in both. Under do(Y) the first member's Z follows
    # the do-value and the second's does not, so they disagree. Under do(X) neither member's Z
    # moves at all, and under do(Z) both are pinned to the same value -- zero disagreement on both.
    # Y is not the first candidate by name, so returning the first would return X and fail here.
    agent = OnlineCausalMBRL(
        VARIABLES, treatment="X", outcome="Z", actions=(0.0, 1.0), n_rollout=16
    )
    agent.ingest(_chain(40, 0))  # only so probe() has a do-value to intervene at
    agent._belief = (
        _member(
            [("Y", "Z")],
            {"X": _constant(0.0), "Y": _constant(0.0), "Z": _affine("Y", 1.0, 0.0)},
        ),
        _member(
            [("Z", "Y")],
            {"X": _constant(0.0), "Y": _affine("Z", 1.0, 0.0), "Z": _constant(0.0)},
        ),
    )
    assert agent.probe() == "Y"


def test_probe_on_a_singleton_belief_raises_naming_the_guard() -> None:
    # Data buffered so that the refusal under test is the singleton guard and not the missing
    # do-value: with one member there are no pairs to score, so a probe() without the guard would
    # rank every candidate at nan and hand back whichever sorted first.
    agent = _agent()
    agent.ingest(_chain(40, 0))
    agent._belief = _disagreeing_belief()[:1]
    with pytest.raises(ValueError, match="structure_uncertain"):
        agent.probe()


def test_probe_refuses_when_no_candidate_shows_any_disagreement() -> None:
    # X -> Y against Y -> X, with Z disconnected from both. The members are different DAGs, but Z
    # is pinned to 0 under do(X) and do(Y) in each of them and to the do-value under do(Z), so no
    # experiment this score can see separates them. Returning a target anyway would report an
    # arbitrary pick as an informative one.
    agent = _agent()
    agent.ingest(_chain(40, 0))
    agent._belief = (
        _member(
            [("X", "Y")],
            {"X": _constant(0.0), "Y": _affine("X", 1.0, 0.0), "Z": _constant(0.0)},
        ),
        _member(
            [("Y", "X")],
            {"X": _affine("Y", 1.0, 0.0), "Y": _constant(0.0), "Z": _constant(0.0)},
        ),
    )
    with pytest.raises(ValueError, match="no target the belief"):
        agent.probe()


def test_probe_without_buffered_rows_raises_naming_the_do_value() -> None:
    # Reachable only by installing a belief without data, as these tests do -- refit() cannot
    # produce one without observational rows -- but the do-value has to come from somewhere.
    agent = _agent()
    agent._belief = _disagreeing_belief()
    with pytest.raises(ValueError, match="do-value"):
        agent.probe()


def test_act_probe_and_structure_uncertain_refuse_an_empty_belief() -> None:
    # The documented loop branches on structure_uncertain(), so an empty belief reporting False
    # would route a caller who never called refit() straight into act(). All three name refit().
    agent = _agent()
    agent.ingest(_chain(50, 0))
    for call in (agent.act, agent.probe, agent.structure_uncertain):
        with pytest.raises(ValueError, match="refit"):
            call()


def test_rollouts_share_one_seed_per_member_so_a_small_effect_survives_the_noise() -> None:
    # The one test here with a NOISY member: Z = 0.01 * X + noise, so the two actions are worth
    # 0.00 and 0.01 against a unit-variance rollout whose 16-sample standard error is ~0.25.
    # Seeding each member's rollouts once means both actions are scored on the same exogenous
    # draws, the noise cancels out of the comparison, and act() returns the better action every
    # time. Unseeded rollouts would draw independently per action and make it a coin flip.
    agent = OnlineCausalMBRL(
        VARIABLES, treatment="X", outcome="Z", actions=(0.0, 1.0), policy="average", n_rollout=16
    )
    agent._belief = (
        _member(
            [("X", "Z")],
            {
                "X": _constant(0.0),
                "Y": _constant(0.0),
                "Z": FunctionalMechanism(["X"], lambda parents, noise: 0.01 * parents["X"] + noise),
            },
        ),
    )
    assert {agent.act() for _ in range(20)} == {1.0}


def test_the_thompson_draw_advances_with_the_step_counter() -> None:
    # Two agents with the same seed that have buffered different amounts of data must not be
    # locked to the same draw: the RNG is derived from the step count as well as the seed.
    quiet = _agent()
    busy = _agent()
    quiet.ingest(_chain(40, 0))
    busy.ingest(_chain(41, 1))
    quiet._belief = _disagreeing_belief()
    busy._belief = _disagreeing_belief()
    assert [quiet.act() for _ in range(12)] != [busy.act() for _ in range(12)]


def test_n_rollout_below_one_is_refused() -> None:
    # Action values are means over see(n_rollout) draws. At n_rollout=0 every mean is nan, nan
    # comparisons are all False, and argmax returns index 0 -- so act() would return a valid-looking
    # action having compared nothing. Discriminating: removing the guard makes this not raise.
    with pytest.raises(ValueError, match="n_rollout"):
        OnlineCausalMBRL(["A", "Y"], treatment="A", outcome="Y", actions=(0.0, 1.0), n_rollout=0)
