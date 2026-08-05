"""OnlineCausalMBRL (task 1): the buffers, ``refit``, and the I-MEC belief.

The world is a noisy chain ``X -> Y -> Z``. Observationally it is Markov-equivalent to
``X <- Y -> Z`` and ``X <- Y <- Z`` (no v-structure to orient), so PC leaves both edges undirected
and the belief holds three members. A perfect ``do(X)`` sample shifts ``Y``'s marginal but leaves
nothing upstream to be invariant, orienting ``X -> Y``; Meek's R1 then propagates ``Y -> Z`` and the
belief collapses to one member. That collapse is what these tests measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.online_causal_mbrl import OnlineCausalMBRL

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
    # Documented caveat: with no belief there is nothing to be uncertain BETWEEN, so this reads
    # False before the first refit. It is not a claim that the structure is known.
    assert not agent.structure_uncertain()


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
