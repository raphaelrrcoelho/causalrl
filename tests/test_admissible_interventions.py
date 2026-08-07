"""AdmissibleInterventions — POMIS recomputed for the levers a given context permits."""

import pytest

from causalrl.identification.intervention_sets import AdmissibleInterventions, pomis
from causalrl.intervention import InterventionSpace, canonical
from causalrl.scm.graph import CausalGraph

# C -> B, C -> D, C -> Y, D -> Y, with B <-> D.
#
# The witness for "restricting the manipulable set is not a filter": unconstrained POMIS is
# [{C, D}], so filtering to the sets that avoid C leaves NOTHING, while the correct constrained
# answer — the POMIS of the latent projection onto {D, Y} — is [{}, {D}]. A filtering
# implementation would tell an agent that can only reach D that it has no move worth making.
_WITNESS = CausalGraph(
    directed_edges=[("C", "B"), ("C", "D"), ("C", "Y"), ("D", "Y")],
    bidirected_edges=[("B", "D")],
)


def test_unconstrained_matches_pomis() -> None:
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    assert set(admissible.sets({"B", "C", "D"})) == set(pomis(_WITNESS, "Y"))


def test_restriction_is_a_projection_not_a_filter() -> None:
    unconstrained = pomis(_WITNESS, "Y")
    assert set(unconstrained) == {frozenset({"C", "D"})}

    naive_filter = [s for s in unconstrained if s <= {"D"}]
    assert naive_filter == []  # what a filtering implementation would return

    admissible = AdmissibleInterventions(_WITNESS, "Y")
    assert set(admissible.sets({"D"})) == {frozenset(), frozenset({"D"})}


def test_a_novel_set_survives_the_arms_composition() -> None:
    # The set {D} that only the projection produces must reach the agent as a real arm.
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    space = InterventionSpace.create({"D": [0, 1]})
    keys = {canonical(arm) for arm in admissible.arms(space)}
    assert keys == {canonical({}), canonical({"D": 0}), canonical({"D": 1})}


def test_arms_are_deduplicated_across_overlapping_sets() -> None:
    # The empty intervention is reachable from the empty set only; no arm may appear twice even
    # when two POMISs share assignments.
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    arms = admissible.arms(InterventionSpace.create({"D": [0, 1]}))
    keys = [canonical(arm) for arm in arms]
    assert len(keys) == len(set(keys))


def test_arms_order_is_deterministic() -> None:
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    space = InterventionSpace.create({"D": [0, 1]})
    first = [canonical(a) for a in admissible.arms(space)]
    second = [canonical(a) for a in AdmissibleInterventions(_WITNESS, "Y").arms(space)]
    assert first == second


def test_results_are_memoised_but_still_correct_across_different_sets() -> None:
    admissible = AdmissibleInterventions(_WITNESS, "Y", cache_size=2)
    for _ in range(3):
        assert set(admissible.sets({"D"})) == {frozenset(), frozenset({"D"})}
        assert set(admissible.sets({"B", "C", "D"})) == set(pomis(_WITNESS, "Y"))
        assert set(admissible.sets({"C"})) == set(pomis(_WITNESS, "Y", manipulable={"C"}))


def test_cache_eviction_does_not_change_answers() -> None:
    small = AdmissibleInterventions(_WITNESS, "Y", cache_size=1)
    large = AdmissibleInterventions(_WITNESS, "Y", cache_size=32)
    for manipulable in ({"D"}, {"C"}, {"B", "C", "D"}, {"D"}, {"C"}):
        assert set(small.sets(manipulable)) == set(large.sets(manipulable))


def test_returned_lists_do_not_alias_the_cache() -> None:
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    first = admissible.sets({"D"})
    first.clear()
    assert set(admissible.sets({"D"})) == {frozenset(), frozenset({"D"})}


def test_the_reward_is_never_manipulable() -> None:
    admissible = AdmissibleInterventions(_WITNESS, "Y")
    assert set(admissible.sets({"D", "Y"})) == set(admissible.sets({"D"}))


def test_reward_must_be_a_node() -> None:
    with pytest.raises(ValueError, match="not a node"):
        AdmissibleInterventions(_WITNESS, "Z")


def test_cache_size_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="cache_size"):
        AdmissibleInterventions(_WITNESS, "Y", cache_size=0)


def test_reward_property_is_exposed() -> None:
    assert AdmissibleInterventions(_WITNESS, "Y").reward == "Y"
