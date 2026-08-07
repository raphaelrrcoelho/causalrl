from causalrl.agents.baselines import OnlineOnlyUCB
from causalrl.envs.suite.dtr import DTREnv
from causalrl.eval.harness import run_episodes
from causalrl.eval.metrics import finite_horizon_regret


def test_run_episodes_returns_returns_list():
    env = DTREnv(seed=0)
    agent = OnlineOnlyUCB(n_states=3, n_actions=2, seed=0)
    returns = run_episodes(agent, env, n_episodes=50, seed=0)
    assert len(returns) == 50
    assert all(r in (0.0, 1.0) for r in returns)


def test_finite_horizon_regret():
    # optimal 0.75/episode, realized [0.75, 0.0, 0.75] -> regret = 0 + 0.75 + 0 = 0.75
    assert abs(finite_horizon_regret([0.75, 0.0, 0.75], optimal_return=0.75) - 0.75) < 1e-9


def test_run_episodes_forwards_transitions_to_agent():
    from causalrl.agents.base import Agent
    from causalrl.envs.suite.dtr import DTREnv
    from causalrl.eval.harness import run_episodes

    class RecordingAgent(Agent):
        def __init__(self):
            self.transitions = []

        def act(self, observation):
            return 0

        def update(self, observation, action, reward):
            pass

        def observe_transition(self, state, action, next_state, done):
            self.transitions.append((state, action, next_state, done))

    agent = RecordingAgent()
    run_episodes(agent, DTREnv(seed=0), n_episodes=3, seed=0)
    # DTR is horizon 1: one transition per episode, all terminal.
    assert len(agent.transitions) == 3
    assert all(done is True for (_s, _a, _ns, done) in agent.transitions)
    assert all(next_state == 2 for (_s, _a, next_state, _d) in agent.transitions)  # DTR terminal


def test_observe_step_encodes_for_a_feature_space_agent() -> None:
    """The representation-neutral driver hook must reach a feature-space agent.

    ``observe_transition`` takes state indices and :class:`~causalrl.FittedQIteration` refuses it
    outright, so before ``observe_step`` there was no hook by which a driver holding raw
    observations could deliver a transition to an agent whose states are vectors.

    Note this agent still invalidates its plan on every observation by design, so it is not
    drivable by :func:`~causalrl.eval.harness.run_episodes`' act-then-observe loop without a refit
    in between -- the hook fixes the representation mismatch, not the batch/online one.
    """
    from causalrl import FittedQIteration, IdentityEncoder

    agent = FittedQIteration(n_actions=2, horizon=2, encoder=IdentityEncoder(("x",)), seed=0)
    agent.observe_step({"state": 0, "x": 0.5}, 1, 2.0, {"state": 1, "x": 1.5}, True)

    buffered = agent.buffered_transitions()
    assert len(buffered) == 1
    assert buffered[0].state.tolist() == [0.5]
    assert buffered[0].next_state.tolist() == [1.5]
    assert buffered[0].action == 1
    assert buffered[0].reward == 2.0


def test_observe_step_default_is_the_tabular_hook() -> None:
    """Every existing agent and driver keeps its exact behaviour: the default forwards."""
    seen: list[tuple[int, int, int, bool]] = []

    class Recorder(OnlineOnlyUCB):
        def observe_transition(self, state: int, action: int, next_state: int, done: bool) -> None:
            seen.append((state, action, next_state, done))

    agent = Recorder(n_states=3, n_actions=2, seed=0)
    agent.observe_step({"state": 2}, 1, 1.0, {"state": 0}, True)
    assert seen == [(2, 1, 0, True)]
