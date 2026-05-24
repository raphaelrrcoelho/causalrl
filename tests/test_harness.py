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
