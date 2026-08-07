"""The agent interfaces: what an agent must implement, and when learning happens.

:class:`Agent` is the minimal contract — choose an action, learn from the reward. It suits an
*online* learner, which is what most of the shipped bandit and value-iteration agents are.

It does not suit the rest. A batch agent gets its whole policy from an offline step (``fit``,
``ingest_offline``) and has nothing to do with a single reward handed to it afterwards; the ten
such agents here each carried an empty ``update`` to satisfy the ABC, which made the interface
claim something untrue of a third of its implementers. :class:`BatchAgent` states that directly
and supplies the no-op once, so the distinction is visible in the type instead of being rediscovered
by reading each class body.
"""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Minimal agent interface: choose an action, then learn from the reward.

    Implement this directly for an **online** learner, one whose policy moves in response to each
    reward it sees. If learning happens in a batch step instead, subclass :class:`BatchAgent`.
    """

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> int: ...

    @abstractmethod
    def update(self, observation: dict[str, Any], action: int, reward: float) -> None: ...

    def observe_transition(  # noqa: B027
        self, state: int, action: int, next_state: int, done: bool
    ) -> None:
        """Optional model-learning hook: observe a `(s, a, s', done)` transition.

        Default no-op. Model-based agents (e.g. DOVI's value iteration) override this to
        build an empirical transition model; reward-only agents ignore it.
        """
        pass

    def observe_step(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
    ) -> None:
        """Observe one transition in the environment's own observation space.

        This is the hook a *driver* should call. :meth:`observe_transition` takes state indices,
        which forces whoever drives the loop to discretise before the agent sees anything -- and an
        agent whose states are feature vectors then has no way to receive the transition at all.
        Here the agent is handed the raw observations and decides for itself how to represent them.

        The default is the tabular reading, so every existing agent and driver keeps its exact
        behaviour: pull ``observation["state"]`` from each endpoint and forward to
        :meth:`observe_transition`. A feature-space agent overrides this to encode instead.
        """
        self.observe_transition(
            int(observation["state"]), action, int(next_observation["state"]), done
        )


class BatchAgent(Agent):
    """An agent whose policy comes from a batch step, not from per-reward updates.

    Subclass this when the policy is produced by an offline fit — a back-door adjustment, a fitted
    backup, a certified plan — and a single ``(observation, action, reward)`` triple carries nothing
    the agent can act on. :meth:`update` is a concrete no-op here so those agents do not each have
    to write one, and so that a reader can tell from the base class alone that calling it is
    expected to do nothing.

    Still an :class:`Agent`, so any harness that drives the online loop keeps working: the batch
    agent simply ignores the rewards it is handed. Refit through the subclass's own entry point
    (``fit``/``ingest_offline``) to move the policy.
    """

    def update(self, observation: dict[str, Any], action: int, reward: float) -> None:
        """No-op: this agent learns in its batch step. See the class docstring."""
