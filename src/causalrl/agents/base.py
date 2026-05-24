from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Minimal agent interface: choose an action, then learn from the reward."""

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
