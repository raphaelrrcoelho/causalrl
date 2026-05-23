from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Minimal agent interface: choose an action, then learn from the reward."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> int: ...

    @abstractmethod
    def update(self, observation: dict[str, Any], action: int, reward: float) -> None: ...
