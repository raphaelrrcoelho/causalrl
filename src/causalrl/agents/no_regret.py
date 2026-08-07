"""No-regret learners over a finite action set (regret matching, multiplicative weights).

These are the learners whose *external regret* — the gain the learner would have made by switching
to any single fixed action for the whole run — grows sublinearly in the horizon. That is exactly the
quantity :func:`~causalrl.magames.cce.cce_regret` measures on a realized joint distribution, so a
population of these agents is what drives it to zero;
:func:`~causalrl.magames.learning.run_no_regret` runs such a population and hands the realized joint
to the certificate layer.

Implementations of published algorithms (no external code is ported):

* :class:`RegretMatching` — S. Hart, A. Mas-Colell, *A Simple Adaptive Procedure Leading to
  Correlated Equilibrium*, Econometrica 68(5):1127-1150, 2000. Play proportionally to the positive
  part of the cumulative regret; Blackwell's approachability theorem (D. Blackwell, *An Analog of
  the Minimax Theorem for Vector Payoffs*, Pacific J. Math. 6(1):1-8, 1956) drives the maximal
  positive external regret to zero. Parameter-free.
* :class:`MultiplicativeWeights` — the Hedge algorithm of Y. Freund, R. Schapire, *A
  Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*, JCSS
  55(1):119-139, 1997 (after N. Littlestone, M. Warmuth, *The Weighted Majority Algorithm*,
  Information and Computation 108(2):212-261, 1994). External regret ``O(sqrt(T log K))``.

The learning signal is :meth:`NoRegretLearner.observe`, which takes the **counterfactual payoff
vector**: what every action would have paid this round against whatever the rest of the environment
actually did. That is the full-information (expert) feedback model the guarantees above are stated
in, and it is available whenever the payoff structure is known — as it is for a finite
:class:`~causalrl.games.CausalGame`. The scalar-reward :meth:`NoRegretLearner.update` hook of
:class:`~causalrl.agents.base.Agent` covers the *bandit* case instead, by turning the one payoff
actually observed into an unbiased estimate of the whole vector through inverse-propensity
weighting: multiplicative weights on that estimate is EXP3 (P. Auer, N. Cesa-Bianchi, Y. Freund,
R. Schapire, *The Nonstochastic Multiarmed Bandit Problem*, SIAM J. Comput. 32(1):48-77, 2002),
which needs the uniform mixing supplied by ``explore`` to keep the estimate's variance bounded.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.agents.base import Agent

__all__ = ["MultiplicativeWeights", "NoRegretLearner", "RegretMatching"]

FloatArray = NDArray[np.float64]


class NoRegretLearner(Agent):
    """Base class for a learner over ``n_actions`` whose external regret grows sublinearly.

    ``explore`` mixes the algorithm's distribution with the uniform one (``(1 - explore) * p +
    explore / K``). Leave it at ``0`` under full-information feedback (:meth:`observe`); set it
    positive under bandit feedback (:meth:`update`), where it bounds the inverse-propensity weights.

    :meth:`act` samples the current mixed strategy. A repeated finite game has no state to condition
    on, so ``observation`` is ignored here; it is present because it is part of the
    :class:`~causalrl.agents.base.Agent` interface, and subclasses that *do* condition on it should
    say so.
    """

    def __init__(self, n_actions: int, *, explore: float = 0.0, seed: int | None = None) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be at least 1")
        if not 0.0 <= explore <= 1.0:
            raise ValueError("explore must lie in [0, 1]")
        self.n_actions = int(n_actions)
        self.explore = float(explore)
        self.rounds = 0
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def _strategy(self) -> FloatArray:
        """The algorithm's own distribution over actions, before the ``explore`` mixing."""

    @abstractmethod
    def _absorb(self, payoffs: FloatArray, strategy: FloatArray) -> None:
        """Fold one counterfactual payoff vector into the algorithm's state."""

    def distribution(self) -> FloatArray:
        """The mixed strategy actually played: the algorithm's, mixed with uniform ``explore``."""
        strategy = self._strategy()
        if self.explore == 0.0:
            return strategy
        return (1.0 - self.explore) * strategy + self.explore / self.n_actions

    def observe(self, payoffs: FloatArray | list[float]) -> None:
        """Learn from the counterfactual payoff vector: ``payoffs[k]`` is what action ``k`` paid."""
        vector = np.asarray(payoffs, dtype=np.float64)
        if vector.shape != (self.n_actions,):
            raise ValueError(f"payoffs must have shape ({self.n_actions},), got {vector.shape}")
        self._absorb(vector, self.distribution())
        self.rounds += 1

    def act(self, observation: dict[str, Any] | None = None) -> int:
        """Sample an action from :meth:`distribution` (``observation`` is not conditioned on)."""
        return int(self._rng.choice(self.n_actions, p=self.distribution()))

    def update(self, observation: dict[str, Any] | None, action: int, reward: float) -> None:
        """Learn from bandit feedback: one realized ``reward`` for the action actually played.

        The single observation is expanded into an unbiased estimate of the full payoff vector by
        inverse-propensity weighting (``reward / p[action]`` on the played coordinate, ``0``
        elsewhere) before the ordinary full-information update. With ``explore > 0`` this is EXP3
        for :class:`MultiplicativeWeights`; ``observation`` is not conditioned on.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside 0..{self.n_actions - 1}")
        played = self.distribution()
        estimate = np.zeros(self.n_actions)
        estimate[action] = float(reward) / max(float(played[action]), 1e-12)
        self._absorb(estimate, played)
        self.rounds += 1


class RegretMatching(NoRegretLearner):
    """Hart & Mas-Colell (2000) regret matching on external regrets: parameter-free, no-regret.

    The cumulative regret of action ``k`` is the total extra payoff it would have earned had it
    replaced the strategy actually played; the next round is drawn proportionally to the positive
    part of that vector (uniformly while no regret is positive).
    """

    def __init__(self, n_actions: int, *, explore: float = 0.0, seed: int | None = None) -> None:
        super().__init__(n_actions, explore=explore, seed=seed)
        self.regret: FloatArray = np.zeros(n_actions)

    def _strategy(self) -> FloatArray:
        positive = np.maximum(self.regret, 0.0)
        total = float(positive.sum())
        if total <= 0.0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return positive / total

    def _absorb(self, payoffs: FloatArray, strategy: FloatArray) -> None:
        self.regret += payoffs - float(strategy @ payoffs)


class MultiplicativeWeights(NoRegretLearner):
    """Hedge / multiplicative weights (Freund & Schapire 1997): ``p[k] ~ exp(eta * cumulative[k])``.

    ``learning_rate`` is ``eta``. Left at ``None`` it is the theory rate
    ``sqrt(8 log K / T) / payoff_range``: with ``horizon=T`` given, the fixed rate tuned to that
    horizon; without one, the anytime rate that uses the number of rounds seen so far instead
    (both from N. Cesa-Bianchi, G. Lugosi, *Prediction, Learning, and Games*, CUP 2006, §2.3).
    ``payoff_range`` is the width of the interval the payoffs live in, which is what the rate must
    be scaled by; pass the game's own range when it is not ``1``.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        learning_rate: float | None = None,
        horizon: int | None = None,
        payoff_range: float = 1.0,
        explore: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions, explore=explore, seed=seed)
        if learning_rate is not None and learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be at least 1")
        if payoff_range <= 0.0:
            raise ValueError("payoff_range must be positive")
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.payoff_range = float(payoff_range)
        self.cumulative: FloatArray = np.zeros(n_actions)

    def _rate(self) -> float:
        if self.learning_rate is not None:
            return self.learning_rate
        elapsed = self.horizon if self.horizon is not None else max(1, self.rounds)
        log_k = np.log(max(2, self.n_actions))
        return float(np.sqrt(8.0 * log_k / elapsed) / self.payoff_range)

    def _strategy(self) -> FloatArray:
        logits = self._rate() * self.cumulative
        logits -= float(logits.max())
        weights = np.exp(logits)
        return weights / float(weights.sum())

    def _absorb(self, payoffs: FloatArray, strategy: FloatArray) -> None:
        self.cumulative += payoffs
