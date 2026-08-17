"""The game whose payoffs were *measured*: a finite normal form plus the error of its own table.

:class:`~causalrl.magames.population.Population` builds a game from a payoff *function*, exact by
construction. The other way a finite game arrives is empirically: fix a pool of strategies, play
every profile of them some number of times, and average — the standard empirical-game-theoretic
construction (M. P. Wellman, *Methods for Empirical Game-Theoretic Analysis*, AAAI 2006). The
resulting table is an estimate, and every downstream object in this module — the polytope, the
learners, the certificate — otherwise reads it as if it were the game itself.

:class:`EmpiricalGame` is that table with its error kept attached. It materialises a
:class:`~causalrl.games.CausalGame` from the means (so
:func:`~causalrl.magames.learning.run_no_regret` and
:func:`~causalrl.magames.cce.certify_cce_do` apply unchanged), and turns the standard errors
into the :class:`~causalrl.magames.cce.PayoffError` those certificates accept. Handing that over is
what keeps a certificate from claiming, off ``n`` simulated matches, something that is only true of
the exact game::

    empirical = EmpiricalGame.from_samples(pool, results)
    run = run_no_regret(empirical.to_game(), 20_000)
    cert = certify_cce_do(
        empirical.to_game(),
        functional,
        no_regret=False,
        epsilon=run.regret,
        payoff_error=empirical.payoff_error(),
    )

The strategy labels *are* the action encoding: strategy ``i`` in the tuple given is integer action
``i`` everywhere else in the module, so ``run.marginal("A1")`` reads back through
:attr:`strategies`. Every agent draws from the same pool, which is the usual round-robin setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from causalrl.estimate._stats import norm_ppf
from causalrl.games import CausalGame
from causalrl.magames.cce import PayoffError
from causalrl.magames.population import AgentType, PayoffTemplate, Population

__all__ = ["EmpiricalGame"]

Profile = tuple[str, ...]
Samples = Mapping[Profile, Sequence[Sequence[float]]]


def _default_agents(arity: int) -> tuple[str, ...]:
    return tuple(f"A{i + 1}" for i in range(arity))


@dataclass(frozen=True)
class EmpiricalGame:
    """A finite game estimated by replication: per-profile payoff means, standard errors and ``n``.

    All three mappings are keyed by a profile of *strategy labels* in agent order. Build one with
    :meth:`from_samples` rather than by hand — it is the constructor that computes the error.
    """

    strategies: tuple[str, ...]
    agents: tuple[str, ...]
    means: Mapping[Profile, tuple[float, ...]]
    stderrs: Mapping[Profile, tuple[float, ...]]
    replications: Mapping[Profile, int]

    @classmethod
    def from_samples(
        cls,
        strategies: Sequence[str],
        samples: Samples,
        *,
        agents: Sequence[str] | None = None,
    ) -> EmpiricalGame:
        """Average a complete round-robin into a game, keeping each cell's standard error.

        ``samples`` maps a profile of strategy labels to the replications observed for it, each
        replication being one payoff per agent. Every profile in the product of ``strategies`` must
        be present with at least two replications: a cell with one sample has no error bar, and an
        object whose contract is to carry the error bar must not invent one.
        """
        pool = tuple(strategies)
        duplicates = sorted({s for s in pool if pool.count(s) > 1})
        if duplicates:
            raise ValueError(f"duplicate strategy labels: {duplicates}")
        arity = len(next(iter(samples))) if samples else 0
        names = tuple(agents) if agents is not None else _default_agents(arity)
        if agents is not None and arity and len(names) != arity:
            raise ValueError(f"{len(names)} agents given but profiles have arity {arity}")

        means: dict[Profile, tuple[float, ...]] = {}
        stderrs: dict[Profile, tuple[float, ...]] = {}
        counts: dict[Profile, int] = {}
        for profile in product(pool, repeat=arity):
            if profile not in samples:
                raise ValueError(f"round-robin is missing the profile {profile}")
            replications = samples[profile]
            if len(replications) < 2:
                raise ValueError(
                    f"profile {profile} has {len(replications)} replication(s); at least 2 are "
                    "needed to estimate its error"
                )
            if any(len(r) != arity for r in replications):
                raise ValueError(
                    f"profile {profile} needs one payoff per agent in every replication"
                )
            observed = np.asarray(replications, dtype=np.float64)
            means[profile] = tuple(float(v) for v in observed.mean(axis=0))
            spread = observed.std(axis=0, ddof=1) / np.sqrt(observed.shape[0])
            stderrs[profile] = tuple(float(v) for v in spread)
            counts[profile] = observed.shape[0]
        return cls(pool, names, means, stderrs, counts)

    def payoff_error(self, *, alpha: float = 0.05, functional_terms: int = 1) -> PayoffError:
        """The cellwise error bound this table licenses, at level ``alpha``.

        A normal interval on each cell's mean, Bonferroni-corrected over every (profile, agent)
        cell so the bound holds simultaneously — which is what the certificate needs, since its
        linear programs touch all of them at once. The reported ``utility`` is the widest such
        half-width, because the relaxation it feeds is a single scalar over all constraints.

        ``functional_terms`` is how many estimated payoffs the functional being certified sums: 1
        for one agent's payoff (the usual case), ``len(agents)`` for welfare, 0 for a functional
        that does not read the payoffs at all.
        """
        if functional_terms < 0:
            raise ValueError("functional_terms must be nonnegative")
        cells = sum(len(v) for v in self.stderrs.values())
        widest = max((max(v, default=0.0) for v in self.stderrs.values()), default=0.0)
        z = float(norm_ppf(1.0 - alpha / (2.0 * max(cells, 1))))
        utility = z * widest
        return PayoffError(utility=utility, functional=functional_terms * utility, alpha=alpha)

    def to_game(self) -> CausalGame:
        """Materialise the normal form over integer actions, using the estimated means as payoffs.

        Strategy ``i`` is action ``i``. The returned game says nothing about its own error — pass
        :meth:`payoff_error` alongside it to :func:`~causalrl.magames.cce.certify_cce_do`.
        """
        actions = tuple(range(len(self.strategies)))
        types = {
            agent: AgentType(name=agent, actions=actions, payoff=self._payoff_for(position))
            for position, agent in enumerate(self.agents)
        }
        return Population(agents=self.agents, types=types).to_game()

    def _payoff_for(self, position: int) -> PayoffTemplate:
        labels = self.strategies

        def payoff(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
            actions = list(others)
            actions.insert(position, own)
            return self.means[tuple(labels[a] for a in actions)][position]

        return payoff
