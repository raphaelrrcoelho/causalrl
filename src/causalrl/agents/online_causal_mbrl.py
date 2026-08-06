"""Learn the SCM while acting: buffers, ``refit``, and the I-MEC belief.

:func:`~causalrl.scm.fit.fit_scm` learns an SCM from a static table -- supervised learning, with no
policy, no reward and no exploration. :class:`OnlineCausalMBRL` is the online counterpart: it keeps
an observational buffer (off-policy, possibly confounded) alongside one interventional buffer per
intervention target, and on :meth:`~OnlineCausalMBRL.refit` recomputes the **interventional Markov
equivalence class** (I-MEC) -- the DAGs consistent with both distributions -- and fits one SCM per
member. That set of SCMs is the agent's belief. Interventions shrink it; observation alone cannot.

This is an implementation of published methods on this library's primitives, not a contribution:

- The alternating loop -- intervene to learn structure, then use that structure to guide the
  policy -- is Sun et al., *Learning by Doing: an online causal reinforcement learning framework
  with causal-aware policy*, Science China Information Sciences (2024),
  `arXiv:2402.04869 <https://arxiv.org/abs/2402.04869>`_. No code is ported from their repository.
- Fusing observational with experimental data is Bareinboim & Forney, *Bandits with Unobserved
  Confounders: A Causal Approach* (MABUC), and Forney, Pearl & Bareinboim, *Counterfactual
  Data-Fusion for Online Reinforcement Learners* -- under unobserved confounding an agent needs
  both quantities, and averaging the confounder out can incur unbounded regret.
- The I-MEC itself is Hauser & Buehlmann, *Characterization and Greedy Learning of Interventional
  Markov Equivalence Classes of DAGs*, JMLR 2012; the invariance principle that orients edges
  incident to an intervention target is Peters, Buehlmann & Meinshausen, *Causal Inference using
  Invariant Prediction*, JRSS-B 2016. Both are already implemented in
  :func:`~causalrl.discovery.discover_interventional`; this module only feeds them data.

What is this library's own stance is the refusal to guess: :func:`~causalrl.scm.fit.fit_scm_mec`
raises rather than truncating an equivalence class it cannot enumerate, and this agent reports
:meth:`~OnlineCausalMBRL.belief_size` and :meth:`~OnlineCausalMBRL.structure_uncertain` as
first-class outputs instead of committing to an arbitrary member.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np

from causalrl.discovery import CPDAG, discover_interventional
from causalrl.scm.fit import fit_scm_mec
from causalrl.scm.scm import StructuralCausalModel

Source = Literal["observational", "interventional"]
Policy = Literal["thompson", "average", "robust"]

_SOURCES: tuple[Source, ...] = ("observational", "interventional")
_POLICIES: tuple[Policy, ...] = ("thompson", "average", "robust")


class RefitRecord(NamedTuple):
    """One :meth:`OnlineCausalMBRL.refit` outcome: the belief size at a transition count.

    The sequence of these is the belief trajectory -- the I-MEC collapsing (or, at finite samples,
    flickering) as experiments arrive. It is recorded rather than assumed monotone: PC is a
    finite-sample procedure and a later CPDAG can be *less* oriented than an earlier one.
    """

    step: int
    belief_size: int


class OnlineCausalMBRL:
    """An agent that learns its SCM from its own interventions plus an off-policy log.

    ``variables`` are the columns every buffered row must carry; ``treatment`` is the node the
    agent intervenes on, ``outcome`` the node it is scored by, and ``actions`` the values it may
    set the treatment to. Data arrives through :meth:`ingest` (bulk, off-policy) or :meth:`observe`
    (one transition, on-policy), and :meth:`refit` turns the buffers into the belief:

    .. code-block:: text

        discover_interventional(observational, interventional, variables)  ->  CPDAG (the I-MEC)
        fit_scm_mec(all rows, cpdag=..., max_members=...)  ->  one fitted SCM per member

    ``policy``, ``actions`` and ``n_rollout`` describe how the belief will be turned into an action;
    the action selection itself (``act``/``probe``) is not part of this class yet, so those three
    are validated and stored here and nothing more. ``refit_every`` is likewise the cadence an
    acting loop should call :meth:`refit` at -- rerunning PC every step is wasteful and makes the
    belief flicker on sampling noise -- and is not self-triggering.

    ``max_members`` is handed to :func:`~causalrl.scm.fit.fit_scm_mec`, whose refusal above the cap
    propagates out of :meth:`refit` deliberately: an equivalence class too large to enumerate is
    information the caller needs, and a silently truncated belief is a misreported one.
    """

    def __init__(
        self,
        variables: Sequence[str],
        *,
        treatment: str,
        outcome: str,
        actions: Sequence[float],
        policy: Policy = "thompson",
        max_members: int = 32,
        refit_every: int = 8,
        n_rollout: int = 512,
        min_interventional_samples: int = 20,
        seed: int = 0,
    ) -> None:
        self.variables = tuple(variables)
        if treatment not in self.variables:
            raise ValueError(f"treatment={treatment!r} is not one of variables={self.variables}")
        if outcome not in self.variables:
            raise ValueError(f"outcome={outcome!r} is not one of variables={self.variables}")
        if not actions:
            raise ValueError("actions is empty: there is nothing for the agent to choose between")
        if policy not in _POLICIES:
            raise ValueError(f"policy={policy!r} is not one of {list(_POLICIES)}")
        self.treatment = treatment
        self.outcome = outcome
        self.actions = tuple(actions)
        self.policy: Policy = policy
        self.max_members = max_members
        self.refit_every = refit_every
        self.n_rollout = n_rollout
        self.min_interventional_samples = min_interventional_samples
        self.seed = seed

        # Columnar buffers: one list per variable, appended to in place. Rebuilding columns from
        # per-row dicts on every refit is the same data at a much worse cost.
        self._observational: dict[str, list[float]] = {v: [] for v in self.variables}
        self._interventional: dict[str, dict[str, list[float]]] = {}
        self._belief: tuple[StructuralCausalModel, ...] = ()
        self._cpdag: CPDAG | None = None
        self._history: list[RefitRecord] = []
        self._steps = 0

    # -- data in ------------------------------------------------------------------------------
    def ingest(
        self,
        data: Mapping[str, np.ndarray],
        *,
        source: Source = "observational",
        target: str | None = None,
    ) -> None:
        """Append a bulk log to the observational buffer, or to ``target``'s do-buffer.

        ``source="interventional"`` requires ``target`` (the node that was set) and
        ``source="observational"`` forbids it. Every variable must be present with equal lengths:
        :func:`~causalrl.discovery.discover_interventional` compares the *marginal of every
        endpoint* between regimes, so a do-sample holding only the reward cannot orient anything.
        """
        if source == "interventional":
            if target is None:
                raise ValueError(
                    "ingest(source='interventional') requires target=<variable>: "
                    "discover_interventional orients the edges incident to each intervention "
                    "target separately, so a do-sample must name the node that was set."
                )
            buffer = self._buffer_for(target)
        elif source == "observational":
            if target is not None:
                raise ValueError(
                    f"ingest(source='observational') forbids target={target!r}: an observational "
                    "sample has no intervention target. Pass source='interventional' to route it "
                    "to that target's buffer."
                )
            buffer = self._observational
        else:
            raise ValueError(f"source={source!r} is not one of {list(_SOURCES)}")
        self._steps += _extend(buffer, data, self.variables)

    def observe(
        self,
        row: Mapping[str, float],
        *,
        intervention: Mapping[str, float] | None = None,
    ) -> None:
        """Append one transition -- to the observational buffer, or to its target's do-buffer.

        With ``intervention``, the row joins the buffer for that target and the do-value is what is
        recorded for it: under a perfect intervention the target was *set*, so ``row`` may omit it.
        Exactly one target is supported per row, because
        :func:`~causalrl.discovery.discover_interventional` orients per single target.
        """
        values = dict(row)
        if intervention is None:
            buffer = self._observational
        else:
            if len(intervention) != 1:
                raise ValueError(
                    f"observe() supports exactly one target per row, got "
                    f"{sorted(intervention)}: discover_interventional orients the edges incident "
                    "to a single intervention target, so a multi-node do() has no buffer to join."
                )
            target = next(iter(intervention))
            buffer = self._buffer_for(target)
            values.update(intervention)
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise ValueError(
                f"row is missing variable(s) {missing}: discover_interventional needs every "
                "variable in every dataset it is handed, so buffers hold full rows."
            )
        for variable in self.variables:
            buffer[variable].append(float(values[variable]))
        self._steps += 1

    def _rows(self, buffer: Mapping[str, list[float]]) -> int:
        """How many transitions a buffer holds (its columns are appended to in lockstep)."""
        return len(buffer[self.variables[0]])

    def _buffer_for(self, target: str) -> dict[str, list[float]]:
        if target not in self.variables:
            raise ValueError(
                f"intervention target {target!r} is not a variable of this agent: {self.variables}"
            )
        if target not in self._interventional:
            self._interventional[target] = {v: [] for v in self.variables}
        return self._interventional[target]

    # -- learning -----------------------------------------------------------------------------
    def refit(self) -> tuple[StructuralCausalModel, ...]:
        """Recompute the I-MEC from the buffers and fit one SCM per member; return the belief.

        Two roles for the same rows, and they are not interchangeable. **Orientation** comes from
        the do-data: :func:`~causalrl.discovery.discover_interventional` runs PC on the
        observational sample for the skeleton and then orients the edges incident to each target by
        the invariance principle, which only the interventional marginals can decide.
        **Mechanisms** are fitted from every row the agent holds, observational and interventional
        concatenated, because a mechanism is invariant across regimes (Peters et al. 2016) and the
        do-rows are extra samples of every node that was not intervened on. The intervened node's
        own mechanism is fitted from rows where it was overridden, which biases *its* marginal --
        harmless here, since planning replaces that mechanism with ``do(treatment=a)`` anyway.

        A target holding fewer than ``min_interventional_samples`` rows is dropped rather than
        passed on. The invariance test compares empirical marginals, so a small do-sample clears
        any shift threshold by chance and an empty one clears it unconditionally — orienting every
        edge incident to that target from evidence the agent does not have.
        :func:`~causalrl.discovery.discover_interventional` refuses such a sample outright; the
        agent filters first so that an early round, when a target has been probed once or twice, is
        an ordinary under-informed refit rather than a raise.

        Raises ``ValueError`` if no observational rows have been buffered (PC needs them for the
        base CPDAG), and lets :func:`~causalrl.scm.fit.fit_scm_mec`'s over-size refusal propagate.
        """
        if not self._rows(self._observational):
            raise ValueError(
                "refit() needs observational rows: discover_interventional runs PC on the "
                "observational sample to build the base CPDAG before the invariance principle "
                "orients edges incident to each intervention target. Ingest an observational log "
                "(or call observe() without intervention=...) first."
            )
        observational = _columns(self._observational)
        interventions = {
            target: _columns(buffer)
            for target, buffer in sorted(self._interventional.items())
            if self._rows(buffer) >= self.min_interventional_samples
        }
        cpdag = discover_interventional(
            observational,
            interventions,
            self.variables,
            min_interventional_samples=self.min_interventional_samples,
        )
        pooled = {
            variable: np.concatenate(
                [observational[variable], *(data[variable] for data in interventions.values())]
            )
            for variable in self.variables
        }
        self._belief = tuple(
            fit_scm_mec(pooled, cpdag=cpdag, max_members=self.max_members, seed=self.seed)
        )
        self._cpdag = cpdag
        self._history.append(RefitRecord(step=self._steps, belief_size=len(self._belief)))
        return self._belief

    # -- what the agent believes --------------------------------------------------------------
    def belief(self) -> tuple[StructuralCausalModel, ...]:
        """The fitted I-MEC members -- empty until the first :meth:`refit`."""
        return self._belief

    def belief_size(self) -> int:
        """How many DAGs the data still cannot choose between."""
        return len(self._belief)

    def structure_uncertain(self) -> bool:
        """Whether more than one I-MEC member survives -- a decision may need the certificate layer.

        Before the first :meth:`refit` the belief is empty and this reads ``False``: there is
        nothing to be uncertain *between* yet. That is not a claim that the structure is known.
        """
        return self.belief_size() > 1

    def history(self) -> tuple[RefitRecord, ...]:
        """An immutable snapshot of the belief trajectory, one :class:`RefitRecord` per refit."""
        return tuple(self._history)

    @property
    def cpdag(self) -> CPDAG | None:
        """The I-essential graph from the last :meth:`refit`; ``None`` before the first."""
        return self._cpdag

    @property
    def steps(self) -> int:
        """Transitions buffered so far, observational and interventional together."""
        return self._steps


def _extend(
    buffer: dict[str, list[float]], data: Mapping[str, np.ndarray], variables: Sequence[str]
) -> int:
    """Append every column of ``data`` to ``buffer``; return the number of rows appended."""
    missing = [v for v in variables if v not in data]
    if missing:
        raise ValueError(
            f"data is missing column(s) for variable(s) {missing}: discover_interventional needs "
            "every variable in every dataset it is handed, so buffers hold full rows."
        )
    columns = {v: np.asarray(data[v], dtype=float).ravel() for v in variables}
    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"data has columns of unequal length: {sorted(lengths)}")
    for variable, column in columns.items():
        buffer[variable].extend(float(value) for value in column)
    return lengths.pop()


def _columns(buffer: Mapping[str, list[float]]) -> dict[str, np.ndarray]:
    """The columnar buffer as the ``Mapping[str, np.ndarray]`` the library's data functions take."""
    return {variable: np.asarray(column, dtype=float) for variable, column in buffer.items()}
