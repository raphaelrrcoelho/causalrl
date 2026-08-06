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
- Acting by drawing one member of the belief and treating it as true -- Thompson sampling over
  structure -- is Ortega & Braun, *Generalized Thompson Sampling for Sequential Decision-Making
  and Causal Inference*, `arXiv:1303.4431 <https://arxiv.org/abs/1303.4431>`_.
- Choosing the next intervention by how much the belief's members disagree about its effect is
  standard active intervention design, where predictive disagreement stands in for expected
  information gain: Scherrer et al., *Learning Neural Causal Models with Active Interventions*,
  `arXiv:2109.02429 <https://arxiv.org/abs/2109.02429>`_, and Zhang et al., *Active learning for
  optimal intervention design in causal models*, Nature Machine Intelligence 2023.

What is this library's own stance is the refusal to guess: :func:`~causalrl.scm.fit.fit_scm_mec`
raises rather than truncating an equivalence class it cannot enumerate, and this agent reports
:meth:`~OnlineCausalMBRL.belief_size` and :meth:`~OnlineCausalMBRL.structure_uncertain` as
first-class outputs instead of committing to an arbitrary member.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal, NamedTuple

import numpy as np

from causalrl.discovery import CPDAG, discover_interventional
from causalrl.scm.fit import fit_scm_mec
from causalrl.scm.scm import StructuralCausalModel

Source = Literal["observational", "interventional"]
Policy = Literal["thompson", "average", "robust"]

_SOURCES: tuple[Source, ...] = ("observational", "interventional")
_POLICIES: tuple[Policy, ...] = ("thompson", "average", "robust")

_TV_BINS = 32
"""Histogram bins :func:`_total_variation` scores two rollout samples on.

Total variation between *continuous* distributions has no sample estimator without a binning
choice, so :meth:`OnlineCausalMBRL.probe` compares two rollouts on a shared histogram over their
pooled range. The count matters only in that every candidate target is scored on the same one: the
score ranks targets against each other rather than reporting an absolute distance, and coarser bins
can only understate a disagreement, never invent one. For an integer-valued outcome whose observed
values span no more steps than this, every level falls in its own bin and the estimate is the exact
discrete total variation.
"""


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

    :meth:`act` turns the belief into an action from ``actions`` -- ``policy`` decides how the
    members' verdicts are combined and ``n_rollout`` how many samples each verdict costs -- and
    :meth:`probe` returns the intervention target the members disagree about most. The loop those
    two are meant for branches on :meth:`structure_uncertain`::

        target = agent.probe() if agent.structure_uncertain() else None
        action = agent.act() if target is None else None

    ``refit_every`` is the cadence that loop should call :meth:`refit` at -- rerunning PC every
    step is wasteful and makes the belief flicker on sampling noise -- and is not self-triggering.

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
        self._draws = 0

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

    # -- acting -------------------------------------------------------------------------------
    def act(self) -> float:
        """Choose an action from ``actions`` by planning inside the belief.

        Every policy scores an action the same way -- ``member.do({treatment: action})``, then
        ``see(n_rollout)`` and the mean of ``outcome`` -- and they differ only in how the members'
        verdicts are combined:

        - ``"thompson"`` draws one member and returns its argmax. Thompson sampling over
          structure (Ortega & Braun, `arXiv:1303.4431 <https://arxiv.org/abs/1303.4431>`_): the
          spread of the belief supplies the exploration, so the policy sharpens exactly as the
          I-MEC collapses and needs no separate schedule.
        - ``"average"`` takes the mean value across members -- marginalising the structure out.
        - ``"robust"`` takes the maximin: score each action by its *worst* member, then take the
          best of those. The one to reach for when a wrong action is expensive, since it is the
          only one of the three that never bets on a particular member being the true DAG.

        Deterministic given ``seed``. The thompson draw comes from an RNG derived from ``seed``,
        the transition count and the number of draws already taken, so a fixed seed replays the
        whole decision sequence, successive calls draw successive members rather than repeating
        one, and two seeds diverge. Rollouts are seeded per member, which also hands every action
        of a member the same exogenous draws (common random numbers), so the gap between two
        actions is not partly the gap between two samples. Ties go to the earliest action in
        ``actions``.

        **This costs rollouts:** ``n_rollout`` samples per action per member scored, so
        ``len(actions) * n_rollout`` under ``"thompson"`` and
        ``len(actions) * belief_size() * n_rollout`` under ``"average"`` and ``"robust"``, every
        call. That is what planning in a model costs instead of reading a value table.

        Raises ``ValueError`` before the first :meth:`refit`, when there is no belief to plan in.
        """
        belief = self._require_belief("act")
        if self.policy == "thompson":
            rng = np.random.default_rng((self.seed, self._steps, self._draws))
            self._draws += 1
            drawn = int(rng.integers(len(belief)))
            values = [
                self._action_value(belief[drawn], action, seed=self.seed + drawn)
                for action in self.actions
            ]
        else:
            by_member = [
                [
                    self._action_value(member, action, seed=self.seed + index)
                    for action in self.actions
                ]
                for index, member in enumerate(belief)
            ]
            across = list(zip(*by_member, strict=True))
            values = (
                [float(np.mean(column)) for column in across]
                if self.policy == "average"
                else [min(column) for column in across]
            )
        return self.actions[max(range(len(self.actions)), key=lambda index: values[index])]

    def probe(self) -> str:
        """Return the intervention target the belief's members disagree about most.

        A **target**, not an action: the answer to "what experiment should I run next", which is
        the other half of the loop :meth:`act` serves. Each candidate is scored by the mean
        pairwise total variation between the members' outcome distributions under ``do(candidate)``
        -- predictive disagreement standing in for expected information gain, as in Scherrer et
        al. (`arXiv:2109.02429 <https://arxiv.org/abs/2109.02429>`_) and Zhang et al. (Nature
        Machine Intelligence 2023). It concentrates experiments on the edges the I-MEC still leaves
        unoriented, because those are the only ones whose ``do()`` distribution the members can
        disagree about; a target every member already predicts identically scores zero and is
        excluded, and if *every* candidate scores zero this raises rather than returning an
        arbitrary one.

        Each candidate is intervened at one representative value: the largest it takes anywhere in
        the agent's buffers. That keeps the do-value inside the observed support, so no fitted
        mechanism is asked to extrapolate, while sitting as far as the data allows from the bottom
        of the range -- a do-value near the middle of a variable's spread perturbs its descendants
        least and makes members that disagree look alike. One value cannot certify agreement,
        though: a zero score means "no disagreement detected at this value", not "these members
        agree everywhere".

        **This costs rollouts:** ``n_rollout`` samples per candidate per member, i.e.
        ``len(variables) * belief_size() * n_rollout`` every call. All members share one exogenous
        stream per candidate (common random numbers), because the score is a difference *between*
        members and independent draws would put a finite-sample floor under it.

        Raises ``ValueError`` before the first :meth:`refit`, on a single-member belief (nothing to
        disambiguate -- :meth:`structure_uncertain` is the guard to branch on), and when no
        candidate scores any disagreement at all.
        """
        belief = self._require_belief("probe")
        if len(belief) == 1:
            raise ValueError(
                "probe() has nothing to disambiguate: the belief holds a single I-MEC member, so "
                "every candidate target has exactly one implied do() distribution and no "
                "experiment can separate anything. structure_uncertain() is the guard to branch "
                "on -- probe() while it reports True, act() once it reports False."
            )
        scores = {
            candidate: self._disagreement(candidate, belief) for candidate in sorted(self.variables)
        }
        best = max(scores, key=lambda candidate: scores[candidate])
        if scores[best] <= 0.0:
            raise ValueError(
                f"probe() found no target the belief's {len(belief)} members disagree about: every "
                f"candidate in {sorted(self.variables)} scored zero total variation between the "
                "members' outcome distributions. Intervening on any of them would be an experiment "
                "whose result every member already predicts identically, so returning one would "
                "report an arbitrary pick as an informative choice. The members differ somewhere "
                "else -- in a mechanism, or at a do-value other than the one probe() uses -- and "
                "that is not something this score can locate."
            )
        return best

    def _require_belief(self, caller: str) -> tuple[StructuralCausalModel, ...]:
        """The belief, or a ``ValueError`` naming ``refit`` when none has been fitted."""
        if not self._belief:
            raise ValueError(
                f"{caller}() needs a belief and none has been fitted yet: call refit() first. An "
                "empty belief is not a structure the agent is certain about, it is one it knows "
                "nothing about, and the documented loop branches on structure_uncertain() -- so "
                "answering from an empty belief would route a caller who never fitted anything "
                "straight into act(), planning in a model that does not exist."
            )
        return self._belief

    def _action_value(self, member: StructuralCausalModel, action: float, *, seed: int) -> float:
        """``member``'s mean ``outcome`` under ``do(treatment=action)``, over ``n_rollout`` rows."""
        return float(np.mean(self._rollout(member, {self.treatment: action}, seed=seed)))

    def _disagreement(self, candidate: str, belief: tuple[StructuralCausalModel, ...]) -> float:
        """Mean pairwise total variation of the members' ``outcome`` under ``do(candidate)``."""
        value = self._probe_value(candidate)
        seed = self.seed + self.variables.index(candidate)
        samples = [self._rollout(member, {candidate: value}, seed=seed) for member in belief]
        return float(np.mean([_total_variation(a, b) for a, b in combinations(samples, 2)]))

    def _probe_value(self, candidate: str) -> float:
        """The largest value ``candidate`` takes across every buffer -- see :meth:`probe`."""
        values = list(self._observational[candidate])
        for buffer in self._interventional.values():
            values.extend(buffer[candidate])
        if not values:
            raise ValueError(
                f"probe() has no buffered row to take a do-value for {candidate!r} from: it "
                "intervenes at the largest value each candidate is observed to take, so that the "
                "do-value stays inside the support the mechanisms were fitted on. Ingest data "
                "before probing."
            )
        return max(values)

    def _rollout(
        self, member: StructuralCausalModel, intervention: Mapping[str, float], *, seed: int
    ) -> np.ndarray:
        """``n_rollout`` draws of ``outcome`` from ``member`` under ``do(intervention)``."""
        sample = member.do(intervention).see(self.n_rollout, seed=seed)
        return np.asarray(sample[self.outcome].detach().numpy(), dtype=float)

    # -- what the agent believes --------------------------------------------------------------
    def belief(self) -> tuple[StructuralCausalModel, ...]:
        """The fitted I-MEC members -- empty until the first :meth:`refit`."""
        return self._belief

    def belief_size(self) -> int:
        """How many DAGs the data still cannot choose between."""
        return len(self._belief)

    def structure_uncertain(self) -> bool:
        """Whether more than one I-MEC member survives -- a decision may need the certificate layer.

        Raises ``ValueError`` before the first :meth:`refit` rather than answering. With an empty
        belief neither answer is honest: there is nothing to be uncertain *between* yet, and
        nothing that says the structure is known either. This is the guard the documented loop
        branches on -- :meth:`probe` while it is ``True``, :meth:`act` once it is ``False`` -- so
        reporting ``False`` from an empty belief would quietly route a caller who never fitted
        anything into the exploit branch.
        """
        return len(self._require_belief("structure_uncertain")) > 1

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


def _total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical total-variation distance between two samples, on a histogram they share.

    Deliberately not :func:`causalrl.discovery._total_variation`, which bins by ``int(value)``.
    That is exact for the integer-valued marginals ``discover_interventional``'s shift test
    compares, but the outcome of a fitted SCM is generally continuous and truncating toward zero
    would merge values that differ -- reporting agreement the members do not have. See
    :data:`_TV_BINS` for what the shared binning does and does not license.
    """
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=_TV_BINS)
    counts_a, _ = np.histogram(a, bins=edges)
    counts_b, _ = np.histogram(b, bins=edges)
    return 0.5 * float(np.abs(counts_a / len(a) - counts_b / len(b)).sum())
