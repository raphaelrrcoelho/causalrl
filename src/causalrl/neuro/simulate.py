"""Data-informed multi-scale cortical simulator with a known ground-truth causal graph.

Causal discovery on real cortical data has no answer key. This module supplies one: a recurrent
spiking network whose synaptic graph, latent common-input structure, and *interventional*
distributions are all known by construction, so any functional-connectivity or abstraction claim
can be scored against ground truth before it is trusted on experimental data.

**Micro (stochastic).** A discrete-time generalised-linear point process — the standard
statistical model for multi-electrode spike trains (Truccolo et al., *J. Neurophysiol.* 2005;
Pillow et al., *Nature* 2008). Each unit's conditional intensity is

    u_i(t) = b_i - rho * a_i(t) + sum_j W[i,j] h_j(t) + sum_m C[i,m] z_m(t)
    P(spike_i at t) = 1 - exp(-dt * exp(u_i(t)))

with ``h_j`` an exponentially filtered, delayed spike history of unit ``j`` (the synaptic kernel),
``a_i`` the unit's own refractory trace, and ``z_m`` **latent** oscillatory drives that are never
recorded. Those latent drives are the point: they induce exactly the *common input* that makes a
correlation-based functional graph report interactions that do not exist.

**Meso (deterministic + stochastic).** Two population observables per area: the population firing
rate, and an LFP proxy taken as the (sign-flipped, low-pass filtered) total synaptic input current
to the area — the standard forward proxy (Mazzoni et al., *PLoS Comput. Biol.* 2008; Einevoll
et al., *Nat. Rev. Neurosci.* 2013). The matched *deterministic* description is the mean-field rate
map :class:`MeanFieldAreaModel`, obtained by averaging the intensity over each area.

**Interventions.** :meth:`SpikingCorticalSimulator.simulate` takes ``do={unit: p}``, which clamps
that unit's spiking to an independent Bernoulli process and cuts its incoming edges — the point-
process analogue of optogenetic silencing (``p=0``) or driving (``p`` large). This yields the true
L2 distribution, which is what :mod:`causalrl.neuro.abstraction` scores a mesoscopic model against.

**Regimes.** ``coupling_gain`` and ``latent_gain`` move the network between the asynchronous-
irregular regime (weak recurrence, no population oscillation) and the synchronous-oscillatory
regime (strong shared drive). The two regimes are where the multi-scale question has different
answers, so both simulators and both certificates are exercised by the same spec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from causalrl.exceptions import CausalRLError
from causalrl.neuro.recording import MultiScaleRecording

__all__ = [
    "CorticalNetworkSpec",
    "MeanFieldAreaModel",
    "SimulationError",
    "SpikingCorticalSimulator",
    "two_area_microcircuit",
]

FloatArray = NDArray[np.float64]


class SimulationError(CausalRLError):
    """A simulator was given an inconsistent specification or an unknown intervention target."""


@dataclass(frozen=True)
class CorticalNetworkSpec:
    """Ground-truth specification of a recurrent cortical microcircuit.

    ``weights[i, j]`` is the synaptic weight of the connection ``j -> i`` (row = target), in units
    of log-intensity per unit of filtered presynaptic activity. ``latent_loadings[i, m]`` is the
    gain of unrecorded common input ``m`` onto unit ``i``.
    """

    unit_names: tuple[str, ...]
    unit_area: Mapping[str, str]
    weights: FloatArray  # (n_units, n_units); weights[i, j] is j -> i
    latent_loadings: FloatArray  # (n_units, n_latent)
    baseline: FloatArray  # (n_units,) log-intensity baseline
    bin_size: float = 0.005  # seconds
    synaptic_delay: int = 1  # bins before a spike reaches its targets
    synaptic_tau: float = 3.0  # bins; exponential decay of the synaptic kernel
    refractory: float = 1.5  # log-intensity penalty per unit of own recent activity
    refractory_tau: float = 1.0  # bins; ~one bin of relative refractoriness
    latent_freqs: FloatArray = field(default_factory=lambda: np.zeros(0))  # Hz, one per latent
    latent_ou_tau: float = 8.0  # bins; timescale of the latent's stochastic component
    latent_noise: float = 0.4  # relative weight of the stochastic vs oscillatory latent component
    max_log_intensity: float = 4.0  # clip, keeps the Bernoulli probability well-defined

    def __post_init__(self) -> None:
        n = len(self.unit_names)
        if len(set(self.unit_names)) != n:
            raise SimulationError("unit_names must be unique")
        if self.weights.shape != (n, n):
            raise SimulationError(f"weights must have shape ({n}, {n})")
        if self.latent_loadings.ndim != 2 or self.latent_loadings.shape[0] != n:
            raise SimulationError(f"latent_loadings must have shape ({n}, n_latent)")
        if self.baseline.shape != (n,):
            raise SimulationError(f"baseline must have shape ({n},)")
        if self.latent_freqs.shape != (self.n_latent,):
            raise SimulationError(f"latent_freqs must have shape ({self.n_latent},)")
        if self.bin_size <= 0.0:
            raise SimulationError("bin_size must be positive")
        if self.synaptic_delay < 1:
            raise SimulationError("synaptic_delay must be at least one bin")
        missing = set(self.unit_names) - set(self.unit_area)
        if missing:
            raise SimulationError(f"unit_area is missing units: {sorted(missing)}")

    @property
    def n_units(self) -> int:
        return len(self.unit_names)

    @property
    def n_latent(self) -> int:
        return int(self.latent_loadings.shape[1])

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.unit_area.values())))

    def synaptic_gain(self) -> float:
        """Total mass of the synaptic kernel, ``sum_tau k(tau)`` — the DC gain of one synapse."""
        return float(1.0 / (1.0 - np.exp(-1.0 / self.synaptic_tau)))

    def refractory_gain(self) -> float:
        """DC gain of the refractory trace — the self-inhibition a unit applies per spike."""
        return float(1.0 / (1.0 - np.exp(-1.0 / self.refractory_tau)))

    def ground_truth_edges(self) -> list[tuple[str, str]]:
        """Directed synaptic edges ``(source, target)``, self-connections excluded."""
        return [
            (self.unit_names[j], self.unit_names[i])
            for i in range(self.n_units)
            for j in range(self.n_units)
            if i != j and self.weights[i, j] != 0.0
        ]

    def ground_truth_confounded_pairs(self) -> list[tuple[str, str]]:
        """Unordered unit pairs sharing at least one latent common input."""
        pairs: list[tuple[str, str]] = []
        loaded = np.abs(self.latent_loadings) > 0.0
        for i in range(self.n_units):
            for j in range(i + 1, self.n_units):
                if bool(np.any(loaded[i] & loaded[j])):
                    pairs.append((self.unit_names[i], self.unit_names[j]))
        return pairs

    def ground_truth_summary_graph(self) -> object:
        """The true summary graph as a ``CyclicCausalGraph`` (directed cycles + latent pairs).

        Imported lazily: :mod:`causalrl.experimental` is outside the stable API, and
        :mod:`causalrl.neuro` must not depend on it at import time.
        """
        from causalrl.experimental.cyclic import CyclicCausalGraph

        return CyclicCausalGraph(
            self.ground_truth_edges(),
            self.ground_truth_confounded_pairs(),
            nodes=list(self.unit_names),
        )

    def area_matrix(self) -> tuple[tuple[str, ...], FloatArray]:
        """``(areas, M)`` with ``M[a, i] = 1/|a|`` when unit ``i`` is in area ``a``.

        This is the canonical micro→meso aggregation map ``tau``: an area's activity is the mean
        activity of its units.
        """
        areas = self.areas
        m = np.zeros((len(areas), self.n_units), dtype=np.float64)
        for a, area in enumerate(areas):
            idx = [i for i, u in enumerate(self.unit_names) if self.unit_area[u] == area]
            if not idx:
                raise SimulationError(f"area {area!r} has no units")
            m[a, idx] = 1.0 / len(idx)
        return areas, m


def two_area_microcircuit(
    *,
    n_per_area: int = 8,
    areas: Sequence[str] = ("M1", "PMd"),
    coupling_gain: float = 1.0,
    latent_gain: float = 1.0,
    n_latent: int = 2,
    oscillation_hz: float = 20.0,
    inhibitory_fraction: float = 0.25,
    connection_density: float = 0.2,
    baseline_rate_hz: float = 12.0,
    bin_size: float = 0.005,
    target_loop_gain: float | None = None,
    seed: int = 0,
) -> CorticalNetworkSpec:
    """A two-area microcircuit with sparse Dale-respecting recurrence and latent beta drive.

    ``coupling_gain`` scales the synaptic weights (recurrence strength) and ``latent_gain`` the
    unrecorded common input; together they select the regime. ``oscillation_hz`` defaults to the
    beta band, the dominant rhythm in motor-cortical Utah-array recordings.

    ``target_loop_gain`` overrides ``coupling_gain`` by rescaling every synapse until the
    *mesoscopic* loop gain — the largest real eigenvalue of the mean-field Jacobian — hits the
    requested value. This is the knob that actually selects the regime, because it fixes the
    quantity stability depends on: the mean dynamics are stable iff the loop gain is below 1, so
    ``0.4`` is comfortably asynchronous and ``1.2`` is unstable. Setting it directly keeps
    individual synapses physiologically modest while still reaching either regime.

    Each latent source projects to a random subset spanning **both** areas, so the resulting
    confounding is not removable by restricting the analysis to one area — the situation that makes
    naive inter-area functional connectivity untrustworthy.
    """
    rng = np.random.default_rng(seed)
    unit_names: list[str] = []
    unit_area: dict[str, str] = {}
    for area in areas:
        for k in range(n_per_area):
            name = f"{area}_{k}"
            unit_names.append(name)
            unit_area[name] = area
    n = len(unit_names)
    if n == 0:
        raise SimulationError("microcircuit needs at least one unit")

    inhibitory = rng.random(n) < inhibitory_fraction
    mask = rng.random((n, n)) < connection_density
    np.fill_diagonal(mask, False)
    magnitude = rng.gamma(shape=2.0, scale=0.09, size=(n, n))
    sign = np.where(inhibitory, -1.6, 1.0)[np.newaxis, :]  # column j: presynaptic sign (Dale)
    weights = coupling_gain * mask * magnitude * sign

    loadings = np.zeros((n, max(n_latent, 0)), dtype=np.float64)
    for m in range(n_latent):
        # Each latent drives ~half the units, deliberately spanning both areas.
        targets = rng.random(n) < 0.5
        if not targets.any():
            targets[rng.integers(n)] = True
        loadings[targets, m] = latent_gain * rng.uniform(0.35, 0.75, size=int(targets.sum()))

    # Baseline chosen so an isolated unit fires at ~baseline_rate_hz.
    p_target = np.clip(baseline_rate_hz * bin_size, 1e-4, 0.4)
    baseline = np.full(n, float(np.log(-np.log1p(-p_target) / bin_size)))
    freqs = np.full(max(n_latent, 0), float(oscillation_hz))
    if n_latent > 1:
        # Slight detuning so the latents are not perfectly collinear.
        freqs = freqs * np.linspace(0.85, 1.15, n_latent)

    spec = CorticalNetworkSpec(
        unit_names=tuple(unit_names),
        unit_area=unit_area,
        weights=weights,
        latent_loadings=loadings,
        baseline=baseline,
        bin_size=bin_size,
        latent_freqs=freqs,
    )
    if target_loop_gain is None:
        return spec
    return _rescale_to_loop_gain(spec, float(target_loop_gain))


def _loop_gain(spec: CorticalNetworkSpec) -> float:
    """Largest real eigenvalue of the mean-field Jacobian — the mesoscopic loop gain."""
    return float(np.max(np.linalg.eigvals(MeanFieldAreaModel(spec).jacobian()).real))


def _rescale_to_loop_gain(
    spec: CorticalNetworkSpec, target: float, *, tol: float = 1e-4, max_iter: int = 200
) -> CorticalNetworkSpec:
    """Scale every synapse until the mesoscopic loop gain equals ``target``.

    The gain is **not** monotone in the scale: past the point where the intensity saturates, the
    slope of the transfer function collapses and the gain falls again, so a large-weight solution
    exists for most targets. That branch is physiologically meaningless — every unit is pinned near
    its maximum rate — so the search scans upward from weak coupling and takes the **first**
    crossing, keeping synapses on the low-weight branch.

    Raises when the target is unreachable on that branch — most often an inhibition-dominated
    circuit asked for a gain no amount of rescaling can produce.
    """
    import dataclasses

    def gain(scale: float) -> float:
        return _loop_gain(dataclasses.replace(spec, weights=spec.weights * scale))

    grid = np.geomspace(1e-3, 1e2, 160)
    gains = [gain(float(s)) for s in grid]
    bracket: tuple[float, float] | None = None
    for k in range(len(grid) - 1):
        if (gains[k] - target) * (gains[k + 1] - target) <= 0.0:
            bracket = (float(grid[k]), float(grid[k + 1]))
            break
    if bracket is None:
        raise SimulationError(
            f"loop gain {target} is unreachable by rescaling on the physiological branch: the "
            f"attainable range is [{min(gains):.3g}, {max(gains):.3g}]. An inhibition-dominated "
            f"circuit cannot reach a high positive gain — lower inhibitory_fraction, raise "
            f"connection_density, or add units per area."
        )
    lo, hi = bracket
    increasing = gain(hi) > gain(lo)
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        g = gain(mid)
        if abs(g - target) < tol:
            lo = hi = mid
            break
        if (g < target) == increasing:
            lo = mid
        else:
            hi = mid
    return dataclasses.replace(spec, weights=spec.weights * math.sqrt(lo * hi))


class SpikingCorticalSimulator:
    """Stochastic micro-scale simulator for a :class:`CorticalNetworkSpec`.

    ``simulate`` returns a :class:`~causalrl.neuro.recording.MultiScaleRecording` carrying both the
    spikes and the derived mesoscopic signals, so a single call produces the multi-scale dataset the
    rest of the pipeline consumes.
    """

    def __init__(self, spec: CorticalNetworkSpec, *, seed: int = 0) -> None:
        self.spec = spec
        self.seed = seed

    def _latent_drive(self, n_bins: int, rng: np.random.Generator) -> FloatArray:
        """``(n_bins, n_latent)`` latent common input: an oscillation plus an OU-like component."""
        spec = self.spec
        if spec.n_latent == 0:
            return np.zeros((n_bins, 0), dtype=np.float64)
        t = np.arange(n_bins, dtype=np.float64) * spec.bin_size
        phase = rng.uniform(0.0, 2.0 * np.pi, size=spec.n_latent)
        oscillation = np.sin(2.0 * np.pi * spec.latent_freqs[np.newaxis, :] * t[:, np.newaxis]
                             + phase[np.newaxis, :])
        noise = rng.standard_normal((n_bins, spec.n_latent))
        decay = float(np.exp(-1.0 / spec.latent_ou_tau))
        ou = np.zeros_like(noise)
        scale = float(np.sqrt(1.0 - decay**2))
        for t_idx in range(1, n_bins):
            ou[t_idx] = decay * ou[t_idx - 1] + scale * noise[t_idx]
        w = float(np.clip(spec.latent_noise, 0.0, 1.0))
        return (1.0 - w) * oscillation + w * ou

    def simulate(
        self,
        n_bins: int,
        *,
        do: Mapping[str, float] | None = None,
        seed: int | None = None,
        burn_in: int = 200,
    ) -> MultiScaleRecording:
        """Simulate ``n_bins`` bins, optionally under ``do={unit: spike probability per bin}``.

        An intervened unit's incoming edges are cut and its spiking becomes an independent
        Bernoulli process — the point-process ``do`` operator. ``burn_in`` bins are simulated and
        discarded so the returned recording starts from the network's stationary regime.
        """
        spec = self.spec
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n = spec.n_units
        total = n_bins + max(burn_in, 0)

        clamped = np.zeros(n, dtype=bool)
        clamp_p = np.zeros(n, dtype=np.float64)
        for name, p in (do or {}).items():
            if name not in spec.unit_names:
                raise SimulationError(f"unknown intervention target: {name!r}")
            if not 0.0 <= float(p) <= 1.0:
                raise SimulationError(f"do({name!r}) must be a probability in [0, 1], got {p!r}")
            i = spec.unit_names.index(name)
            clamped[i] = True
            clamp_p[i] = float(p)

        latent = self._latent_drive(total, rng)
        latent_input = latent @ spec.latent_loadings.T  # (total, n_units)

        syn_decay = float(np.exp(-1.0 / spec.synaptic_tau))
        ref_decay = float(np.exp(-1.0 / spec.refractory_tau))
        delay = spec.synaptic_delay

        spikes = np.zeros((total, n), dtype=np.int64)
        recurrent = np.zeros((total, n), dtype=np.float64)  # synaptic input current per unit
        h = np.zeros(n, dtype=np.float64)  # filtered presynaptic activity
        a = np.zeros(n, dtype=np.float64)  # own refractory trace
        uniform = rng.random((total, n))

        for t in range(total):
            if t >= delay:
                h = syn_decay * h + spikes[t - delay].astype(np.float64)
            else:
                h = syn_decay * h
            drive = spec.weights @ h  # (n,), weights[i, j] is j -> i
            recurrent[t] = drive
            u = spec.baseline - spec.refractory * a + drive + latent_input[t]
            p = -np.expm1(-spec.bin_size * np.exp(_soft_clip(u, spec.max_log_intensity)))
            p = np.where(clamped, clamp_p, p)
            fired = uniform[t] < p
            spikes[t] = fired.astype(np.int64)
            a = ref_decay * a + fired.astype(np.float64)

        sl = slice(max(burn_in, 0), total)
        spikes_out = spikes[sl]
        meso, meso_names = self._mesoscopic(
            spikes_out, recurrent[sl] + latent_input[sl], clamped=clamped
        )
        return MultiScaleRecording(
            spikes=spikes_out,
            unit_names=spec.unit_names,
            bin_size=spec.bin_size,
            meso=meso,
            meso_names=meso_names,
            unit_area=dict(spec.unit_area),
            metadata={
                "generator": "SpikingCorticalSimulator",
                "do": dict(do or {}),
                "seed": self.seed if seed is None else seed,
                "n_latent": spec.n_latent,
            },
        )

    def _mesoscopic(
        self, spikes: NDArray[np.int64], current: FloatArray, *, clamped: NDArray[np.bool_]
    ) -> tuple[FloatArray, tuple[str, ...]]:
        """Population rate and LFP proxy per area, both low-pass filtered on the shared time base.

        The LFP proxy is the sign-flipped mean synaptic input current of the area's units — the
        standard forward proxy. Clamped units contribute no synaptic input of their own, matching
        the mutilated mechanism.
        """
        spec = self.spec
        areas, agg = spec.area_matrix()
        rate = spikes.astype(np.float64) @ agg.T  # (n_bins, n_areas)
        masked_current = np.where(clamped[np.newaxis, :], 0.0, current)
        lfp = -(masked_current @ agg.T)
        smooth_tau = max(spec.synaptic_tau, 1.0)
        rate_s = _lowpass(rate, smooth_tau)
        lfp_s = _lowpass(lfp, smooth_tau)
        names = tuple(f"rate:{a}" for a in areas) + tuple(f"lfp:{a}" for a in areas)
        return np.concatenate([rate_s, lfp_s], axis=1), names


def _soft_clip(u: FloatArray, limit: float) -> FloatArray:
    """Smooth, monotone bound on the log-intensity: ``limit * tanh(u / limit)``.

    A hard ``np.clip`` bounds the rate but sets the derivative to exactly zero beyond the limit,
    which makes a saturated network report a mesoscopic loop gain of 0 — an artefact of the clip,
    not a property of the circuit. The smooth version agrees with the identity well below the limit
    and keeps the Jacobian meaningful everywhere.
    """
    return float(limit) * np.tanh(u / float(limit))


def _lowpass(x: FloatArray, tau: float) -> FloatArray:
    """Causal one-pole low-pass filter along axis 0 (the mesoscopic observation kernel)."""
    decay = float(np.exp(-1.0 / max(tau, 1e-9)))
    out = np.empty_like(x)
    acc = np.zeros(x.shape[1], dtype=np.float64)
    for t in range(x.shape[0]):
        acc = decay * acc + (1.0 - decay) * x[t]
        out[t] = acc
    return out


class MeanFieldAreaModel:
    """Deterministic mesoscopic model: the area-level mean-field rate map of a spec.

    Averaging the micro intensity over each area and replacing spike trains by their rates gives
    the fixed-point equation

        r = phi(mu(r) + sigma^2(r)/2),      phi(u) = (1 - exp(-dt * exp(u))) / dt

    where ``mu`` collects the baseline, the mean refractory self-inhibition, and the recurrent
    coupling ``B_area[a, b]`` (mean total weight from area ``b`` onto a unit of area ``a``, times
    the synaptic DC gain).

    The optional ``sigma²/2`` term is the **diffusion-approximation correction**: the intensity is
    ``exp(u)``, so replacing a fluctuating input by its mean underestimates the rate by
    ``exp(sigma²/2)`` (Jensen), with ``sigma²`` collecting the variance injected by the latent
    common drive and by finite-size fluctuation of the recurrent input.

    It is **off by default**, on the evidence: with the refractory self-inhibition modelled (which
    is the term that actually matters — omitting it biases the predicted rate upward by tens of
    percent), the plain mean field already tracks the spiking network to well under 1 Hz, and the
    first-order correction then overshoots by 1-2 Hz. Its Gaussian-input assumption is the likely
    culprit; the latent drive here is a bounded oscillation, not a Gaussian. Turn it on for
    regimes with genuinely large, near-Gaussian input fluctuation, and check it against the
    simulator before trusting it — which is what :mod:`causalrl.neuro.abstraction` is for.

    This is the "deterministic simulation" counterpart of the spiking model, and the *macro model*
    whose interventional predictions :mod:`causalrl.neuro.abstraction` scores against the micro
    simulator.
    """

    def __init__(self, spec: CorticalNetworkSpec, *, include_fluctuations: bool = False) -> None:
        self.spec = spec
        self.include_fluctuations = include_fluctuations
        areas, agg = spec.area_matrix()
        self.areas: tuple[str, ...] = areas
        sizes = np.array(
            [sum(1 for u in spec.unit_names if spec.unit_area[u] == a) for a in areas],
            dtype=np.float64,
        )
        # Mean total weight received by a unit of area a from all units of area b.
        indicator = (agg > 0.0).astype(np.float64)
        self.coupling: FloatArray = (agg @ spec.weights @ indicator.T) * spec.synaptic_gain()
        self.sizes: FloatArray = sizes
        self.baseline: FloatArray = agg @ spec.baseline
        # Self-inhibition: a unit's own refractory trace is part of the mean field, not noise.
        # Omitting it biases the predicted rate upward by tens of percent at cortical rates.
        self.self_inhibition: float = spec.refractory * spec.refractory_gain()
        # Mean latent loading per area; the latent's own mean is zero, so it enters only through
        # the (regime-dependent) effective gain, kept explicit for sensitivity analysis.
        self.latent_mean_loading: FloatArray = agg @ spec.latent_loadings
        # Variance the latent drive injects into a typical unit's log-intensity. The latent is
        # (1-w)*sin + w*OU with independent parts, so Var(z) = (1-w)²/2 + w².
        w = float(np.clip(spec.latent_noise, 0.0, 1.0))
        var_z = (1.0 - w) ** 2 / 2.0 + w**2
        self.latent_variance: FloatArray = agg @ (spec.latent_loadings**2).sum(axis=1) * var_z
        # Per-area, per-source-area recurrent variance coefficient: Var(h_j) = r_j*dt*sum_tau k²,
        # so Var(sum_j W_ij h_j) = (sum_j W_ij²) * r*dt*sum_tau k² under independence.
        kernel_energy = 1.0 / (1.0 - np.exp(-2.0 / spec.synaptic_tau))
        self.recurrent_variance_coef: FloatArray = (
            agg @ (spec.weights**2) @ indicator.T
        ) * kernel_energy

    def _phi(self, u: FloatArray) -> FloatArray:
        dt = self.spec.bin_size
        return -np.expm1(-dt * np.exp(_soft_clip(u, self.spec.max_log_intensity))) / dt

    def _drive(self, rates: FloatArray) -> FloatArray:
        """Mean log-intensity per area at ``rates`` (Hz): baseline, refractory, recurrence."""
        per_bin = rates * self.spec.bin_size
        return self.baseline - self.self_inhibition * per_bin + self.coupling @ per_bin

    def variance(self, rates: FloatArray) -> FloatArray:
        """Log-intensity variance per area: latent common drive plus recurrent fluctuation."""
        if not self.include_fluctuations:
            return np.zeros_like(rates)
        per_bin = rates * self.spec.bin_size
        return self.latent_variance + self.recurrent_variance_coef @ per_bin

    def _map(self, rates: FloatArray) -> FloatArray:
        """One step of the mesoscopic rate map, with the diffusion-approximation correction."""
        return self._phi(self._drive(rates) + 0.5 * self.variance(rates))

    def equilibrium(
        self,
        *,
        do: Mapping[str, float] | None = None,
        max_iter: int = 2000,
        tol: float = 1e-10,
        damping: float = 0.5,
    ) -> dict[str, float]:
        """Fixed-point area rates (spikes/s), optionally under ``do={area: rate}``.

        ``do`` pins an area's rate and cuts its incoming coupling — the equilibrium ``do()``
        operator on the mesoscopic model.

        Solved by damped Newton on ``map(r) - r = 0``, **not** by iterating the map. The two
        agree wherever the dynamics are stable, but part company exactly where this model is
        most interesting: when the loop gain exceeds 1 the fixed point still exists and is still
        the equilibrium ``do()`` answer, while iterating the map runs away from it. Reporting an
        iterate as "the equilibrium" there would confuse the two semantics whose agreement
        :mod:`causalrl.neuro.abstraction` exists to test. Falls back to damped iteration only
        if the Newton step is singular.
        """
        pinned: dict[int, float] = {}
        for name, value in (do or {}).items():
            if name not in self.areas:
                raise SimulationError(f"unknown macro intervention target: {name!r}")
            pinned[self.areas.index(name)] = float(value)
        n = len(self.areas)
        free = [i for i in range(n) if i not in pinned]
        base = self._phi(self.baseline)
        for idx, value in pinned.items():
            base[idx] = value
        if not free:
            return {a: float(v) for a, v in zip(self.areas, base, strict=True)}

        best: FloatArray | None = None
        best_norm = float("inf")
        for factor in (1.0, 2.0, 0.5, 4.0, 0.25, 8.0, 0.1, 16.0):
            start = base.copy()
            start[free] = np.maximum(base[free] * factor, 1e-6)
            r, norm = self._newton(start, pinned, free, tol=tol, max_iter=max_iter)
            if norm < best_norm:
                best, best_norm = r, norm
            if norm < tol:
                break
        if best is None or best_norm >= tol:
            # No root found from any start; fall back to damped iteration and return its iterate,
            # which is the honest answer when the map has no fixed point in the positive orthant.
            r = base.copy()
            for _ in range(max_iter):
                nxt = self._map(r)
                for idx, value in pinned.items():
                    nxt[idx] = value
                nxt = damping * nxt + (1.0 - damping) * r
                if float(np.max(np.abs(nxt - r))) < tol:
                    r = nxt
                    break
                r = nxt
            best = r
        return {a: float(v) for a, v in zip(self.areas, best, strict=True)}

    def equilibria(
        self,
        *,
        do: Mapping[str, float] | None = None,
        tol: float = 1e-8,
        max_iter: int = 200,
        distinct_hz: float = 0.5,
    ) -> list[dict[str, float]]:
        """All distinct fixed points found by multi-start Newton, low rate first.

        Multiplicity matters causally: when several equilibria exist, which one the network
        settles in depends on its history and basin, and *no* plain macro SCM can represent that
        selection (Blom, Bongers & Mooij, UAI 2019). A macro ``do()`` is then answering a question
        the model is not equipped to answer, and
        :func:`causalrl.neuro.abstraction.certify_abstraction` hedges accordingly.
        """
        pinned: dict[int, float] = {}
        for name, value in (do or {}).items():
            if name not in self.areas:
                raise SimulationError(f"unknown macro intervention target: {name!r}")
            pinned[self.areas.index(name)] = float(value)
        n = len(self.areas)
        free = [i for i in range(n) if i not in pinned]
        base = self._phi(self.baseline)
        for idx, value in pinned.items():
            base[idx] = value
        if not free:
            return [{a: float(v) for a, v in zip(self.areas, base, strict=True)}]
        roots: list[FloatArray] = []
        ceiling = 1.0 / self.spec.bin_size
        for factor in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 50.0, 200.0):
            start = base.copy()
            start[free] = np.clip(base[free] * factor, 1e-6, ceiling)
            r, norm = self._newton(start, pinned, free, tol=tol, max_iter=max_iter)
            if norm >= tol:
                continue
            if not any(float(np.max(np.abs(r - seen))) < distinct_hz for seen in roots):
                roots.append(r)
        roots.sort(key=lambda r: float(np.mean(r)))
        return [{a: float(v) for a, v in zip(self.areas, r, strict=True)} for r in roots]

    def _newton(
        self,
        start: FloatArray,
        pinned: Mapping[int, float],
        free: Sequence[int],
        *,
        tol: float,
        max_iter: int,
    ) -> tuple[FloatArray, float]:
        """Newton on ``map(r) - r = 0`` in log-rate coordinates (positivity is then automatic).

        Returns ``(rates, residual_norm)``; the caller decides whether that counts as converged.
        """
        idx = list(free)
        x: FloatArray = np.log(np.maximum(start[idx], 1e-9)).astype(np.float64)

        def rates_of(xv: FloatArray) -> FloatArray:
            r = start.copy()
            for k, value in pinned.items():
                r[k] = value
            r[idx] = np.exp(xv)
            return r

        def residual(xv: FloatArray) -> FloatArray:
            r = rates_of(xv)
            return (self._map(r) - r)[idx]

        for _ in range(max_iter):
            res = residual(x)
            norm = float(np.linalg.norm(res))
            if norm < tol:
                return rates_of(x), norm
            r = rates_of(x)
            # d(residual)/dx = (J - I) * diag(r), by the chain rule through r = exp(x).
            base = self.jacobian_at(r)[np.ix_(idx, idx)] - np.eye(len(idx))
            jac = base * r[idx][np.newaxis, :]
            try:
                step = np.linalg.solve(jac, -res)
            except np.linalg.LinAlgError:
                return rates_of(x), norm
            # At most e^2 per iteration, which keeps Newton from bolting.
            step = np.clip(step, -2.0, 2.0).astype(np.float64)
            scale = 1.0
            for _ in range(40):
                trial: FloatArray = (x + scale * step).astype(np.float64)
                if float(np.linalg.norm(residual(trial))) < norm:
                    x = trial
                    break
                scale *= 0.5
            else:
                return rates_of(x), norm
        res = residual(x)
        return rates_of(x), float(np.linalg.norm(res))

    def jacobian(self, rates: Mapping[str, float] | None = None) -> FloatArray:
        """Linearisation ``B`` of the mean-field map at ``rates`` (default: the free equilibrium).

        Feeding this to :class:`~causalrl.experimental.cyclic.LinearCyclicSCM` gives the
        mesoscopic *equilibrium* semantics; its ``stability_margin`` decides whether the
        mesoscopic equilibrium is the causally correct object (the T1 condition).

        Computed by central differences on :meth:`_map`, so the fluctuation correction's own
        rate-dependence is differentiated too rather than dropped.
        """
        eq = self.equilibrium() if rates is None else dict(rates)
        return self.jacobian_at(np.array([eq[a] for a in self.areas], dtype=np.float64))

    def jacobian_at(self, r: FloatArray) -> FloatArray:
        """Central-difference Jacobian of the mesoscopic map at the given rate vector."""
        n = r.shape[0]
        jac = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            step = max(1e-5, 1e-5 * abs(float(r[k])))
            up, dn = r.copy(), r.copy()
            up[k] += step
            dn[k] -= step
            jac[:, k] = (self._map(up) - self._map(dn)) / (2.0 * step)
        return jac
