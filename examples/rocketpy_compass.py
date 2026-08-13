"""COMPASS on a rocket: learn which simulator parameter is wrong, then recover its value.

Reproduces the method of

    Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis,
    Bingqing Chen, Ding Zhao. "What Went Wrong? Closing the Sim-to-Real Gap via Differentiable
    Causal Discovery." Conference on Robot Learning (CoRL), 2023. arXiv:2306.15864.

**Scope, stated up front.** This is *not* a replication of the paper's numbers. COMPASS is evaluated
on robosuite/MuJoCo manipulation tasks (air hockey, drop, pose) and reproducing those results needs
their environments. This reproduces the **algorithm** on a different domain, which is the part worth
having here: a rocket whose simulator is wrong in a way we chose, so both halves of the method have
a ground truth to be scored against. Unlike the rest of ``examples/``, the authors' reference
implementation (MIT, https://github.com/XilunZhangRobo/COMPASS-Sim2Real) was consulted for the mask
parameterisation, loss and optimisation loop; causalrl's usual "from the paper only" rule does not
apply to this file and saying so is cheaper than implying otherwise.

**The problem it solves.** Every team knows their simulator is wrong and tunes it back to reality by
hand -- a drag multiplier here, a mass tweak there -- chosen so last flight matches. Which parameters
to tune is picked case by case and does not scale. COMPASS learns the mapping from environment
parameters to the *sim-to-real gap*, and learns a sparse causal mask over that mapping at the same
time, so the parameters that matter are discovered rather than nominated.

**The algorithm, as implemented here.**

1. Roll out the real system once to get a reference trajectory summary.
2. Domain-randomise the simulator's parameters and roll out, recording, per sample, the parameter
   vector and the resulting gap against the reference.
3. Fit :class:`CausalSim2Real`: a shared encoder over ``(value, index-embedding)`` pairs, a binary
   mask sampled with a hard Gumbel-softmax, and a shared decoder over the masked features. Its loss
   is MSE on the gap plus an L1-style sparsity penalty on the mask -- applied only to the parameter
   rows, never the action rows, because an action is not a candidate cause of a modelling error.
4. Freeze the model, threshold the mask at 0.5, and gradient-descend on the *parameters* to drive
   the predicted gap to zero. The simulator is never differentiated: the gradient flows through the
   learned model, which is what makes the method usable on a black-box simulator like RocketPy.

**What can be scored here that cannot be scored on a real robot.** We choose which parameters are
wrong, so the mask can be checked against the truth (did it find the miscalibrated ones and reject
the rest?) and the recovered values can be checked against the values we injected.

    pip install "causalrl[rocketpy,torch]"
    python examples/rocketpy_compass.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from rocketpy import Environment, Flight, SolidMotor
    from rocketpy import Rocket as RocketPyRocket
except ImportError as exc:  # pragma: no cover - the example is opt-in
    raise SystemExit(
        "This example needs RocketPy: pip install 'causalrl[rocketpy,torch]'\n"
        "RocketPy is an optional extra -- causalrl itself never imports it."
    ) from exc

ELEVATION = 1400.0
BURN_TIME = 3.0
BASE_THRUST = ((0.0, 0.0), (0.1, 1600.0), (2.6, 1500.0), (BURN_TIME, 0.0))
BRAKE_FLOOR = 700.0
DEPLOYMENTS = (0.0, 0.25, 0.5, 0.75)  # the "actions" the gap is measured across
SEED = 0

PARAMETERS = ("airframe_cd", "mass", "thrust_scale", "brake_cd", "wind")
# What the simulator currently believes.
NOMINAL = {"airframe_cd": 0.50, "mass": 14.0, "thrust_scale": 1.00, "brake_cd": 1.20, "wind": 8.0}
# What reality actually is. Two parameters are wrong and three are already right, which is the
# discrimination the mask has to make: finding the wrong ones is easy if you flag everything.
REAL = {"airframe_cd": 0.68, "mass": 14.0, "thrust_scale": 1.00, "brake_cd": 1.85, "wind": 8.0}
WRONG = tuple(p for p in PARAMETERS if abs(REAL[p] - NOMINAL[p]) > 1e-9)
# Randomisation half-widths, in the units of each parameter.
SPREAD = {"airframe_cd": 0.25, "mass": 3.0, "thrust_scale": 0.18, "brake_cd": 0.9, "wind": 6.0}

N_SAMPLES = 140
EPOCHS = 4000
SPARSITY_WEIGHT = 0.01
OPTIMIZE_STEPS = 2000


@dataclass(frozen=True)
class Vehicle:
    airframe_cd: float
    mass: float
    thrust_scale: float
    brake_cd: float
    wind: float


def vehicle_from(values: dict[str, float]) -> Vehicle:
    return Vehicle(**{name: float(values[name]) for name in PARAMETERS})


_environments: dict[float, Environment] = {}


def environment_for(wind: float) -> Environment:
    key = round(wind, 3)
    if key not in _environments:
        env = Environment(latitude=32.99, longitude=-106.97, elevation=ELEVATION)
        env.set_atmospheric_model(
            type="custom_atmosphere",
            wind_u=[(0, key), (4000, key * 1.6)],
            wind_v=[(0, 0.0), (4000, 0.0)],
        )
        _environments[key] = env
    return _environments[key]


def rollout(vehicle: Vehicle, deployment: float) -> np.ndarray:
    """Trajectory summary: apogee, speed at the brake floor, and time to apogee.

    A summary rather than the full trajectory the paper differences. The gap has to be a fixed-length
    vector for the decoder to predict, and three physically distinct quantities is enough to make the
    identification non-trivial while keeping a flight campaign affordable.
    """
    seen: dict[str, float] = {}
    thrust = [(t, f * vehicle.thrust_scale) for t, f in BASE_THRUST]
    motor = SolidMotor(
        thrust_source=thrust,
        dry_mass=1.6,
        dry_inertia=(0.12, 0.12, 0.002),
        nozzle_radius=0.033,
        grain_number=4,
        grain_density=1815,
        grain_outer_radius=0.033,
        grain_initial_inner_radius=0.015,
        grain_initial_height=0.12,
        grain_separation=0.005,
        grains_center_of_mass_position=0.0,
        center_of_dry_mass_position=0.0,
        nozzle_position=-0.25,
        burn_time=BURN_TIME,
        throat_radius=0.011,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = RocketPyRocket(
        radius=0.0635,
        mass=vehicle.mass,
        inertia=(6.3, 6.3, 0.034),
        power_off_drag=vehicle.airframe_cd,
        power_on_drag=vehicle.airframe_cd,
        center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.25)
    rocket.add_nose(length=0.55, kind="von karman", position=1.28)
    rocket.add_trapezoidal_fins(4, root_chord=0.12, tip_chord=0.06, span=0.11, position=-1.05)
    rocket.add_tail(top_radius=0.0635, bottom_radius=0.0435, length=0.06, position=-1.19)

    def controller(time_, sampling_rate, state, history, observed, interactive, *rest):
        brakes = interactive[0] if isinstance(interactive, (list, tuple)) else interactive
        altitude, speed = state[2] - ELEVATION, state[5]
        active = altitude > BRAKE_FLOOR and speed > 0
        brakes.deployment_level = deployment if active else 0.0
        if active:
            seen.setdefault("speed", float(speed))
        return (time_, brakes.deployment_level)

    rocket.add_air_brakes(
        drag_coefficient_curve=lambda level, mach: vehicle.brake_cd * level,
        controller_function=controller,
        sampling_rate=10,
        clamp=True,
        reference_area=None,
        override_rocket_drag=False,
    )
    flight = Flight(
        rocket=rocket,
        environment=environment_for(vehicle.wind),
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )
    return np.array(
        [
            float(flight.apogee) - ELEVATION,
            seen.get("speed", 0.0),
            float(flight.apogee_time),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSim2Real(nn.Module):
    """Shared encoder/decoder over indexed inputs, with a Gumbel-sampled binary mask between them.

    The mask is the causal graph: ``mask[i, j] = 1`` says input ``i`` is allowed to influence gap
    component ``j``. It is *learned jointly with* the regression rather than fixed in advance, which
    is the paper's central move -- the alternative is nominating tunable parameters by hand, which is
    the practice COMPASS exists to replace.

    ``n_actions`` trailing input rows are excluded from the sparsity penalty. An action is a thing
    the experiment varied, not a candidate explanation for the simulator being wrong, so penalising
    its edges would trade fit against a hypothesis nobody is testing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        n_actions: int,
        emb_dim: int = 10,
        causal_dim: int = 32,
        hidden: int = 64,
        sparse_weight: float = SPARSITY_WEIGHT,
    ) -> None:
        super().__init__()
        self.input_dim, self.output_dim, self.n_actions = input_dim, output_dim, n_actions
        self.sparse_weight = sparse_weight
        self.encoder = MLP(emb_dim + 1, causal_dim, hidden)
        self.encoder_emb = nn.Embedding(input_dim, emb_dim)
        self.decoder = MLP(causal_dim + emb_dim, 1, hidden)
        self.decoder_emb = nn.Embedding(output_dim, emb_dim)
        self.mask_logits = nn.Parameter(3.0 * torch.ones(input_dim, output_dim))

    def sample_mask(self, threshold: float | None = None) -> torch.Tensor:
        probability = torch.sigmoid(self.mask_logits)
        if threshold is not None:
            return (probability > threshold).float()
        stacked = torch.stack([1.0 - probability, probability], dim=-1).clamp_min(1e-9).log()
        return F.gumbel_softmax(stacked, tau=1.0, hard=True, dim=-1)[..., 1]

    def forward(self, inputs: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        index = torch.arange(self.input_dim)
        encoded = self.encoder(
            torch.cat(
                [
                    inputs.unsqueeze(-1),
                    self.encoder_emb(index).expand(inputs.shape[0], -1, -1),
                ],
                dim=-1,
            )
        )
        masked = torch.einsum("bnc,ns->bsc", encoded, self.sample_mask(threshold))
        out_index = torch.arange(self.output_dim)
        decoded = torch.cat(
            [masked, self.decoder_emb(out_index).expand(inputs.shape[0], -1, -1)], dim=-1
        )
        return self.decoder(decoded).squeeze(-1)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
        mse = F.mse_loss(predicted, target)
        sparsity = torch.sigmoid(self.mask_logits[: self.input_dim - self.n_actions, :]).mean()
        return mse + self.sparse_weight * sparsity, float(mse)


def collect(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Domain-randomise the simulator and record (parameters, action, gap-against-reality)."""
    real = vehicle_from(REAL)
    reference = {d: rollout(real, d) for d in DEPLOYMENTS}

    parameters, actions, gaps = [], [], []
    for _ in range(N_SAMPLES):
        drawn = {
            name: NOMINAL[name] + float(rng.uniform(-SPREAD[name], SPREAD[name]))
            for name in PARAMETERS
        }
        drawn["airframe_cd"] = max(0.1, drawn["airframe_cd"])
        drawn["mass"] = max(6.0, drawn["mass"])
        drawn["thrust_scale"] = max(0.5, drawn["thrust_scale"])
        drawn["brake_cd"] = max(0.1, drawn["brake_cd"])
        drawn["wind"] = max(0.0, drawn["wind"])
        vehicle = vehicle_from(drawn)
        deployment = float(rng.choice(DEPLOYMENTS))
        parameters.append([drawn[name] for name in PARAMETERS])
        actions.append([deployment])
        gaps.append(rollout(vehicle, deployment) - reference[deployment])
    return np.array(parameters), np.array(actions), np.array(gaps)


def standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (x - mean) / scale, mean, scale


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    print("Reality differs from the simulator in a way we chose, so both halves can be scored.")
    for name in PARAMETERS:
        mark = "  <- WRONG" if name in WRONG else ""
        print(f"   {name:14s} sim {NOMINAL[name]:7.3f}   real {REAL[name]:7.3f}{mark}")
    print(f"\n1. Domain-randomise and fly {N_SAMPLES} simulated rollouts against reality.")
    parameters, actions, gaps = collect(rng)
    print(
        f"   collected in {time.perf_counter() - started:.0f}s; "
        f"gap magnitude {np.abs(gaps).mean(axis=0).round(1)}"
    )

    scaled_parameters, mean, scale = standardise(parameters)
    scaled_gaps, gap_mean, gap_scale = standardise(gaps)
    inputs = torch.tensor(np.hstack([scaled_parameters, actions]), dtype=torch.float32)
    targets = torch.tensor(scaled_gaps, dtype=torch.float32)

    model = CausalSim2Real(inputs.shape[1], targets.shape[1], n_actions=actions.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"\n2. Fit the gap model and its mask jointly ({EPOCHS} epochs).")
    for epoch in range(EPOCHS):
        optimiser.zero_grad()
        loss, mse = model.loss(model(inputs), targets)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % 1000 == 0:
            print(f"   epoch {epoch + 1:5d}  mse {mse:.4f}")

    with torch.no_grad():
        kept = model.sample_mask(threshold=0.5)[: len(PARAMETERS)].sum(dim=1)
    print("\n3. What the mask kept -- the discovered causes of the gap:")
    selected = []
    for index, name in enumerate(PARAMETERS):
        edges = int(kept[index].item())
        verdict = "KEPT" if edges > 0 else "pruned"
        truth = "(really wrong)" if name in WRONG else "(really fine)"
        if edges > 0:
            selected.append(name)
        print(f"   {name:14s} {verdict:7s} {edges}/{targets.shape[1]} edges  {truth}")
    hits = sorted(set(selected) & set(WRONG))
    false_positives = sorted(set(selected) - set(WRONG))
    print(f"   found {hits}; false positives {false_positives}")

    print(
        f"\n4. Recover the real parameter values by descent through the frozen model "
        f"({OPTIMIZE_STEPS} steps)."
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    guess = torch.tensor(
        ((np.array([NOMINAL[n] for n in PARAMETERS]) - mean) / scale), dtype=torch.float32
    ).requires_grad_(True)
    inner = torch.optim.Adam([guess], lr=5e-3)
    action_grid = torch.tensor([[d] for d in DEPLOYMENTS], dtype=torch.float32)
    zero = torch.zeros(len(DEPLOYMENTS), targets.shape[1])
    for _ in range(OPTIMIZE_STEPS):
        inner.zero_grad()
        batch = torch.cat([guess.expand(len(DEPLOYMENTS), -1), action_grid], dim=1)
        # Drive the predicted gap to zero: the parameters that make the simulator match reality.
        objective = F.mse_loss(model(batch, threshold=0.5), zero)
        objective.backward()
        inner.step()

    recovered = guess.detach().numpy() * scale + mean
    print(f"   {'parameter':14s} {'sim':>8s} {'recovered':>10s} {'real':>8s} {'closed':>8s}")
    for index, name in enumerate(PARAMETERS):
        before = abs(NOMINAL[name] - REAL[name])
        after = abs(float(recovered[index]) - REAL[name])
        closed = "-" if before < 1e-9 else f"{100 * (1 - after / before):5.0f}%"
        print(
            f"   {name:14s} {NOMINAL[name]:8.3f} {recovered[index]:10.3f} "
            f"{REAL[name]:8.3f} {closed:>8s}"
        )
    print(f"\nTotal wall clock {time.perf_counter() - started:.0f}s.")


if __name__ == "__main__":
    main()
