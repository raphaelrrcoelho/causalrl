"""Multi-scale neural causality: spikes, population signals, and certified abstraction.

This package is causalrl's front-end for electrophysiology. It closes the three gaps between the
library's identification machinery and what cortical recordings actually are:

1. **Time series, not i.i.d. draws.** :mod:`causalrl.neuro.timeseries` adds lagged (PCMCI-style)
   discovery with a contemporaneous FCI slice, so autocorrelation stops manufacturing edges and
   latent common input is reported as ``<->`` instead of a fabricated direction.
2. **Point processes and continuous signals, not discrete tables.**
   :mod:`causalrl.neuro.citests` supplies point-process GLM, partial-correlation and k-NN
   conditional-independence tests behind the shipped :class:`~causalrl.discovery.CITest`
   protocol, so PC and FCI run unchanged on spike counts and LFP.
3. **Two scales, not one.** :mod:`causalrl.neuro.abstraction` measures and certifies whether a
   mesoscopic model's ``do()`` agrees with the spiking network's interventional behaviour — the
   micro/meso question, posed as a causal-abstraction commutation check.

:mod:`causalrl.neuro.simulate` provides the ground truth all of this is validated against: a
recurrent spiking network with a known synaptic graph, known unrecorded common input, and true
interventional distributions. :mod:`causalrl.neuro.connectivity` turns a discovered graph into
per-edge sensitivity certificates, and :mod:`causalrl.neuro.io` bridges Neo/Elephant recordings in.

**Stability:** like :mod:`causalrl.experimental`, this package is not yet API-frozen; it is
outside causalrl's semver guarantees until promoted.
"""

from __future__ import annotations

from causalrl.neuro.abstraction import (
    AbstractionReport,
    AreaRateAbstraction,
    InterventionOutcome,
    MicroIntervention,
    SimulatedMicroSystem,
    abstraction_error,
    area_rates,
    certify_abstraction,
    default_interventions,
    mean_field_stability_margin,
)
from causalrl.neuro.citests import (
    CITestResult,
    KnnCMITest,
    PartialCorrelationTest,
    PoissonGLMTest,
)
from causalrl.neuro.connectivity import (
    CommonInputSensitivity,
    FunctionalConnectivity,
    certify_functional_edge,
    common_input_tipping_point,
    functional_connectivity,
    observed_shared_variance,
)
from causalrl.neuro.io import (
    ALLEN_QUALITY,
    DATASETS,
    DatasetSpec,
    DatasetUnavailableError,
    from_neo_block,
    from_nwb_ecephys,
    from_spike_trains,
    load_dataset,
)
from causalrl.neuro.recording import (
    MultiScaleRecording,
    RecordingError,
    bin_spike_times,
)
from causalrl.neuro.simulate import (
    CorticalNetworkSpec,
    MeanFieldAreaModel,
    SimulationError,
    SpikingCorticalSimulator,
    two_area_microcircuit,
)
from causalrl.neuro.stimulus import (
    EpochTable,
    contiguous_blocks,
    read_epochs,
    stimulus_regressors,
)
from causalrl.neuro.timeseries import (
    ConditionedCITest,
    LaggedGraph,
    LaggedLink,
    discover_lagged,
    lag_name,
    lagged_frame,
)

__all__ = [
    "ALLEN_QUALITY",
    "DATASETS",
    "AbstractionReport",
    "AreaRateAbstraction",
    "CITestResult",
    "CommonInputSensitivity",
    "ConditionedCITest",
    "CorticalNetworkSpec",
    "DatasetSpec",
    "DatasetUnavailableError",
    "EpochTable",
    "FunctionalConnectivity",
    "InterventionOutcome",
    "KnnCMITest",
    "LaggedGraph",
    "LaggedLink",
    "MeanFieldAreaModel",
    "MicroIntervention",
    "MultiScaleRecording",
    "PartialCorrelationTest",
    "PoissonGLMTest",
    "RecordingError",
    "SimulatedMicroSystem",
    "SimulationError",
    "SpikingCorticalSimulator",
    "abstraction_error",
    "area_rates",
    "bin_spike_times",
    "certify_abstraction",
    "certify_functional_edge",
    "common_input_tipping_point",
    "contiguous_blocks",
    "default_interventions",
    "discover_lagged",
    "from_neo_block",
    "from_nwb_ecephys",
    "from_spike_trains",
    "functional_connectivity",
    "lag_name",
    "lagged_frame",
    "load_dataset",
    "mean_field_stability_margin",
    "observed_shared_variance",
    "read_epochs",
    "stimulus_regressors",
    "two_area_microcircuit",
]
