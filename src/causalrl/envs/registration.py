"""Gymnasium env registration for causalrl demo environments.

Call :func:`register_envs` (or simply ``import causalrl``, which calls it automatically)
to add all registrable demo environments to the Gymnasium registry so they can be
instantiated with :func:`gymnasium.make`.

Registered IDs
--------------
``causalrl/StructuralCausalBandit-v0``
    The confounded chain bandit (``U->X1->X2->X3->Y``, ``X1<->Y``).  Created via
    :func:`~causalrl.envs.suite.scbandit.make_confounded_chain_env` with default arguments
    (``n_mc=2000``).

``causalrl/FrontdoorBandit-v0``
    The front-door / cholesterol bandit (``X->Z->Y``, ``X<->Y``, ``Z`` non-manipulable).
    Created via :func:`~causalrl.envs.suite.scbandit.make_frontdoor_env`.

Notes
-----
The following environment classes are *not* registered because they require constructor
arguments that have no sensible universal default and cannot be described purely via
``kwargs`` in the registry:

* ``ConfoundedGridworld`` — requires at least ``size``.
* ``MABUCEnv`` — requires ``n_states``, ``n_actions``, ``horizon``.
* ``SequentialDTREnv``, ``SequentialMABUCEnv``, ``DTREnv``, ``CounterfactualBanditEnv`` —
  these work fine without args but their Gymnasium Discrete/Dict spaces are parameterised by
  class-level attributes; they *are* constructable, but including them here would create
  misleading IDs.  Use them directly: ``gymnasium.make`` is not the only way to instantiate.

Registration is idempotent: a module-level flag prevents double-registration warnings from
Gymnasium when the module is imported more than once in the same process.
"""

from __future__ import annotations

import gymnasium

# Module-level flag: registration is performed at most once per process.
# Lowercase to avoid pyright's reportConstantRedefinition on bool re-assignment.
_registered: bool = False


def register_envs() -> None:
    """Register causalrl demo environments in the Gymnasium registry.

    Calling this function more than once in the same process is safe (idempotent).

    After calling this function (or importing ``causalrl``), you can use::

        import gymnasium
        import causalrl  # triggers register_envs()

        env = gymnasium.make("causalrl/StructuralCausalBandit-v0")
        vec = gymnasium.make_vec("causalrl/StructuralCausalBandit-v0", num_envs=2)
    """
    global _registered
    if _registered:
        return

    # gymnasium.register has an incompletely typed `kwargs` dict parameter in the stubs;
    # suppress the resulting pyright reportUnknownMemberType at each call site.
    gymnasium.register(  # type: ignore[reportUnknownMemberType]
        id="causalrl/StructuralCausalBandit-v0",
        entry_point="causalrl.envs.suite.scbandit:make_confounded_chain_env",
        # disable_env_checker keeps CI fast; callers can re-enable via check_env.
        disable_env_checker=True,
    )

    gymnasium.register(  # type: ignore[reportUnknownMemberType]
        id="causalrl/FrontdoorBandit-v0",
        entry_point="causalrl.envs.suite.scbandit:make_frontdoor_env",
        disable_env_checker=True,
    )

    _registered = True
