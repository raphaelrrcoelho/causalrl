"""Scale path: certify a policy from an external offline-RL trainer against hidden confounding.

causalrl supplies the causal layer, not a new trainer. Train a policy however you like (e.g. with
``d3rlpy`` — see :func:`causalrl.scale.d3rlpy.to_mdp_dataset`), then hand its chosen actions to
:func:`causalrl.ope.certify.certify_policy` to bound whether its value improvement over the
logging/behaviour policy survives hidden confounding, and — with ``alpha`` — whether it clears the
finite-sample downside gate of :func:`causalrl.conformal_action_value`.

``certify_policy`` itself lives in :mod:`causalrl.ope.certify` with the rest of off-policy
evaluation and is re-exported here so the ``causalrl.scale`` story — train elsewhere, certify
here — still reads end to end.
"""

from __future__ import annotations

from causalrl.ope.certify import certify_policy

__all__ = ["certify_policy"]
