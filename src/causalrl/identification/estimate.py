"""PolicyValueContrast: the typed seam the decision certificate consumes.

Any estimator that can name, per logged unit i, the reward ``Y_i``, the nominal logging propensity
``e0(a_i | x_i)``, and two target policies' action probabilities ``pi_on(a_i | x_i)``,
``pi_off(a_i | x_i)`` at the logged action, can be certified against hidden confounding by packing
those into a :class:`PolicyValueContrast` and calling :func:`causalrl.certify_estimate`. The
marginal-sensitivity-model layer is applied to the logging propensities; the point contrast is the
self-normalised IPS difference at ``Gamma = 1``. A binary-arm reduction (``treated`` +
``confounder_bins`` / ``mi_cap``) additionally enables the structural pivotality layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyValueContrast:
    """An off-policy value contrast ``V(pi_on) - V(pi_off)`` in the terms certify_estimate needs.

    Supply MSM evidence (``logging_propensities`` + ``target_on`` + ``target_off``) and/or
    structural evidence (``treated`` + ``confounder_bins`` or ``mi_cap``); at least one layer
    must be runnable. See :func:`causalrl.certify_estimate`.
    """

    outcomes: Sequence[float]
    logging_propensities: Sequence[float] | None = None
    target_on: Sequence[float] | None = None
    target_off: Sequence[float] | None = None
    treated: Sequence[int] | None = None
    confounder_bins: Sequence[int] | None = None
    mi_cap: float | None = None

    def __post_init__(self) -> None:
        y = np.asarray(self.outcomes, dtype=float)
        if y.size == 0:
            raise ValueError("outcomes must be non-empty")
        n = int(y.size)
        if self.logging_propensities is not None:
            if self.target_on is None or self.target_off is None:
                raise ValueError("logging_propensities requires target_on and target_off")
            e0 = np.asarray(self.logging_propensities, dtype=float)
            on = np.asarray(self.target_on, dtype=float)
            off = np.asarray(self.target_off, dtype=float)
            if not (e0.size == on.size == off.size == n):
                raise ValueError(
                    "outcomes, logging_propensities, target_on, target_off must be equal length"
                )
            if not bool(np.all((e0 > 0.0) & (e0 <= 1.0))):
                raise ValueError("logging_propensities must lie in (0, 1]")
        if self.treated is not None:
            f = np.asarray(self.treated).astype(bool)
            if f.size != n:
                raise ValueError("treated must match outcomes length")
            if not (f.any() and (~f).any()):
                raise ValueError("both arms must be present in `treated`")
        if not (self.has_msm or self.has_pivotality):
            raise ValueError(
                "supply at least one evidence source: logging_propensities (+ target_on/off) for "
                "the MSM layer, or treated + confounder_bins / mi_cap for the pivotality layer"
            )

    @property
    def has_msm(self) -> bool:
        """Whether the MSM sensitivity layer can run (logging propensities supplied)."""
        return self.logging_propensities is not None

    @property
    def has_pivotality(self) -> bool:
        """Whether the structural sign-robustness layer can run (binary arms + a channel bound)."""
        return self.treated is not None and (
            self.confounder_bins is not None or self.mi_cap is not None
        )

    @classmethod
    def from_binary(
        cls,
        outcomes: Sequence[float],
        treated: Sequence[int],
        *,
        propensities: Sequence[float] | None = None,
        confounder_bins: Sequence[int] | None = None,
        mi_cap: float | None = None,
    ) -> PolicyValueContrast:
        """The raw-logs case: one-hot arms ``pi_on = 1{treated}``, ``pi_off = 1{control}``.

        Exactly the contrast :func:`causalrl.certify_decision` builds internally. The MSM layer runs
        only when ``propensities`` are given; the pivotality layer runs when ``confounder_bins`` or
        ``mi_cap`` is given.
        """
        f = np.asarray(treated).astype(float)
        on = f.tolist() if propensities is not None else None
        off = (1.0 - f).tolist() if propensities is not None else None
        return cls(
            outcomes=outcomes,
            logging_propensities=propensities,
            target_on=on,
            target_off=off,
            treated=list(treated),
            confounder_bins=confounder_bins,
            mi_cap=mi_cap,
        )
