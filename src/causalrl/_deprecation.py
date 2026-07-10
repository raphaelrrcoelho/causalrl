"""Deprecation helpers (invariant I9): the pre-2.0 certificate-default-flip ``FutureWarning``.

In causalrl 2.0 the shipped inferential routines that have a certificate-returning variant will
return a :class:`~causalrl.certify.certificate.Certificate` **by default**. That is a breaking
change, so a ``FutureWarning`` is introduced in a pre-2.0 minor (held ≥ 1 minor before the flip,
I9): a caller that has not opted out (``return_certificate`` unset) is told the return type will
change and how to pin it. The warning is a leaf utility so ``identification`` can emit it without
importing ``certify`` (no import cycle).
"""

from __future__ import annotations

import warnings

FLIP_VERSION = "2.0"


def warn_certificate_default_flip(routine: str, certified: str, *, stacklevel: int = 3) -> None:
    """Warn that ``routine`` will return a ``Certificate`` by default in causalrl ``FLIP_VERSION``.

    Emitted only when the caller left ``return_certificate`` unset. ``stacklevel=3`` points the
    warning at the user's call site (past the routine body and this helper).
    """
    warnings.warn(
        f"In causalrl {FLIP_VERSION}, {routine}() will return a Certificate by default. "
        f"Pass return_certificate=False to keep the current return type, or "
        f"return_certificate=True (equivalently, call {certified}()) to opt into it now. "
        f"This warning is removed at the {FLIP_VERSION} flip.",
        FutureWarning,
        stacklevel=stacklevel,
    )
