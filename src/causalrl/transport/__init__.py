"""Transport (plan §4/§7.5): the shipped decision layer + the data-plane estimation layer.

Re-exports the torch-free transportability decision (``transport_formula`` / ``is_transportable`` /
``SelectionDiagram``) from :mod:`causalrl.identification.transport` and adds
``certify_transported_effect``, which estimates the transported mean from source/target data and
returns a unified :class:`~causalrl.certify.certificate.Certificate`. The SCM-based
``transported_effect`` stays in the shipped module.
"""

from causalrl.identification.transport import (
    SelectionDiagram,
    TransportFormula,
    is_transportable,
    transport_estimand,
    transport_formula,
)
from causalrl.transport.estimate import certify_transported_effect, transport_gcomp

__all__ = [
    "SelectionDiagram",
    "TransportFormula",
    "certify_transported_effect",
    "is_transportable",
    "transport_estimand",
    "transport_formula",
    "transport_gcomp",
]
