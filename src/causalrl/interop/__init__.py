"""Adapters from third-party causal-inference estimates to causalrl's PolicyValueContrast seam.

Each adapter duck-types (or lazily imports) its third-party library inside the function, so
importing ``causalrl`` — or this subpackage — never requires DoWhy / EconML. Install them with the
``causalrl[interop]`` extra to run the worked examples.
"""
