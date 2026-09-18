"""The package's error hierarchy.

One base class, so a caller can catch everything this package raises with a
single `except` and still discriminate when it wants to. The alternative —
bare `ValueError` here, `KeyError` there — forces callers to either catch
`Exception` or to know the implementation.

`SurfaceFeaturesError` derives from `ValueError` so that existing handlers for
bad input keep working.
"""

from __future__ import annotations


class SurfaceFeaturesError(ValueError):
    """Anything this package refuses to do."""


class CorpusError(SurfaceFeaturesError):
    """The file exists but is not the corpus this experiment expects."""


class SpacyUnavailable(SurfaceFeaturesError):
    """spaCy, or the Spanish model, is not installed."""
