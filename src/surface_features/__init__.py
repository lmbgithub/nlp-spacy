"""Surface linguistic features for hostility detection in Spanish comments.

The package is deliberately split so that the measurement code never imports
spaCy: `tokens` defines the minimal token protocol the feature extractor needs,
`features` and `stats` operate on that protocol alone, and `spacy_backend` is
the only module that knows a real NLP pipeline exists. That boundary is what
makes the whole feature set testable against hand-built tokens with known-exact
expected values.
"""

from __future__ import annotations

from surface_features.errors import CorpusError, SpacyUnavailable, SurfaceFeaturesError
from surface_features.features import (
    FEATURE_NAMES,
    SurfaceFeatures,
    extract_features,
)
from surface_features.stats import cohens_d, effect_table, magnitude
from surface_features.tokens import Token

__all__ = [
    "FEATURE_NAMES",
    "CorpusError",
    "SpacyUnavailable",
    "SurfaceFeatures",
    "SurfaceFeaturesError",
    "Token",
    "cohens_d",
    "effect_table",
    "extract_features",
    "magnitude",
]
