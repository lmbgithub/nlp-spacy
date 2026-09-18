"""Feature extraction: one parsed comment in, one row of ratios out."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from surface_features.tokens import EMOTIONAL_PUNCTUATION, Token

# The columns a classifier is allowed to see. `n_words` is deliberately absent:
# it is reported for diagnostics but feeding raw length to the model would let
# it learn "long comments are hostile", which is the confound the ratios exist
# to remove.
FEATURE_NAMES: tuple[str, ...] = (
    "verbs",
    "adjectives",
    "uppercase",
    "emotional_punctuation",
)

#: Every numeric column produced, including the diagnostic length columns.
ALL_COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, "n_words", "n_tokens")


@dataclass(frozen=True, slots=True)
class SurfaceFeatures:
    """Surface features for a single comment.

    The four modelled features are ratios over *word* tokens (punctuation
    excluded). Counts would measure comment length instead: hostile comments
    could simply be longer, and a raw adjective count would then rank them
    higher for a reason that has nothing to do with hostility.
    """

    verbs: float
    adjectives: float
    uppercase: float
    emotional_punctuation: float
    n_words: int
    n_tokens: int

    def as_vector(self) -> list[float]:
        """The modelled features only, in `FEATURE_NAMES` order."""
        return [getattr(self, name) for name in FEATURE_NAMES]


def extract_features(tokens: Iterable[Token]) -> SurfaceFeatures | None:
    """Compute surface features, or `None` for a comment with no word tokens.

    Returning `None` rather than a row of zeros is the important choice. A
    comment that is nothing but emoji or punctuation has no denominator; giving
    it zeros would not mark it as missing, it would assert that it contains no
    verbs and no adjectives, and thousands of such rows drag every group mean
    toward zero. The caller drops them and reports how many were dropped.
    """

    tokens = list(tokens)
    words: Sequence[Token] = [t for t in tokens if not t.is_punct]
    n_words = len(words)
    if n_words == 0:
        return None

    return SurfaceFeatures(
        verbs=_ratio(words, lambda t: t.pos_ == "VERB", n_words),
        adjectives=_ratio(words, lambda t: t.pos_ == "ADJ", n_words),
        # Shouting: an all-caps alphabetic word. The `is_alpha` guard keeps
        # acronyms-by-accident like "10K" and bare numerals out of the count.
        uppercase=_ratio(
            words, lambda t: t.is_alpha and t.text.isupper() and len(t.text) > 1, n_words
        ),
        # Affective punctuation is counted over *all* tokens (it was filtered
        # out of `words`) but normalised by word count, so the feature reads as
        # "marks per word" and stays comparable across comment lengths.
        emotional_punctuation=_ratio(
            tokens, lambda t: t.is_punct and _is_emotional(t.text), n_words
        ),
        n_words=n_words,
        n_tokens=len(tokens),
    )


def extract_many(
    documents: Iterable[Iterable[Token]],
) -> tuple[list[SurfaceFeatures], int]:
    """Extract features for a corpus, returning the rows and the dropped count."""

    rows: list[SurfaceFeatures] = []
    dropped = 0
    for doc in documents:
        row = extract_features(doc)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    return rows, dropped


def _ratio(tokens: Iterable[Token], predicate, denominator: int) -> float:
    return sum(1 for t in tokens if predicate(t)) / denominator


def _is_emotional(text: str) -> bool:
    # A run like "!!!" is one token in some tokenizers and three in others;
    # treating any token made entirely of affective marks as a single hit keeps
    # the feature from depending on that tokenizer detail.
    return bool(text) and all(c in EMOTIONAL_PUNCTUATION for c in text)
