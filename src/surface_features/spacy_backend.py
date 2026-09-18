"""The only module that knows spaCy exists.

Everything else in the package works on the `Token` protocol, so the model is
an implementation detail that can be swapped for hand-built tokens in tests and
in the offline example.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from surface_features.errors import SpacyUnavailable
from surface_features.tokens import Token

DEFAULT_MODEL = "es_core_news_md"

# Only the tagger is needed for part-of-speech ratios. Disabling the rest is
# not a micro-optimisation: on a corpus this size the parser alone roughly
# doubles wall-clock time for information no feature reads.
DISABLED_COMPONENTS = ("parser", "lemmatizer", "attribute_ruler", "ner")


def load_pipeline(model: str = DEFAULT_MODEL, *, disable: Sequence[str] | None = None):
    """Load the spaCy pipeline, with an error that says how to fix it."""

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SpacyUnavailable(
            "spaCy is not installed; run: pip install -r requirements.txt"
        ) from exc

    disabled = list(DISABLED_COMPONENTS if disable is None else disable)
    try:
        return spacy.load(model, disable=disabled)
    except OSError as exc:  # pragma: no cover - depends on the environment
        raise SpacyUnavailable(
            f"spaCy model {model!r} is not installed; "
            f"run: python -m spacy download {model}"
        ) from exc


def parse(
    pipeline, texts: Iterable[str], *, batch_size: int = 200
) -> Iterator[list[Token]]:
    """Parse texts in batches, yielding token lists.

    `pipeline.pipe` rather than calling the pipeline per text: it batches the
    model forward pass and is several times faster on a corpus of this size.
    """

    for doc in pipeline.pipe(texts, batch_size=batch_size):
        yield list(doc)
