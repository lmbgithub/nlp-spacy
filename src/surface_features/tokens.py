"""The minimal token interface the feature extractor depends on.

spaCy's `Token` satisfies this protocol structurally, so the real pipeline can
be passed straight through. `SimpleToken` is the deterministic fake used by the
tests: hand-built tokens with known part-of-speech tags let every ratio be
checked against a value computed by hand, which is the only way to know the
extractor is measuring the corpus rather than its own bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Punctuation that carries affect rather than structure. Spanish opens
# exclamations and questions with inverted marks, so both forms count.
EMOTIONAL_PUNCTUATION = frozenset("!¡¿?")


@runtime_checkable
class Token(Protocol):
    """Structural subset of `spacy.tokens.Token` used by this package."""

    @property
    def text(self) -> str: ...

    @property
    def pos_(self) -> str: ...

    @property
    def is_punct(self) -> bool: ...

    @property
    def is_alpha(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class SimpleToken:
    """A token with no NLP pipeline behind it.

    `is_punct` and `is_alpha` default to being derived from the text so that a
    test can write `SimpleToken("idiota", "ADJ")` and get sensible flags, while
    still being able to override them to reproduce a tagger's disagreement.
    """

    text: str
    pos_: str = "X"
    _is_punct: bool | None = None
    _is_alpha: bool | None = None

    @property
    def is_punct(self) -> bool:
        if self._is_punct is not None:
            return self._is_punct
        return bool(self.text) and not any(c.isalnum() for c in self.text)

    @property
    def is_alpha(self) -> bool:
        if self._is_alpha is not None:
            return self._is_alpha
        return self.text.isalpha()


def tokenize(text: str, pos: dict[str, str] | None = None) -> list[SimpleToken]:
    """Whitespace-split `text` into `SimpleToken`s, tagging from a lookup table.

    This is not a tokenizer worth using for anything real; it exists so tests
    and the offline example can build token sequences from readable strings.
    """

    pos = pos or {}
    return [SimpleToken(word, pos.get(word.lower(), "X")) for word in text.split()]
