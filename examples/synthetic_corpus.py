"""Run the measurement pipeline end to end with no corpus, model, or network.

Two synthetic groups are generated with a *known* difference: the hostile group
is given a higher adjective rate and more shouting, and no difference at all in
verb rate. A pipeline that cannot recover the planted effects — large for
adjectives and uppercase, negligible for verbs — is measuring its own bugs
rather than the data, so this doubles as the sanity check the README describes.

    python examples/synthetic_corpus.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surface_features import experiment
from surface_features.stats import format_table
from surface_features.tokens import SimpleToken

RNG = random.Random(20260916)


def make_comment(*, adjective_rate: float, shout_rate: float) -> list[SimpleToken]:
    """Twenty word tokens with the requested adjective and shouting rates."""
    tokens: list[SimpleToken] = []
    for _ in range(20):
        # Verbs are drawn first and at a fixed rate in both groups, so the
        # adjective rate cannot silently displace them — otherwise the "no
        # difference" control would show an effect that was an artefact of the
        # generator rather than of the data.
        if RNG.random() < 0.25:
            word, pos = "dice", "VERB"
        elif RNG.random() < adjective_rate:
            word, pos = "pesimo", "ADJ"
        else:
            word, pos = "cosa", "NOUN"
        if RNG.random() < shout_rate:
            word = word.upper()
        tokens.append(SimpleToken(word, pos))
    tokens.append(SimpleToken("!", "PUNCT"))
    return tokens


def main() -> None:
    documents, labels = [], []
    for _ in range(400):
        documents.append(make_comment(adjective_rate=0.30, shout_rate=0.40))
        labels.append(True)
        documents.append(make_comment(adjective_rate=0.05, shout_rate=0.02))
        labels.append(False)

    data = experiment.build(documents, labels)
    print(
        f"comments {len(data)}  hostile {data.positive_rate:.0%}  "
        f"dropped {data.dropped}\n"
    )
    print(format_table(experiment.effects(data)))
    print("\nPlanted: adjectives and uppercase differ, verbs do not.")


if __name__ == "__main__":
    main()
