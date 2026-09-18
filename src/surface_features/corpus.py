"""Reading the annotated comment corpus.

The corpus is a ~391 MB semicolon-delimited CSV. It is read with the standard
library `csv` module and yielded row by row rather than loaded with
`pandas.read_csv`: only three of its columns are ever used, and streaming keeps
peak memory proportional to one row instead of to the file.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from surface_features.errors import CorpusError

TEXT_COLUMN = "CONTENIDO A ANALIZAR"
KIND_COLUMN = "TIPO DE MENSAJE"
INTENSITY_COLUMN = "INTENSIDAD"
COMMENT_KIND = "COMENTARIO"

REQUIRED_COLUMNS = (TEXT_COLUMN, KIND_COLUMN, INTENSITY_COLUMN)


@dataclass(frozen=True, slots=True)
class Comment:
    text: str
    hostile: bool
    intensity: float


def read_comments(
    path: str | Path,
    *,
    threshold: float = 0.0,
    encoding: str = "utf-8",
) -> Iterator[Comment]:
    """Yield the comment rows of the corpus, labelled by intensity threshold.

    `hostile` is `intensity > threshold`, not `>=`: the annotation uses 0 for
    "no hostility", so an inclusive comparison would label the entire corpus
    hostile. Rows whose intensity is blank or unparseable are skipped rather
    than defaulted to 0 — an unlabelled row is missing data, not a negative.
    """

    with Path(path).open("r", encoding=encoding, newline="") as handle:
        yield from _read_stream(handle, threshold=threshold)


def _read_stream(handle: TextIO, *, threshold: float) -> Iterator[Comment]:
    # The corpus has rows with very long free-text fields; the default field
    # limit rejects them outright.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    reader = csv.DictReader(handle, delimiter=";")

    fieldnames = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        raise CorpusError(
            f"corpus is missing required column(s): {', '.join(missing)}; "
            f"found {', '.join(fieldnames) or '<no header>'}"
        )

    for row in reader:
        if (row.get(KIND_COLUMN) or "").strip().upper() != COMMENT_KIND:
            continue
        intensity = _parse_intensity(row.get(INTENSITY_COLUMN))
        if intensity is None:
            continue
        text = (row.get(TEXT_COLUMN) or "").strip()
        if not text:
            continue
        yield Comment(text=text, hostile=intensity > threshold, intensity=intensity)


def _parse_intensity(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # The export is Spanish-locale in places, so a decimal comma appears.
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None
