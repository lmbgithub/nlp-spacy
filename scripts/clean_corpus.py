"""Repair the raw cp1252 corpus export and rewrite it as clean UTF-8.

The original export has three separate problems, fixed in order:

1. Stray bytes (0xBF, 0xA1, 0xB3) that break decoding outright.
2. Records split across several physical lines, which a CSV reader sees as
   short, malformed rows.
3. Mojibake — text that was encoded as UTF-8 and then decoded as latin-1, so
   "debía" arrives as "debÃ­a".

Run it once; the experiment reads only the cleaned file.

    python scripts/clean_corpus.py comentarios.csv comentarios_limpio_utf8.csv
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

# Bytes that appear in the export but are not valid in the declared encoding.
BAD_BYTES = frozenset({0xBF, 0xA1, 0xB3})

# A physical line starting with one of these begins a new record; anything else
# is a continuation of the record above it.
RECORD_STARTS = ("MEDIO;SOPORTE;", "EL PA? ;WEB;")


def strip_bad_bytes(source: Path, target: Path) -> None:
    """Drop the undecodable bytes at the binary level, before any decoding."""
    with source.open("rb") as handle_in, target.open("wb") as handle_out:
        for line in handle_in:
            handle_out.write(bytes(b for b in line if b not in BAD_BYTES))


def rejoin_records(source: Path, target: Path, *, encoding: str = "latin-1") -> int:
    """Glue continuation lines back onto the record they belong to."""
    joined = 0
    with (
        source.open("r", encoding=encoding) as handle_in,
        target.open("w", encoding=encoding) as handle_out,
    ):
        record = ""
        for line in handle_in:
            if line.startswith(RECORD_STARTS):
                if record:
                    handle_out.write(record + "\n")
                    joined += 1
                record = ""
            record += re.sub(r"^\n", "", line)
        if record:
            handle_out.write(record + "\n")
            joined += 1
    return joined


def repair_mojibake(value: str) -> str:
    """Undo one round-trip of UTF-8 bytes decoded as latin-1.

    Text that never suffered the round trip is returned untouched: the encode
    step raises for any character outside latin-1, and the decode step raises
    for byte sequences that are not valid UTF-8.
    """
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="raw cp1252 export")
    parser.add_argument("target", type=Path, help="cleaned UTF-8 output")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        stripped = Path(tmp) / "stripped.csv"
        rejoined = Path(tmp) / "rejoined.csv"
        strip_bad_bytes(args.source, stripped)
        records = rejoin_records(stripped, rejoined)

        with (
            rejoined.open("r", encoding="latin-1") as handle_in,
            args.target.open("w", encoding="utf-8") as handle_out,
        ):
            for line in handle_in:
                handle_out.write(repair_mojibake(line))

    print(f"wrote {args.target} ({records:,} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
