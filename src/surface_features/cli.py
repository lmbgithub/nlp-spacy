"""Command line entry point: run the whole experiment over a corpus file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from surface_features import experiment
from surface_features.corpus import CorpusError, read_comments
from surface_features.spacy_backend import (
    DEFAULT_MODEL,
    SpacyUnavailable,
    load_pipeline,
    parse,
)
from surface_features.stats import format_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-features",
        description="Measure whether surface features separate hostile Spanish comments.",
    )
    parser.add_argument("corpus", type=Path, help="semicolon-delimited annotated CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="spaCy model name")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N comments (for a quick run)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="a comment counts as hostile when INTENSIDAD exceeds this",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="report effect sizes only; skips the scikit-learn dependency",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        comments = list(
            _take(read_comments(args.corpus, threshold=args.threshold), args.limit)
        )
    except FileNotFoundError:
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    except CorpusError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not comments:
        print("no usable comments in corpus", file=sys.stderr)
        return 1

    try:
        pipeline = load_pipeline(args.model)
    except SpacyUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    documents = parse(pipeline, (c.text for c in comments))
    data = experiment.build(documents, (c.hostile for c in comments))

    print(
        f"comments analysed  {len(data):,}  "
        f"(dropped {data.dropped:,} with no word tokens)"
    )
    print(f"hostile            {sum(data.labels):,} ({data.positive_rate * 100:.1f}%)\n")
    print(format_table(experiment.effects(data)))

    if not args.no_classifier:
        result = experiment.classify(data, seed=args.seed)
        print(
            f"\nmajority-class accuracy {result.majority_accuracy:.3f}"
            f"\nrandom forest accuracy  {result.model_accuracy:.3f}"
            f"\nrandom forest ROC-AUC   {result.roc_auc:.3f}"
            f"\nbeats baseline          {result.beats_baseline}"
        )
        print("\nfeature importances")
        for name, value in experiment.importances(data, seed=args.seed):
            print(f"  {name:<24}{value:.4f}")

    return 0


def _take(iterable, limit: int | None):
    if limit is None:
        yield from iterable
        return
    for index, item in enumerate(iterable):
        if index >= limit:
            return
        yield item


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
