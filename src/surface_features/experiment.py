"""The experiment itself: features, effect sizes, and a classifier floor."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from surface_features.errors import SurfaceFeaturesError
from surface_features.features import (
    ALL_COLUMNS,
    FEATURE_NAMES,
    SurfaceFeatures,
    extract_features,
)
from surface_features.stats import Effect, effect_table


@dataclass(frozen=True, slots=True)
class LabelledFeatures:
    """Extracted rows plus their labels, kept parallel by construction."""

    rows: tuple[SurfaceFeatures, ...]
    labels: tuple[bool, ...]
    dropped: int = 0

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.labels):
            raise SurfaceFeaturesError(
                f"rows and labels must be the same length; "
                f"got {len(self.rows)} and {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def positive_rate(self) -> float:
        if not self.labels:
            return float("nan")
        return sum(self.labels) / len(self.labels)

    def column(self, name: str, *, hostile: bool) -> list[float]:
        """One feature column restricted to one group."""
        if name not in ALL_COLUMNS:
            raise SurfaceFeaturesError(
                f"unknown column {name!r}; expected one of {ALL_COLUMNS}"
            )
        return [
            float(getattr(row, name))
            for row, label in zip(self.rows, self.labels, strict=True)
            if bool(label) is hostile
        ]

    def matrix(self) -> list[list[float]]:
        """Design matrix over `FEATURE_NAMES`, in row order."""
        return [row.as_vector() for row in self.rows]


def build(documents: Iterable[Sequence], labels: Iterable[bool]) -> LabelledFeatures:
    """Extract features and keep only the rows that survived extraction.

    Labels are consumed alongside documents so a dropped document takes its
    label with it. Extracting first and zipping afterwards is the classic way
    to silently shift every label by the number of dropped rows.
    """

    rows: list[SurfaceFeatures] = []
    kept: list[bool] = []
    dropped = 0
    for tokens, label in zip(documents, labels, strict=False):
        row = extract_features(tokens)
        if row is None:
            dropped += 1
            continue
        rows.append(row)
        kept.append(bool(label))
    return LabelledFeatures(tuple(rows), tuple(kept), dropped)


def effects(data: LabelledFeatures, columns: Sequence[str] = ALL_COLUMNS) -> list[Effect]:
    """Cohen's *d* per column, hostile against everything else."""
    return effect_table(
        {
            name: (data.column(name, hostile=True), data.column(name, hostile=False))
            for name in columns
        }
    )


@dataclass(frozen=True, slots=True)
class ClassifierResult:
    """A classifier compared against the only baseline that matters."""

    majority_accuracy: float
    model_accuracy: float
    roc_auc: float

    @property
    def beats_baseline(self) -> bool:
        """Accuracy alone cannot answer this on an imbalanced corpus.

        A model that predicts the majority class everywhere already scores the
        base rate. Requiring the AUC to clear 0.5 by a margin is the check that
        the features carry ranking information at all.
        """
        return self.model_accuracy > self.majority_accuracy and self.roc_auc > 0.55


def classify(
    data: LabelledFeatures, *, seed: int = 0, test_size: float = 0.25
) -> ClassifierResult:
    """Fit a random forest on the surface features and score it honestly.

    Imported lazily so that the statistics half of this package stays usable
    without scikit-learn installed.
    """

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    x = data.matrix()
    y = list(data.labels)
    if len(set(y)) < 2:
        raise SurfaceFeaturesError("cannot classify: the corpus contains a single class")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)

    return ClassifierResult(
        majority_accuracy=baseline.score(x_test, y_test),
        model_accuracy=model.score(x_test, y_test),
        roc_auc=roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
    )


def importances(data: LabelledFeatures, *, seed: int = 0) -> list[tuple[str, float]]:
    """Feature importances, largest first — a second opinion on the ranking."""

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(data.matrix(), list(data.labels))
    pairs = list(
        zip(FEATURE_NAMES, (float(v) for v in model.feature_importances_), strict=True)
    )
    return sorted(pairs, key=lambda p: p[1], reverse=True)
