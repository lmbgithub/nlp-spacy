"""The experiment wiring: label alignment, grouping, and the baseline check."""

import math

import pytest

from surface_features import experiment
from surface_features.errors import SurfaceFeaturesError
from surface_features.experiment import ClassifierResult, LabelledFeatures
from surface_features.tokens import SimpleToken as T

HOSTILE = [T("PESIMO", "ADJ"), T("idiota", "ADJ"), T("!", "PUNCT")]
CALM = [T("gracias", "NOUN"), T("dice", "VERB")]
EMPTY = [T("!"), T("?")]


def test_dropped_documents_take_their_label_with_them():
    # The classic bug this guards: extracting first and zipping afterwards
    # shifts every label by the number of dropped rows.
    data = experiment.build([HOSTILE, EMPTY, CALM], [True, True, False])
    assert len(data) == 2
    assert data.dropped == 1
    assert data.labels == (True, False)


def test_all_documents_dropped_leaves_an_empty_result():
    data = experiment.build([EMPTY, EMPTY], [True, False])
    assert len(data) == 0
    assert data.dropped == 2
    assert math.isnan(data.positive_rate)


def test_positive_rate_is_the_base_rate():
    data = experiment.build([HOSTILE, CALM, CALM, CALM], [True, False, False, False])
    assert data.positive_rate == 0.25


def test_labels_and_rows_must_be_parallel():
    with pytest.raises(ValueError, match="same length"):
        LabelledFeatures(rows=(), labels=(True,))


def test_column_splits_by_group():
    data = experiment.build([HOSTILE, CALM], [True, False])
    assert data.column("adjectives", hostile=True) == [1.0]
    assert data.column("adjectives", hostile=False) == [0.0]


def test_column_rejects_an_unknown_name():
    data = experiment.build([HOSTILE], [True])
    with pytest.raises(SurfaceFeaturesError, match="unknown column"):
        data.column("sentiment", hostile=True)


def test_matrix_rows_line_up_with_labels():
    data = experiment.build([HOSTILE, CALM], [True, False])
    matrix = data.matrix()
    assert len(matrix) == len(data.labels) == 2


def test_effects_recover_a_planted_difference():
    # Each document varies slightly: identical rows have zero within-group
    # variance, which makes every pooled standard deviation — and therefore
    # every effect size — undefined.
    documents, labels = [], []
    for i in range(20):
        documents.append(HOSTILE + [T("cosa", "NOUN")] * (i % 3))
        labels.append(True)
        documents.append(CALM + [T("cosa", "NOUN")] * (i % 3))
        labels.append(False)
    table = {
        e.feature: e for e in experiment.effects(experiment.build(documents, labels))
    }
    assert table["adjectives"].magnitude == "large"
    assert table["adjectives"].difference > 0
    assert table["uppercase"].difference > 0


def test_effects_on_a_single_class_corpus_are_undefined():
    data = experiment.build([HOSTILE] * 5, [True] * 5)
    assert all(math.isnan(e.cohens_d) for e in experiment.effects(data))


def test_classify_refuses_a_single_class_corpus():
    data = experiment.build([HOSTILE] * 5, [True] * 5)
    with pytest.raises(ValueError, match="single class"):
        experiment.classify(data)


def test_beating_the_baseline_requires_more_than_accuracy():
    # 92% accuracy on a 92% majority corpus is the baseline, not a result.
    tie = ClassifierResult(majority_accuracy=0.92, model_accuracy=0.92, roc_auc=0.80)
    assert not tie.beats_baseline


def test_beating_the_baseline_requires_ranking_information():
    # Accuracy up by a hair, AUC at chance: the model learnt the base rate.
    noise = ClassifierResult(majority_accuracy=0.92, model_accuracy=0.93, roc_auc=0.50)
    assert not noise.beats_baseline


def test_a_real_result_clears_both_checks():
    good = ClassifierResult(majority_accuracy=0.92, model_accuracy=0.94, roc_auc=0.71)
    assert good.beats_baseline


def test_build_stops_at_the_shorter_of_documents_and_labels():
    data = experiment.build([HOSTILE, CALM, CALM], [True])
    assert len(data) == 1
