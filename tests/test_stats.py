"""Effect sizes, including every case where the statistic is undefined."""

import math

import pytest

from surface_features.stats import (
    Effect,
    cohens_d,
    effect_table,
    format_table,
    magnitude,
    mean,
    variance,
)


def test_cohens_d_recovers_a_known_value():
    # Two groups one pooled standard deviation apart: d == 1 exactly.
    a = [2.0, 4.0]  # mean 3, sample variance 2
    b = [0.0, 2.0]  # mean 1, sample variance 2
    assert cohens_d(a, b) == pytest.approx(2 / math.sqrt(2))


def test_identical_groups_have_zero_effect():
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_effect_changes_sign_when_the_groups_swap():
    a, b = [3.0, 5.0], [1.0, 2.0]
    assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))


def test_single_observation_group_is_undefined_not_zero():
    # Reporting 0.0 here would read as "measured, no effect".
    assert math.isnan(cohens_d([1.0], [1.0, 2.0, 3.0]))


def test_empty_group_is_undefined():
    assert math.isnan(cohens_d([], [1.0, 2.0]))


def test_two_constant_groups_have_no_pooled_spread():
    assert math.isnan(cohens_d([5.0, 5.0], [1.0, 1.0]))


def test_one_constant_group_still_yields_a_finite_effect():
    assert not math.isnan(cohens_d([5.0, 5.0], [1.0, 3.0]))


def test_variance_needs_two_observations():
    assert math.isnan(variance([1.0]))
    assert variance([1.0, 3.0]) == 2.0


def test_mean_of_nothing_is_undefined():
    assert math.isnan(mean([]))


@pytest.mark.parametrize(
    "value,label",
    [
        (0.0, "negligible"),
        (0.09, "negligible"),
        (0.1, "small"),
        (0.19, "small"),
        (0.2, "modest"),
        (0.49, "modest"),
        (0.5, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
        (12.0, "large"),
    ],
)
def test_magnitude_bands_at_their_boundaries(value, label):
    assert magnitude(value) == label


def test_magnitude_uses_the_absolute_size():
    assert magnitude(-0.9) == "large"


def test_undefined_effect_is_labelled_rather_than_binned():
    assert magnitude(float("nan")) == "undefined"


def test_effect_table_sorts_by_absolute_effect():
    table = effect_table(
        {
            "small": ([1.0, 1.1], [1.0, 1.1]),
            "big": ([10.0, 11.0], [0.0, 1.0]),
            "negative": ([0.0, 1.0], [4.0, 5.0]),
        }
    )
    assert table[0].feature == "big"
    assert [e.feature for e in table][:2] == ["big", "negative"]


def test_undefined_effects_sort_last():
    table = effect_table(
        {
            "undefined": ([1.0, 1.0], [2.0, 2.0]),
            "real": ([3.0, 5.0], [1.0, 2.0]),
        }
    )
    assert table[0].feature == "real"
    assert math.isnan(table[-1].cohens_d)


def test_effect_exposes_the_raw_difference():
    e = Effect("f", positive_mean=0.5, negative_mean=0.2, cohens_d=1.0)
    assert e.difference == pytest.approx(0.3)
    assert e.magnitude == "large"


def test_format_table_renders_one_line_per_feature():
    text = format_table(effect_table({"a": ([1.0, 2.0], [3.0, 4.0])}))
    assert text.splitlines()[0].startswith("feature")
    assert len(text.splitlines()) == 3
