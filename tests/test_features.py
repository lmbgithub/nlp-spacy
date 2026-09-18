"""Every ratio is checked against a value computed by hand."""

import pytest

from surface_features.features import (
    ALL_COLUMNS,
    FEATURE_NAMES,
    extract_features,
    extract_many,
)
from surface_features.tokens import SimpleToken as T


def test_ratios_match_hand_computed_values():
    # 4 words: 1 verb, 1 adjective, 1 shouted word; 1 affective punctuation mark.
    tokens = [
        T("dice", "VERB"),
        T("pesimo", "ADJ"),
        T("TODO", "NOUN"),
        T("cosa", "NOUN"),
        T("!", "PUNCT"),
    ]
    f = extract_features(tokens)
    assert f.n_words == 4
    assert f.n_tokens == 5
    assert f.verbs == 0.25
    assert f.adjectives == 0.25
    assert f.uppercase == 0.25
    assert f.emotional_punctuation == 0.25


def test_a_comment_with_no_word_tokens_returns_none():
    assert extract_features([T("!"), T("?"), T("...")]) is None


def test_an_empty_document_returns_none():
    assert extract_features([]) is None


def test_single_word_document_is_a_valid_extreme():
    f = extract_features([T("idiota", "ADJ")])
    assert f.adjectives == 1.0
    assert f.n_words == 1


def test_punctuation_is_excluded_from_the_denominator():
    without = extract_features([T("dice", "VERB"), T("cosa", "NOUN")])
    with_punct = extract_features([T("dice", "VERB"), T("cosa", "NOUN"), T(","), T(".")])
    assert with_punct.verbs == without.verbs == 0.5


def test_ratios_not_counts_so_length_does_not_inflate_the_feature():
    short = extract_features([T("pesimo", "ADJ"), T("cosa", "NOUN")])
    long = extract_features([T("pesimo", "ADJ"), T("cosa", "NOUN")] * 10)
    assert short.adjectives == long.adjectives


def test_single_letter_capital_is_not_shouting():
    # "A" as a preposition is not a raised voice; requiring length > 1 avoids
    # scoring ordinary Spanish prose as shouting.
    f = extract_features([T("A", "ADP"), T("casa", "NOUN")])
    assert f.uppercase == 0.0


def test_digits_are_not_counted_as_shouting():
    f = extract_features([T("10K", "NUM"), T("casa", "NOUN")])
    assert f.uppercase == 0.0


def test_a_run_of_marks_counts_once_regardless_of_tokenization():
    joined = extract_features([T("basta", "VERB"), T("!!!", "PUNCT")])
    assert joined.emotional_punctuation == 1.0


def test_structural_punctuation_is_not_emotional():
    f = extract_features([T("cosa", "NOUN"), T(","), T(";"), T(".")])
    assert f.emotional_punctuation == 0.0


def test_emotional_punctuation_can_exceed_one_per_word():
    f = extract_features([T("basta", "VERB"), T("!"), T("?"), T("¡")])
    assert f.emotional_punctuation == 3.0


def test_as_vector_follows_the_declared_feature_order():
    f = extract_features([T("dice", "VERB")])
    assert f.as_vector() == [getattr(f, name) for name in FEATURE_NAMES]
    assert len(f.as_vector()) == len(FEATURE_NAMES)


def test_length_columns_are_not_offered_to_the_model():
    assert "n_words" not in FEATURE_NAMES
    assert "n_words" in ALL_COLUMNS


def test_all_columns_matches_the_dataclass_fields():
    # The column list is used to index into rows; if a field is added without
    # being listed here, every consumer silently ignores it.
    from dataclasses import fields

    f = extract_features([T("dice", "VERB")])
    assert {field.name for field in fields(f)} == set(ALL_COLUMNS)


def test_features_are_immutable():
    f = extract_features([T("dice", "VERB")])
    with pytest.raises(AttributeError):
        f.verbs = 1.0


def test_extract_many_reports_how_many_documents_were_dropped():
    rows, dropped = extract_many([[T("dice", "VERB")], [T("!")], []])
    assert len(rows) == 1
    assert dropped == 2


def test_extract_many_on_an_empty_corpus():
    assert extract_many([]) == ([], 0)
