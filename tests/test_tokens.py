"""The deterministic fake must behave like the thing it stands in for."""

from surface_features.tokens import EMOTIONAL_PUNCTUATION, SimpleToken, Token, tokenize


def test_simple_token_satisfies_the_protocol():
    assert isinstance(SimpleToken("hola", "INTJ"), Token)


def test_punctuation_is_derived_from_the_text():
    assert SimpleToken("!").is_punct
    assert not SimpleToken("hola").is_punct


def test_alpha_is_derived_from_the_text():
    assert SimpleToken("hola").is_alpha
    assert not SimpleToken("10K").is_alpha
    assert not SimpleToken("!").is_alpha


def test_empty_text_is_neither_punctuation_nor_alphabetic():
    empty = SimpleToken("")
    assert not empty.is_punct
    assert not empty.is_alpha


def test_flags_can_be_overridden_to_reproduce_a_tagger_disagreement():
    token = SimpleToken("hola", "NOUN", _is_punct=True)
    assert token.is_punct


def test_emotional_punctuation_covers_the_inverted_spanish_marks():
    assert {"¡", "¿"} <= set(EMOTIONAL_PUNCTUATION)


def test_tokenize_tags_from_the_lookup_and_defaults_to_x():
    tokens = tokenize("el coche corre", {"corre": "VERB"})
    assert [t.pos_ for t in tokens] == ["X", "X", "VERB"]


def test_tokenize_on_empty_text_yields_nothing():
    assert tokenize("") == []
