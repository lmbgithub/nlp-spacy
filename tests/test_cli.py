"""Command line behaviour that does not need spaCy installed."""

import pytest

from surface_features.cli import build_parser, main

HEADER = "TIPO DE MENSAJE;INTENSIDAD;CONTENIDO A ANALIZAR\n"


def test_defaults():
    args = build_parser().parse_args(["corpus.csv"])
    assert args.threshold == 0.0
    assert args.seed == 0
    assert args.limit is None


def test_missing_corpus_exits_with_a_usage_code(tmp_path, capsys):
    assert main([str(tmp_path / "absent.csv")]) == 2
    assert "corpus not found" in capsys.readouterr().err


def test_malformed_corpus_reports_the_missing_column(tmp_path, capsys):
    path = tmp_path / "corpus.csv"
    path.write_text("A;B\n1;2\n", encoding="utf-8")
    assert main([str(path)]) == 2
    assert "missing required column" in capsys.readouterr().err


def test_empty_corpus_is_a_failure_not_an_empty_report(tmp_path, capsys):
    path = tmp_path / "corpus.csv"
    path.write_text(HEADER, encoding="utf-8")
    assert main([str(path)]) == 1
    assert "no usable comments" in capsys.readouterr().err


def test_limit_is_rejected_when_not_an_integer():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["corpus.csv", "--limit", "many"])
