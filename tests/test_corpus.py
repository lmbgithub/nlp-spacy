"""Corpus reading: the labelling rule and every row that must be skipped."""

import pytest

from surface_features.corpus import CorpusError, read_comments

HEADER = "TIPO DE MENSAJE;INTENSIDAD;CONTENIDO A ANALIZAR\n"


def write(tmp_path, body, header=HEADER):
    path = tmp_path / "corpus.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_reads_comment_rows(tmp_path):
    path = write(tmp_path, "COMENTARIO;1.5;esto es una prueba\n")
    (comment,) = read_comments(path)
    assert comment.text == "esto es una prueba"
    assert comment.intensity == 1.5
    assert comment.hostile


def test_non_comment_rows_are_skipped(tmp_path):
    path = write(tmp_path, "NOTICIA;3.0;titular\nCOMENTARIO;0.0;respuesta\n")
    assert [c.text for c in read_comments(path)] == ["respuesta"]


def test_zero_intensity_is_not_hostile(tmp_path):
    # The threshold is strict: `>= 0` would label the entire corpus hostile.
    path = write(tmp_path, "COMENTARIO;0.0;tranquilo\n")
    assert not next(iter(read_comments(path))).hostile


def test_threshold_is_configurable(tmp_path):
    path = write(tmp_path, "COMENTARIO;1.0;molesto\n")
    assert next(iter(read_comments(path))).hostile
    assert not next(iter(read_comments(path, threshold=1.0))).hostile


def test_blank_intensity_is_missing_data_not_a_negative(tmp_path):
    path = write(tmp_path, "COMENTARIO;;sin anotar\n")
    assert list(read_comments(path)) == []


def test_unparseable_intensity_is_skipped(tmp_path):
    path = write(tmp_path, "COMENTARIO;n/a;raro\n")
    assert list(read_comments(path)) == []


def test_decimal_comma_is_accepted(tmp_path):
    path = write(tmp_path, "COMENTARIO;2,5;locale espanol\n")
    assert next(iter(read_comments(path))).intensity == 2.5


def test_empty_text_is_skipped(tmp_path):
    path = write(tmp_path, "COMENTARIO;3.0;   \n")
    assert list(read_comments(path)) == []


def test_message_kind_matching_ignores_case_and_padding(tmp_path):
    path = write(tmp_path, " comentario ;1.0;hola\n")
    assert len(list(read_comments(path))) == 1


def test_missing_column_names_the_column(tmp_path):
    path = write(tmp_path, "COMENTARIO;1.0\n", header="TIPO DE MENSAJE;INTENSIDAD\n")
    with pytest.raises(CorpusError, match="CONTENIDO A ANALIZAR"):
        list(read_comments(path))


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_comments(tmp_path / "absent.csv"))


def test_header_only_corpus_yields_nothing(tmp_path):
    assert list(read_comments(write(tmp_path, ""))) == []


def test_reading_is_lazy(tmp_path):
    # A 391 MB corpus must not be materialised to inspect its first row.
    path = write(tmp_path, "".join(f"COMENTARIO;1.0;fila {i}\n" for i in range(1000)))
    first = next(iter(read_comments(path)))
    assert first.text == "fila 0"
