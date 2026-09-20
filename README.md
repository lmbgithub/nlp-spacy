# nlp-spacy

Detecting hostile Spanish comments using cheap surface features — verb ratio, adjective ratio, shouting, emotional punctuation.

The project asks whether those features separate hostile comments from ordinary ones well enough to be worth using. Differences are reported as effect sizes rather than differences in means, and a simple classifier provides a floor that shows whether the surviving features carry anything a model can actually use.

**Core package: standard library only.** spaCy and scikit-learn are needed to run against the real corpus; neither is needed for the test suite or the offline example. **82 tests.**

## Skills demonstrated

**NLP** — part-of-speech feature extraction over a protocol-typed token stream, ratio normalisation to remove a length confound, Spanish-specific affective punctuation handling, spaCy pipeline configured to the components actually used

**Statistics** — Cohen's *d* with pooled standard deviation, magnitude bands with an explicit negligible floor, NaN propagation for undefined statistics rather than a zero sentinel, effect ranking that sorts undefined last

**ML evaluation** — majority-class baseline as the comparison, ROC-AUC alongside accuracy on an imbalanced corpus, a classifier used as a floor check rather than as a product

**Data engineering** — streaming CSV over a 391 MB file so peak memory is one row, locale-aware decimal parsing, a repair script for mojibake and stray cp1252 bytes

**Software engineering** — one-way boundary: the measurement code never imports spaCy, so 82 tests run in a second against hand-built tokens with values computed by hand

**Tooling** — ruff, pre-commit, CI matrix on 3.10/3.11/3.12 with a lint job

## What the run looks like

The experiment needs a 391 MB corpus that is not in this repository, so the transcript below is from the offline example, which generates two synthetic groups with a *known* planted difference: a higher adjective rate and more shouting in the hostile group, and deliberately **no** difference in verb rate.

```
$ python examples/synthetic_corpus.py
comments 800  hostile 50%  dropped 0

feature                    hostile     other   cohen_d  magnitude
-----------------------------------------------------------------
uppercase                   0.3880    0.0198    4.4493  large
adjectives                  0.2231    0.0360    2.5839  large
verbs                       0.2509    0.2527   -0.0199  negligible
emotional_punctuation       0.0500    0.0500       nan  undefined
n_words                    20.0000   20.0000       nan  undefined
n_tokens                   21.0000   21.0000       nan  undefined

Planted: adjectives and uppercase differ, verbs do not.
```

Two planted effects recovered, one planted non-effect correctly reported as negligible, and the three genuinely constant columns reported as *undefined* rather than as zero. A measurement tool that has never been run against known truth is measuring its own bugs; this is the run that rules that out.

Against the real corpus:

```
$ python -m surface_features comentarios_limpio_utf8.csv --limit 50000
```

## The six decisions worth discussing

**1. Ratios, not counts.** Hostile comments may simply be longer. A raw adjective count would then rank them higher for a reason that has nothing to do with hostility. Every feature is divided by the word-token count, and the length columns are reported separately so the confound is visible instead of assumed away. `n_words` is deliberately excluded from `FEATURE_NAMES` — the classifier is never handed the length it could learn instead of the signal.

**2. Effect size, not significance.** With tens of thousands of comments almost any difference in means clears a significance test. Cohen's *d* asks the question that matters — how large is the gap relative to the spread within each group — and the bands add a `negligible` floor below 0.1, where the distributions overlap so heavily that no classifier can exploit the difference. "Negligible" is a more useful verdict than "small but significant".

**3. An empty comment is dropped, not zeroed.** A comment that is nothing but emoji has no denominator. Returning a row of zeros would not mark it missing; it would assert that it contains no verbs and no adjectives, and thousands of such rows drag every group mean toward zero. `extract_features` returns `None` and the caller reports the dropped count.

**4. An undefined statistic stays undefined.** Fewer than two observations in a group, or two constant groups, leave Cohen's *d* with no pooled spread. `cohens_d` returns NaN and the table labels it `undefined`. Returning `0.0` would read as "measured, no effect" — the opposite of the truth — and NaN rows are sorted last so an undefined statistic can never head the ranking.

**5. The classifier is scored against the majority baseline, not against zero.** On an imbalanced corpus a model that predicts "not hostile" everywhere already scores the base rate. `ClassifierResult.beats_baseline` requires accuracy above the majority classifier *and* ROC-AUC clearly above chance, because accuracy alone cannot distinguish a real result from a model that learnt the base rate.

**6. Labels travel with their documents.** `experiment.build` consumes documents and labels together and discards the label of any dropped row. Extracting features first and zipping labels afterwards silently shifts every label by the number of dropped rows — a bug that produces plausible, entirely wrong effect sizes. There is a test named after it.

## Design

```
src/surface_features/
  tokens.py          the minimal token protocol + SimpleToken, the deterministic fake
  features.py        one comment in, one row of ratios out
  stats.py           Cohen's d, magnitude bands, the effect table
  corpus.py          streaming CSV reader and the labelling rule
  spacy_backend.py   the only module that imports spaCy
  experiment.py      wiring: build, effects, classifier floor
  cli.py             argument parsing and the report
```

The boundary that matters: **the measurement code never imports spaCy.** `features` and `stats` depend only on the `Token` protocol, which spaCy's own `Token` satisfies structurally. That is what lets every ratio be checked against hand-built tokens with values computed by hand, and it is why the test suite runs in under a second with no model downloaded.

`corpus.py` streams with the standard library `csv` module rather than `pandas.read_csv`: three columns of a 391 MB file are used, and streaming keeps peak memory proportional to one row instead of to the file.

## Usage

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # tests and the offline example
pip install -r requirements.txt    # spaCy + scikit-learn, for the real corpus
python -m spacy download es_core_news_md
```

```bash
pytest -q
python examples/synthetic_corpus.py
python -m surface_features comentarios_limpio_utf8.csv --limit 50000
python -m surface_features comentarios_limpio_utf8.csv --no-classifier
```

As a library:

```python
from surface_features import extract_features
from surface_features.tokens import SimpleToken

row = extract_features([SimpleToken("PESIMO", "ADJ"), SimpleToken("!", "PUNCT")])
print(row.uppercase, row.emotional_punctuation)   # 1.0 1.0
```

`main.ipynb` runs the same experiment as a narrative, importing the package rather than redefining it.

## Dataset

Not included (~391 MB). The corpus is a set of Spanish news-site comments, each annotated with an `INTENSIDAD` (intensity) score; a comment counts as hostile when its intensity exceeds zero.

Place the file as `comentarios_limpio_utf8.csv` in the project root. It must be semicolon-delimited, UTF-8, and carry the columns `TIPO DE MENSAJE`, `INTENSIDAD` and `CONTENIDO A ANALIZAR`; the reader names any column it cannot find rather than failing on a `KeyError` a thousand rows later.

If your copy is the raw cp1252 export, `scripts/clean_corpus.py` strips the stray bytes, repairs the mojibake and rewrites it as UTF-8:

```bash
python scripts/clean_corpus.py comentarios.csv comentarios_limpio_utf8.csv
```

Then install the Spanish model:

```bash
python -m spacy download es_core_news_md
```

Any corpus with those three columns works; nothing in the code is specific to this one.

## Scope

- **No embeddings.** Surface features are a floor, not a solution. They capture how something is written, not what it says, and a hostile comment in calm lowercase prose defeats every feature here. That gap is what embeddings exist to fill; knowing its size is worth more than assuming it.
- **No hyperparameter search.** The random forest is a floor check, not a product. Tuning it would obscure what the features do and do not carry.
- **No bootstrap confidence intervals on *d*.** At corpus scale the point estimate is stable enough for a ranking; at the sample sizes here an interval would be decoration.
- **No per-annotator agreement analysis.** The label is a threshold on a continuous score, which collapses mild and extreme hostility into one class — a real limitation of the experiment, stated rather than fixed.
- **No multi-language support.** The pipeline and the punctuation set are Spanish-specific by design.

## License

MIT — see [LICENSE](LICENSE).
