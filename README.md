# wikipedia_analysis

Predicting the quality class of Wikipedia articles (stub, start, C, B, good, featured) for English,
French and Russian Wikipedia, with models that need little or no language-specific feature engineering.

The repository started in 2016 as R code on ~20k English articles, then grew into LSTM/CNN models on
raw text (see [`legacy/`](legacy/README.md)). The current version is a Python pipeline that redoes the
experiments with current tooling, fixes problems in the old evaluation, and benchmarks simple
baselines against frozen and fine-tuned multilingual transformers.

> A separate study, on LLM-style text in new English Wikipedia articles (2018-2026), lives in [`ai_text/`](ai_text/README.md). Background reading for an essay on Wikipedia's role in the LLM era is in [`wikipedia_role/`](wikipedia_role/README.md).

## Data

| Language | Articles | Classes |
|---|---|---|
| English | 29,468 | stub, start, c, b, ga, fa |
| French | 8,834 | e, bd, b, ba, a, adq |
| Russian | 7,888 | IV, III, II, I, GA, FA, SA |

Classes are roughly balanced. Labels come from the WikiProject assessments published with the ORES
`wp10` model (2015); each article has one labelled revision, fetched by timestamp. The raw wikitext is about 1.5 GB and is no
longer in the working tree. It stays in the commit `a8a849c`, and `scripts/prepare_data.py`
extracts it from there into `data/processed/*.parquet`. The class order used for ordinal metrics
is in `wikiquality/data.py`; for French and Russian it is our reading of the scales.

## Label leakage

Featured and good articles carry status templates in their wikitext (`{{Featured article}}`,
`{{Bon article}}`, `{{Избранная статья}}`, ...), and stubs carry stub templates (`{{Asia-geo-stub}}`,
`{{Ébauche}}`, ...). A model trained on the raw text can read the label instead of judging quality.
`wikiquality.features.scrub` removes these templates and the matching categories, and all results
below are reported for scrubbed text. The effect is modest for the classic models (TF-IDF macro-F1
on English 0.612 raw vs 0.595 scrubbed, see `results/baselines_*.json`) but large enough that the
scrubbed numbers are the ones to cite.

## Results

Stratified 80/20 split, seed 2017, macro-F1 on the held-out 20% (full corpora, scrubbed text).
QWK is quadratic weighted kappa under the class order above.

| Model | en F1 | en QWK | fr F1 | fr QWK | ru F1 | ru QWK |
|---|---|---|---|---|---|---|
| Length only (log chars, logistic regression) | 0.448 | 0.78 | 0.432 | 0.78 | 0.431 | 0.64 |
| 19 structural features (refs, headings, links, ...) + LightGBM | 0.570 | 0.84 | 0.534 | 0.82 | 0.556 | 0.82 |
| TF-IDF word 1-2 grams over wikitext + linear SVM | 0.595 | 0.86 | 0.506 | 0.81 | 0.562 | 0.82 |
| XLM-R base fine-tuned (first 512 tokens of prose, 3 epochs) | 0.536 | 0.83 | 0.410 | 0.74 | 0.430 | 0.70 |

Frozen `multilingual-e5-small` embeddings (first 256 tokens of the prose, logistic regression), on a
random subset of about 4,000 articles per language with its own split, so compare within the block only:

| Model (4k-article subset) | en F1 | fr F1 | ru F1 |
|---|---|---|---|
| Frozen e5-small + logistic regression | 0.382 | 0.378 | 0.336 |
| Structural features + LightGBM | 0.512 | 0.523 | 0.552 |
| Structural features + e5-small embeddings + LightGBM | 0.524 | 0.528 | 0.527 |
| TF-IDF + linear SVM | 0.539 | 0.505 | 0.557 |

What this shows so far:

- Article length alone already gets QWK 0.64-0.78. Quality classes are largely a size and structure
  signal, and the remaining gains come from references, headings and link density.
- Semantic embeddings of the lead section carry little of that signal. A frozen encoder is clearly
  worse than counting structure, and adding it to the structural features does not help reliably.
- Fine-tuned XLM-R base (`scripts/finetune.py`, T4 GPU, fp16, lr 2e-5, batch 16, 3 epochs, best epoch by
  validation macro-F1, same test split as the baselines) does not beat the cheap baselines in any language:
  0.536 vs 0.595 (en), 0.410 vs 0.534 (fr), 0.430 vs 0.562 (ru). Its QWK is lower too.
  The gap is largest on the smaller French and Russian sets. Validation F1 was still rising at epoch 3
  on all three, so these runs are under-trained; the model also only sees the first 512 tokens of prose,
  so it has no view of length, references or headings, which are what the baselines exploit.
  Longer training, a long-context model, or feeding structural features alongside the text may change
  the picture, but none of that has been tried. One seed per language, no confidence intervals.
- There is no comparison with the current ORES/Lift Wing `articlequality` model yet.

## Reproducing

```sh
pip install -e .
python scripts/prepare_data.py                 # extract corpora from commit a8a849c
python scripts/run_baselines.py                # length, structural, TF-IDF (about 40 min on 4 CPUs)
python scripts/embed.py --max-len 256 --max-docs 4000
python scripts/run_embeddings.py
python scripts/finetune.py --lang en --model xlm-roberta-base   # GPU, about 50 min for en on a T4
```

Results are written as JSON to `results/`.

## Limits

- One labelled revision per article, dated between roughly 2008 and 2016, so results describe that period.
- Lead-section embeddings were limited to 256 tokens to fit CPU time.
- The original 2017 LSTM evaluation fitted the vocabulary separately on the test set, so its accuracies
  from that time are not comparable with the numbers here.

## License

GPL-2.0, see `LICENSE`.
