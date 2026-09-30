"""Length, structural-feature and TF-IDF baselines, with and without label-leak scrubbing.

    python scripts/run_baselines.py --langs en fr ru
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikiquality import data, features  # noqa: E402
from wikiquality.metrics import evaluate  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = data.ROOT / "results"
MAX_CHARS = 30000


def structural_features(df, lang, scrubbed):
    cache = data.PROCESSED / f"struct_{lang}_{'scrub' if scrubbed else 'raw'}.parquet"
    if cache.exists():
        import pandas as pd

        return pd.read_parquet(cache)
    texts = df["wikitext"].map(features.scrub) if scrubbed else df["wikitext"]
    feats = features.structural_frame(texts)
    feats.to_parquet(cache)
    return feats


def run(lang):
    df = data.load(lang)
    train, test = data.split(df)
    y = df["y"].to_numpy()
    out = {}
    for scrubbed in (False, True):
        tag = "scrubbed" if scrubbed else "raw"
        feats = structural_features(df, lang, scrubbed)
        res = {}

        x = np.log1p(feats[["n_chars"]].to_numpy())
        clf = LogisticRegression(max_iter=1000).fit(x[train], y[train])
        res["length_only"] = evaluate(y[test], clf.predict(x[test]))

        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=data.SEED, verbose=-1)
        clf.fit(feats.iloc[train], y[train])
        res["structural_lgbm"] = evaluate(y[test], clf.predict(feats.iloc[test]))

        texts = df["wikitext"].map(features.scrub) if scrubbed else df["wikitext"]
        texts = texts.str.slice(0, MAX_CHARS)
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=300000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|\{\{|\[\[|==")
        xt = tfidf.fit_transform(texts.iloc[train])
        svm = LinearSVC(C=0.5).fit(xt, y[train])
        res["tfidf_linear_svm"] = evaluate(y[test], svm.predict(tfidf.transform(texts.iloc[test])))

        out[tag] = res
        print(lang, tag, {k: round(v["macro_f1"], 3) for k, v in res.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=data.LANGS)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for lang in args.langs:
        res = run(lang)
        (RESULTS / f"baselines_{lang}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
