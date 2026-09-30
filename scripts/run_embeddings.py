"""Frozen-encoder results, compared with the classic baselines on the same documents and split.

    python scripts/run_embeddings.py --langs en fr ru
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikiquality import data, features  # noqa: E402
from wikiquality.metrics import evaluate  # noqa: E402

warnings.filterwarnings("ignore")


def run(lang):
    z = np.load(data.PROCESSED / f"emb_{lang}.npz")
    df = data.load(lang).set_index("doc_id").loc[z["doc_id"]].reset_index()
    struct = pd.read_parquet(data.PROCESSED / f"struct_{lang}_scrub.parquet")
    struct.index = data.load(lang)["doc_id"].to_numpy()
    struct = struct.loc[z["doc_id"]].reset_index(drop=True)
    emb, y = z["emb"], df["y"].to_numpy()
    train, test = data.split(df)
    res = {"n_docs": len(df)}

    clf = LogisticRegression(C=10, max_iter=2000).fit(emb[train], y[train])
    res["e5_small_frozen_lr"] = evaluate(y[test], clf.predict(emb[test]))

    lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=data.SEED, verbose=-1)
    m = lg().fit(struct.iloc[train], y[train])
    res["structural_lgbm"] = evaluate(y[test], m.predict(struct.iloc[test]))

    both = np.hstack([struct.to_numpy(), emb])
    m = lg().fit(both[train], y[train])
    res["structural_plus_e5_lgbm"] = evaluate(y[test], m.predict(both[test]))

    texts = df["wikitext"].map(features.scrub).str.slice(0, 30000)
    tf = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=300000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|\{\{|\[\[|==")
    svm = LinearSVC(C=0.5).fit(tf.fit_transform(texts.iloc[train]), y[train])
    res["tfidf_linear_svm"] = evaluate(y[test], svm.predict(tf.transform(texts.iloc[test])))
    print(lang, {k: round(v["macro_f1"], 3) for k, v in res.items() if k != "n_docs"}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=data.LANGS)
    args = ap.parse_args()
    for lang in args.langs:
        (data.ROOT / "results" / f"embeddings_{lang}.json").write_text(json.dumps(run(lang), indent=2))


if __name__ == "__main__":
    main()
