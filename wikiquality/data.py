"""Corpus definitions, loading and train/test splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# Quality classes from lowest to highest quality. The order of the French and
# Russian scales is our reading of the wikiproject assessment scales and is
# only used for the ordinal metrics; accuracy and macro-F1 do not depend on it.
CLASS_ORDER = {
    "en": ["stub", "start", "c", "b", "ga", "fa"],
    "fr": ["e", "bd", "b", "ba", "a", "adq"],
    "ru": ["IV", "III", "II", "I", "GA", "FA", "SA"],
}
LANGS = list(CLASS_ORDER)
SEED = 2017


def load(lang):
    """Return the processed corpus of one language as a DataFrame.

    Columns: lang, doc_id, label, y (index in CLASS_ORDER), wikitext.
    """
    path = PROCESSED / f"{lang}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run scripts/prepare_data.py first")
    df = pd.read_parquet(path)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASS_ORDER[lang])})
    return df


def split(df, test_size=0.2, val_size=0.0, seed=SEED):
    """Stratified train/(val)/test split. Returns index arrays (positional)."""
    idx = range(len(df))
    train, test = train_test_split(idx, test_size=test_size, stratify=df["y"], random_state=seed)
    if val_size:
        train, val = train_test_split(
            train, test_size=val_size, stratify=df["y"].iloc[train], random_state=seed
        )
        return list(train), list(val), list(test)
    return list(train), list(test)
