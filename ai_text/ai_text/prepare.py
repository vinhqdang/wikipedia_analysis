"""Turn raw creation-revision dumps into the analysis table and the detector input.

    python -m ai_text.prepare
"""
import re
from pathlib import Path

import pandas as pd

from .lexical import marker_counts
from .text import n_words, plain_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
MIN_WORDS = 150
PER_MONTH = {0: 100, 118: 60}
_REDIRECT = re.compile(r"^\s*#\s*(?:redirect|REDIRECT)", re.I)


def build():
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("*.parquet"))]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df["has_text"] = df["has_text"].astype(bool)
    df["text"] = df["text"].fillna("")
    df["is_redirect"] = df["text"].map(lambda t: bool(_REDIRECT.match(t)))
    prose = df["text"].map(plain_text)
    df["prose"] = prose.str.slice(0, 6000)
    df["n_words"] = prose.map(n_words)
    counts = df["prose"].map(marker_counts)
    df["n_marker_words"] = counts.map(lambda c: c[0])
    df["n_marker_phrases"] = counts.map(lambda c: c[1])
    df["year"] = df["month"].str.slice(0, 4).astype(int)
    return df.drop(columns=["text"])


def main():
    df = build()
    df.to_parquet(OUT / "docs.parquet", index=False)
    eligible = df[df["has_text"] & ~df["is_redirect"] & (df["n_words"] >= MIN_WORDS)]
    parts = []
    for (month, ns), g in eligible.groupby(["month", "ns"]):
        parts.append(g.sample(n=min(len(g), PER_MONTH[ns]), random_state=2026))
    sel = pd.concat(parts, ignore_index=True)
    sel[["revid", "month", "ns", "title", "prose", "n_words"]].to_parquet(OUT / "detect_input.parquet", index=False)
    print(len(df), "events;", len(eligible), "eligible;", len(sel), "selected for detection")


if __name__ == "__main__":
    main()
