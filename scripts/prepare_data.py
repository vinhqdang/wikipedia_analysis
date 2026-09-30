"""Build data/processed/{en,fr,ru}.parquet from the 2016-2017 text dumps.

The raw article text is no longer in the working tree. It is kept in the git
tag `legacy-2017`; by default this script extracts it from there.

    python scripts/prepare_data.py                 # extract from the tag
    python scripts/prepare_data.py --source DIR    # DIR contains {en,fr,ru}wiki/
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikiquality.data import LANGS, PROCESSED, ROOT  # noqa: E402

TAG = "legacy-2017"


def extract_from_tag(dest):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = f"git -C {ROOT} archive {TAG} lang_model | tar -x -C {dest}"
    subprocess.run(cmd, shell=True, check=True)
    return dest / "lang_model"


def build(lang, source):
    d = source / f"{lang}wiki"
    ids = sorted(int(p.name) for p in (d / "text").iterdir() if p.name.isdigit())
    labels = (d / f"{lang}wikilabel").read_text().splitlines()
    if len(ids) != len(labels):
        raise ValueError(f"{lang}: {len(ids)} documents but {len(labels)} labels")
    texts = [(d / "text" / str(i)).read_text(encoding="utf-8", errors="replace") for i in ids]
    return pd.DataFrame({"lang": lang, "doc_id": ids, "label": labels, "wikitext": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="directory holding enwiki/, frwiki/, ruwiki/")
    args = ap.parse_args()
    source = args.source or extract_from_tag(ROOT / "data" / "raw")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        df = build(lang, source)
        df.to_parquet(PROCESSED / f"{lang}.parquet", index=False)
        print(lang, len(df), df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
