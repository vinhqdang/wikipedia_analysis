"""Run on a GPU: generate machine-written references, then Binoculars-score everything.

    python scripts/detect.py --n-ref 300
Reads data/detect_input.parquet; writes data/scores.parquet and data/reference.parquet.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_text.binoculars import Binoculars  # noqa: E402
from ai_text.generate_reference import generate  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
GENERATORS = ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen3-4B-Instruct-2507"]
LEAD_WORDS = 250


def lead(text):
    return " ".join(text.split()[:LEAD_WORDS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=300)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    docs = pd.read_parquet(DATA / "detect_input.parquet")
    docs["text"] = docs["prose"].map(lead)

    ref_path = DATA / "reference.parquet"
    if not args.skip_generation:
        pre = docs[(docs["month"] < "2022-06") & (docs["ns"] == 0)].sample(args.n_ref, random_state=1)
        rows = []
        for model in GENERATORS:
            outs = generate(model, pre["title"].tolist())
            rows += [{"generator": model, "title": t, "text": lead(o)} for t, o in zip(pre["title"], outs)]
            print("generated", model, flush=True)
        pd.DataFrame(rows).to_parquet(ref_path, index=False)
    ref = pd.read_parquet(ref_path)

    det = Binoculars()
    ref["score"] = det.score(ref["text"].tolist())
    ref.to_parquet(ref_path, index=False)
    print("reference scored", flush=True)
    docs["score"] = det.score(docs["text"].tolist())
    docs[["revid", "month", "ns", "n_words", "score"]].to_parquet(DATA / "scores.parquet", index=False)
    print("scored", len(docs))


if __name__ == "__main__":
    main()
