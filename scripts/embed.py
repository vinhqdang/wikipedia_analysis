"""Frozen multilingual encoder embeddings (mean pooled) of the scrubbed article lead.

    python scripts/embed.py --langs en fr ru --max-docs 10000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikiquality import data, features  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"


def embed(texts, tok, model, max_len, batch_size):
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        ids = order[start : start + batch_size]
        enc = tok(["query: " + texts[i] for i in ids], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1)
        out[ids] = torch.nn.functional.normalize(pooled, dim=-1).numpy()
        if (start // batch_size) % 20 == 0:
            print(f"  {start}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=data.LANGS)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-docs", type=int, default=0, help="stratified cap per language (0 = all)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    for lang in args.langs:
        df = data.load(lang)
        if args.max_docs and len(df) > args.max_docs:
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.sample(n=round(args.max_docs * len(g) / len(df)), random_state=data.SEED)
            ).sort_values("doc_id")
        texts = [features.plain_text(features.scrub(t))[:6000] for t in df["wikitext"]]
        print(lang, len(texts), flush=True)
        emb = embed(texts, tok, model, args.max_len, args.batch_size)
        np.savez(data.PROCESSED / f"emb_{lang}.npz", doc_id=df["doc_id"].to_numpy(), emb=emb)


if __name__ == "__main__":
    main()
