"""Fine-tune a multilingual transformer on one language. Meant for a GPU.

    python scripts/finetune.py --lang en --model xlm-roberta-base --epochs 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikiquality import data, features  # noqa: E402
from wikiquality.metrics import evaluate  # noqa: E402


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=data.LANGS)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N documents (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(data.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = data.load(args.lang)
    if args.limit:
        df = df.sample(args.limit, random_state=data.SEED).reset_index(drop=True)
    train, val, test = data.split(df, test_size=0.2, val_size=0.1)
    texts = [features.plain_text(features.scrub(t))[:8000] for t in df["wikitext"]]
    y = df["y"].to_numpy()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(data.CLASS_ORDER[args.lang])).to(device)

    def make_loader(idx, shuffle):
        enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
        enc["labels"] = torch.tensor(y[idx])
        items = [{k: v[i] for k, v in enc.items()} for i in range(len(idx))]
        return DataLoader(items, batch_size=args.batch_size, shuffle=shuffle, collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})

    train_dl, val_dl, test_dl = make_loader(train, True), make_loader(val, False), make_loader(test, False)
    use_cuda = device == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_cuda and amp_dtype == torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, best_state = -1, None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_cuda):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
        val_f1 = evaluate(y[val], predict(model, val_dl, device))["macro_f1"]
        print(f"epoch {epoch + 1}: val macro-F1 {val_f1:.4f}", flush=True)
        if val_f1 > best:
            best, best_state = val_f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = evaluate(y[test], predict(model, test_dl, device))
    print(json.dumps(res, indent=2))
    out = data.ROOT / "results"
    out.mkdir(exist_ok=True)
    name = args.model.split("/")[-1]
    (out / f"finetune_{name}_{args.lang}.json").write_text(json.dumps({"val_macro_f1": best, "test": res, "args": vars(args)}, indent=2))


if __name__ == "__main__":
    main()
