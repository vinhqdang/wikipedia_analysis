"""Monthly/quarterly estimates from the scored sample.

    python -m ai_text.analyze
Writes results/summary.json and results/quarterly.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
FPR_TARGET = 0.01
CALIBRATION_END = "2022-01"  # months before this fix the threshold; 2022-01..2022-10 check the FPR
CHATGPT = "2022-12"


def quarter(month):
    y, m = month.split("-")
    return f"{y}Q{(int(m) - 1) // 3 + 1}"


def rogan_gladen(flag_rate, fpr, tpr):
    return np.clip((flag_rate - fpr) / (tpr - fpr), 0, 1)


def bootstrap_rate(flags, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    flags = np.asarray(flags, dtype=float)
    draws = rng.choice(flags, size=(n_boot, len(flags))).mean(axis=1)
    return flags.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    RESULTS.mkdir(exist_ok=True)
    sc = pd.read_parquet(DATA / "scores.parquet")
    ref = pd.read_parquet(DATA / "reference.parquet")
    sc["quarter"] = sc["month"].map(quarter)
    ns0 = sc[sc["ns"] == 0]

    calib = ns0[ns0["month"] < CALIBRATION_END]
    tau = float(np.quantile(calib["score"], FPR_TARGET))  # low score = machine-like
    holdout = ns0[(ns0["month"] >= CALIBRATION_END) & (ns0["month"] < CHATGPT)]
    fpr_holdout = float((holdout["score"] < tau).mean())
    human_ref = ns0[ns0["month"] < CHATGPT]
    tpr = {g: float((s["score"] < tau).mean()) for g, s in ref.groupby("generator")}

    rows = []
    for q, g in ns0.groupby("quarter"):
        rate, lo, hi = bootstrap_rate(g["score"] < tau)
        rows.append({"quarter": q, "n": len(g), "flagged": rate, "lo": lo, "hi": hi})
    q = pd.DataFrame(rows)
    tpr_mid = float(np.median(list(tpr.values())))
    fpr_use = max(fpr_holdout, FPR_TARGET)
    q["prevalence_est"] = rogan_gladen(q["flagged"], fpr_use, tpr_mid)
    q.to_csv(RESULTS / "quarterly.csv", index=False)

    docs = pd.read_parquet(DATA / "docs.parquet")
    docs["quarter"] = docs["month"].map(quarter)
    summary = {
        "threshold": tau, "calibration_n": len(calib), "holdout_n": len(holdout), "fpr_holdout": fpr_holdout,
        "tpr_by_generator": tpr, "n_scored": len(sc),
        "post_chatgpt_flag_rate": float((ns0[ns0["month"] >= CHATGPT]["score"] < tau).mean()),
        "pre_chatgpt_flag_rate": float((human_ref["score"] < tau).mean()),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
