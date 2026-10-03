"""Headline numbers, lexical series and the figure for the report.

    python -m ai_text.report
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .analyze import CHATGPT, DATA, RESULTS, bootstrap_rate, quarter

INK, MUTED, GRID, ACCENT = "#1f2933", "#6b7785", "#e3e7ea", "#2a6f97"


def marker_flag(df):
    return (df["n_marker_phrases"] > 0) | (df["n_marker_words"] >= 2)


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    tau = summary["threshold"]
    sc = pd.read_parquet(DATA / "scores.parquet")
    docs = pd.read_parquet(DATA / "docs.parquet")
    m = sc.merge(docs[["revid", "n_marker_words", "n_marker_phrases"]], on="revid")
    m = m[m["ns"] == 0].copy()
    m["quarter"] = m["month"].map(quarter)
    m["flag"] = m["score"] < tau
    m["marker"] = marker_flag(m)

    pre, post = m[m["month"] < CHATGPT], m[m["month"] >= "2024-01"]
    out = {"tau": tau, "n_pre": len(pre), "n_post_2024plus": len(post)}
    for name, g in (("pre", pre), ("post2024", post)):
        rate, lo, hi = bootstrap_rate(g["flag"])
        mk, mlo, mhi = bootstrap_rate(g["marker"])
        out[name] = {"flag_rate": rate, "flag_ci": [lo, hi], "marker_rate": mk, "marker_ci": [mlo, mhi],
                     "score_median": float(g["score"].median())}
    out["ks_pre_vs_2024plus"] = dict(zip(("stat", "p"), map(float, stats.ks_2samp(pre["score"], post["score"]))))
    # TPR on real-world text that would still allow 5% prevalence (upper CI of the post flag rate)
    fpr = max(summary["fpr_holdout"], 0.01)
    hi = out["post2024"]["flag_ci"][1]
    out["tpr_needed_for_5pct_prevalence"] = float((hi - 0.95 * fpr) / 0.05)
    out["prevalence_if_tpr"] = {str(t): float(np.clip((out["post2024"]["flag_rate"] - fpr) / (t - fpr), 0, 1))
                                for t in (0.9, 0.7, 0.5, 0.3)}
    mk_pos = post[post["marker"]]
    out["marker_positive_post"] = {"n": len(mk_pos), "flag_rate": float(mk_pos["flag"].mean()),
                                   "median_score": float(mk_pos["score"].median()),
                                   "other_flag_rate": float(post[~post["marker"]]["flag"].mean())}
    (RESULTS / "headline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    m["half"] = m["month"].str[:4] + "H" + np.where(m["month"].str[5:7].astype(int) <= 6, "1", "2")
    rows = []
    for h, g in m.groupby("half"):
        r = {"half": h, "n": len(g)}
        for col in ("flag", "marker"):
            rate, lo, hi = bootstrap_rate(g[col])
            r.update({col: rate, col + "_lo": lo, col + "_hi": hi})
        rows.append(r)
    hs = pd.DataFrame(rows)
    hs.to_csv(RESULTS / "halfyearly.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    x = np.arange(len(hs))
    labels = list(hs["half"])
    cut = labels.index("2022H2") + 0.5
    for ax, col, title in ((axes[0], "flag", "Flagged by Binoculars (1% false-positive threshold)"),
                           (axes[1], "marker", "Contains LLM-style wording")):
        y = 100 * hs[col]
        err = np.vstack([y - 100 * hs[col + "_lo"], 100 * hs[col + "_hi"] - y])
        ax.errorbar(x, y, yerr=err, color=ACCENT, lw=2, marker="o", ms=5, mfc=ACCENT, mec="white", mew=1,
                    elinewidth=1, ecolor=ACCENT, capsize=0)
        ax.axvline(cut, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xticks([i for i, l in enumerate(labels) if l.endswith("H1")], [l[:4] for l in labels if l.endswith("H1")])
        ax.set_ylim(0, 8)
        ax.set_ylabel("% of new articles")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        ax.text(cut + 0.15, 7.6, "after ChatGPT", color=MUTED, fontsize=8.5, va="top")
    axes[0].axhline(1, color=MUTED, lw=1)
    axes[0].text(0.0, 7.6, "line at 1%: expected from human text alone", color=MUTED, fontsize=8.5, va="top")
    fig.suptitle("New English Wikipedia articles, half-yearly samples (n = %d, 95%% bootstrap intervals)" % len(m), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "halfyearly.png", dpi=160)


if __name__ == "__main__":
    main()
