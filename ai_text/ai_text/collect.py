"""Sample page creations on English Wikipedia and fetch the text of each creation revision.

For every month from 2018-09 (when the creation log starts) we draw random hours, list the
page-creation log events in those hours for main (ns 0) and Draft (ns 118) pages, then fetch
the creation revision (text, size, tags) and the page's current state (a creation revision that can no longer be fetched means the page was deleted). Output: data/raw/YYYY-MM.parquet, one file per month, resumable.

    python -m ai_text.collect --start 2018-09 --end 2026-09 --hours 12
"""
import argparse
import calendar
import datetime as dt
import random
import time
from pathlib import Path

import pandas as pd

from .wikiapi import api

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
NAMESPACES = {0, 118}
TEXT_CAP = 15000


def hour_windows(month, n, seed):
    year, mon = map(int, month.split("-"))
    hours = calendar.monthrange(year, mon)[1] * 24
    rng = random.Random(f"{seed}-{month}")
    start = dt.datetime(year, mon, 1)
    return [start + dt.timedelta(hours=h) for h in sorted(rng.sample(range(hours), n))]


def creation_events(start):
    end = start + dt.timedelta(hours=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    out, cont = [], {}
    while True:
        r = api(list="logevents", letype="create", lestart=end.strftime(fmt), leend=start.strftime(fmt),
                ledir="older", lelimit=500, leprop="ids|title|timestamp|user", **cont)
        for e in r.get("query", {}).get("logevents", []):
            if e.get("ns") in NAMESPACES and e.get("revid") and e.get("logpage"):
                out.append({"revid": e["revid"], "pageid": e["logpage"], "title": e["title"], "ns": e["ns"],
                            "created": e["timestamp"], "user": e.get("user")})
        if "continue" not in r:
            return out
        cont = r["continue"]


def batches(xs, n=50):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def fetch_revisions(revids):
    got = {}
    for chunk in batches(revids):
        r = api(prop="revisions", revids="|".join(map(str, chunk)), rvprop="ids|size|content|tags", rvslots="main")
        for p in r.get("query", {}).get("pages", []):
            for rev in p.get("revisions", []):
                got[rev["revid"]] = {"size": rev.get("size"), "tags": ",".join(rev.get("tags", [])),
                                     "text": (rev.get("slots", {}).get("main", {}).get("content") or "")[:TEXT_CAP]}
    return got


def fetch_state(pageids):
    got = {}
    for chunk in batches(sorted(set(pageids))):
        r = api(prop="info", pageids="|".join(map(str, chunk)))
        for p in r.get("query", {}).get("pages", []):
            got[p["pageid"]] = {"exists_now": not p.get("missing", False), "redirect_now": bool(p.get("redirect")),
                                "ns_now": p.get("ns"), "length_now": p.get("length"), "title_now": p.get("title")}
    return got


def collect_month(month, hours, seed):
    events = []
    for w in hour_windows(month, hours, seed):
        events += creation_events(w)
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events).drop_duplicates("revid")
    revs = fetch_revisions(df["revid"].tolist())
    df["has_text"] = df["revid"].isin(revs)
    for col in ("size", "tags", "text"):
        df[col] = df["revid"].map(lambda r: revs.get(r, {}).get(col))
    df["month"] = month
    return df


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-09")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--hours", type=int, default=8)
    ap.add_argument("--seed", default="2026")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    pre = list(months(args.start, "2021-12"))[::2]  # sparser baseline period
    post = list(months("2022-01", args.end))
    for month in pre + post:
        path = RAW / f"{month}.parquet"
        if path.exists():
            continue
        t = time.time()
        df = collect_month(month, args.hours, args.seed)
        df.to_parquet(path, index=False)
        print(month, len(df), int(df["has_text"].sum()) if len(df) else 0, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
