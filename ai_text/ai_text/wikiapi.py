"""Minimal polite MediaWiki API client (serial, honours Retry-After)."""
import time

import requests

UA = {"User-Agent": "wikipedia_analysis-research/0.2 (https://github.com/vinhqdang/wikipedia_analysis; serial reads, low volume)"}
_session = requests.Session()
_session.headers.update(UA)


def api(lang="en", **params):
    params.update(action="query", format="json", formatversion=2)
    url = f"https://{lang}.wikipedia.org/w/api.php"
    for attempt in range(12):
        try:
            r = _session.get(url, params=params, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("retry-after", 5)) + 2 * attempt)
            continue
        r.raise_for_status()
        time.sleep(0.3)
        return r.json()
    raise RuntimeError("API kept failing")
