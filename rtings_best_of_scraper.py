#!/usr/bin/env python3
"""
RTINGS Best-Of TV List Scraper
==============================
Captures RTINGS' "Best TVs of YYYY" recommendation list — main 6 picks
(Best, Upper Mid-Range, Mid-Range, Lower Mid-Range, Budget, Cheap) plus
Notable Mentions — and writes:
  - data/rtings_best_of_tvs.csv      current snapshot (overwritten each run)
  - data/rtings_best_of_history.csv  append-only weekly history

The list is published at https://www.rtings.com/tv/reviews/best/tvs-on-the-market
and updated frequently. The page hydrates a Next.js-style data blob into a
`data-props` attribute; we parse that JSON rather than scraping rendered DOM.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

URL = "https://www.rtings.com/tv/reviews/best/tvs-on-the-market"
DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_CSV = DATA_DIR / "rtings_best_of_tvs.csv"
HISTORY_CSV = DATA_DIR / "rtings_best_of_history.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15",
    "Accept": "text/html,application/xhtml+xml",
}
_session = os.getenv("RTINGS_SESSION", "")
if _session:
    HEADERS["Cookie"] = f"_rtings_session={_session}"


def fetch_page() -> str:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as c:
        r = c.get(URL)
        r.raise_for_status()
        return r.text


def extract_recommendation_blob(page_html: str) -> dict:
    """Find the data-props blob that contains page_data.page.recommendation.

    There are several `<div data-part="..." data-props="...">` blocks on the
    page; the one carrying the recommendation list is the only one with all
    six recommendation IDs. We pick the largest data-props blob that parses
    as JSON and contains `product_recommendations`.
    """
    candidates = re.findall(r'data-props="([^"]+)"', page_html)
    best = None
    for raw in candidates:
        decoded = html.unescape(raw)
        if "product_recommendations" not in decoded:
            continue
        try:
            obj = json.loads(decoded)
        except json.JSONDecodeError:
            continue
        if best is None or len(decoded) > best[1]:
            best = (obj, len(decoded))
    if best is None:
        raise RuntimeError("Could not locate recommendation data-props blob")
    return best[0]["page_data"]["page"]["recommendation"]


def _derive_url_part(product: dict) -> str:
    """Mentions don't carry `full_url_part` like main picks do — derive it
    from `page.url` (e.g. "/tv/reviews/lg/g5-oled" → "lg-g5-oled") so the
    join with `tv_database_with_prices.csv` works for both pick types.
    """
    if product.get("full_url_part"):
        return product["full_url_part"]
    page_url = (product.get("page") or {}).get("url", "")
    m = re.match(r"/tv/reviews/(.+?)/?$", page_url)
    return m.group(1).replace("/", "-") if m else ""


def extract_picks(rec: dict, snapshot_date: str) -> list[dict]:
    rows: list[dict] = []
    for rank, item in enumerate(rec.get("product_recommendations", []), start=1):
        product = item.get("product") or {}
        rows.append({
            "snapshot_date": snapshot_date,
            "rank": rank,
            "category": item.get("title", ""),
            "subtitle": item.get("subtitle", ""),
            "product_id": product.get("id"),
            "fullname": product.get("fullname", ""),
            "url_part": _derive_url_part(product),
            "review_url": (product.get("page") or {}).get("url", ""),
            "is_mention": False,
        })
    for rank, item in enumerate(rec.get("recommendation_mentions", []), start=1):
        product = item.get("product") or {}
        rows.append({
            "snapshot_date": snapshot_date,
            "rank": rank,
            "category": "Notable Mention",
            "subtitle": item.get("text", "") or item.get("subtitle", ""),
            "product_id": product.get("id"),
            "fullname": product.get("fullname", ""),
            "url_part": _derive_url_part(product),
            "review_url": (product.get("page") or {}).get("url", ""),
            "is_mention": True,
        })
    return rows


def append_history(snapshot_rows: list[dict]) -> None:
    """Append today's snapshot to the rolling history. Dedup keeps last per
    (snapshot_date, category, rank) so re-running the same day overwrites."""
    snap = pd.DataFrame(snapshot_rows)
    if HISTORY_CSV.exists():
        existing = pd.read_csv(HISTORY_CSV)
        combined = pd.concat([existing, snap], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["snapshot_date", "category", "rank"], keep="last"
        )
    else:
        combined = snap
    combined.to_csv(HISTORY_CSV, index=False)


def main() -> int:
    print("=" * 70)
    print("RTINGS Best-Of TVs Scraper")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Session cookie: {'configured' if _session else 'NOT SET'}")
    print("=" * 70)

    page = fetch_page()
    print(f"Fetched {len(page):,} bytes from {URL}")

    rec = extract_recommendation_blob(page)
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    rows = extract_picks(rec, snapshot_date)
    if not rows:
        print("ERROR: no recommendations parsed from page")
        return 1

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_CSV, index=False)
    append_history(rows)

    main_picks = df[~df["is_mention"]]
    mentions = df[df["is_mention"]]
    print(f"\nMain picks ({len(main_picks)}):")
    for _, r in main_picks.iterrows():
        print(f"  {r['rank']}. {r['category']:30}  {r['fullname']}")
    print(f"\nNotable mentions ({len(mentions)}):")
    for _, r in mentions.iterrows():
        print(f"  - {r['fullname']}")

    print(f"\nSaved snapshot: {SNAPSHOT_CSV}")
    print(f"Updated history: {HISTORY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
