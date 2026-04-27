#!/usr/bin/env python3
"""
One-shot backfill for data/rtings_best_of_history.csv.

Reconstructs RTINGS Best-Of TV picks for each known confirmation date
using the changelog scraped from
https://www.rtings.com/tv/reviews/best/tvs-on-the-market.

Source dates (from `recommendation_updates`):
  2025-07-17  big change — current 6-category structure established;
              Best Bright Room + Home Theater categories dropped
  2025-10-27  small change — Mid-Range LG B4 → TCL QM8K, Cheap Q651G → QD6QF
  2025-12-12  confirmed unchanged
  2026-02-04  confirmed unchanged
  2026-03-27  confirmed unchanged

Pre-2025-07-17 not backfilled — Mid-Range pick prior to the July update
is ambiguous (changelog only mentions models replaced *in* July,
nothing about earlier periods), and we have no anchor confirmation date
to attach a snapshot to.

Notable Mentions are NOT backfilled — RTINGS only logs "we updated the
Notable Mentions" without specifying what changed.

Idempotent: run repeatedly without duplication. Existing snapshot rows
for the target dates are dropped and re-inserted.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_CSV = ROOT / "data" / "tv_database_with_prices.csv"
HIST_CSV = ROOT / "data" / "rtings_best_of_history.csv"

CATEGORIES = [
    "Best TV",
    "Best Upper Mid-Range TV",
    "Best Mid-Range TV",
    "Best Lower Mid-Range TV",
    "Best Budget TV",
    "Best Cheap TV",
]

# Each historical configuration: list of fullnames in CATEGORIES order.
JULY_CONFIG = [
    "Samsung S95F OLED",
    "Samsung S90F OLED",
    "LG B4 OLED",
    "TCL QM7K",
    "TCL QM6K",
    "TCL Q651G",
]
OCT_CONFIG = [
    "Samsung S95F OLED",
    "Samsung S90F OLED",
    "TCL QM8K",
    "TCL QM7K",
    "TCL QM6K",
    "Hisense QD6QF",
]

# date → list of fullnames per category
SNAPSHOTS = {
    "2025-07-17": JULY_CONFIG,
    "2025-10-27": OCT_CONFIG,
    "2025-12-12": OCT_CONFIG,
    "2026-02-04": OCT_CONFIG,
    "2026-03-27": OCT_CONFIG,
}


def build_rows() -> pd.DataFrame:
    db = pd.read_csv(DB_CSV)
    lookup = db.set_index("fullname")[
        ["product_id", "url_part"]
    ].to_dict("index")

    rows = []
    for date, config in SNAPSHOTS.items():
        for rank, fullname in enumerate(config, start=1):
            entry = lookup.get(fullname, {})
            rows.append({
                "snapshot_date": date,
                "rank": rank,
                "category": CATEGORIES[rank - 1],
                "subtitle": "",  # not preserved historically
                "product_id": entry.get("product_id"),
                "fullname": fullname,
                "url_part": entry.get("url_part", ""),
                "review_url": f"/tv/reviews/{entry.get('url_part','').replace('-', '/', 1)}"
                              if entry.get("url_part") else "",
                "is_mention": False,
            })
    return pd.DataFrame(rows)


def main() -> None:
    backfill = build_rows()
    target_dates = set(SNAPSHOTS.keys())

    if HIST_CSV.exists():
        existing = pd.read_csv(HIST_CSV)
        # Drop any rows for the dates we're about to write so we're idempotent
        kept = existing[~existing["snapshot_date"].isin(target_dates)]
        combined = pd.concat([kept, backfill], ignore_index=True)
    else:
        combined = backfill

    combined = combined.sort_values(["snapshot_date", "is_mention", "rank"])
    combined.to_csv(HIST_CSV, index=False)

    print(f"Wrote {len(backfill)} backfilled rows for {len(target_dates)} dates")
    print(f"Total history rows: {len(combined)}")
    summary = combined.groupby("snapshot_date").size()
    print("\nRows per snapshot:")
    for date, n in summary.items():
        print(f"  {date}: {n}")


if __name__ == "__main__":
    main()
