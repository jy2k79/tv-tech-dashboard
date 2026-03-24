#!/usr/bin/env python3
"""
RTINGS Data Scraper
===================
Scrapes product review data from RTINGS.com's internal API endpoints.
Supports multiple product silos (TV, Monitor) via silo_config.py.

Endpoints used (all POST, JSON body, no auth required):
  - table_tool__products_list  — Product identity, brand, slugs, SKUs, bench version
  - table_tool__test_results   — Measured values and scores per test
  - table_tool__ratings        — Usage/performance aggregate scores
  - table_tool__prices         — Retail pricing per SKU

Requirements:
    pip install httpx pandas openpyxl

Usage:
    python rtings_scraper.py              # Default: scrape TVs
    python rtings_scraper.py --silo tv    # Explicit: scrape TVs
    python rtings_scraper.py --silo monitor  # Scrape monitors
"""

import argparse
import re
import httpx
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from silo_config import get_silo, TV

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "https://www.rtings.com/api/v2/safe"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# RTINGS member session cookie — unblurs measurements and scores
# Set via RTINGS_SESSION env var or .env file. ~30 day expiry, refresh by
# logging into rtings.com and copying _rtings_session cookie value.
RTINGS_SESSION = os.getenv("RTINGS_SESSION", "")
if RTINGS_SESSION:
    HEADERS["Cookie"] = f"_rtings_session={RTINGS_SESSION}"

# Delay between API requests (seconds) — be polite to RTINGS
REQUEST_DELAY = 2.0

# Legacy module-level constants — kept for backwards compatibility with
# any code that imports them directly. Pipeline code should use silo_config.
TEST_IDS = TV["test_ids"]
USAGE_IDS = TV["usage_ids"]
FALLBACK_BENCH_IDS = TV["fallback_bench_ids"]
OUTPUT_DIR = TV["paths"]["scraped_csv"].parent
SPD_DIR = TV["paths"]["spd_images"]


# =============================================================================
# API HELPERS
# =============================================================================
def api_post(endpoint: str, variables: dict, client: httpx.Client) -> dict:
    """Make a POST request to an RTINGS API endpoint."""
    url = f"{BASE_URL}/{endpoint}"
    payload = {"variables": variables}
    resp = client.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# DYNAMIC BENCH DISCOVERY
# =============================================================================
def discover_bench_ids(client: httpx.Client, silo_cfg: dict) -> list[str]:
    """Discover test bench IDs at or above the silo's minimum version.

    Calls the column_options endpoint to read the list of test benches,
    then filters to those at or above silo_cfg["min_bench_version"].
    Falls back to silo_cfg["fallback_bench_ids"] on error.
    """
    VERSION_PATTERN = re.compile(r'^v(\d+)\.(\d+)(?:\.(\d+))?$')
    min_version = tuple(silo_cfg["min_bench_version"])
    fallback = list(silo_cfg["fallback_bench_ids"])

    try:
        print(f"Discovering test bench IDs for {silo_cfg['display_name']}...")
        silo = fetch_column_options(client, silo_cfg)
        benches = silo.get("test_benches", [])

        matched_ids = []
        for bench in benches:
            display_name = bench.get("display_name", "")
            match = VERSION_PATTERN.match(display_name)
            if not match:
                continue
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
            if version >= min_version:
                matched_ids.append(str(bench["id"]))
                print(f"  Found bench: {display_name} (ID {bench['id']})")

        if not matched_ids:
            print(f"  WARNING: No v{'.'.join(str(v) for v in min_version)}+ benches found — using fallback")
            return fallback

        print(f"  Discovered {len(matched_ids)} bench IDs: {matched_ids}")
        return matched_ids

    except Exception as e:
        print(f"  WARNING: Bench discovery failed ({e}) — using fallback")
        return fallback


# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_products(client: httpx.Client, bench_ids: list[str]) -> list[dict]:
    """Fetch all products for the given test bench IDs."""
    print(f"Fetching product list for bench IDs: {bench_ids}...")
    data = api_post("table_tool__products_list", {
        "test_bench_ids": bench_ids,
        "named_version": "public",
        "is_admin": False,
    }, client)
    products = data["data"]["products"]
    print(f"  -> {len(products)} products found")
    return products


def fetch_test_results(client: httpx.Client, bench_ids: list[str],
                       silo_cfg: dict) -> list[dict]:
    """Fetch test results for all key measurements."""
    test_ids = list(silo_cfg["test_ids"].keys())
    print(f"Fetching test results for {len(test_ids)} tests...")
    time.sleep(REQUEST_DELAY)
    data = api_post("table_tool__test_results", {
        "test_bench_ids": bench_ids,
        "original_ids": test_ids,
        "named_version": "public",
        "is_admin": False,
    }, client)
    results = data["data"]["test_results"]
    print(f"  -> {len(results)} test result records")
    return results


def fetch_ratings(client: httpx.Client, bench_ids: list[str],
                  silo_cfg: dict) -> list[dict]:
    """Fetch usage/performance ratings."""
    usage_ids = list(silo_cfg["usage_ids"].keys())
    print(f"Fetching ratings for {len(usage_ids)} usage/performance scores...")
    time.sleep(REQUEST_DELAY)
    data = api_post("table_tool__ratings", {
        "test_bench_ids": bench_ids,
        "original_ids": usage_ids,
        "named_version": "public",
    }, client)
    ratings = data["data"]["ratings"]
    print(f"  -> {len(ratings)} rating records")
    return ratings


def fetch_column_options(client: httpx.Client, silo_cfg: dict) -> dict:
    """Fetch the full column/test schema for a silo."""
    print(f"Fetching column options for {silo_cfg['silo_url_part']}...")
    time.sleep(REQUEST_DELAY)
    data = api_post("table_tool__column_options", {
        "silo_url_part": silo_cfg["silo_url_part"],
        "named_version": "public",
    }, client)
    return data["data"]["silo"]


# =============================================================================
# DATA ASSEMBLY
# =============================================================================
def build_product_records(products: list[dict]) -> dict[str, dict]:
    """Build base product records keyed by product ID."""
    records = {}
    for p in products:
        pid = p["id"]
        bench = p["review"]["test_bench"]
        records[pid] = {
            "product_id": pid,
            "fullname": p["fullname"],
            "brand": p["brand_name"],
            "url_part": p["full_url_part"],
            "review_url": f"https://www.rtings.com{p['page']['url']}",
            "image_url": p.get("image", ""),
            "test_bench_id": bench["id"],
            "test_bench_version": bench["display_name"],
            "released_at": p.get("approximate_released_at", ""),
            "last_updated_at": p.get("last_updated_at", ""),
            "first_published_at": p["page"].get("first_published_at", ""),
            "reviewed_sku_id": p.get("reviewed_sku_id", ""),
            "sizes_available": "; ".join(
                f"{s['variation']} ({s['name']})" for s in p.get("variant_skus", [])
            ),
            "sku_count": len(p.get("variant_skus", [])),
        }
    return records


def merge_test_results(records: dict[str, dict], test_results: list[dict],
                       silo_cfg: dict):
    """Merge test results into product records."""
    test_id_map = silo_cfg["test_ids"]
    for tr in test_results:
        pid = tr["product_id"]
        if pid not in records:
            continue
        oid = tr["original_id"]
        col_name = test_id_map.get(oid, f"test_{oid}")

        if tr["kind"] == "picture":
            # For picture-type tests, store the asset URL
            records[pid][col_name] = tr.get("asset_url", "")
            if tr.get("asset_thumb_url"):
                records[pid][f"{col_name}_thumb"] = tr["asset_thumb_url"]
        elif tr["kind"] in ("number",):
            records[pid][col_name] = tr.get("value")
            records[pid][f"{col_name}_score"] = tr.get("score")
            records[pid][f"{col_name}_display"] = tr.get("rendered_value", "")
        elif tr["kind"] == "word":
            records[pid][col_name] = tr.get("value", "")
        else:
            records[pid][col_name] = tr.get("rendered_value") or tr.get("value", "")


def merge_ratings(records: dict[str, dict], ratings: list[dict],
                  silo_cfg: dict):
    """Merge usage/performance ratings into product records."""
    usage_id_map = silo_cfg["usage_ids"]
    for r in ratings:
        pid = r["product_id"]
        if pid not in records:
            continue
        oid = r["original_id"]
        col_name = usage_id_map.get(oid, f"usage_{oid}")
        records[pid][col_name] = r.get("score")


# =============================================================================
# SPD IMAGE DOWNLOAD
# =============================================================================
def download_spd_images(records: dict[str, dict], output_dir: Path,
                        old_bench_map: dict[str, str] | None = None):
    """Download SPD images for all products that have them.

    If old_bench_map is provided, delete cached SPD images for any product
    whose test_bench_id changed (forces re-download with new bench data).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Invalidate cached SPDs for products that moved to a new bench
    invalidated = 0
    if old_bench_map:
        for pid, rec in records.items():
            old_bench = old_bench_map.get(pid)
            new_bench = str(rec.get("test_bench_id", ""))
            if old_bench and new_bench and old_bench != new_bench:
                safe_name = rec["url_part"].replace("/", "-")
                cached = output_dir / f"{safe_name}_spd.jpg"
                if cached.exists():
                    cached.unlink()
                    invalidated += 1
                    print(f"  Invalidated cached SPD: {rec['fullname']} (bench {old_bench} → {new_bench})")
        if invalidated:
            print(f"  Invalidated {invalidated} cached SPD images due to bench changes")

    spd_products = [
        (pid, r) for pid, r in records.items()
        if r.get("spd_image") and r["spd_image"].strip()
    ]
    print(f"\nDownloading SPD images for {len(spd_products)} products...")

    downloaded = 0
    skipped = 0
    stale = 0
    failed = 0

    with httpx.Client(headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://www.rtings.com/",
    }) as client:
        for i, (pid, rec) in enumerate(spd_products):
            url = rec["spd_image"]
            safe_name = rec["url_part"].replace("/", "-")
            filename = f"{safe_name}_spd.jpg"
            filepath = output_dir / filename

            if filepath.exists():
                # Staleness check: compare cached file size with remote Content-Length
                try:
                    head = client.head(url, timeout=10)
                    remote_size = int(head.headers.get("content-length", 0))
                    local_size = filepath.stat().st_size
                    if remote_size > 0 and abs(remote_size - local_size) > 100:
                        print(f"  Stale SPD detected: {rec['fullname']} "
                              f"(local={local_size}B, remote={remote_size}B)")
                        filepath.unlink()
                        stale += 1
                    else:
                        skipped += 1
                        rec["spd_image_local"] = str(filepath)
                        continue
                except Exception:
                    # HEAD failed — keep cached version
                    skipped += 1
                    rec["spd_image_local"] = str(filepath)
                    continue

            try:
                print(f"  [{i+1}/{len(spd_products)}] {rec['fullname']}...")
                resp = client.get(url, timeout=15)
                resp.raise_for_status()
                filepath.write_bytes(resp.content)
                rec["spd_image_local"] = str(filepath)
                downloaded += 1
                time.sleep(1.5)  # Be polite
            except Exception as e:
                print(f"    ERROR: {e}")
                rec["spd_image_local"] = ""
                failed += 1

    print(f"  Downloaded: {downloaded}, Skipped (cached): {skipped}, "
          f"Stale (re-downloaded): {stale}, Failed: {failed}")


# =============================================================================
# OUTPUT
# =============================================================================
def save_outputs(records: dict[str, dict], silo_cfg: dict,
                 bench_ids: list[str] | None = None):
    """Save scraped data to CSV, JSON, and Excel."""
    paths = silo_cfg["paths"]
    output_dir = paths["scraped_csv"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(records.values())

    # Sort by brand, then fullname
    df = df.sort_values(["brand", "fullname"]).reset_index(drop=True)

    # Add metadata
    df["scraped_at"] = datetime.now(timezone.utc).isoformat()

    # Column ordering: identity first, then scores, then measurements
    identity_cols = [
        "product_id", "fullname", "brand", "url_part", "review_url",
        "test_bench_id", "test_bench_version",
        "released_at", "last_updated_at", "first_published_at",
        "sizes_available", "sku_count",
    ]
    panel_cols = [
        "panel_type", "backlight_type", "resolution", "native_refresh_rate",
        "dimming_zone_count",
    ]
    usage_vals = set(silo_cfg["usage_ids"].values())
    usage_cols = sorted([c for c in df.columns if c in usage_vals])
    # Everything else
    known = set(identity_cols + panel_cols + usage_cols + ["scraped_at", "image_url", "reviewed_sku_id"])
    other_cols = sorted([c for c in df.columns if c not in known])

    ordered_cols = identity_cols + panel_cols + usage_cols + other_cols + ["scraped_at"]
    # Only include columns that exist
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    # Add any remaining columns
    remaining = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)
    df = df[ordered_cols]

    # Save CSV
    csv_path = paths["scraped_csv"]
    df.to_csv(csv_path, index=False)
    print(f"  CSV:   {csv_path} ({len(df)} rows)")

    # Save timestamped snapshot
    snapshot_path = csv_path.with_name(f"{csv_path.stem}_{timestamp}.csv")
    df.to_csv(snapshot_path, index=False)
    print(f"  Snapshot: {snapshot_path}")

    # Save JSON (for programmatic access)
    json_path = paths["scraped_json"]
    json_records = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(json_path, "w") as f:
        json.dump({
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "silo": silo_cfg["name"],
            "test_bench_ids": bench_ids or silo_cfg["fallback_bench_ids"],
            "total_products": len(json_records),
            "products": json_records,
        }, f, indent=2)
    print(f"  JSON:  {json_path}")

    # Save Excel
    xlsx_path = paths["scraped_xlsx"]
    sheet_name = f"RTINGS {silo_cfg['display_name']} Data"
    df.to_excel(xlsx_path, index=False, sheet_name=sheet_name[:31])
    print(f"  Excel: {xlsx_path}")

    return df


# =============================================================================
# SUMMARY
# =============================================================================
def print_summary(df: pd.DataFrame, silo_cfg: dict):
    """Print a summary of the scraped data."""
    label = silo_cfg["display_name"]
    min_ver = ".".join(str(v) for v in silo_cfg["min_bench_version"])

    print("\n" + "=" * 70)
    print(f"SCRAPE SUMMARY — {label}")
    print("=" * 70)

    print(f"\nTotal {label} (v{min_ver}+ only): {len(df)}")
    print(f"\nBy test bench version:")
    for ver, count in df["test_bench_version"].value_counts().items():
        print(f"  {ver}: {count}")

    print(f"\nBy brand:")
    for brand, count in df["brand"].value_counts().items():
        print(f"  {brand}: {count}")

    print(f"\nBy panel type:")
    if "panel_type" in df.columns:
        for pt, count in df["panel_type"].value_counts(dropna=False).items():
            print(f"  {pt}: {count}")

    print(f"\nBy backlight type:")
    if "backlight_type" in df.columns:
        for bt, count in df["backlight_type"].value_counts(dropna=False).items():
            print(f"  {bt}: {count}")

    if "spd_image" in df.columns:
        has_spd = df["spd_image"].notna() & (df["spd_image"] != "")
        print(f"\nSPD images available: {has_spd.sum()} / {len(df)}")

    # Print the first usage score column as a sanity check
    first_usage = next(iter(silo_cfg["usage_ids"].values()), None)
    if first_usage and first_usage in df.columns:
        col = df[first_usage].dropna()
        if len(col) > 0:
            print(f"\n{first_usage} score range: {col.min():.1f} - {col.max():.1f}")
            print(f"{first_usage} mean: {col.mean():.1f}")

    if "dimming_zone_count" in df.columns:
        zones = pd.to_numeric(df["dimming_zone_count"], errors="coerce").dropna()
        if len(zones) > 0:
            print(f"\nDimming zone count range: {zones.min():.0f} - {zones.max():.0f}")


# =============================================================================
# MAIN
# =============================================================================
def main(silo_cfg: dict | None = None):
    """Run the scraper for a given silo (defaults to TV)."""
    if silo_cfg is None:
        silo_cfg = TV

    paths = silo_cfg["paths"]
    label = silo_cfg["display_name"]
    min_ver = ".".join(str(v) for v in silo_cfg["min_bench_version"])
    spd_dir = paths["spd_images"]

    print("=" * 70)
    print(f"RTINGS {label} Data Scraper")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Session cookie: {'configured' if RTINGS_SESSION else 'NOT SET — data will be blurred'}")
    print("=" * 70)

    # Load old data to detect bench changes (for SPD cache invalidation)
    old_csv = paths["scraped_csv"]
    old_bench_map = {}
    if old_csv.exists():
        try:
            old_df = pd.read_csv(old_csv, usecols=["product_id", "test_bench_id"])
            old_bench_map = dict(zip(
                old_df["product_id"].astype(str),
                old_df["test_bench_id"].astype(str),
            ))
            print(f"Loaded old bench map for {len(old_bench_map)} products")
        except Exception as e:
            print(f"WARNING: Could not load old bench map: {e}")

    with httpx.Client(headers=HEADERS) as client:
        # Step 0: Discover bench IDs dynamically
        bench_ids = discover_bench_ids(client, silo_cfg)
        print(f"\nTarget: {label} Test Bench v{min_ver}+ (IDs: {bench_ids})")

        # Step 1: Fetch products
        products = fetch_products(client, bench_ids)

        # Step 2: Fetch test results
        test_results = fetch_test_results(client, bench_ids, silo_cfg)

        # Step 3: Fetch ratings
        ratings = fetch_ratings(client, bench_ids, silo_cfg)

    # Check if data came back unblurred (session cookie working)
    n_unblurred_ratings = sum(1 for r in ratings if r.get("unblurred"))
    n_blurred_ratings = sum(1 for r in ratings if not r.get("unblurred"))
    session_ok = n_unblurred_ratings > 0
    if session_ok:
        print(f"\nSession OK: {n_unblurred_ratings}/{len(ratings)} ratings unblurred")
    else:
        print(f"\nWARNING: All {n_blurred_ratings} ratings are blurred — session cookie expired or missing")
    # Write flag for weekly_update.py to read
    paths["session_flag"].parent.mkdir(parents=True, exist_ok=True)
    paths["session_flag"].write_text("1" if session_ok else "0")

    # Step 4: Assemble records
    print("\nAssembling product records...")
    records = build_product_records(products)
    merge_test_results(records, test_results, silo_cfg)
    merge_ratings(records, ratings, silo_cfg)
    print(f"  -> {len(records)} complete product records")

    # Step 5: Download SPD images (invalidate cache for bench-changed products)
    download_spd_images(records, spd_dir, old_bench_map=old_bench_map)

    # Step 6: Save outputs
    print("\nSaving output files...")
    df = save_outputs(records, silo_cfg, bench_ids=bench_ids)

    # Step 7: Summary
    print_summary(df, silo_cfg)

    # Step 8: Save raw API responses for debugging/reference
    raw_dir = paths["raw_api"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "products.json", "w") as f:
        json.dump(products, f, indent=2)
    with open(raw_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    with open(raw_dir / "ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)
    print(f"\nRaw API responses saved to: {raw_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    from silo_config import SILOS

    parser = argparse.ArgumentParser(description="RTINGS Data Scraper")
    parser.add_argument(
        "--silo", default="tv", choices=list(SILOS.keys()),
        help="Product silo to scrape (default: tv)",
    )
    args = parser.parse_args()
    main(silo_cfg=get_silo(args.silo))
