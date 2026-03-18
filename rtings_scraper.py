#!/usr/bin/env python3
"""
RTINGS TV Data Scraper
======================
Scrapes TV review data from RTINGS.com's internal API endpoints.
Only ingests TVs reviewed with Test Bench v2.0 or higher.

Endpoints used (all POST, JSON body, no auth required):
  - table_tool__products_list  — TV identity, brand, slugs, SKUs, bench version
  - table_tool__test_results   — Measured values and scores per test
  - table_tool__ratings        — Usage/performance aggregate scores
  - table_tool__prices         — Retail pricing per SKU

Requirements:
    pip install httpx pandas openpyxl

Usage:
    python rtings_scraper.py
"""

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

# Fallback bench IDs if dynamic discovery fails
# 210 = v2.1, 197 = v2.0.1, 227 = v2.2
FALLBACK_BENCH_IDS = ["227", "210", "197"]

# Delay between API requests (seconds) — be polite to RTINGS
REQUEST_DELAY = 2.0

# Output paths
OUTPUT_DIR = Path("data")
SPD_DIR = Path("spd_images")

# =============================================================================
# TEST IDS — key measurements to pull from test_results endpoint
# =============================================================================
# These are the original_id values from RTINGS' test schema.
# Discovered via the table_tool__column_options endpoint.

TEST_IDS = {
    # Identity / panel info
    "217": "panel_type",            # LCD, OLED, etc.
    "216": "panel_sub_type",        # QD-OLED, WOLED, IPS, VA, etc.
    "215": "backlight_type",        # Full-Array, Edge-Lit, etc.
    "208": "resolution",            # 4k, 8k
    "219": "native_refresh_rate",   # 60Hz, 120Hz, 144Hz

    # Brightness
    "141": "hdr_peak_2pct_nits",        # Peak 2% Window (HDR)
    "461": "hdr_peak_10pct_nits",       # Peak 10% Window (HDR)
    "462": "hdr_peak_25pct_nits",       # Peak 25% Window (HDR)
    "96":  "hdr_peak_50pct_nits",       # Peak 50% Window (HDR)
    "463": "hdr_peak_100pct_nits",      # Peak 100% Window (HDR)
    "609": "sdr_peak_2pct_nits",        # Peak 2% Window (SDR)
    "610": "sdr_peak_10pct_nits",       # Peak 10% Window (SDR)
    "619": "sdr_real_scene_peak_nits",  # Real Scene Peak (SDR)

    # Contrast / black level
    "11":  "native_contrast",
    "647": "contrast_ratio",

    # Dimming
    "16981": "dimming_zone_count",

    # Color
    "28334": "sdr_dci_p3_coverage_pct",    # CIELAB DCI-P3 Coverage
    "28336": "sdr_bt2020_coverage_pct",    # CIELAB BT.2020 Coverage
    "927":   "hdr_bt2020_coverage_itp_pct", # 10,000 cd/m² BT.2020 Coverage ITP

    # Response time
    "30862": "first_response_time_ms",
    "30860": "total_response_time_ms",

    # Input lag
    "12237": "input_lag_1080p_ms",  # 1080p @ Max Refresh Rate
    "12239": "input_lag_4k_ms",     # 4k @ Max Refresh Rate

    # Connectivity
    "487":  "hdmi_ports",
    "2684": "hdmi_21_speed",

    # VRR
    "2190": "vrr_support",
    "5039": "hdmi_forum_vrr",

    # SPD image (picture type — returns asset_url)
    "26239": "spd_image",

    # Backlight chart image
    "133": "backlight_chart",
}

# Usage/performance score IDs
USAGE_IDS = {
    "1":     "mixed_usage",
    "12":    "home_theater",
    "32477": "bright_room",
    "9":     "sports",
    "10":    "gaming",
    "32475": "brightness_score",
    "32565": "black_level_score",
    "32566": "color_score",
    "32496": "game_mode_responsiveness",
}


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
def discover_v2_bench_ids(client: httpx.Client) -> list[str]:
    """Discover all v2.0+ test bench IDs from the RTINGS API.

    Calls the column_options endpoint (already implemented at fetch_column_options)
    to read the list of test benches, then filters to v2.0+.
    Falls back to FALLBACK_BENCH_IDS on error.
    """
    VERSION_PATTERN = re.compile(r'^v(\d+)\.(\d+)(?:\.(\d+))?$')
    MIN_VERSION = (2, 0, 0)

    try:
        print("Discovering test bench IDs from RTINGS API...")
        silo = fetch_column_options(client)
        benches = silo.get("test_benches", [])

        v2_ids = []
        for bench in benches:
            display_name = bench.get("display_name", "")
            match = VERSION_PATTERN.match(display_name)
            if not match:
                continue
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
            if version >= MIN_VERSION:
                v2_ids.append(str(bench["id"]))
                print(f"  Found bench: {display_name} (ID {bench['id']})")

        if not v2_ids:
            print("  WARNING: No v2.0+ benches found — falling back to FALLBACK_BENCH_IDS")
            return list(FALLBACK_BENCH_IDS)

        print(f"  Discovered {len(v2_ids)} v2.0+ bench IDs: {v2_ids}")
        return v2_ids

    except Exception as e:
        print(f"  WARNING: Bench discovery failed ({e}) — falling back to FALLBACK_BENCH_IDS")
        return list(FALLBACK_BENCH_IDS)


# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_products(client: httpx.Client, bench_ids: list[str]) -> list[dict]:
    """Fetch all TV products for the given test bench IDs."""
    print(f"Fetching product list for bench IDs: {bench_ids}...")
    data = api_post("table_tool__products_list", {
        "test_bench_ids": bench_ids,
        "named_version": "public",
        "is_admin": False,
    }, client)
    products = data["data"]["products"]
    print(f"  -> {len(products)} products found")
    return products


def fetch_test_results(client: httpx.Client, bench_ids: list[str]) -> list[dict]:
    """Fetch test results for all key measurements."""
    test_ids = list(TEST_IDS.keys())
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


def fetch_ratings(client: httpx.Client, bench_ids: list[str]) -> list[dict]:
    """Fetch usage/performance ratings."""
    usage_ids = list(USAGE_IDS.keys())
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


def fetch_column_options(client: httpx.Client) -> dict:
    """Fetch the full column/test schema for reference."""
    print("Fetching column options schema...")
    time.sleep(REQUEST_DELAY)
    data = api_post("table_tool__column_options", {
        "silo_url_part": "tv",
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


def merge_test_results(records: dict[str, dict], test_results: list[dict]):
    """Merge test results into product records."""
    for tr in test_results:
        pid = tr["product_id"]
        if pid not in records:
            continue
        oid = tr["original_id"]
        col_name = TEST_IDS.get(oid, f"test_{oid}")

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


def merge_ratings(records: dict[str, dict], ratings: list[dict]):
    """Merge usage/performance ratings into product records."""
    for r in ratings:
        pid = r["product_id"]
        if pid not in records:
            continue
        oid = r["original_id"]
        col_name = USAGE_IDS.get(oid, f"usage_{oid}")
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

    print(f"  Downloaded: {downloaded}, Skipped (cached): {skipped}, Failed: {failed}")


# =============================================================================
# OUTPUT
# =============================================================================
def save_outputs(records: dict[str, dict], output_dir: Path, bench_ids: list[str] | None = None):
    """Save scraped data to CSV, JSON, and Excel."""
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
    usage_cols = sorted([c for c in df.columns if c in USAGE_IDS.values()])
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
    csv_path = output_dir / "rtings_tv_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV:   {csv_path} ({len(df)} rows)")

    # Save timestamped snapshot
    snapshot_path = output_dir / f"rtings_tv_data_{timestamp}.csv"
    df.to_csv(snapshot_path, index=False)
    print(f"  Snapshot: {snapshot_path}")

    # Save JSON (for programmatic access)
    json_path = output_dir / "rtings_tv_data.json"
    # Convert to serializable format
    json_records = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(json_path, "w") as f:
        json.dump({
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "test_bench_ids": bench_ids or FALLBACK_BENCH_IDS,
            "total_products": len(json_records),
            "products": json_records,
        }, f, indent=2)
    print(f"  JSON:  {json_path}")

    # Save Excel
    xlsx_path = output_dir / "rtings_tv_data.xlsx"
    df.to_excel(xlsx_path, index=False, sheet_name="RTINGS TV Data")
    print(f"  Excel: {xlsx_path}")

    return df


# =============================================================================
# SUMMARY
# =============================================================================
def print_summary(df: pd.DataFrame):
    """Print a summary of the scraped data."""
    print("\n" + "=" * 70)
    print("SCRAPE SUMMARY")
    print("=" * 70)

    print(f"\nTotal TVs (v2.0+ only): {len(df)}")
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

    if "mixed_usage" in df.columns:
        print(f"\nMixed Usage score range: {df['mixed_usage'].min():.1f} - {df['mixed_usage'].max():.1f}")
        print(f"Mixed Usage mean: {df['mixed_usage'].mean():.1f}")

    if "dimming_zone_count" in df.columns:
        zones = pd.to_numeric(df["dimming_zone_count"], errors="coerce").dropna()
        if len(zones) > 0:
            print(f"\nDimming zone count range: {zones.min():.0f} - {zones.max():.0f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("RTINGS TV Data Scraper")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Session cookie: {'configured' if RTINGS_SESSION else 'NOT SET — data will be blurred'}")
    print("=" * 70)

    # Load old data to detect bench changes (for SPD cache invalidation)
    old_csv = OUTPUT_DIR / "rtings_tv_data.csv"
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
        # Step 0: Discover v2.0+ bench IDs dynamically
        bench_ids = discover_v2_bench_ids(client)
        print(f"\nTarget: Test Bench v2.0+ (IDs: {bench_ids})")

        # Step 1: Fetch products
        products = fetch_products(client, bench_ids)

        # Step 2: Fetch test results
        test_results = fetch_test_results(client, bench_ids)

        # Step 3: Fetch ratings
        ratings = fetch_ratings(client, bench_ids)

    # Check if data came back unblurred (session cookie working)
    n_unblurred_ratings = sum(1 for r in ratings if r.get("unblurred"))
    n_blurred_ratings = sum(1 for r in ratings if not r.get("unblurred"))
    session_ok = n_unblurred_ratings > 0
    if session_ok:
        print(f"\nSession OK: {n_unblurred_ratings}/{len(ratings)} ratings unblurred")
    else:
        print(f"\nWARNING: All {n_blurred_ratings} ratings are blurred — session cookie expired or missing")
    # Write flag for weekly_update.py to read
    (OUTPUT_DIR / ".session_ok").write_text("1" if session_ok else "0")

    # Step 4: Assemble records
    print("\nAssembling product records...")
    records = build_product_records(products)
    merge_test_results(records, test_results)
    merge_ratings(records, ratings)
    print(f"  -> {len(records)} complete product records")

    # Step 5: Download SPD images (invalidate cache for bench-changed products)
    download_spd_images(records, SPD_DIR, old_bench_map=old_bench_map)

    # Step 6: Save outputs
    print("\nSaving output files...")
    df = save_outputs(records, OUTPUT_DIR, bench_ids=bench_ids)

    # Step 7: Summary
    print_summary(df)

    # Step 8: Save raw API responses for debugging/reference
    raw_dir = OUTPUT_DIR / "raw"
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
    main()
