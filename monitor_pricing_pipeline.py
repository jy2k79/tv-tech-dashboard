#!/usr/bin/env python3
"""
Monitor Pricing Pipeline — Amazon (Keepa) + Best Buy
=====================================================
Reuses RTINGS API extraction and pricing functions from pricing_pipeline.py
with monitor-specific configuration (sizes, model patterns, merge logic).

Usage:
    python monitor_pricing_pipeline.py
"""

import os
import re
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Import shared functions from TV pricing pipeline
from pricing_pipeline import (
    fetch_rtings_prices as _fetch_rtings_prices_tv,
    extract_amazon_asin,
    extract_bestbuy_sku,
    build_product_retailer_map,
    fetch_keepa_prices,
    fetch_bestbuy_prices,
    RTINGS_API,
)
from silo_config import MONITOR

import httpx
import time

DATA_DIR = Path("data")

# Valid monitor sizes (inches)
VALID_MONITOR_SIZES = {
    24, 25, 27, 28, 30, 32, 34, 38, 40, 42, 45, 49, 55, 57,
}

# Screen area lookup for monitors (diagonal inches → m²)
# Must account for different aspect ratios:
#   16:9 → W/H = 16/9
#   21:9 (ultrawide) → W/H = 21/9
#   32:9 (super ultrawide) → W/H = 32/9
# area = diag² × (W/H) / (W² + H²) × W (in m²)
def _screen_area_m2(diag_inches, aspect_w=16, aspect_h=9):
    """Calculate screen area in m² from diagonal and aspect ratio."""
    diag_m = diag_inches * 0.0254
    ratio = aspect_w / aspect_h
    width_m = diag_m * ratio / math.sqrt(1 + ratio**2)
    height_m = diag_m / math.sqrt(1 + ratio**2)
    return width_m * height_m


# Pre-compute common monitor sizes with their typical aspect ratios
MONITOR_SCREEN_AREA_M2 = {
    # 16:9 monitors
    24: _screen_area_m2(24),
    25: _screen_area_m2(25),
    27: _screen_area_m2(27),
    28: _screen_area_m2(28),
    30: _screen_area_m2(30),
    32: _screen_area_m2(32),
    40: _screen_area_m2(40),
    42: _screen_area_m2(42),
    45: _screen_area_m2(45),
    55: _screen_area_m2(55),
    # 21:9 ultrawide
    34: _screen_area_m2(34, 21, 9),
    38: _screen_area_m2(38, 21, 9),
    # 32:9 super ultrawide
    49: _screen_area_m2(49, 32, 9),
    57: _screen_area_m2(57, 32, 9),
}


def _get_monitor_bench_ids() -> list[str]:
    """Derive bench IDs from monitor scraper output."""
    csv_path = MONITOR["paths"]["scraped_csv"]
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["test_bench_id"])
            ids = sorted(df["test_bench_id"].dropna().astype(int).unique().astype(str).tolist())
            if ids:
                print(f"  Bench IDs from monitor scraper: {ids}")
                return ids
        except Exception as e:
            print(f"  WARNING: Could not read bench IDs: {e}")
    fallback = MONITOR["fallback_bench_ids"]
    print(f"  Using fallback bench IDs: {fallback}")
    return list(fallback)


def fetch_rtings_monitor_prices():
    """Fetch affiliate link data from RTINGS API for monitor products."""
    print("Fetching RTINGS monitor affiliate links...")
    bench_ids = _get_monitor_bench_ids()
    with httpx.Client(timeout=30) as client:
        resp_us = client.post(
            f"{RTINGS_API}table_tool__prices",
            json={"variables": {"test_bench_ids": bench_ids, "country_id": "2"}}
        )
        us_data = resp_us.json()["data"]["test_bench_prices"]
        print(f"  US prices: {len(us_data)} records")

        time.sleep(2)

        resp_ca = client.post(
            f"{RTINGS_API}table_tool__prices",
            json={"variables": {"test_bench_ids": bench_ids, "country_id": "1"}}
        )
        ca_data = resp_ca.json()["data"]["test_bench_prices"]
        print(f"  CA prices: {len(ca_data)} records")

    return us_data, ca_data


def extract_monitor_size(text, monitor_db=None, product_id=None):
    """Extract screen size from text or monitor database.

    Monitors often have size in their measured display_size field rather than
    in the model number, so we prefer the database value when available.
    """
    # Try database first
    if monitor_db is not None and product_id is not None:
        row = monitor_db[monitor_db['product_id'].astype(str) == str(product_id)]
        if len(row) > 0:
            ds = row.iloc[0].get('display_size')
            if pd.notna(ds):
                try:
                    size = round(float(ds))
                    if size in VALID_MONITOR_SIZES:
                        return size
                except (ValueError, TypeError):
                    pass

    if not text:
        return None
    text = unquote(str(text))

    # Pattern 1: XX-Inch, XX Inch, XX", Class XX
    for pattern in [
        r'(\d{2,3})[\s-]*(?:inch|in\b|"|\'\'|class)',
        r'class[\s-]*(\d{2,3})',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            size = int(match.group(1))
            if size in VALID_MONITOR_SIZES:
                return size

    # Pattern 2: Monitor model patterns with size
    # e.g. S27FG810, AW2725DF (27"), PG32UCDM (32"), U4025QW (40")
    # Look for 2-digit size after brand prefix letters
    match = re.search(r'(?:^|[\s/])(?:[A-Z]{1,3})(\d{2})', text, re.IGNORECASE)
    if match:
        size = int(match.group(1))
        if size in VALID_MONITOR_SIZES:
            return size

    return None


def detect_aspect_ratio(monitor_db, product_id):
    """Detect aspect ratio for a monitor product from database."""
    row = monitor_db[monitor_db['product_id'].astype(str) == str(product_id)]
    if len(row) > 0:
        ar = row.iloc[0].get('aspect_ratio', '')
        if pd.notna(ar):
            ar_str = str(ar)
            if '32:9' in ar_str:
                return (32, 9)
            if '21:9' in ar_str:
                return (21, 9)
    return (16, 9)


def merge_monitor_pricing(retailer_df, keepa_df, bestbuy_df):
    """Merge pricing data into monitor database."""
    print("\nMerging monitor pricing data...")

    monitor_db = pd.read_csv(MONITOR["paths"]["database"])
    print(f"  Monitor database: {len(monitor_db)} monitors")

    # Merge Keepa data
    if len(keepa_df) > 0:
        retailer_df = retailer_df.merge(
            keepa_df, on='amazon_asin', how='left', suffixes=('', '_keepa')
        )

    # Merge Best Buy data
    if len(bestbuy_df) > 0:
        retailer_df = retailer_df.merge(
            bestbuy_df, on='bestbuy_sku', how='left', suffixes=('', '_bb')
        )

    # Extract sizes — prefer database display_size, then parse from titles
    for idx, row in retailer_df.iterrows():
        if pd.notna(row.get('size_inches')):
            continue
        pid = str(row['product_id'])
        size = extract_monitor_size(None, monitor_db, pid)
        if size:
            retailer_df.at[idx, 'size_inches'] = size
            continue
        for col in ['keepa_title', 'bb_name']:
            if col in retailer_df.columns:
                size = extract_monitor_size(row.get(col))
                if size:
                    retailer_df.at[idx, 'size_inches'] = size
                    break

    # Build per-size pricing table
    price_rows = []
    for _, row in retailer_df.iterrows():
        size = row.get('size_inches')
        pid = str(row['product_id'])

        avail = str(row.get('availability', ''))
        is_out_of_stock = avail == 'out_of_stock'

        # Collect current sale prices
        candidates = []
        if 'buy_box_price' in row and pd.notna(row.get('buy_box_price')) and row['buy_box_price'] > 0:
            candidates.append(('amazon_buy_box', row['buy_box_price']))
        if 'amazon_price' in row and pd.notna(row.get('amazon_price')) and row['amazon_price'] > 0:
            candidates.append(('amazon', row['amazon_price']))
        if 'new_3p_price' in row and pd.notna(row.get('new_3p_price')) and row['new_3p_price'] > 0:
            candidates.append(('amazon_3p', row['new_3p_price']))
        if 'bb_sale_price' in row and pd.notna(row.get('bb_sale_price')) and row['bb_sale_price'] > 0:
            candidates.append(('bestbuy', row['bb_sale_price']))

        if not candidates and pd.notna(row.get('rtings_price')) and row['rtings_price'] > 0:
            candidates.append(('rtings', row['rtings_price']))

        best_price = None
        price_source = None
        if candidates and not is_out_of_stock:
            price_source, best_price = min(candidates, key=lambda x: x[1])

        # Compute price per m² using correct aspect ratio
        price_per_m2 = None
        if best_price and pd.notna(size):
            size_int = int(size)
            if size_int in MONITOR_SCREEN_AREA_M2:
                area = MONITOR_SCREEN_AREA_M2[size_int]
            else:
                # Compute dynamically with detected aspect ratio
                ar_w, ar_h = detect_aspect_ratio(monitor_db, pid)
                area = _screen_area_m2(size_int, ar_w, ar_h)
            price_per_m2 = best_price / area

        price_rows.append({
            'product_id': pid,
            'sku_id': row['sku_id'],
            'size_inches': size,
            'amazon_asin': row.get('amazon_asin'),
            'bestbuy_sku': row.get('bestbuy_sku'),
            'best_price': best_price,
            'price_source': price_source,
            'amazon_price': row.get('amazon_price') if 'amazon_price' in row else None,
            'buy_box_price': row.get('buy_box_price') if 'buy_box_price' in row else None,
            'list_price': row.get('list_price') if 'list_price' in row else None,
            'bestbuy_price': row.get('bb_sale_price') if 'bb_sale_price' in row else None,
            'bestbuy_regular': row.get('bb_regular_price') if 'bb_regular_price' in row else None,
            'rtings_price': row.get('rtings_price'),
            'price_per_m2': price_per_m2,
            'price_date': datetime.now().strftime('%Y-%m-%d'),
        })

    prices_df = pd.DataFrame(price_rows)

    # Save detailed price table
    prices_out = DATA_DIR / "monitor_prices.csv"
    prices_df.to_csv(prices_out, index=False)
    print(f"  Price records: {len(prices_df)}")

    # Append to rolling price history
    append_monitor_price_history(prices_df, monitor_db)

    # Per-product pricing summary
    # Monitors typically come in one size, so we just pick the best price
    summary_rows = []
    for pid in monitor_db['product_id'].unique():
        pid_str = str(pid)
        prod_prices = prices_df[prices_df['product_id'] == pid_str]

        if len(prod_prices) == 0:
            summary_rows.append({'product_id': pid, 'sizes_with_price': 0})
            continue

        priced_rows = prod_prices[prod_prices['best_price'].notna()]
        if len(priced_rows) == 0:
            summary_rows.append({
                'product_id': pid,
                'sizes_with_price': len(prod_prices['size_inches'].dropna().unique()),
            })
            continue

        # For monitors, most are single-size. Use the best (lowest) price.
        best_row = priced_rows.loc[priced_rows['best_price'].idxmin()]

        summary_rows.append({
            'product_id': pid,
            'price_size': best_row.get('size_inches'),
            'price_best': best_row.get('best_price'),
            'price_source': best_row.get('price_source'),
            'price_amazon': best_row.get('amazon_price'),
            'price_buybox': best_row.get('buy_box_price'),
            'price_bestbuy': best_row.get('bestbuy_price'),
            'price_list': best_row.get('list_price'),
            'price_per_m2': best_row.get('price_per_m2'),
            'sizes_with_price': len(priced_rows['size_inches'].dropna().unique()),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Merge into monitor database
    monitor_db['product_id'] = monitor_db['product_id'].astype(int)
    summary_df['product_id'] = summary_df['product_id'].astype(int)
    merged = monitor_db.merge(summary_df, on='product_id', how='left')

    # Save enriched database
    enriched_out = MONITOR["paths"]["database_with_prices"]
    merged.to_csv(enriched_out, index=False)
    print(f"\n  Enriched database saved: {enriched_out}")

    # Print summary
    print_monitor_pricing_summary(merged, prices_df)

    return merged, prices_df


def append_monitor_price_history(prices_df, monitor_db):
    """Append today's price snapshot to monitor price history."""
    history_path = MONITOR["paths"]["price_history"]
    tech_map = dict(zip(monitor_db['product_id'].astype(str), monitor_db['color_architecture']))

    snapshot = prices_df[prices_df['best_price'].notna()].copy()
    snapshot = snapshot[['product_id', 'size_inches', 'best_price', 'price_source', 'price_date']].copy()
    snapshot.rename(columns={'price_date': 'snapshot_date'}, inplace=True)
    snapshot['color_architecture'] = snapshot['product_id'].map(tech_map)
    snapshot = snapshot[['snapshot_date', 'product_id', 'size_inches', 'best_price',
                         'price_source', 'color_architecture']]

    if history_path.exists():
        existing = pd.read_csv(history_path)
        # Refresh stale color_architecture labels using current database
        existing['product_id'] = existing['product_id'].astype(str)
        existing['color_architecture'] = existing['product_id'].map(tech_map)
        combined = pd.concat([existing, snapshot], ignore_index=True)
        # Collapse multi-SKU rows to the cheapest price per (date, product, size)
        # so the history matches how monitor_database_with_prices.csv picks
        # `price_best` (min across SKUs). Before this fix, `keep='last'` picked
        # whichever SKU happened to be last in the file, overstating prices
        # when multiple SKUs existed (e.g., Apple Studio XDR 4 SKUs at 27").
        combined = (
            combined.sort_values('best_price')
                    .drop_duplicates(
                        subset=['snapshot_date', 'product_id', 'size_inches'],
                        keep='first')
        )
        combined['snapshot_date'] = pd.to_datetime(combined['snapshot_date'])
        combined['_iso_year'] = combined['snapshot_date'].dt.isocalendar().year.astype(int)
        combined['_iso_week'] = combined['snapshot_date'].dt.isocalendar().week.astype(int)
        latest_per_week = (combined.groupby(['_iso_year', '_iso_week'])['snapshot_date']
                          .max().reset_index(name='_latest'))
        combined = combined.merge(latest_per_week, on=['_iso_year', '_iso_week'])
        combined = combined[combined['snapshot_date'] == combined['_latest']]
        combined = combined.drop(columns=['_iso_year', '_iso_week', '_latest'])
        combined['snapshot_date'] = combined['snapshot_date'].dt.strftime('%Y-%m-%d')
        combined.to_csv(history_path, index=False)
    else:
        snapshot.to_csv(history_path, index=False)

    print(f"  Monitor price history updated: {history_path} ({len(snapshot)} rows)")


def print_monitor_pricing_summary(merged, prices_df):
    """Print monitor pricing summary by technology."""
    print(f"\n{'='*70}")
    print("MONITOR PRICING SUMMARY")
    print("=" * 70)

    priced = merged[merged['price_best'].notna()]
    print(f"\nMonitors with pricing: {len(priced)}/{len(merged)}")

    if len(priced) == 0:
        return

    print(f"\nPrice by color_architecture:")
    print(f"  {'Technology':<15s} {'Count':>5s} {'Min':>8s} {'Median':>8s} {'Max':>8s} {'$/m²':>8s}")
    print(f"  {'─'*15} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for tech in ['QD-OLED', 'WOLED', 'QD-LCD', 'KSF', 'WLED']:
        t = priced[priced['color_architecture'] == tech]
        if len(t) == 0:
            continue
        prices = t['price_best'].dropna()
        m2 = t['price_per_m2'].dropna()
        m2_str = f"${m2.median():>7.0f}" if len(m2) > 0 else "    N/A"
        print(f"  {tech:<15s} {len(t):>5d} ${prices.min():>7.0f} ${prices.median():>7.0f} "
              f"${prices.max():>7.0f} {m2_str}")


def main():
    print("=" * 70)
    print("MONITOR PRICING PIPELINE")
    print("=" * 70)

    # Step 1: Extract retailer IDs from RTINGS
    us_data, ca_data = fetch_rtings_monitor_prices()
    retailer_df = extract_all_monitor_ids(us_data, ca_data)

    # Step 2: Keepa (Amazon) prices — write to monitor-specific file
    keepa_df = fetch_keepa_prices(retailer_df,
                                   output_path=DATA_DIR / "monitor_keepa_prices.csv")

    # Step 3: Best Buy prices — write to monitor-specific file
    bestbuy_df = fetch_bestbuy_prices(retailer_df,
                                      output_path=DATA_DIR / "monitor_bestbuy_prices.csv")

    # Step 4: Merge
    merge_monitor_pricing(retailer_df, keepa_df, bestbuy_df)


def extract_all_monitor_ids(us_data, ca_data):
    """Build and save the retailer ID mapping for monitors."""
    # build_product_retailer_map is already imported — it's the generic part
    products = build_product_retailer_map(us_data, ca_data)

    rows = []
    for pid, skus in products.items():
        for sid, sku in skus.items():
            # Extract size from SKU variation name
            size = None
            for entry in us_data + ca_data:
                if str(entry['product_id']) == pid and str(entry['sku_id']) == sid:
                    variation = entry.get('variation', '')
                    size = extract_monitor_size(variation)
                    break

            rows.append({
                'product_id': pid,
                'sku_id': sid,
                'size_inches': size,
                'amazon_asin': sku['amazon_asin'],
                'bestbuy_sku': sku['bestbuy_sku'],
                'rtings_price': sku['price_rtings'],
                'availability': sku['availability'],
            })

    retailer_df = pd.DataFrame(rows)

    n_amz = retailer_df['amazon_asin'].notna().sum()
    n_bb = retailer_df['bestbuy_sku'].notna().sum()
    n_products = retailer_df['product_id'].nunique()
    print(f"\n  Total SKU variants: {len(retailer_df)}")
    print(f"  Products: {n_products}")
    print(f"  Amazon ASINs: {n_amz} SKUs")
    print(f"  Best Buy SKUs: {n_bb} SKUs")

    retailer_out = DATA_DIR / "monitor_product_retailer_ids.csv"
    retailer_df.to_csv(retailer_out, index=False)
    print(f"  Saved: {retailer_out}")

    return retailer_df


if __name__ == '__main__':
    main()
