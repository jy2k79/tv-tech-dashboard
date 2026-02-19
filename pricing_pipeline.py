#!/usr/bin/env python3
"""
Pricing Pipeline — Amazon (Keepa) + Best Buy
=============================================
Extracts Amazon ASINs and Best Buy SKUs from RTINGS affiliate links,
pulls price history from Keepa API and current prices from Best Buy API,
then merges into the TV database.

Requirements:
    pip install httpx pandas python-dotenv openpyxl

Usage:
    python pricing_pipeline.py                  # Full pipeline
    python pricing_pipeline.py --extract-only   # Just extract ASINs/SKUs
    python pricing_pipeline.py --keepa-only     # Just pull Keepa data
    python pricing_pipeline.py --bestbuy-only   # Just pull Best Buy data
"""

import os
import re
import sys
import json
import time
import httpx
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

KEEPA_API_KEY = os.getenv("KEEPA_API_KEY", "")
BESTBUY_API_KEY = os.getenv("BESTBUY_API_KEY", "")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RTINGS_API = "https://www.rtings.com/api/v2/safe/"
TEST_BENCH_IDS = ["210", "197"]

# Screen area lookup (diagonal inches → m²) for price/m² calculations
# Using 16:9 aspect ratio: area = (diag * 0.0254)² * (16/√(16²+9²)) * (9/√(16²+9²))
SCREEN_AREA_M2 = {
    32: 0.22, 40: 0.34, 42: 0.38, 43: 0.40, 48: 0.50,
    50: 0.54, 55: 0.65, 58: 0.72, 60: 0.77, 65: 0.91,
    70: 1.06, 75: 1.21, 77: 1.28, 80: 1.38, 83: 1.49,
    85: 1.56, 86: 1.59, 98: 2.07, 100: 2.15,
}


# ============================================================================
# STEP 1: Extract ASINs and Best Buy SKUs from RTINGS
# ============================================================================
def fetch_rtings_prices():
    """Fetch affiliate link data from RTINGS API for all products."""
    print("Fetching RTINGS affiliate links...")
    with httpx.Client(timeout=30) as client:
        # US prices
        resp_us = client.post(
            f"{RTINGS_API}table_tool__prices",
            json={"variables": {"test_bench_ids": TEST_BENCH_IDS, "country_id": "2"}}
        )
        us_data = resp_us.json()["data"]["test_bench_prices"]
        print(f"  US prices: {len(us_data)} records")

        time.sleep(2)

        # Canada prices (for additional Amazon ASINs)
        resp_ca = client.post(
            f"{RTINGS_API}table_tool__prices",
            json={"variables": {"test_bench_ids": TEST_BENCH_IDS, "country_id": "1"}}
        )
        ca_data = resp_ca.json()["data"]["test_bench_prices"]
        print(f"  CA prices: {len(ca_data)} records")

    return us_data, ca_data


def extract_amazon_asin(tracking_url):
    """Extract ASIN from Amazon affiliate tracking URL."""
    if not tracking_url:
        return None

    decoded = unquote(tracking_url)

    # Pattern 1: /gp/product/ASIN or /dp/ASIN
    match = re.search(r'/(?:gp/product|dp)/([A-Z0-9]{10})', decoded)
    if match:
        return match.group(1)

    # Pattern 2: ASIN in query parameter
    match = re.search(r'[?&]asin=([A-Z0-9]{10})', decoded)
    if match:
        return match.group(1)

    return None


def extract_bestbuy_sku(tracking_url):
    """Extract Best Buy SKU from affiliate tracking URL."""
    if not tracking_url:
        return None

    decoded = unquote(tracking_url)

    # Pattern 1: prodsku=XXXXXXX in query string
    match = re.search(r'prodsku=(\d{6,8})', decoded)
    if match:
        return match.group(1)

    # Pattern 2: /XXXXXXX.p in URL path
    match = re.search(r'/(\d{7})\.p', decoded)
    if match:
        return match.group(1)

    return None


# Standard consumer TV sizes. Excludes 90/99 which are model numbers (QN90D), not sizes.
VALID_TV_SIZES = {32, 40, 42, 43, 48, 50, 55, 58, 60, 65, 70, 75, 77, 80, 83, 85, 86, 98, 100}


def extract_size_from_text(text):
    """Extract screen size in inches from text (title, URL, product name).

    Looks for patterns like '65 Inch', '65-Inch', '65"', 'QN65S95D', etc.
    Returns integer size or None.
    """
    if not text:
        return None
    text = unquote(str(text))

    # Pattern 1: XX-Inch, XX Inch, XX", Class XX
    for pattern in [
        r'(\d{2,3})[\s-]*(?:inch|in\b|"|\'\'|class)',
        r'class[\s-]*(\d{2,3})',
        r'(\d{2,3})[\s-]*class',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            size = int(match.group(1))
            if size in VALID_TV_SIZES:
                return size

    # Pattern 2: Model number prefixes like QN65S95D, XR55A95L, KD75X85K
    # Must be followed by a letter + at least 2 more chars to distinguish from
    # model names like QN90D where 90 is the model series, not size.
    match = re.search(r'(?:QN|XR|KD|UN|KQ|KE|FW|NS|TC)(\d{2})[A-Z]\w{2,}', text, re.IGNORECASE)
    if match:
        size = int(match.group(1))
        if size in VALID_TV_SIZES:
            return size

    return None


def build_product_retailer_map(us_data, ca_data):
    """Build a mapping of product_id → {sku_id → retailer info}."""
    products = {}

    for record in us_data + ca_data:
        pid = str(record['product_id'])
        sid = str(record['sku_id'])
        ecom = record.get('ecommerce', {})
        ecom_id = str(ecom.get('id', ''))
        store_name = ecom.get('name', '')
        tracking_url = record.get('tracking', {}).get('url', '')
        price = record.get('price')
        availability = record.get('availability', '')

        if pid not in products:
            products[pid] = {}

        if sid not in products[pid]:
            products[pid][sid] = {
                'sku_id': sid,
                'price_rtings': price,
                'availability': availability,
                'amazon_asin': None,
                'bestbuy_sku': None,
                'walmart_url': None,
                'retailers': [],
            }

        sku = products[pid][sid]

        # Extract identifiers
        if ecom_id == '2':  # Amazon.com
            asin = extract_amazon_asin(tracking_url)
            if asin:
                sku['amazon_asin'] = asin
        elif ecom_id == '3':  # Amazon.ca
            asin = extract_amazon_asin(tracking_url)
            if asin and not sku['amazon_asin']:
                sku['amazon_asin'] = asin  # fallback to CA ASIN if no US
        elif ecom_id == '1':  # Best Buy
            bb_sku = extract_bestbuy_sku(tracking_url)
            if bb_sku:
                sku['bestbuy_sku'] = bb_sku

        sku['retailers'].append({
            'store': store_name,
            'ecommerce_id': ecom_id,
            'price': price,
            'tracking_url': tracking_url,
        })

    return products


def extract_all_product_ids(us_data, ca_data):
    """Extract and save all ASINs and Best Buy SKUs."""
    products = build_product_retailer_map(us_data, ca_data)

    # Load TV data for product names
    tv_df = pd.read_csv(DATA_DIR / "rtings_tv_data.csv")
    tv_names = dict(zip(tv_df['product_id'].astype(str), tv_df['fullname']))

    rows = []
    for pid, skus in products.items():
        name = tv_names.get(pid, f'Product {pid}')
        for sid, sku in skus.items():
            # Try to extract size from all available text sources
            size = None
            for r in sku['retailers']:
                size = extract_size_from_text(r.get('tracking_url', ''))
                if size:
                    break
            rows.append({
                'product_id': pid,
                'fullname': name,
                'sku_id': sid,
                'size_inches': size,
                'amazon_asin': sku['amazon_asin'],
                'bestbuy_sku': sku['bestbuy_sku'],
                'rtings_price': sku['price_rtings'],
                'availability': sku['availability'],
                'num_retailers': len(set(r['store'] for r in sku['retailers'])),
            })

    df = pd.DataFrame(rows)

    # Summary stats
    n_asins = df['amazon_asin'].notna().sum()
    n_bb = df['bestbuy_sku'].notna().sum()
    n_products_asin = df[df['amazon_asin'].notna()]['product_id'].nunique()
    n_products_bb = df[df['bestbuy_sku'].notna()]['product_id'].nunique()
    n_products = df['product_id'].nunique()

    print(f"\n  Total SKU variants: {len(df)}")
    print(f"  Products: {n_products}")
    print(f"  Amazon ASINs: {n_asins} SKUs across {n_products_asin}/{n_products} products")
    print(f"  Best Buy SKUs: {n_bb} SKUs across {n_products_bb}/{n_products} products")

    # Save
    out = DATA_DIR / "product_retailer_ids.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")

    return df


# ============================================================================
# STEP 2: Keepa Amazon Price History
# ============================================================================
def keepa_time_to_datetime(keepa_min):
    """Convert Keepa time (minutes since 2011-01-01) to datetime."""
    epoch = datetime(2011, 1, 1)
    return epoch + timedelta(minutes=int(keepa_min))


def fetch_keepa_prices(retailer_df):
    """Pull price history from Keepa for all Amazon ASINs."""
    if not KEEPA_API_KEY:
        print("ERROR: KEEPA_API_KEY not set in .env")
        return pd.DataFrame()

    asins = retailer_df[retailer_df['amazon_asin'].notna()]['amazon_asin'].unique().tolist()
    print(f"\nFetching Keepa price history for {len(asins)} unique ASINs...")

    # Check token balance
    token_resp = httpx.get(f"https://api.keepa.com/token?key={KEEPA_API_KEY}")
    token_data = token_resp.json()
    tokens = token_data.get('tokensLeft', 0)
    print(f"  Tokens available: {tokens}")

    # Each product query costs 1 token. We can batch up to 100 ASINs per request.
    # At 1 token per ASIN, we need len(asins) tokens.
    if tokens < len(asins):
        print(f"  WARNING: Need {len(asins)} tokens, have {tokens}. Will process what we can.")

    all_price_records = []
    batch_size = 50  # Process 50 ASINs at a time (fewer per batch for reliability)

    for i in range(0, len(asins), batch_size):
        batch = asins[i:i + batch_size]
        asin_str = ",".join(batch)

        print(f"  Batch {i // batch_size + 1}: {len(batch)} ASINs ({i + 1}-{i + len(batch)} of {len(asins)})...")

        try:
            resp = httpx.get(
                f"https://api.keepa.com/product",
                params={
                    "key": KEEPA_API_KEY,
                    "domain": "1",
                    "asin": asin_str,
                    "stats": "1",
                    "history": "1",
                    "buybox": "1",
                },
                timeout=60,
            )
            data = resp.json()

            tokens_left = data.get('tokensLeft', 0)
            print(f"    Tokens remaining: {tokens_left}")

            if 'error' in data:
                print(f"    ERROR: {data['error']}")
                continue

            for product in data.get('products', []):
                asin = product.get('asin', '')
                title = product.get('title', '')
                brand = product.get('brand', '')
                csv_data = product.get('csv', [])
                stats = product.get('stats', {})

                if not csv_data:
                    continue

                # Extract current prices from stats
                current_prices = stats.get('current', []) if stats else []

                # Price type indices: 0=AMAZON, 1=NEW_3P, 2=USED, 16=BUY_BOX, 18=LIST_PRICE
                price_map = {
                    0: 'amazon_price',
                    1: 'new_3p_price',
                    16: 'buy_box_price',
                    18: 'list_price',
                }

                record = {
                    'amazon_asin': asin,
                    'keepa_title': title,
                    'keepa_brand': brand,
                }

                # Get current prices
                for idx, col_name in price_map.items():
                    if current_prices and idx < len(current_prices):
                        val = current_prices[idx]
                        record[col_name] = val / 100.0 if val and val > 0 else None
                    else:
                        record[col_name] = None

                # Get price history summary (min, avg, max over last 90 days)
                at_interval = stats.get('atIntervalStart', []) if stats else []
                for idx, col_name in price_map.items():
                    if idx < len(csv_data) and csv_data[idx]:
                        history = csv_data[idx]
                        # Extract last 90 days of prices
                        recent_prices = []
                        now_keepa = int((datetime.now() - datetime(2011, 1, 1)).total_seconds() / 60)
                        cutoff = now_keepa - (90 * 24 * 60)  # 90 days ago in keepa minutes

                        for j in range(0, len(history) - 1, 2):
                            ts, price = history[j], history[j + 1]
                            if ts >= cutoff and price > 0:
                                recent_prices.append(price / 100.0)

                        if recent_prices:
                            record[f"{col_name}_90d_min"] = min(recent_prices)
                            record[f"{col_name}_90d_avg"] = np.mean(recent_prices)
                            record[f"{col_name}_90d_max"] = max(recent_prices)

                    # Number of history data points
                    if idx < len(csv_data) and csv_data[idx]:
                        record[f"{col_name}_history_points"] = len(csv_data[idx]) // 2

                all_price_records.append(record)

        except Exception as e:
            print(f"    ERROR: {e}")

        # Rate limit: Keepa recommends waiting between requests
        if i + batch_size < len(asins):
            time.sleep(2)

    df = pd.DataFrame(all_price_records)
    if len(df) > 0:
        out = DATA_DIR / "keepa_prices.csv"
        df.to_csv(out, index=False)
        print(f"\n  Keepa data saved: {out}")
        print(f"  Products with data: {len(df)}")
        print(f"  Products with Amazon price: {df['amazon_price'].notna().sum()}")
        print(f"  Products with Buy Box price: {df['buy_box_price'].notna().sum()}")

    return df


# ============================================================================
# STEP 3: Best Buy API Current Prices
# ============================================================================
def fetch_bestbuy_prices(retailer_df):
    """Pull current prices from Best Buy API for all Best Buy SKUs."""
    if not BESTBUY_API_KEY:
        print("\nSkipping Best Buy API (no BESTBUY_API_KEY in .env)")
        print("  Register for free at: https://developer.bestbuy.com/")
        return pd.DataFrame()

    skus = retailer_df[retailer_df['bestbuy_sku'].notna()]['bestbuy_sku'].unique().tolist()
    print(f"\nFetching Best Buy prices for {len(skus)} unique SKUs...")

    all_records = []
    batch_size = 20  # Best Buy API allows up to 100 SKUs per query

    with httpx.Client(timeout=30) as client:
        for i in range(0, len(skus), batch_size):
            batch = skus[i:i + batch_size]
            sku_filter = "|".join(f"sku={s}" for s in batch)

            try:
                resp = client.get(
                    f"https://api.bestbuy.com/v1/products({sku_filter})",
                    params={
                        "apiKey": BESTBUY_API_KEY,
                        "format": "json",
                        "show": "sku,name,salePrice,regularPrice,onSale,"
                                "dollarSavings,percentSavings,priceUpdateDate,"
                                "url,modelNumber,upc",
                        "pageSize": 100,
                    }
                )
                data = resp.json()

                for product in data.get('products', []):
                    all_records.append({
                        'bestbuy_sku': str(product.get('sku', '')),
                        'bb_name': product.get('name', ''),
                        'bb_model': product.get('modelNumber', ''),
                        'bb_upc': product.get('upc', ''),
                        'bb_regular_price': product.get('regularPrice'),
                        'bb_sale_price': product.get('salePrice'),
                        'bb_on_sale': product.get('onSale', False),
                        'bb_dollar_savings': product.get('dollarSavings'),
                        'bb_percent_savings': product.get('percentSavings'),
                        'bb_price_updated': product.get('priceUpdateDate'),
                        'bb_url': product.get('url', ''),
                    })

                print(f"  Batch {i // batch_size + 1}: {len(data.get('products', []))} products")

            except Exception as e:
                print(f"  Batch {i // batch_size + 1} ERROR: {e}")

            # Rate limit: 5 req/sec max
            time.sleep(0.25)

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        out = DATA_DIR / "bestbuy_prices.csv"
        df.to_csv(out, index=False)
        print(f"\n  Best Buy data saved: {out}")
        print(f"  Products with price: {df['bb_sale_price'].notna().sum()}")

    return df


# ============================================================================
# STEP 4: Merge pricing into TV database
# ============================================================================
def merge_pricing(retailer_df, keepa_df, bestbuy_df):
    """Merge all pricing data and compute derived metrics."""
    print("\nMerging pricing data...")

    # Load the TV database
    tv_db = pd.read_csv(DATA_DIR / "tv_database.csv")
    print(f"  TV database: {len(tv_db)} TVs")

    # Merge retailer IDs
    # For each product, get the best price across all SKUs
    # Group by product_id to get one row per product

    # First, merge Keepa data onto retailer_df via amazon_asin
    if len(keepa_df) > 0:
        retailer_df = retailer_df.merge(
            keepa_df, on='amazon_asin', how='left', suffixes=('', '_keepa')
        )

    # Merge Best Buy data onto retailer_df via bestbuy_sku
    if len(bestbuy_df) > 0:
        retailer_df = retailer_df.merge(
            bestbuy_df, on='bestbuy_sku', how='left', suffixes=('', '_bb')
        )

    # Parse size from Keepa titles and Best Buy product names
    for idx, row in retailer_df[retailer_df['size_inches'].isna()].iterrows():
        for col in ['keepa_title', 'bb_name']:
            if col in retailer_df.columns:
                size = extract_size_from_text(row.get(col))
                if size:
                    retailer_df.at[idx, 'size_inches'] = size
                    break

    # Build per-size pricing table
    price_rows = []
    for _, row in retailer_df.iterrows():
        size = row.get('size_inches')
        pid = str(row['product_id'])

        # Skip out-of-stock items
        avail = str(row.get('availability', ''))
        is_out_of_stock = avail == 'out_of_stock'

        # Collect all current sale prices from each retailer.
        # Keepa amazon_price = Amazon's current selling price (NOT MSRP).
        # Keepa new_3p_price = lowest 3rd-party new price on Amazon.
        # Best Buy bb_sale_price = current sale price (discounted if on_sale).
        # RTINGS rtings_price = current retail price from affiliate link (may be stale).
        # Keepa list_price = MSRP — stored as reference but NOT used as best price.
        candidates = []
        if 'buy_box_price' in row and pd.notna(row.get('buy_box_price')) and row['buy_box_price'] > 0:
            candidates.append(('amazon_buy_box', row['buy_box_price']))
        if 'amazon_price' in row and pd.notna(row.get('amazon_price')) and row['amazon_price'] > 0:
            candidates.append(('amazon', row['amazon_price']))
        if 'new_3p_price' in row and pd.notna(row.get('new_3p_price')) and row['new_3p_price'] > 0:
            candidates.append(('amazon_3p', row['new_3p_price']))
        if 'bb_sale_price' in row and pd.notna(row.get('bb_sale_price')) and row['bb_sale_price'] > 0:
            candidates.append(('bestbuy', row['bb_sale_price']))

        # Use RTINGS price only as fallback when no live API prices available
        if not candidates and pd.notna(row.get('rtings_price')) and row['rtings_price'] > 0:
            candidates.append(('rtings', row['rtings_price']))

        # Pick the LOWEST current sale price across all retailers
        best_price = None
        price_source = None
        if candidates and not is_out_of_stock:
            price_source, best_price = min(candidates, key=lambda x: x[1])

        # Compute price per m²
        price_per_m2 = None
        if best_price and size and size in SCREEN_AREA_M2:
            price_per_m2 = best_price / SCREEN_AREA_M2[size]

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
            'screen_area_m2': SCREEN_AREA_M2.get(size) if size else None,
            'price_date': datetime.now().strftime('%Y-%m-%d'),
        })

    prices_df = pd.DataFrame(price_rows)

    # Save detailed price table
    prices_out = DATA_DIR / "tv_prices.csv"
    prices_df.to_csv(prices_out, index=False)
    print(f"  Price records: {len(prices_df)}")
    print(f"  Saved: {prices_out}")

    # Append to rolling price history
    append_price_history(prices_df, tv_db)

    # Also create a per-product summary (best price at most common comparison size = 65")
    summary_rows = []
    for pid in tv_db['product_id'].unique():
        pid_str = str(pid)
        prod_prices = prices_df[prices_df['product_id'] == pid_str]

        if len(prod_prices) == 0:
            summary_rows.append({
                'product_id': pid,
                'sizes_with_price': 0,
            })
            continue

        # Prefer 65" price, then closest to 65"
        sizes_available = prod_prices['size_inches'].dropna().unique()
        best_row = None
        if 65 in sizes_available:
            best_row = prod_prices[prod_prices['size_inches'] == 65].iloc[0]
        elif len(sizes_available) > 0:
            closest = min(sizes_available, key=lambda x: abs(x - 65))
            best_row = prod_prices[prod_prices['size_inches'] == closest].iloc[0]
        else:
            # Use any row with a price
            priced = prod_prices[prod_prices['best_price'].notna()]
            if len(priced) > 0:
                best_row = priced.iloc[0]

        if best_row is not None:
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
                'sizes_with_price': len(sizes_available),
                'all_asins': ','.join(str(a) for a in prod_prices['amazon_asin'].dropna().unique()),
                'all_bb_skus': ','.join(str(s) for s in prod_prices['bestbuy_sku'].dropna().unique()),
            })
        else:
            summary_rows.append({
                'product_id': pid,
                'sizes_with_price': 0,
            })

    summary_df = pd.DataFrame(summary_rows)

    # Merge summary into TV database
    tv_db['product_id'] = tv_db['product_id'].astype(int)
    summary_df['product_id'] = summary_df['product_id'].astype(int)
    merged = tv_db.merge(summary_df, on='product_id', how='left')

    # Compute price per Mixed Usage point
    if 'mixed_usage' in merged.columns:
        merged['price_per_mixed_use'] = merged.apply(
            lambda r: r['price_best'] / r['mixed_usage']
            if pd.notna(r.get('price_best')) and pd.notna(r.get('mixed_usage')) and r['mixed_usage'] > 0
            else None,
            axis=1
        )

    # Save enriched database
    enriched_out = DATA_DIR / "tv_database_with_prices.csv"
    merged.to_csv(enriched_out, index=False)
    print(f"\n  Enriched database saved: {enriched_out}")

    enriched_xlsx = DATA_DIR / "tv_database_with_prices.xlsx"
    merged.to_excel(enriched_xlsx, index=False, sheet_name="TV Database")
    print(f"  Excel version saved: {enriched_xlsx}")

    # Print summary
    print_pricing_summary(merged, prices_df)

    return merged, prices_df


def append_price_history(prices_df, tv_db):
    """Append today's price snapshot to the rolling history file."""
    history_path = DATA_DIR / "price_history.csv"

    # Build lean history rows by joining technology from tv_db
    tech_map = dict(zip(tv_db['product_id'].astype(str), tv_db['color_architecture']))

    snapshot = prices_df[prices_df['best_price'].notna()].copy()
    snapshot = snapshot[['product_id', 'size_inches', 'best_price', 'price_source', 'price_date']].copy()
    snapshot.rename(columns={'price_date': 'snapshot_date'}, inplace=True)
    snapshot['color_architecture'] = snapshot['product_id'].map(tech_map)
    snapshot = snapshot[['snapshot_date', 'product_id', 'size_inches', 'best_price',
                         'price_source', 'color_architecture']]

    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, snapshot], ignore_index=True)
        combined.drop_duplicates(
            subset=['snapshot_date', 'product_id', 'size_inches'],
            keep='last', inplace=True,
        )
        combined.to_csv(history_path, index=False)
    else:
        snapshot.to_csv(history_path, index=False)

    if len(snapshot) > 0:
        print(f"  Price history updated: {history_path} "
              f"({len(snapshot)} rows for {snapshot['snapshot_date'].iloc[0]})")
    else:
        print(f"  Price history updated: {history_path} (0 rows)")


def print_pricing_summary(tv_df, prices_df):
    """Print pricing summary by technology."""
    print(f"\n{'='*70}")
    print("PRICING SUMMARY")
    print("=" * 70)

    priced = tv_df[tv_df['price_best'].notna()]
    print(f"\nTVs with pricing: {len(priced)}/{len(tv_df)}")

    if len(priced) == 0:
        return

    print(f"\nPrice by color_architecture (best available price):")
    print(f"  {'Technology':<15s} {'Count':>5s} {'Min':>8s} {'Median':>8s} {'Max':>8s} {'$/m²':>8s}")
    print(f"  {'─'*15} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for tech in ['QD-OLED', 'WOLED', 'QD-LCD', 'Pseudo QD', 'KSF', 'WLED']:
        t = priced[priced['color_architecture'] == tech]
        if len(t) == 0:
            continue
        prices = t['price_best'].dropna()
        m2 = t['price_per_m2'].dropna()
        print(f"  {tech:<15s} {len(t):>5d} ${prices.min():>7.0f} ${prices.median():>7.0f} "
              f"${prices.max():>7.0f} ${m2.median():>7.0f}" if len(m2) > 0 else
              f"  {tech:<15s} {len(t):>5d} ${prices.min():>7.0f} ${prices.median():>7.0f} "
              f"${prices.max():>7.0f}     N/A")

    # Size breakdown
    print(f"\nDetailed size pricing: {len(prices_df)} total price records")
    sized = prices_df[prices_df['size_inches'].notna() & prices_df['best_price'].notna()]
    if len(sized) > 0:
        for size in sorted(sized['size_inches'].unique()):
            s = sized[sized['size_inches'] == size]
            print(f"  {int(size)}\":  {len(s)} SKUs, "
                  f"${s['best_price'].min():.0f} - ${s['best_price'].max():.0f}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("TV PRICING PIPELINE")
    print("=" * 70)

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    # Step 1: Extract retailer IDs
    us_data, ca_data = fetch_rtings_prices()
    retailer_df = extract_all_product_ids(us_data, ca_data)

    if mode == '--extract-only':
        return

    # Step 2: Keepa (Amazon) prices
    keepa_df = pd.DataFrame()
    if mode != '--bestbuy-only':
        keepa_df = fetch_keepa_prices(retailer_df)

    # Step 3: Best Buy prices
    bestbuy_df = pd.DataFrame()
    if mode != '--keepa-only':
        bestbuy_df = fetch_bestbuy_prices(retailer_df)

    # Step 4: Merge everything
    merge_pricing(retailer_df, keepa_df, bestbuy_df)


if __name__ == '__main__':
    main()
