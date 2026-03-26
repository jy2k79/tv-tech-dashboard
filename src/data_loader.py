"""Data loading, enrichment, and caching for the Display Technology Dashboard."""

import math
import streamlit as st
import pandas as pd
from pathlib import Path
from silo_config import is_samsung_woled_sku
from src.charts import TECH_ORDER

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Product type configuration
# ---------------------------------------------------------------------------
PRODUCT_CONFIGS = {
    "TVs": {
        "data_file": "tv_database_with_prices.csv",
        "prices_file": "tv_prices.csv",
        "history_file": "price_history.csv",
        "item_label": "TVs",
        "item_singular": "TV",
        "score_cols": ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"],
        "primary_score": "mixed_usage",
        "has_model_year": True,
        "has_8k_exclusion": True,
        "has_samsung_woled": True,
        "has_price_per_score": True,
        "price_per_score_col": "price_per_mixed_use",
        "price_per_score_label": "$ per Mixed Usage Point",
        "input_lag_col": "input_lag_4k_ms",
        "input_lag_label": "4K Input Lag (ms)",
        "profile_page": "TV Profiles",
        "extra_score_cols": ["brightness_score", "contrast_ratio_score", "color_score",
                             "black_level_score", "native_contrast_score"],
        "advantage_metrics": [
            ("hdr_peak_10pct_nits", "HDR Peak Brightness\n(10%, nits)"),
            ("hdr_bt2020_coverage_itp_pct", "HDR BT.2020 Coverage\n(ITP %)"),
            ("sdr_dci_p3_coverage_pct", "SDR DCI-P3 Coverage\n(%)"),
            ("contrast_ratio_score", "Contrast Ratio Score"),
            ("color_score", "Color Score"),
            ("brightness_score", "Brightness Score"),
        ],
    },
    "Monitors": {
        "data_file": "monitor_database_with_prices.csv",
        "prices_file": "monitor_prices.csv",
        "history_file": "monitor_price_history.csv",
        "item_label": "Monitors",
        "item_singular": "Monitor",
        "score_cols": ["pc_gaming", "console_gaming", "office", "editing"],
        "primary_score": "pc_gaming",
        "has_model_year": True,
        "has_8k_exclusion": False,
        "has_samsung_woled": False,
        "has_price_per_score": False,
        "price_per_score_col": None,
        "price_per_score_label": None,
        "input_lag_col": "input_lag_native_ms",
        "input_lag_label": "Native Input Lag (ms)",
        "profile_page": "Monitor Profiles",
        "extra_score_cols": ["brightness_score", "color_accuracy"],
        "advantage_metrics": [
            ("hdr_peak_10pct_nits", "HDR Peak Brightness\n(10%, nits)"),
            ("hdr_bt2020_coverage_itp_pct", "HDR BT.2020 Coverage\n(ITP %)"),
            ("total_response_time_ms", "Total Response Time\n(ms)"),
            ("brightness_score", "Brightness Score"),
            ("input_lag_native_ms", "Native Input Lag\n(ms)"),
            ("console_gaming", "Console Gaming Score"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Screen area lookups
# ---------------------------------------------------------------------------
def _area(diag, aw=16, ah=9):
    d = diag * 0.0254
    r = aw / ah
    return d * r / math.sqrt(1 + r**2) * d / math.sqrt(1 + r**2)

# TV sizes — must match pricing_pipeline.py SCREEN_AREA_M2
TV_SCREEN_AREA = {
    32: 0.22, 40: 0.34, 42: 0.38, 43: 0.40, 48: 0.50,
    50: 0.54, 55: 0.65, 58: 0.72, 60: 0.77, 65: 0.91,
    70: 1.06, 75: 1.21, 77: 1.28, 80: 1.38, 83: 1.49,
    85: 1.56, 86: 1.59, 98: 2.07, 100: 2.15,
}

# Monitor sizes — formula-computed, matching monitor_pricing_pipeline.py
MONITOR_SCREEN_AREA = {s: _area(s) for s in [24, 25, 27, 28, 30, 32, 40, 42, 45, 55]}
MONITOR_SCREEN_AREA[34] = _area(34, 21, 9)
MONITOR_SCREEN_AREA[38] = _area(38, 21, 9)
MONITOR_SCREEN_AREA[49] = _area(49, 32, 9)
MONITOR_SCREEN_AREA[57] = _area(57, 32, 9)


def get_screen_area_map(product_type):
    """Return the appropriate screen area lookup for a product type."""
    if product_type == "Monitors":
        return MONITOR_SCREEN_AREA
    return TV_SCREEN_AREA


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def _load_single(data_file, score_cols, extra_score_cols,
                 price_per_score_col, input_lag_col):
    """Load and prepare a single product-type CSV."""
    path = DATA_DIR / data_file
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["first_published_at", "last_updated_at", "released_at", "scraped_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    numeric_cols = [
        "price_best", "price_per_m2",
        "hdr_peak_10pct_nits", "hdr_peak_2pct_nits",
        "sdr_real_scene_peak_nits", "native_contrast",
        "dimming_zone_count", "price_size",
        "green_fwhm_nm", "red_fwhm_nm", "blue_fwhm_nm",
        "hdr_bt2020_coverage_itp_pct", "sdr_dci_p3_coverage_pct",
        "first_response_time_ms", "total_response_time_ms",
    ] + list(score_cols) + list(extra_score_cols)
    if price_per_score_col:
        numeric_cols.append(price_per_score_col)
    if input_lag_col:
        numeric_cols.append(input_lag_col)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["color_architecture"] = pd.Categorical(
        df["color_architecture"], categories=TECH_ORDER, ordered=True
    )
    return df


def load_data(product_type="TVs"):
    if product_type == "All Products":
        frames = []
        for pt, cfg in PRODUCT_CONFIGS.items():
            sub = _load_single(cfg["data_file"], cfg["score_cols"],
                               cfg["extra_score_cols"], cfg["price_per_score_col"],
                               cfg["input_lag_col"])
            if len(sub) > 0:
                if "product_type" not in sub.columns:
                    sub["product_type"] = cfg["name"] if "name" in cfg else pt.lower().rstrip("s")
                frames.append(sub)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()
    cfg = PRODUCT_CONFIGS[product_type]
    return _load_single(cfg["data_file"], cfg["score_cols"],
                        cfg["extra_score_cols"], cfg["price_per_score_col"],
                        cfg["input_lag_col"])


@st.cache_data
def load_size_prices(product_type="TVs"):
    cfg = PRODUCT_CONFIGS[product_type]
    path = DATA_DIR / cfg["prices_file"]
    if path.exists():
        df = pd.read_csv(path)
        for col in ["best_price", "amazon_price", "bestbuy_price", "rtings_price",
                     "list_price", "price_per_m2", "size_inches"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data
def load_price_history(product_type="TVs"):
    cfg = PRODUCT_CONFIGS[product_type]
    path = DATA_DIR / cfg["history_file"]
    if path.exists():
        hist = pd.read_csv(path)
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
        hist["best_price"] = pd.to_numeric(hist["best_price"], errors="coerce")
        return hist
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Price enrichment
# ---------------------------------------------------------------------------
def _is_samsung_woled_row(row, name_map, tech_map):
    """Check if a price_history row is a Samsung WOLED-panel SKU."""
    pid = str(row.get("product_id", ""))
    tech = tech_map.get(pid, "")
    name = name_map.get(pid, "")
    size = int(row["size_inches"]) if pd.notna(row.get("size_inches")) else 0
    return is_samsung_woled_sku(name, tech, size)


@st.cache_data
def _enrich_history_core(hist, screen_area_map):
    """Cached: compute $/m² and time columns on price_history."""
    h = hist.copy()
    h["screen_area_m2"] = h["size_inches"].map(screen_area_map)
    h["price_per_m2"] = h["best_price"] / h["screen_area_m2"]
    h["year"] = h["snapshot_date"].dt.year
    h["month"] = h["snapshot_date"].dt.to_period("M").astype(str)
    h["quarter"] = h["snapshot_date"].dt.to_period("Q").astype(str)
    h["iso_year"] = h["snapshot_date"].dt.isocalendar().year.astype(int)
    h["iso_week"] = h["snapshot_date"].dt.isocalendar().week.astype(int)
    return h


def enrich_history(hist, main_df=None, *, screen_area_map=None):
    """Add $/m² and time columns to price_history data.

    Excludes Samsung WOLED-panel SKUs from QD-OLED products if main_df is provided.
    Returns (enriched_df, n_woled_excluded).
    """
    if screen_area_map is None:
        screen_area_map = TV_SCREEN_AREA
    if len(hist) == 0:
        return hist, 0
    h = _enrich_history_core(hist, screen_area_map)

    n_woled_excluded = 0
    if main_df is not None and len(main_df) > 0:
        _name_map = dict(zip(main_df["product_id"].astype(str), main_df["fullname"]))
        _tech_map = dict(zip(main_df["product_id"].astype(str), main_df["color_architecture"]))
        woled_mask = h.apply(lambda r: _is_samsung_woled_row(r, _name_map, _tech_map), axis=1)
        n_woled_excluded = h[woled_mask].groupby(["product_id", "size_inches"]).ngroups
        h = h[~woled_mask]

    return h, n_woled_excluded


def compute_m2_from_history(hist, product_ids=None, snapshot="latest"):
    """Compute per-product median $/m² from price_history.

    Returns dict of product_id → median $/m².
    """
    if len(hist) == 0:
        return {}
    h = hist.dropna(subset=["price_per_m2"]).copy()
    if snapshot == "latest":
        h = h[h["snapshot_date"] == h["snapshot_date"].max()]
    elif snapshot == "ytd":
        current_year = h["snapshot_date"].max().year
        h = h[h["year"] == current_year]
    elif snapshot != "all":
        if "Q" in str(snapshot):
            h = h[h["quarter"] == snapshot]
        else:
            h = h[h["month"] == snapshot]
    if product_ids is not None:
        h = h[h["product_id"].astype(str).isin(product_ids)]
    if len(h) == 0:
        return {}
    return h.groupby("product_id")["price_per_m2"].mean().to_dict()
