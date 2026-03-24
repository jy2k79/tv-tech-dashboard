#!/usr/bin/env python3
"""
Display Technology Dashboard
=======================================
Interactive Streamlit dashboard for exploring TV and Monitor display
technologies, pricing, and performance metrics.

Branding: Nanosys (Inter font, brand color palette)

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Display Technology Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Password gate — requires password set in .streamlit/secrets.toml
# ---------------------------------------------------------------------------
def check_password():
    """Return True if the user has entered the correct password."""
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return True
    pwd = st.text_input("Password", type="password", key="pwd_input")
    if pwd:
        if pwd == st.secrets.get("app_password", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    else:
        st.info("Enter the dashboard password to continue.")
    return False

if not check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Nanosys branding — Inter font, larger base sizes, heavier weights
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
h1 { font-weight: 800 !important; font-size: 2.2rem !important; }
h2 { font-weight: 700 !important; font-size: 1.5rem !important; }
h3 { font-weight: 700 !important; font-size: 1.25rem !important; }
strong { font-weight: 700 !important; }
button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 1rem !important; }
[data-testid="stMetricValue"] { font-weight: 700 !important; font-size: 1.3rem !important; }
[data-testid="stMetricLabel"] { font-weight: 600 !important; }
.stSidebar [data-testid="stMarkdownContainer"] { font-weight: 500; }
/* Suppress tooltips */
[role="tooltip"] {
    display: none !important;
}
/* Tighter sidebar spacing */
.stSidebar [data-testid="stCheckbox"] { margin-bottom: -10px; }
/* Mobile: iPhone Pro Max (~430px) */
@media (max-width: 480px) {
    [data-testid="stSidebar"] { min-width: 0 !important; }
    .stPlotlyChart { margin-left: -0.5rem; margin-right: -0.5rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stTabs"] button { font-size: 0.85rem !important; padding: 6px 8px !important; }
    .stSidebar [data-testid="stCheckbox"] { margin-bottom: -14px; }
}
</style>
""", unsafe_allow_html=True)

# Fix Streamlit sidebar toggle showing raw ligature text like
# "keyboard_double_arrow_right". Uses components.html() which runs real JS
# in an iframe that can reach into the parent document.
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    var ligatures = {
        'keyboard_double_arrow_right': '\u25B6',
        'keyboard_double_arrow_left': '\u25C0',
        'arrow_right': '\u25B8',
        'arrow_drop_down': '\u25BE',
        'arrow_forward_ios': '\u25B8',
        'expand_more': '\u25BE',
        'expand_less': '\u25B4',
        'chevron_right': '\u25B8'
    };

    function fix() {
        var walker = doc.createTreeWalker(
            doc.body, NodeFilter.SHOW_TEXT, null, false
        );
        var node;
        while (node = walker.nextNode()) {
            var t = node.textContent.trim();
            if (ligatures[t]) {
                var el = node.parentElement;
                if (el && !el.getAttribute('data-fixed')) {
                    el.setAttribute('data-fixed', '1');
                    el.textContent = '';
                    var span = doc.createElement('span');
                    span.textContent = ligatures[t];
                    span.style.fontSize = '12px';
                    span.style.color = '#999';
                    span.style.cursor = 'pointer';
                    el.appendChild(span);
                }
            }
        }
    }

    // Inject colored dots into sidebar checkbox labels
    var colorMap = {
        'QD-OLED': '#FF009F', 'WOLED': '#4B40EB', 'QD-LCD': '#FFC700',
        'Pseudo QD': '#FF7E43', 'KSF': '#90BFFF', 'WLED': '#A8BDD0',
        'OLED': '#4B40EB', 'LCD': '#FFC700'
    };

    function addDots() {
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return;
        var labels = sidebar.querySelectorAll('[data-testid="stCheckbox"] label p, [data-testid="stCheckbox"] label span');
        labels.forEach(function(el) {
            var text = el.textContent.trim();
            if (colorMap[text] && !el.getAttribute('data-dotted')) {
                el.setAttribute('data-dotted', '1');
                var dot = doc.createElement('span');
                dot.style.cssText = 'display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle;background:' + colorMap[text];
                el.insertBefore(dot, el.firstChild);
            }
        });
    }

    // Run repeatedly to catch Streamlit re-renders
    fix(); addDots();
    setTimeout(function() { fix(); addDots(); }, 300);
    setTimeout(function() { fix(); addDots(); }, 1000);
    setTimeout(function() { fix(); addDots(); }, 2000);
    setTimeout(function() { fix(); addDots(); }, 4000);

    // Also watch for DOM changes
    var obs = new MutationObserver(function() { fix(); addDots(); });
    obs.observe(doc.body, { childList: true, subtree: true, characterData: true });
})();
</script>
""", height=0)

# Plotly defaults for consistent styling
PL = dict(font=dict(family="Inter, sans-serif", size=14))
MARKER = dict(size=11)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# Canonical technology ordering (left-to-right: low → high QD content)
TECH_ORDER = ["WLED", "KSF", "Pseudo QD", "QD-LCD", "WOLED", "QD-OLED"]

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
    },
}


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
        # Load both and concatenate
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

# Screen area lookup for $/m² calculations (shared with pricing_pipeline.py)
# Screen area lookups.
# TV values match pricing_pipeline.py (historically established, internally consistent).
# Monitor values use exact formula with per-product aspect ratios.
import math as _math
def _area(diag, aw=16, ah=9):
    d = diag * 0.0254
    r = aw / ah
    return d * r / _math.sqrt(1 + r**2) * d / _math.sqrt(1 + r**2)

# TV sizes — must match pricing_pipeline.py SCREEN_AREA_M2
_TV_SCREEN_AREA = {
    32: 0.22, 40: 0.34, 42: 0.38, 43: 0.40, 48: 0.50,
    50: 0.54, 55: 0.65, 58: 0.72, 60: 0.77, 65: 0.91,
    70: 1.06, 75: 1.21, 77: 1.28, 80: 1.38, 83: 1.49,
    85: 1.56, 86: 1.59, 98: 2.07, 100: 2.15,
}
# Monitor sizes — formula-computed, matching monitor_pricing_pipeline.py
_MONITOR_SCREEN_AREA = {s: _area(s) for s in [24, 25, 27, 28, 30, 32, 40, 42, 45, 55]}
_MONITOR_SCREEN_AREA[34] = _area(34, 21, 9)
_MONITOR_SCREEN_AREA[38] = _area(38, 21, 9)
_MONITOR_SCREEN_AREA[49] = _area(49, 32, 9)
_MONITOR_SCREEN_AREA[57] = _area(57, 32, 9)

# Selected at runtime based on product type (set after sidebar selection)
_SCREEN_AREA_M2_GLOBAL = _TV_SCREEN_AREA  # default, updated below


# Samsung OLED sizes that use WOLED panels despite QD-OLED classification.
# Samsung Display only makes QD-OLED at 55", 65", 77". All other sizes are WOLED.
# Covers 2024-2026 lineup. Must match pricing_pipeline.py SAMSUNG_WOLED_SIZES.
_ALL_WOLED = {42, 48, 55, 65, 77, 83, 85, 98, 100}
_SAMSUNG_WOLED_SIZES = {
    "S90": {42, 48, 83}, "S95": {48, 83}, "S99": {83},
    "S85H": _ALL_WOLED, "S85D": _ALL_WOLED,
    "S82": _ALL_WOLED, "S83": _ALL_WOLED,
    "S85F": {77, 83},
}


def _is_samsung_woled_row(row, name_map: dict, tech_map: dict) -> bool:
    """Check if a price_history row is a Samsung WOLED-panel SKU."""
    pid = str(row.get("product_id", ""))
    tech = tech_map.get(pid, "")
    if tech != "QD-OLED":
        return False
    name = name_map.get(pid, "")
    if "Samsung" not in name:
        return False
    size = int(row["size_inches"]) if pd.notna(row.get("size_inches")) else 0
    for model, sizes in _SAMSUNG_WOLED_SIZES.items():
        if model in name and size in sizes:
            return True
    return False


@st.cache_data
def _enrich_history_core(hist: pd.DataFrame) -> pd.DataFrame:
    """Cached: compute $/m² and time columns on price_history."""
    h = hist.copy()
    h["screen_area_m2"] = h["size_inches"].map(_SCREEN_AREA_M2_GLOBAL)
    h["price_per_m2"] = h["best_price"] / h["screen_area_m2"]
    h["year"] = h["snapshot_date"].dt.year
    h["month"] = h["snapshot_date"].dt.to_period("M").astype(str)
    h["quarter"] = h["snapshot_date"].dt.to_period("Q").astype(str)
    h["iso_year"] = h["snapshot_date"].dt.isocalendar().year.astype(int)
    h["iso_week"] = h["snapshot_date"].dt.isocalendar().week.astype(int)
    return h


def enrich_history(hist: pd.DataFrame, main_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add $/m² and time columns to price_history data. Call once at load time.

    Excludes Samsung WOLED-panel SKUs from QD-OLED products if main_df is provided.
    """
    if len(hist) == 0:
        return hist
    h = _enrich_history_core(hist)

    # Exclude Samsung WOLED-panel sizes from QD-OLED pricing
    if main_df is not None and len(main_df) > 0:
        _name_map = dict(zip(main_df["product_id"].astype(str), main_df["fullname"]))
        _tech_map = dict(zip(main_df["product_id"].astype(str), main_df["color_architecture"]))
        woled_mask = h.apply(lambda r: _is_samsung_woled_row(r, _name_map, _tech_map), axis=1)
        global _n_woled_excluded
        # Count unique (product_id, size) combos, not total history rows
        _n_woled_excluded = h[woled_mask].groupby(["product_id", "size_inches"]).ngroups
        h = h[~woled_mask]

    return h

_n_woled_excluded = 0


def compute_m2_from_history(hist: pd.DataFrame, product_ids: set | None = None,
                            snapshot: str = "latest") -> dict:
    """Compute per-product median $/m² from price_history.

    Args:
        hist: enriched price_history DataFrame (must have price_per_m2 column)
        product_ids: optional set of product_id strings to include
        snapshot: time window to aggregate over —
            "latest"  = most recent snapshot only (default, for bar charts)
            "ytd"     = year-to-date
            "all"     = all available history
            or a period string like "2026Q1", "2026-03" for quarter/month

    Returns dict of product_id → median $/m².
    This is the single source of truth for all $/m² displays.
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
        # Try quarter (e.g. "2026Q1") or month (e.g. "2026-03")
        if "Q" in str(snapshot):
            h = h[h["quarter"] == snapshot]
        else:
            h = h[h["month"] == snapshot]

    if product_ids is not None:
        h = h[h["product_id"].astype(str).isin(product_ids)]

    if len(h) == 0:
        return {}

    return h.groupby("product_id")["price_per_m2"].mean().to_dict()


# Post-processing happens after product type selection (below sidebar)

# ---------------------------------------------------------------------------
# Nanosys brand color palette (colorblind-safe selections)
# ---------------------------------------------------------------------------
TECH_COLORS = {
    "QD-OLED": "#FF009F",    # Nanosys magenta
    "WOLED": "#4B40EB",      # Nanosys violet
    "QD-LCD": "#FFC700",     # Nanosys gold
    "Pseudo QD": "#FF7E43",  # Nanosys orange
    "KSF": "#90BFFF",        # Nanosys sky blue
    "WLED": "#A8BDD0",       # muted blue-gray
}

DISPLAY_TYPE_COLORS = {
    "OLED": "#4B40EB",
    "LCD": "#FFC700",
}

# Human-readable label map (used across all pages)
LABEL_MAP = {
    "native_contrast": "Native Contrast Ratio",
    "hdr_peak_10pct_nits": "HDR Peak Brightness (10% window, nits)",
    "hdr_peak_2pct_nits": "HDR Peak Brightness (2% window, nits)",
    "sdr_real_scene_peak_nits": "SDR Real Scene Peak (nits)",
    "hdr_bt2020_coverage_itp_pct": "HDR BT.2020 Coverage (%)",
    "sdr_dci_p3_coverage_pct": "SDR DCI-P3 Coverage (%)",
    "input_lag_4k_ms": "4K Input Lag (ms)",
    "input_lag_native_ms": "Native Input Lag (ms)",
    "total_response_time_ms": "Total Response Time (ms)",
    "first_response_time_ms": "First Response Time (ms)",
    "contrast_ratio_score": "Contrast Ratio Score",
    "black_level_score": "Black Level Score",
    "color_score": "Color Score",
    "color_accuracy": "Color Accuracy",
    "brightness_score": "Brightness Score",
    "native_contrast_score": "Native Contrast Score",
    "mixed_usage": "Mixed Usage",
    "home_theater": "Home Theater",
    "gaming": "Gaming",
    "sports": "Sports",
    "bright_room": "Bright Room",
    "pc_gaming": "PC Gaming",
    "console_gaming": "Console Gaming",
    "office": "Office",
    "editing": "Editing",
    "price_best": "Price ($)",
    "price_per_m2": "Price per m\u00b2",
    "price_per_mixed_use": "$ per Mixed Usage Point",
    "color_architecture": "Technology",
    "green_fwhm_nm": "Green FWHM (nm)",
    "red_fwhm_nm": "Red FWHM (nm)",
    "blue_fwhm_nm": "Blue FWHM (nm)",
}


def friendly(col):
    """Return a human-friendly label for a column name."""
    return LABEL_MAP.get(col, col.replace("_", " ").title())


def axis_range(col, data_df=None):
    """Return [min, max] range for a column based on the current filtered data.

    Ranges are set explicitly so that axes stay stable when the user
    toggles legend items (hides/shows a technology), while still
    scaling properly to the currently filtered data on each redraw.
    """
    score_cols = {
        "mixed_usage", "home_theater", "gaming", "sports", "bright_room",
        "brightness_score", "contrast_ratio_score", "color_score",
        "black_level_score", "native_contrast_score",
        "pc_gaming", "console_gaming", "office", "editing", "color_accuracy",
    }
    if col in score_cols:
        return [0, 10.5]
    pct_cols = {"sdr_dci_p3_coverage_pct", "hdr_bt2020_coverage_itp_pct", "sdr_bt2020_coverage_pct"}
    if col in pct_cols:
        return [0, 105]
    # Use filtered data (fdf) by default so the chart scales to the current view
    src = data_df if data_df is not None else fdf
    if col not in src.columns:
        return None
    vals = pd.to_numeric(src[col], errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    price_cols = {"price_best", "price_per_m2", "price_per_mixed_use", "best_price"}
    if col in price_cols:
        return [0, vmax * 1.1]
    pad = (vmax - vmin) * 0.05 if vmax > vmin else 1
    return [max(0, vmin - pad), vmax + pad]


# ---------------------------------------------------------------------------
# Sidebar — Branding & Global Filters
# ---------------------------------------------------------------------------
_logo_path = Path(__file__).parent / "logos" / "Nanosys Logo White Text 4X.png"
if _logo_path.exists():
    st.sidebar.image(str(_logo_path), use_container_width=True)
    st.sidebar.markdown("<p style='text-align:center;color:#999;font-size:0.85em;margin-top:-8px'>Display Technology Intelligence</p>",
                        unsafe_allow_html=True)

st.sidebar.divider()

# --- Product type selector ---
_product_types = list(PRODUCT_CONFIGS.keys()) + ["All Products"]
product_type = st.sidebar.radio("Product Type", _product_types, index=0,
                                 key="product_type", horizontal=True)
_is_blended = product_type == "All Products"
PCFG = PRODUCT_CONFIGS.get(product_type, PRODUCT_CONFIGS["TVs"])
_SCREEN_AREA_M2_GLOBAL = _MONITOR_SCREEN_AREA if product_type == "Monitors" else _TV_SCREEN_AREA
st.sidebar.divider()

# Load data for selected product type
df = load_data(product_type)
if _is_blended:
    # Ensure product_type column exists
    if "product_type" not in df.columns:
        df["product_type"] = "tv"  # fallback
    # Enrich TV subset $/m² from history so it matches the TV-only view
    _tv_hist = load_price_history("TVs")
    if len(_tv_hist) > 0:
        _tv_hist_enriched = enrich_history(_tv_hist, main_df=df[df["product_type"] == "tv"])
        _tv_m2_map = compute_m2_from_history(_tv_hist_enriched)
        _tv_mask = df["product_type"] == "tv"
        df.loc[_tv_mask, "price_per_m2"] = df.loc[_tv_mask, "product_id"].map(
            lambda pid: _tv_m2_map.get(pid) or _tv_m2_map.get(str(pid))
        )
    prices_df = pd.DataFrame()
    history_df = pd.DataFrame()
else:
    prices_df = load_size_prices(product_type)
    history_df = load_price_history(product_type)

# Post-processing: enrich history, exclude 8K, derive model_year
_n_woled_excluded = 0
if not _is_blended:
    if PCFG["has_samsung_woled"]:
        history_df = enrich_history(history_df, main_df=df)
    elif len(history_df) > 0:
        history_df = _enrich_history_core(history_df)

    # Overwrite price_per_m2 from history for TVs only.
    # TV pipeline benefits from history-derived median across all sizes.
    # Monitor pipeline already computes correct per-product $/m² with
    # per-product aspect ratios — overwriting would use the generic
    # _SCREEN_AREA_M2_GLOBAL lookup which doesn't account for ultrawides.
    if len(history_df) > 0 and product_type == "TVs":
        _m2_map = compute_m2_from_history(history_df)
        df["price_per_m2"] = df["product_id"].map(
            lambda pid: _m2_map.get(pid) or _m2_map.get(str(pid))
        )

# Exclude 8K products globally (tiny segment that skews averages)
_n_8k = 0
if "resolution" in df.columns:
    _n_8k = (df["resolution"] == "8k").sum()
    if _n_8k > 0:
        df = df[df["resolution"] != "8k"].reset_index(drop=True)

if "released_at" in df.columns:
    df["model_year"] = df["released_at"].dt.year

# --- Monthly report download ---
_reports_dir = Path(__file__).parent / "data" / "reports"
_report_files = sorted(_reports_dir.glob("display_intelligence_*.pdf"), reverse=True) if _reports_dir.exists() else []
if _report_files:
    _latest_report = _report_files[0]
    _report_label = _latest_report.stem.replace("display_intelligence_", "").replace("_", " ")
    with open(_latest_report, "rb") as _f:
        st.sidebar.download_button(
            f"Monthly Report ({_report_label})",
            _f.read(),
            file_name=_latest_report.name,
            mime="application/pdf",
            use_container_width=True,
        )
    st.sidebar.divider()

st.sidebar.title("Filters")

# --- Technology checkboxes with color dots ---
all_techs = df["color_architecture"].cat.categories.tolist()
st.sidebar.markdown("**Color Architecture**")
tech_all = st.sidebar.checkbox("All technologies", value=True, key="tech_all")
selected_techs = []
for tech in all_techs:
    if st.sidebar.checkbox(tech, value=tech_all, key=f"tech_{tech}"):
        selected_techs.append(tech)

# --- Display Type checkboxes ---
st.sidebar.markdown("**Display Type**")
all_display_types = sorted(df["display_type"].dropna().unique())
dt_all = st.sidebar.checkbox("All display types", value=True, key="dt_all")
selected_display_types = []
for dt in all_display_types:
    if st.sidebar.checkbox(dt, value=dt_all, key=f"dt_{dt}"):
        selected_display_types.append(dt)

# --- Brand checkboxes ---
all_brands = sorted(df["brand"].dropna().unique())
st.sidebar.markdown("**Brands**")
brand_all = st.sidebar.checkbox("All brands", value=True, key="brand_all")
show_brands = st.sidebar.checkbox("Show brand list", value=False, key="show_brand_list")
selected_brands = []
if show_brands:
    for brand in all_brands:
        checked = st.sidebar.checkbox(brand, value=brand_all, key=f"brand_{brand}")
        if checked:
            selected_brands.append(brand)
else:
    if brand_all:
        selected_brands = list(all_brands)
    else:
        selected_brands = []

st.sidebar.divider()

# --- Price range ---
price_min = float(df["price_best"].min()) if df["price_best"].notna().any() else 0
price_max = float(df["price_best"].max()) if df["price_best"].notna().any() else 10000
price_range = st.sidebar.slider(
    "Price Range ($)", min_value=0, max_value=int(price_max) + 500,
    value=(0, int(price_max) + 500), step=50,
)
_unpriced_label = "products" if _is_blended else PCFG["item_label"].lower()
include_unpriced = st.sidebar.checkbox(f"Include {_unpriced_label} without pricing", value=True)

# --- Model Year checkboxes (if available) ---
selected_years = []
if PCFG["has_model_year"] and "model_year" in df.columns:
    available_years = sorted(df["model_year"].dropna().unique().astype(int).tolist(), reverse=True)
    if available_years:
        st.sidebar.markdown("**Model Year**")
        year_all = st.sidebar.checkbox("All years", value=True, key="year_all")
        for yr in available_years:
            if st.sidebar.checkbox(str(yr), value=year_all, key=f"year_{yr}"):
                selected_years.append(yr)

# --- Build filter mask ---
mask = (
    df["color_architecture"].isin(selected_techs)
    & df["display_type"].isin(selected_display_types)
    & df["brand"].isin(selected_brands)
)
if selected_years and "model_year" in df.columns:
    mask = mask & (df["model_year"].isin(selected_years) | df["model_year"].isna())
if include_unpriced:
    mask = mask & (df["price_best"].isna() | df["price_best"].between(price_range[0], price_range[1]))
else:
    mask = mask & df["price_best"].between(price_range[0], price_range[1])

fdf = df[mask].copy()
_filter_label = "Products" if _is_blended else PCFG["item_label"]
st.sidebar.markdown(f"**Showing {len(fdf)}/{len(df)} {_filter_label}**")

# Temporal dataframe — all filters EXCEPT year, for year-over-year analysis
temporal_mask = (
    df["color_architecture"].isin(selected_techs)
    & df["display_type"].isin(selected_display_types)
    & df["brand"].isin(selected_brands)
)
if include_unpriced:
    temporal_mask = temporal_mask & (df["price_best"].isna() | df["price_best"].between(price_range[0], price_range[1]))
else:
    temporal_mask = temporal_mask & df["price_best"].between(price_range[0], price_range[1])
tdf = df[temporal_mask].copy()
caveats = []
if _n_8k > 0:
    caveats.append(f"{_n_8k} 8K sets excluded from all metrics")
if _n_woled_excluded > 0:
    caveats.append(f"{_n_woled_excluded} Samsung WOLED-panel SKUs excluded from QD-OLED pricing")
if caveats:
    st.sidebar.caption(" · ".join(caveats))

# Support deep-linking via ?page=...
if _is_blended:
    ALL_PAGES = ["Cross-Product Analysis"]
    page = "Cross-Product Analysis"
else:
    ALL_PAGES = ["Overview", "Technology Explorer", "Price Analyzer", "Temporal Analysis", "Comparison Tool", PCFG["profile_page"]]
    qp_page = st.query_params.get("page", None)
    default_idx = ALL_PAGES.index(qp_page) if qp_page in ALL_PAGES else 0
    page = st.sidebar.radio("View", ALL_PAGES, index=default_idx)

# --- Version info (bottom of sidebar) ---
_VERSION = "2.0"
_CHANGELOG_TEXT = """\
**2.0 — 2026-03-24**
- Monitor support (70 monitors, v2.1.2+)
- "All Products" cross-product analysis view
- Master RTINGS Score by technology
- QD Adoption dashboard with brand breakdown
- $/m\u00b2 comparison across TVs vs Monitors
- Brand strategy heatmap & FWHM cross-validation
- QD SKU Tracker weekly email export
- SPD calibration hardening

**1.0 — 2026-02-15**
- Initial TV dashboard with 85+ TVs
- SPD-based technology classification
- Keepa + Best Buy pricing pipeline
- 6 pages: Overview, Tech Explorer, Price Analyzer, Temporal, Comparison, Profiles
"""
st.sidebar.divider()
st.sidebar.markdown(
    f"<p style='text-align:center;color:#555;font-size:0.8em;margin-bottom:2px'>Version {_VERSION}</p>",
    unsafe_allow_html=True,
)
if st.sidebar.button("What's new?", use_container_width=True):
    st.sidebar.markdown(_CHANGELOG_TEXT)


# ============================================================================
# PAGE: Overview
# ============================================================================
if page == "Overview":
    st.title(f"{PCFG['item_singular']} Display Technology Dashboard")
    _bench_label = "v2.0+" if product_type == "TVs" else "v2.1.2+"
    st.caption(f"Database: {len(df)} {PCFG['item_label']} — test bench {_bench_label} · "
               f"Data covers RTINGS-reviewed models only, not the full {PCFG['item_singular'].lower()} market")

    priced = fdf[fdf["price_best"].notna()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(PCFG["item_label"], len(fdf))
    c2.metric("With Pricing", len(priced))
    c3.metric("Brands", fdf["brand"].nunique())
    c4.metric("Avg Price", f"${priced['price_best'].mean():,.0f}" if len(priced) else "N/A")
    c5.metric("Avg $/m\u00b2", f"${priced['price_per_m2'].mean():,.0f}" if len(priced) else "N/A")

    st.divider()

    # --- Hero charts: Pricing & Performance at a glance ---
    hero1, hero2 = st.columns(2)
    with hero1:
        st.subheader("Technology Cost per m\u00b2")
        if len(priced) > 0:
            m2_hero = (priced.dropna(subset=["price_per_m2"])
                       .groupby("color_architecture", observed=False)["price_per_m2"]
                       .mean().reset_index())
            m2_hero.columns = ["Technology", "Avg $/m\u00b2"]
            wled_base = m2_hero.loc[m2_hero["Technology"] == "WLED", "Avg $/m\u00b2"]
            wled_hero = float(wled_base.iloc[0]) if len(wled_base) > 0 else None
            fig = px.bar(m2_hero, x="Technology", y="Avg $/m\u00b2", color="Technology",
                         color_discrete_map=TECH_COLORS, text="Avg $/m\u00b2",
                         category_orders={"Technology": TECH_ORDER})
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                              textfont_size=13, textfont_weight=600, cliponaxis=False)
            if wled_hero:
                fig.add_hline(y=wled_hero, line_dash="dot",
                              line_color=TECH_COLORS.get("WLED", "#888"),
                              opacity=0.4)
            fig.update_layout(showlegend=False, height=370, **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No priced {PCFG['item_label'].lower()} in current filter.")

    with hero2:
        _primary = PCFG["primary_score"]
        _primary_label = friendly(_primary)
        st.subheader(f"{_primary_label} Score by Technology")
        if _primary in fdf.columns:
            fig = px.box(fdf, x="color_architecture", y=_primary,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         labels={_primary: f"{_primary_label} Score", "color_architecture": ""},
                         points="all")
            fig.update_layout(showlegend=False, height=370,
                              yaxis=dict(range=axis_range(_primary)), **PL)
            fig.update_traces(marker=dict(size=7))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Color Architecture Distribution")
        tech_counts = fdf["color_architecture"].value_counts().reindex(TECH_ORDER).dropna().reset_index()
        tech_counts.columns = ["Technology", "Count"]
        fig = px.bar(tech_counts, x="Technology", y="Count", color="Technology",
                     color_discrete_map=TECH_COLORS, text="Count",
                     category_orders={"Technology": TECH_ORDER})
        fig.update_layout(showlegend=False, height=350, **PL)
        fig.update_traces(textposition="outside", textfont_size=14, textfont_weight=600,
                          cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Display Type & Brand")
        brand_tech = fdf.groupby(["brand", "display_type"]).size().reset_index(name="Count")
        fig = px.bar(brand_tech, x="brand", y="Count", color="display_type",
                     color_discrete_map=DISPLAY_TYPE_COLORS,
                     labels={"brand": "Brand", "display_type": "Display Type"})
        fig.update_layout(height=350, legend_title_text="", **PL)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Price Distribution")
        if len(priced) > 0:
            fig = px.histogram(priced, x="price_best", nbins=25,
                               color="color_architecture", color_discrete_map=TECH_COLORS,
                               category_orders={"color_architecture": TECH_ORDER},
                               labels={"price_best": "Price ($)", "color_architecture": "Technology"})
            fig.update_layout(height=350, barmode="stack", legend_title_text="",
                              xaxis=dict(range=axis_range("price_best")), **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No priced {PCFG['item_label'].lower()} in current filter.")

    with col4:
        st.subheader("Price by Technology")
        if len(priced) > 0:
            fig = px.box(priced, x="color_architecture", y="price_best",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         labels={"price_best": "Price ($)", "color_architecture": "Technology"},
                         hover_name="fullname", points="all")
            fig.update_layout(showlegend=False, height=350,
                              yaxis=dict(range=axis_range("price_best")), **PL)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No priced {PCFG['item_label'].lower()} in current filter.")

    st.subheader("Usage Score Overview")
    score_cols = [c for c in PCFG["score_cols"] if c in fdf.columns]
    score_data = fdf[["fullname", "color_architecture"] + score_cols].melt(
        id_vars=["fullname", "color_architecture"], value_vars=score_cols,
        var_name="Usage", value_name="Score"
    )
    score_data["Usage"] = score_data["Usage"].map(friendly)
    fig = px.box(score_data, x="Usage", y="Score", color="color_architecture",
                 color_discrete_map=TECH_COLORS,
                 category_orders={"color_architecture": TECH_ORDER},
                 labels={"color_architecture": "Technology"})
    fig.update_layout(height=400, legend_title_text="",
                      yaxis=dict(range=axis_range("mixed_usage")), **PL)
    fig.update_traces(marker=dict(size=7))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: Technology Explorer
# ============================================================================
elif page == "Technology Explorer":
    st.title("Technology Explorer")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Technology Table", "SPD Analysis", "Panel Metrics",
        "QD Advantage", "Mixed Usage Drivers", "Value Analysis",
    ])

    # --- Tab 1: Technology Table ---
    with tab1:
        st.subheader("Display Technology Classification")
        tech_cols = [
            "fullname", "brand", "display_type", "color_architecture",
            "backlight_type_v2", "dimming_zone_count", "qd_present",
            "qd_material", "spd_verified", "marketing_label",
            "panel_sub_type", "panel_type",
        ]
        available_cols = [c for c in tech_cols if c in fdf.columns]
        display_df = fdf[available_cols].sort_values(["color_architecture", "brand", "fullname"])
        st.dataframe(display_df, use_container_width=True, height=600)

    # --- Tab 2: SPD Analysis ---
    with tab2:
        st.subheader("SPD Peak Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Green Peak FWHM by Technology**")
            fig = px.strip(fdf, x="color_architecture", y="green_fwhm_nm",
                           color="color_architecture", color_discrete_map=TECH_COLORS,
                           category_orders={"color_architecture": TECH_ORDER},
                           hover_name="fullname",
                           labels={"green_fwhm_nm": "Green FWHM (nm)", "color_architecture": ""})
            fig.add_hline(y=28, line_dash="dash", line_color="gray",
                          annotation_text="QD-LCD threshold (28nm)",
                          annotation_font_size=12)
            fig.add_hline(y=40, line_dash="dash", line_color="gray",
                          annotation_text="Pseudo QD threshold (40nm)",
                          annotation_font_size=12)
            fig.update_layout(showlegend=False, height=450,
                              yaxis=dict(range=[0, 60]), **PL)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Red Peak FWHM by Technology**")
            valid_red = fdf[fdf["red_fwhm_nm"].notna()]
            fig = px.strip(valid_red, x="color_architecture", y="red_fwhm_nm",
                           color="color_architecture", color_discrete_map=TECH_COLORS,
                           category_orders={"color_architecture": TECH_ORDER},
                           hover_name="fullname",
                           labels={"red_fwhm_nm": "Red FWHM (nm)", "color_architecture": ""})
            fig.add_hline(y=10, line_dash="dash", line_color="gray",
                          annotation_text="KSF narrow (<10nm)",
                          annotation_font_size=12)
            fig.add_hline(y=40, line_dash="dash", line_color="gray",
                          annotation_text="Broad threshold",
                          annotation_font_size=12)
            fig.update_layout(showlegend=False, height=450,
                              yaxis=dict(range=[0, 60]), **PL)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Green vs Red FWHM — Technology Clusters**")
        valid_both = fdf[fdf["green_fwhm_nm"].notna() & fdf["red_fwhm_nm"].notna()]
        fig = px.scatter(valid_both, x="green_fwhm_nm", y="red_fwhm_nm",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         hover_name="fullname", hover_data=["brand", "marketing_label"],
                         labels={"green_fwhm_nm": "Green FWHM (nm)", "red_fwhm_nm": "Red FWHM (nm)"})
        fig.add_shape(type="rect", x0=0, x1=28, y0=0, y1=28,
                       line=dict(color="rgba(255,199,0,0.5)", dash="dash"),
                       fillcolor="rgba(255,199,0,0.05)")
        fig.add_annotation(x=14, y=2, text="QD-LCD zone", showarrow=False,
                           font=dict(color="rgba(255,199,0,0.8)", size=13))
        fig.update_layout(height=500, legend_title_text="Technology",
                          xaxis=dict(range=[0, 60]),
                          yaxis=dict(range=[0, 60]), **PL)
        fig.update_traces(marker=MARKER)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Panel Metrics ---
    with tab3:
        st.subheader("Panel Performance by Technology")

        _all_metric_options = [
            "native_contrast", "hdr_peak_10pct_nits", "sdr_real_scene_peak_nits",
            "hdr_bt2020_coverage_itp_pct", "sdr_dci_p3_coverage_pct",
            PCFG["input_lag_col"], "total_response_time_ms",
        ]
        metric_options = [m for m in _all_metric_options if m in fdf.columns]
        metric = st.selectbox(
            "Metric",
            metric_options,
            format_func=friendly,
        )

        valid = fdf[fdf[metric].notna()].copy()

        # Native contrast: OLEDs have infinite values (perfect blacks).
        # Show only finite data in the box plot; add colored bars at top for OLEDs.
        if metric == "native_contrast" and np.isinf(valid[metric]).any():
            inf_techs = valid.loc[np.isinf(valid[metric]), "color_architecture"].unique()
            finite = valid[np.isfinite(valid[metric])]
            fig = px.box(finite, x="color_architecture", y=metric,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         points="all", hover_name="fullname",
                         labels={metric: friendly(metric), "color_architecture": ""})
            y_range = axis_range(metric, data_df=finite)
            fig.update_layout(showlegend=False, height=500,
                              yaxis=dict(range=y_range), **PL)
            # Add colored "∞" bars at the top for each OLED technology
            ymax = y_range[1] if y_range else 10000
            for tech in inf_techs:
                xi = TECH_ORDER.index(tech) if tech in TECH_ORDER else -1
                if xi < 0:
                    continue
                color = TECH_COLORS.get(tech, "#888")
                fig.add_shape(type="rect", x0=xi - 0.35, x1=xi + 0.35,
                              y0=ymax * 0.92, y1=ymax * 0.98,
                              fillcolor=color, opacity=0.8, line_width=0)
                fig.add_annotation(text="∞", x=xi, y=ymax * 0.95,
                                   showarrow=False,
                                   font=dict(size=18, color="white", family="Inter"))
        else:
            fig = px.box(valid, x="color_architecture", y=metric,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         points="all", hover_name="fullname",
                         labels={metric: friendly(metric), "color_architecture": ""})
            fig.update_layout(showlegend=False, height=500,
                              yaxis=dict(range=axis_range(metric)), **PL)
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Average Scores by Technology**")
        score_cols = PCFG["score_cols"] + PCFG["extra_score_cols"]
        available_scores = [c for c in score_cols if c in fdf.columns]
        avg_scores = fdf.groupby("color_architecture")[available_scores].mean()
        avg_scores.columns = [friendly(c) for c in avg_scores.columns]
        fig = px.imshow(avg_scores.T, text_auto=".1f", color_continuous_scale="Viridis",
                        labels={"x": "Technology", "y": "Score", "color": "Value"},
                        aspect="auto")
        fig.update_layout(height=400, **PL)
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: QD Advantage ---
    with tab4:
        st.subheader("Quantum Dot Performance Advantage")
        st.caption("How QD technologies compare on key picture quality metrics")

        advantage_metrics = [
            ("hdr_peak_10pct_nits", "HDR Peak Brightness\n(10%, nits)"),
            ("hdr_bt2020_coverage_itp_pct", "HDR BT.2020 Coverage\n(ITP %)"),
            ("sdr_dci_p3_coverage_pct", "SDR DCI-P3 Coverage\n(%)"),
            ("contrast_ratio_score", "Contrast Ratio Score"),
            ("color_score", "Color Score"),
            ("brightness_score", "Brightness Score"),
        ]

        for row_start in range(0, 6, 3):
            cols = st.columns(3)
            for j, (metric_col, metric_label) in enumerate(advantage_metrics[row_start:row_start + 3]):
                with cols[j]:
                    means = (
                        fdf.groupby("color_architecture")[metric_col]
                        .mean().reset_index()
                        .sort_values(metric_col, ascending=True)
                    )
                    means.columns = ["Technology", "Value"]
                    fig = px.bar(
                        means, y="Technology", x="Value", orientation="h",
                        color="Technology", color_discrete_map=TECH_COLORS,
                        text=means["Value"].apply(lambda v: f"{v:.0f}" if v > 20 else f"{v:.1f}"),
                    )
                    # Pad x-axis so "outside" text labels aren't clipped
                    x_max = means["Value"].max()
                    x_pad = x_max * 0.25 if x_max > 0 else 1
                    fig.update_layout(
                        title=dict(text=metric_label, font=dict(size=14, family="Inter, sans-serif")),
                        showlegend=False, height=280,
                        margin=dict(l=0, r=10, t=40, b=0),
                        xaxis=dict(title="", range=[0, x_max + x_pad]),
                        yaxis_title="",
                        font=dict(family="Inter, sans-serif", size=13),
                    )
                    fig.update_traces(textposition="outside", textfont_size=13, textfont_weight=600,
                                      cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Response time comparison (LCD technologies only)
        st.markdown("**Response Time: QD-LCD vs KSF & Pseudo QD**")
        st.caption("Lower = faster. QD-LCD shows statistically significant speed advantage over KSF and Pseudo QD phosphor-based technologies.")

        lcd_techs = ["QD-LCD", "Pseudo QD", "KSF", "WLED"]
        lcd_data = fdf[fdf["color_architecture"].isin(lcd_techs)].copy()

        resp_metrics = [
            ("total_response_time_ms", "Total Response Time (ms)"),
            ("first_response_time_ms", "First Response Time (ms)"),
        ]
        rcols = st.columns(2)
        for k, (resp_col, resp_label) in enumerate(resp_metrics):
            with rcols[k]:
                valid_r = lcd_data[lcd_data[resp_col].notna()]
                fig = px.box(
                    valid_r, x="color_architecture", y=resp_col,
                    color="color_architecture", color_discrete_map=TECH_COLORS,
                    category_orders={"color_architecture": TECH_ORDER},
                    points="all", hover_name="fullname",
                    labels={resp_col: resp_label, "color_architecture": ""},
                )
                # Add mean annotation per tech
                for tech in lcd_techs:
                    t_vals = valid_r[valid_r["color_architecture"] == tech][resp_col]
                    if len(t_vals) > 0:
                        fig.add_annotation(
                            x=tech, y=t_vals.max() + 0.5,
                            text=f"avg {t_vals.mean():.1f}ms",
                            showarrow=False,
                            font=dict(size=12, weight=600),
                        )
                fig.update_layout(showlegend=False, height=380,
                                  yaxis=dict(range=axis_range(resp_col)), **PL)
                fig.update_traces(marker=dict(size=9))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Technology radar overlay
        st.markdown("**Technology Profile Radar**")
        st.caption("Metrics normalized 0\u20131 across technologies (higher = better on all axes)")
        radar_metrics = [
            ("brightness_score", "Brightness"),
            ("contrast_ratio_score", "Contrast"),
            ("color_score", "Color"),
            ("black_level_score", "Black Level"),
            ("hdr_bt2020_coverage_itp_pct", "HDR Gamut"),
        ]
        tech_means = fdf.groupby("color_architecture").agg(
            {m[0]: "mean" for m in radar_metrics}
        )
        # Response speed: invert so higher = faster (better)
        resp_mean = fdf.groupby("color_architecture")["total_response_time_ms"].mean()
        max_resp = resp_mean.max()
        tech_means["response_speed"] = max_resp - resp_mean  # linear inversion (higher = faster)

        radar_labels = [m[1] for m in radar_metrics] + ["Response Speed"]
        radar_cols = [m[0] for m in radar_metrics] + ["response_speed"]

        # Min-max normalize
        for col in radar_cols:
            col_min = tech_means[col].min()
            col_max = tech_means[col].max()
            if col_max > col_min:
                tech_means[col] = (tech_means[col] - col_min) / (col_max - col_min)
            else:
                tech_means[col] = 0.5

        fig = go.Figure()
        for tech in tech_means.index:
            values = [tech_means.loc[tech, c] for c in radar_cols]
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values, theta=radar_labels + [radar_labels[0]],
                fill="toself", name=str(tech), opacity=0.45,
                line=dict(color=TECH_COLORS.get(str(tech), "#888"), width=3),
                fillcolor=TECH_COLORS.get(str(tech), "#888"),
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.05],
                                tickfont=dict(size=12)),
                angularaxis=dict(tickfont=dict(size=14, weight=600)),
            ),
            height=520, legend_title_text="Technology",
            legend=dict(font=dict(size=13)),
            font=dict(family="Inter, sans-serif", size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        # QD vs Non-QD headline callouts
        st.divider()
        st.markdown("**QD vs Non-QD: Headline Advantages**")
        qd_techs = ["QD-OLED", "QD-LCD"]
        qd_mask = fdf["color_architecture"].isin(qd_techs)
        headline_metrics = [
            ("hdr_peak_10pct_nits", "HDR Peak Brightness"),
            ("hdr_bt2020_coverage_itp_pct", "HDR Color Gamut"),
            (PCFG["primary_score"], friendly(PCFG["primary_score"]) + " Score"),
            ("brightness_score", "Brightness Score"),
        ]
        hcols = st.columns(len(headline_metrics))
        for i, (hm_col, hm_label) in enumerate(headline_metrics):
            qd_val = fdf[qd_mask][hm_col].mean()
            non_val = fdf[~qd_mask][hm_col].mean()
            pct = ((qd_val - non_val) / non_val * 100) if non_val else 0
            with hcols[i]:
                st.metric(
                    hm_label,
                    f"{qd_val:.0f}" if qd_val > 20 else f"{qd_val:.1f}",
                    delta=f"+{pct:.0f}% vs non-QD",
                )

    # --- Tab 5: Score Drivers ---
    with tab5:
        _ps = PCFG["primary_score"]
        _ps_label = friendly(_ps)
        st.subheader(f"What Drives {_ps_label} Scores?")
        st.caption(f"Correlation analysis: which metrics predict overall {PCFG['item_singular'].lower()} performance")

        _all_corr_metrics = {
            "contrast_ratio_score": "Contrast Ratio Score",
            "black_level_score": "Black Level Score",
            "color_score": "Color Score",
            "color_accuracy": "Color Accuracy",
            "hdr_bt2020_coverage_itp_pct": "HDR BT.2020 Coverage",
            "brightness_score": "Brightness Score",
            "sdr_dci_p3_coverage_pct": "DCI-P3 Coverage",
            "native_contrast_score": "Native Contrast Score",
            "hdr_peak_10pct_nits": "HDR Peak Brightness",
            "hdr_peak_2pct_nits": "HDR Peak (2%)",
            "sdr_real_scene_peak_nits": "SDR Peak Brightness",
            "total_response_time_ms": "Response Time",
            PCFG["input_lag_col"]: friendly(PCFG["input_lag_col"]),
        }
        corr_metrics = {k: v for k, v in _all_corr_metrics.items() if k in fdf.columns}
        corr_data = []
        for col, label in corr_metrics.items():
            if col in fdf.columns and _ps in fdf.columns:
                valid = fdf[[_ps, col]].dropna()
                if len(valid) > 5:
                    r = valid[_ps].corr(valid[col])
                    corr_data.append({"Metric": label, "col": col, "Correlation": r})

        corr_df = pd.DataFrame(corr_data).sort_values("Correlation")
        corr_df["Direction"] = corr_df["Correlation"].apply(
            lambda x: "Positive" if x >= 0 else "Negative"
        )

        fig = px.bar(corr_df, y="Metric", x="Correlation", orientation="h",
                     color="Direction",
                     color_discrete_map={"Positive": "#4B40EB", "Negative": "#FF009F"},
                     text=corr_df["Correlation"].apply(lambda x: f"{x:.2f}"))
        fig.add_vline(x=0, line_color="white", line_width=1)
        fig.update_layout(
            height=450, showlegend=False,
            xaxis=dict(range=[-1, 1], title=f"Pearson Correlation with {_ps_label}"),
            yaxis_title="",
            margin=dict(l=0, r=60, t=10, b=0),
            **PL,
        )
        fig.update_traces(textposition="outside", textfont_size=13, textfont_weight=600,
                          cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        scol1, scol2 = st.columns(2)

        # Scatter plots — only show if both columns exist
        _driver_x_cols = ["contrast_ratio_score", "total_response_time_ms"]
        _avail_drivers = [c for c in _driver_x_cols if c in fdf.columns and _ps in fdf.columns]
        if _avail_drivers:
            _dcols = st.columns(len(_avail_drivers))
            for _di, _dx in enumerate(_avail_drivers):
                with _dcols[_di]:
                    valid = fdf[[_dx, _ps, "color_architecture", "fullname"]].dropna()
                    if len(valid) > 3:
                        r = valid[_dx].corr(valid[_ps])
                        st.markdown(f"**{friendly(_dx)} vs {_ps_label}** (r = {r:.2f})")
                        fig = px.scatter(valid, x=_dx, y=_ps,
                                         color="color_architecture", color_discrete_map=TECH_COLORS,
                                         category_orders={"color_architecture": TECH_ORDER},
                                         hover_name="fullname",
                                         labels={_dx: friendly(_dx), _ps: _ps_label})
                        x_arr = valid[_dx].values
                        y_arr = valid[_ps].values
                        m, b = np.polyfit(x_arr, y_arr, 1)
                        x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
                        r2 = np.corrcoef(x_arr, y_arr)[0, 1] ** 2
                        fig.add_trace(go.Scatter(
                            x=x_line, y=m * x_line + b, mode="lines",
                            name=f"r\u00b2 = {r2:.2f}",
                            line=dict(color="rgba(255,255,255,0.5)", dash="dash", width=2),
                        ))
                        fig.update_layout(height=420, showlegend=True, legend_title_text="",
                                          xaxis=dict(range=axis_range(_dx)),
                                          yaxis=dict(range=axis_range(_ps)), **PL)
                        fig.update_traces(marker=MARKER, selector=dict(mode="markers"))
                        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**Technology Positioning Map**")
        st.caption("Bubble size = Mixed Usage score. Top-right = best brightness + widest gamut.")
        pos_valid = fdf[
            fdf["hdr_peak_10pct_nits"].notna() & fdf["hdr_bt2020_coverage_itp_pct"].notna()
        ].copy()
        if len(pos_valid) > 0:
            fig = px.scatter(
                pos_valid, x="hdr_peak_10pct_nits", y="hdr_bt2020_coverage_itp_pct",
                color="color_architecture", color_discrete_map=TECH_COLORS,
                category_orders={"color_architecture": TECH_ORDER},
                size=PCFG["primary_score"] if PCFG["primary_score"] in pos_valid.columns else None,
                size_max=22,
                hover_name="fullname", hover_data=["brand", "price_best"],
                labels={
                    "hdr_peak_10pct_nits": "HDR Peak Brightness (nits)",
                    "hdr_bt2020_coverage_itp_pct": "HDR BT.2020 Coverage (%)",
                },
            )
            med_x = pos_valid["hdr_peak_10pct_nits"].median()
            med_y = pos_valid["hdr_bt2020_coverage_itp_pct"].median()
            fig.add_hline(y=med_y, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.add_vline(x=med_x, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.update_layout(height=540, legend_title_text="Technology",
                              xaxis=dict(range=axis_range("hdr_peak_10pct_nits")),
                              yaxis=dict(range=axis_range("hdr_bt2020_coverage_itp_pct")), **PL)
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 6: Value Analysis ---
    with tab6:
        st.subheader("Value Analysis: Cost per Performance Point")
        st.caption("Lower $/point = more performance for your money")

        val_priced = fdf[fdf["price_per_mixed_use"].notna()].copy()
        if len(val_priced) == 0:
            st.warning(f"No priced {PCFG['item_label'].lower()} match the current filters.")
        else:
            fig = px.box(val_priced, x="color_architecture", y="price_per_mixed_use",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         points="all", hover_name="fullname",
                         labels={"price_per_mixed_use": "$ per Mixed Usage Point",
                                 "color_architecture": ""})
            fig.update_layout(showlegend=False, height=440,
                              yaxis=dict(range=axis_range("price_per_mixed_use")), **PL)
            fig.update_traces(marker=dict(size=9))
            fig.add_annotation(
                x=0.5, y=-0.12, xref="paper", yref="paper",
                text="Lower = better value", showarrow=False,
                font=dict(size=13, color="rgba(255,255,255,0.5)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            _ps = PCFG["primary_score"]
            _ps_label = friendly(_ps)
            st.markdown("**Value Frontier: Price vs Performance**")
            st.caption(f"The dashed line traces the \"efficient frontier\" — {PCFG['item_label'].lower()} that offer "
                       f"the highest {_ps_label} score for their price. Any {PCFG['item_singular'].lower()} on or near "
                       f"the line is the best performance you can buy at that budget. "
                       f"{PCFG['item_label']} far below the line are overpriced for what they deliver.")
            _hover = ["price_size", "brand"]
            if PCFG["price_per_score_col"] and PCFG["price_per_score_col"] in val_priced.columns:
                _hover.insert(0, PCFG["price_per_score_col"])
            fig = px.scatter(val_priced, x="price_best", y=_ps,
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             category_orders={"color_architecture": TECH_ORDER},
                             hover_name="fullname",
                             hover_data=_hover,
                             labels={"price_best": "Price ($)", _ps: f"{_ps_label} Score"})
            sorted_v = val_priced.sort_values(_ps, ascending=False)
            frontier = []
            min_price = float("inf")
            for _, row in sorted_v.iterrows():
                if row["price_best"] <= min_price:
                    frontier.append(row)
                    min_price = row["price_best"]
            if frontier:
                ffront = pd.DataFrame(frontier).sort_values("price_best")
                fig.add_trace(go.Scatter(
                    x=ffront["price_best"], y=ffront[_ps],
                    mode="lines+markers", name="Value Frontier",
                    line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=2),
                    marker=dict(size=7, color="rgba(255,255,255,0.6)"),
                ))
            fig.update_layout(height=520, legend_title_text="Technology",
                              xaxis=dict(range=axis_range("price_best")),
                              yaxis=dict(range=axis_range(_ps)), **PL)
            fig.update_traces(marker=MARKER, selector=dict(mode="markers"))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            if PCFG["has_price_per_score"] and PCFG["price_per_score_col"] in val_priced.columns:
                _pps = PCFG["price_per_score_col"]
                st.markdown(f"**Top 20 Best Value {PCFG['item_label']}** (lowest {PCFG['price_per_score_label']})")
                top_val = val_priced.sort_values(_pps).head(20)
            else:
                st.markdown(f"**Top 20 Best Value {PCFG['item_label']}** (lowest price)")
                top_val = val_priced.sort_values("price_best").head(20)
                _pps = None
            _val_cols = ["fullname", "color_architecture", "price_best", _ps]
            _val_headers = [PCFG["item_singular"], "Technology", "Price", _ps_label]
            if _pps and _pps in top_val.columns:
                _val_cols.append(_pps)
                _val_headers.append("$/Point")
            for _vc in ["hdr_peak_10pct_nits", "hdr_bt2020_coverage_itp_pct", "price_size"]:
                if _vc in top_val.columns:
                    _val_cols.append(_vc)
                    _val_headers.append(friendly(_vc).split("(")[0].strip())
            val_table = top_val[[c for c in _val_cols if c in top_val.columns]].copy()
            val_table.columns = _val_headers[:len(val_table.columns)]
            val_table["Price"] = val_table["Price"].apply(lambda x: f"${x:,.0f}")
            if "$/Point" in val_table.columns:
                val_table["$/Point"] = val_table["$/Point"].apply(lambda x: f"${x:,.0f}")
            val_table[_ps_label] = val_table[_ps_label].apply(lambda x: f"{x:.1f}")
            val_table["HDR Brightness"] = val_table["HDR Brightness"].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "\u2014")
            val_table["BT.2020 %"] = val_table["BT.2020 %"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "\u2014")
            val_table["Size"] = val_table["Size"].apply(
                lambda x: f'{int(x)}"' if pd.notna(x) else "?")
            st.dataframe(val_table, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Average $/Point by Technology**")
            overall_avg = val_priced["price_per_mixed_use"].mean()
            tech_means_val = val_priced.groupby("color_architecture")["price_per_mixed_use"].mean()
            mcols = st.columns(len(tech_means_val))
            for i, (tech, mean_val) in enumerate(tech_means_val.sort_values().items()):
                delta = mean_val - overall_avg
                with mcols[i]:
                    st.metric(str(tech), f"${mean_val:,.0f}/pt",
                              delta=f"${delta:+,.0f} vs avg", delta_color="inverse")


# ============================================================================
# PAGE: Price Analyzer
# ============================================================================
elif page == "Price Analyzer":
    st.title("Price Analyzer")

    priced = fdf[fdf["price_best"].notna()].copy()
    if len(priced) == 0:
        st.warning(f"No priced {PCFG['item_label'].lower()} match the current filters.")
        st.stop()

    # Merge Amazon channels: amazon + amazon_3p → Amazon
    if "price_source" in priced.columns:
        priced["channel"] = priced["price_source"].replace({
            "amazon": "Amazon", "amazon_3p": "Amazon",
            "bestbuy": "Best Buy", "rtings": "RTINGS (affiliate)",
        })

    # --- Headline: Technology Cost per m² with WLED premium ---
    st.subheader("Technology Cost per m\u00b2")
    st.caption("Median price per square meter by display technology, with premium over WLED baseline. "
               "Per-product median across all available sizes. Data limited to RTINGS-reviewed models.")

    m2_data = (priced.dropna(subset=["price_per_m2"])
               .groupby("color_architecture", observed=False)["price_per_m2"]
               .mean().reset_index())
    m2_data.columns = ["Technology", "Avg $/m\u00b2"]
    wled_baseline = m2_data.loc[m2_data["Technology"] == "WLED", "Avg $/m\u00b2"]
    wled_val = float(wled_baseline.iloc[0]) if len(wled_baseline) > 0 else None

    fig = px.bar(m2_data, x="Technology", y="Avg $/m\u00b2", color="Technology",
                 color_discrete_map=TECH_COLORS, text="Avg $/m\u00b2",
                 category_orders={"Technology": TECH_ORDER})
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                      textfont_size=14, textfont_weight=600, cliponaxis=False)
    if wled_val:
        fig.add_hline(y=wled_val, line_dash="dot", line_color=TECH_COLORS.get("WLED", "#888"),
                      opacity=0.4)
    fig.update_layout(showlegend=False, height=400, **PL)
    st.plotly_chart(fig, use_container_width=True)

    # Premium metrics row
    m2_col = "Avg $/m\u00b2"
    if wled_val and wled_val > 0:
        techs_with_m2 = m2_data[m2_data["Technology"] != "WLED"].sort_values(m2_col)
        mcols = st.columns(len(techs_with_m2))
        for i, (_, row) in enumerate(techs_with_m2.iterrows()):
            med_val = row[m2_col]
            premium_pct = (med_val - wled_val) / wled_val * 100
            with mcols[i]:
                st.metric(str(row["Technology"]),
                          f"${med_val:,.0f}/m\u00b2",
                          delta=f"+{premium_pct:.0f}% vs WLED")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Value Map", "Price/m\u00b2", "Best Deals", "Price Trends", "Channels"])

    with tab1:
        score_metric = st.selectbox(
            "Score metric",
            [c for c in PCFG["score_cols"] if c in fdf.columns],
            format_func=friendly,
            key="value_score",
        )
        score_label = friendly(score_metric)
        st.caption("The dashed \"Value Frontier\" traces the best-performing TV at each price point. "
                   "TVs on the line are the best you can buy at that budget; TVs below it are overpriced for their score.")

        fig = px.scatter(priced, x="price_best", y=score_metric,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         hover_name="fullname",
                         hover_data=["brand", "channel", "price_size"],
                         labels={"price_best": "Price ($)", score_metric: score_label})
        fig.update_layout(height=550, legend_title_text="Technology",
                          xaxis=dict(range=axis_range("price_best")),
                          yaxis=dict(range=axis_range(score_metric)), **PL)
        fig.update_traces(marker=MARKER)

        sorted_p = priced.dropna(subset=[score_metric]).sort_values(score_metric, ascending=False)
        frontier = []
        min_price = float("inf")
        for _, row in sorted_p.iterrows():
            if row["price_best"] <= min_price:
                frontier.append(row)
                min_price = row["price_best"]
        if frontier:
            frontier_df = pd.DataFrame(frontier).sort_values("price_best")
            fig.add_trace(go.Scatter(
                x=frontier_df["price_best"], y=frontier_df[score_metric],
                mode="lines", name="Value Frontier",
                line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=2),
            ))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Price per Square Meter")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(priced.sort_values("price_per_m2"),
                         x="fullname", y="price_per_m2",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         hover_data=["price_best", "price_size", "brand"],
                         labels={"price_per_m2": "$/m\u00b2", "fullname": ""})
            fig.update_layout(height=500, showlegend=False, xaxis_tickangle=-45,
                              yaxis=dict(range=axis_range("price_per_m2")), **PL)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(priced, x="color_architecture", y="price_per_m2",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         points="all", hover_name="fullname",
                         labels={"price_per_m2": "$/m\u00b2", "color_architecture": ""})
            fig.update_layout(height=500, showlegend=False,
                              yaxis=dict(range=axis_range("price_per_m2")), **PL)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        # Per-size $/m² comparison by technology
        st.divider()
        st.subheader("$/m² by Screen Size and Technology")
        st.caption("Median price per m² at each screen size, broken down by technology. "
                   "Shows whether the cost gap between technologies holds across all sizes.")
        if len(history_df) > 0:
            # Use latest snapshot from price_history (already enriched)
            latest_snap = history_df["snapshot_date"].max()
            snap = history_df[history_df["snapshot_date"] == latest_snap].copy()
            # Apply sidebar filters
            snap = snap[snap["product_id"].astype(str).isin(set(fdf["product_id"].astype(str)))]
            snap = snap.dropna(subset=["price_per_m2", "size_inches"])
            snap["size"] = snap["size_inches"].astype(int)

            if len(snap) > 0:
                # Only show sizes with data from 2+ technologies
                common_sizes = [43, 50, 55, 65, 75, 85, 98]
                snap_common = snap[snap["size"].isin(common_sizes)]

                size_tech = (snap_common.groupby(["size", "color_architecture"])["price_per_m2"]
                             .mean().reset_index())
                size_tech.columns = ["Size", "Technology", "Avg $/m\u00b2"]
                size_tech["Size"] = size_tech["Size"].astype(str) + '"'

                fig = px.bar(size_tech, x="Size", y="Avg $/m\u00b2",
                             color="Technology", color_discrete_map=TECH_COLORS,
                             barmode="group",
                             category_orders={
                                 "Technology": TECH_ORDER,
                                 "Size": [f'{s}"' for s in common_sizes],
                             },
                             text="Avg $/m\u00b2")
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                                  textfont_size=10, cliponaxis=False)
                fig.update_layout(height=500, legend_title_text="Technology", **PL)
                st.plotly_chart(fig, use_container_width=True)

                # Also show the raw data table
                pivot = size_tech.pivot(index="Technology", columns="Size", values="Avg $/m\u00b2")
                pivot = pivot.reindex([t for t in TECH_ORDER if t in pivot.index])
                pivot = pivot[[f'{s}"' for s in common_sizes if f'{s}"' in pivot.columns]]
                st.dataframe(
                    pivot.style.format("${:,.0f}", na_rep="—"),
                    use_container_width=True,
                )
        else:
            st.info("No price history available for per-size breakdown.")

        if len(prices_df) > 0:
            st.divider()
            st.subheader("Price by Screen Size")
            sized = prices_df[prices_df["best_price"].notna() & prices_df["size_inches"].notna()].copy()
            if len(sized) > 0:
                if "price_source" in sized.columns:
                    sized["channel"] = sized["price_source"].replace({
                        "amazon": "Amazon", "amazon_3p": "Amazon",
                        "bestbuy": "Best Buy", "rtings": "RTINGS (affiliate)",
                    })
                fig = px.scatter(sized, x="size_inches", y="best_price",
                                 color="channel",
                                 hover_data=["amazon_asin", "bestbuy_sku"],
                                 labels={"size_inches": "Screen Size (inches)",
                                         "best_price": "Price ($)"})
                fig.update_layout(height=400, legend_title_text="Source",
                                  xaxis=dict(range=axis_range("size_inches", prices_df)),
                                  yaxis=dict(range=axis_range("best_price", prices_df)), **PL)
                fig.update_traces(marker=MARKER)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"Best Value {PCFG['item_label']}")
        value_metric = st.selectbox(
            "Optimize for",
            [c for c in PCFG["score_cols"] if c in fdf.columns],
            format_func=friendly,
            key="deal_metric",
        )
        value_label = friendly(value_metric)

        priced = priced.copy()
        priced["value_index"] = priced[value_metric] / priced["price_best"] * 1000
        priced = priced.sort_values("value_index", ascending=False)

        st.markdown(f"**Top 15 by {value_label} per $1,000**")
        deal_cols = ["fullname", "color_architecture", "price_best", "price_size",
                     value_metric, "value_index", "channel"]
        display = priced[deal_cols].head(15).copy()
        display.columns = ["TV", "Technology", "Price", "Size", value_label,
                           f"{value_label}/k$", "Source"]
        display["Price"] = display["Price"].apply(lambda x: f"${x:,.0f}")
        display[f"{value_label}/k$"] = display[f"{value_label}/k$"].apply(lambda x: f"{x:.1f}")
        display["Size"] = display["Size"].apply(lambda x: f'{int(x)}"' if pd.notna(x) else "?")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("**Best Value per Technology**")
        best_per_tech = []
        for tech in all_techs:
            tech_tvs = priced[priced["color_architecture"] == tech]
            if len(tech_tvs) > 0:
                best = tech_tvs.iloc[0]
                best_per_tech.append({
                    "Technology": tech,
                    "Best Value TV": best["fullname"],
                    "Price": f"${best['price_best']:,.0f}",
                    value_label: f"{best[value_metric]:.1f}",
                    f"{value_label}/k$": f"{best['value_index']:.1f}",
                })
        if best_per_tech:
            st.dataframe(pd.DataFrame(best_per_tech), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Avg $/m\u00b2 by Technology Over Time")

        if len(history_df) == 0:
            st.info("No price history available yet. Run the pricing pipeline "
                    "(ideally every Monday) to accumulate weekly data.")
        else:
            # Time granularity selector
            granularity = st.selectbox(
                "Time granularity",
                ["Weekly", "Monthly", "Quarterly", "YTD"],
                key="price_trend_granularity",
            )

            # Apply same filters as the rest of the dashboard (sidebar + 8K exclusion)
            filtered_pids = set(fdf["product_id"].astype(str))
            hist_filtered = history_df[
                history_df["color_architecture"].isin(selected_techs)
                & history_df["product_id"].astype(str).isin(filtered_pids)
            ].copy()
            if len(hist_filtered) == 0:
                st.info("No price history for selected technologies.")
            else:
                # history_df is pre-enriched with price_per_m2, iso_year, iso_week, month, quarter
                hist_m2 = hist_filtered.dropna(subset=["price_per_m2"])

                # Use only the latest snapshot per week
                latest_per_wk = (hist_m2.groupby(["iso_year", "iso_week"])["snapshot_date"]
                                 .max().reset_index(name="_latest"))
                hist_m2 = hist_m2.merge(latest_per_wk, on=["iso_year", "iso_week"])
                hist_m2 = hist_m2[hist_m2["snapshot_date"] == hist_m2["_latest"]].drop(columns=["_latest"])

                # Set groupby keys based on granularity
                if granularity == "Weekly":
                    time_cols = ["iso_year", "iso_week"]
                elif granularity == "Monthly":
                    time_cols = ["month"]
                elif granularity == "Quarterly":
                    time_cols = ["quarter"]
                else:  # YTD
                    time_cols = ["year"]

                # Per-product mean $/m² across sizes, then per-tech mean across products
                prod_agg = (
                    hist_m2.groupby(time_cols + ["product_id", "color_architecture"])["price_per_m2"]
                    .mean().reset_index()
                )
                period_agg = (
                    prod_agg.groupby(time_cols + ["color_architecture"])["price_per_m2"]
                    .mean().reset_index()
                )

                # Build sortable period label
                if granularity == "Weekly":
                    n_years = period_agg["iso_year"].nunique()
                    if n_years <= 1:
                        period_agg["Period"] = "Wk " + period_agg["iso_week"].astype(str)
                        period_agg["_sort"] = period_agg["iso_week"]
                    else:
                        period_agg["Period"] = period_agg["iso_year"].astype(str) + " Wk " + period_agg["iso_week"].astype(str)
                        period_agg["_sort"] = period_agg["iso_year"] * 100 + period_agg["iso_week"]
                elif granularity == "Monthly":
                    period_agg["Period"] = period_agg["month"]
                    period_agg["_sort"] = period_agg["month"]
                elif granularity == "Quarterly":
                    period_agg["Period"] = period_agg["quarter"]
                    period_agg["_sort"] = period_agg["quarter"]
                else:  # YTD
                    period_agg["Period"] = period_agg["year"].astype(str)
                    period_agg["_sort"] = period_agg["year"]

                period_agg = period_agg.sort_values("_sort")
                period_agg.rename(columns={"color_architecture": "Technology",
                                           "price_per_m2": "Avg $/m\u00b2"}, inplace=True)

                n_periods = period_agg["Period"].nunique()

                fig = px.line(
                    period_agg, x="Period", y="Avg $/m\u00b2",
                    color="Technology", color_discrete_map=TECH_COLORS,
                    markers=True,
                    labels={"Avg $/m\u00b2": "Avg $/m\u00b2", "Period": ""},
                    category_orders={"Period": period_agg["Period"].unique().tolist(),
                                     "Technology": TECH_ORDER},
                )
                fig.update_traces(marker=dict(size=10))
                fig.update_layout(height=500, legend_title_text="Technology",
                                  xaxis=dict(type="category"), **PL)
                if n_periods == 1:
                    st.caption(f"Only one {granularity.lower()} period of data.")
                st.plotly_chart(fig, use_container_width=True)

                # Optional size filter for a second chart
                available_sizes = sorted(hist_m2["size_inches"].dropna().unique())
                if len(available_sizes) > 1:
                    size_filter = st.selectbox(
                        "Filter by screen size",
                        ["All sizes"] + [f'{int(s)}"' for s in available_sizes],
                        key="history_size_filter",
                    )
                    if size_filter != "All sizes":
                        size_val = int(size_filter.replace('"', ''))
                        hist_sized = hist_m2[hist_m2["size_inches"] == size_val]
                        sized_agg = (
                            hist_sized.groupby(time_cols + ["color_architecture"])["price_per_m2"]
                            .mean().reset_index()
                        )
                        # Reuse same period labeling logic
                        if granularity == "Weekly":
                            if sized_agg["iso_year"].nunique() <= 1:
                                sized_agg["Period"] = "Wk " + sized_agg["iso_week"].astype(str)
                                sized_agg["_sort"] = sized_agg["iso_week"]
                            else:
                                sized_agg["Period"] = sized_agg["iso_year"].astype(str) + " Wk " + sized_agg["iso_week"].astype(str)
                                sized_agg["_sort"] = sized_agg["iso_year"] * 100 + sized_agg["iso_week"]
                        elif granularity == "Monthly":
                            sized_agg["Period"] = sized_agg["month"]
                            sized_agg["_sort"] = sized_agg["month"]
                        elif granularity == "Quarterly":
                            sized_agg["Period"] = sized_agg["quarter"]
                            sized_agg["_sort"] = sized_agg["quarter"]
                        else:
                            sized_agg["Period"] = sized_agg["year"].astype(str)
                            sized_agg["_sort"] = sized_agg["year"]
                        sized_agg = sized_agg.sort_values("_sort")
                        sized_agg.rename(columns={"color_architecture": "Technology",
                                                  "price_per_m2": "Avg $/m\u00b2"}, inplace=True)
                        fig2 = px.line(
                            sized_agg, x="Period", y="Avg $/m\u00b2",
                            color="Technology", color_discrete_map=TECH_COLORS,
                            markers=True,
                            labels={"Avg $/m\u00b2": f'Avg $/m\u00b2 \u2014 {size_filter}', "Period": ""},
                            category_orders={"Period": sized_agg["Period"].unique().tolist(),
                                             "Technology": TECH_ORDER},
                        )
                        fig2.update_traces(marker=dict(size=10))
                        fig2.update_layout(height=450, legend_title_text="Technology",
                                          xaxis=dict(type="category"), **PL)
                        st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.subheader("Channel Analysis")
        st.caption("Where to find the best prices by display technology. "
                   "Amazon combines 1P (sold by Amazon) and 3P (third-party sellers). "
                   "RTINGS affiliate prices are used as fallback when no live API data is available.")

        if "channel" not in priced.columns:
            st.info("No channel data available.")
        else:
            # --- Channel wins by technology ---
            chan_tech = (priced.groupby(["color_architecture", "channel"], observed=True)
                        .size().reset_index(name="count"))
            # Compute % within each technology
            tech_totals = chan_tech.groupby("color_architecture", observed=True)["count"].transform("sum")
            chan_tech["pct"] = (chan_tech["count"] / tech_totals * 100).round(0)

            CHANNEL_COLORS = {
                "Amazon": "#FF9900",
                "Best Buy": "#0046BE",
                "RTINGS (affiliate)": "#666666",
            }

            fig = px.bar(chan_tech, x="color_architecture", y="pct", color="channel",
                         color_discrete_map=CHANNEL_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         text="pct", barmode="stack",
                         labels={"color_architecture": "Technology", "pct": "% of Best Prices",
                                 "channel": "Channel"})
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                              textfont_size=12)
            fig.update_layout(height=400, legend_title_text="Channel",
                              yaxis=dict(range=[0, 105], title="% of Best Prices"), **PL)
            st.plotly_chart(fig, use_container_width=True)

            # --- QD channel breakdown ---
            st.markdown("**Best Channel for QD-Based TVs**")
            qd_techs = ["QD-LCD", "QD-OLED", "Pseudo QD"]
            qd_priced = priced[priced["color_architecture"].isin(qd_techs)]
            if len(qd_priced) > 0:
                qd_channels = (qd_priced.groupby("channel")
                               .agg(wins=("channel", "size"),
                                    avg_price=("price_best", "mean"),
                                    median_price=("price_best", "median"))
                               .sort_values("wins", ascending=False)
                               .reset_index())
                qd_channels["avg_price"] = qd_channels["avg_price"].apply(lambda x: f"${x:,.0f}")
                qd_channels["median_price"] = qd_channels["median_price"].apply(lambda x: f"${x:,.0f}")
                qd_channels.columns = ["Channel", "Best Price Wins", "Avg Price", "Median Price"]
                st.dataframe(qd_channels, use_container_width=True, hide_index=True)

            # --- Size-level channel analysis from tv_prices.csv ---
            if len(prices_df) > 0:
                st.markdown("**Channel Coverage by Size**")
                st.caption("Number of size/model combinations with pricing from each channel, "
                           "across the full tv_prices dataset.")
                sized = prices_df[prices_df["best_price"].notna()].copy()
                if "price_source" in sized.columns:
                    sized["channel"] = sized["price_source"].replace({
                        "amazon": "Amazon", "amazon_3p": "Amazon",
                        "bestbuy": "Best Buy", "rtings": "RTINGS (affiliate)",
                    })
                    size_chan = (sized.groupby(["size_inches", "channel"])
                                .size().reset_index(name="count"))
                    fig = px.bar(size_chan, x="size_inches", y="count", color="channel",
                                 color_discrete_map=CHANNEL_COLORS,
                                 barmode="group",
                                 labels={"size_inches": "Screen Size (inches)",
                                         "count": "Price Points", "channel": "Channel"})
                    fig.update_layout(height=400, legend_title_text="Channel", **PL)
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: Temporal Analysis
# ============================================================================
elif page == "Temporal Analysis":
    st.title("Temporal Analysis")
    st.caption(
        "Year-over-year technology trends. This page ignores the sidebar "
        "Model Year filter so all years are always visible for comparison."
    )

    MIN_SAMPLES = 1  # minimum TVs per (tech, year) group to show aggregated data

    # Valid years present in the temporal dataframe
    _year_counts = tdf["model_year"].dropna().value_counts().sort_index()
    _valid_years = sorted(_year_counts[_year_counts >= 1].index.astype(int).tolist())
    _n_valid_years = len(_valid_years)

    if _n_valid_years == 0:
        st.warning(f"No {PCFG['item_label'].lower()} with release date information available.")
        st.stop()

    # Build per-(tech, year) aggregation used across multiple charts
    _avail_scores = [c for c in PCFG["score_cols"] if c in tdf.columns]
    _score_agg = {f"avg_{c}": (c, "mean") for c in _avail_scores}
    _ty = (
        tdf.dropna(subset=["model_year"])
        .groupby(["color_architecture", "model_year"])
        .agg(n=("fullname", "size"), avg_price_m2=("price_per_m2", "mean"), **_score_agg)
        .reset_index()
    )
    _ty["model_year"] = _ty["model_year"].astype(int)
    # Filter to groups meeting minimum sample threshold
    _ty = _ty[_ty["n"] >= MIN_SAMPLES].copy()

    tab_perf, tab_price, tab_qd = st.tabs(["Performance Trends", "Pricing Trends", "QD Material Trends"])

    # ------------------------------------------------------------------
    # Tab 1: Performance Trends
    # ------------------------------------------------------------------
    with tab_perf:
        # Chart 1 — Avg primary score by Technology by Year (grouped bar)
        _ps = PCFG["primary_score"]
        _ps_agg = f"avg_{_ps}"
        _ps_label = friendly(_ps)
        st.subheader(f"Average {_ps_label} by Technology & Year")
        _ch1 = _ty.dropna(subset=[_ps_agg]).copy() if _ps_agg in _ty.columns else pd.DataFrame()
        if len(_ch1) == 0:
            st.info(f"Not enough scored {PCFG['item_label'].lower()} per technology/year.")
        else:
            _ch1["year_str"] = _ch1["model_year"].astype(str)
            _ch1["label"] = _ch1[_ps_agg].apply(lambda v: f"{v:.1f}")
            fig1 = px.bar(
                _ch1,
                x="year_str",
                y=_ps_agg,
                color="color_architecture",
                barmode="group",
                text="label",
                hover_data={"n": True, "color_architecture": True, "year_str": False},
                color_discrete_map=TECH_COLORS,
                category_orders={
                    "color_architecture": TECH_ORDER,
                    "year_str": [str(y) for y in _valid_years],
                },
                labels={
                    "year_str": "Model Year",
                    _ps_agg: f"Avg {_ps_label} Score",
                    "color_architecture": "Technology",
                    "n": "Sample Size",
                },
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(
                yaxis=dict(range=[0, 10.5]),
                height=480,
                **PL,
            )
            st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # Chart 2 — Score Trajectory (line + strip, user-selectable metric)
        st.subheader("Score Trajectory by Technology")
        _metric_options = {friendly(c): c for c in _avail_scores}
        _selected_metric_label = st.selectbox(
            "Metric", list(_metric_options.keys()), index=0
        )
        _selected_metric = _metric_options[_selected_metric_label]

        _dots = tdf.dropna(subset=[_selected_metric, "model_year"]).copy()
        _dots["model_year"] = _dots["model_year"].astype(int)

        if len(_dots) == 0:
            st.info(f"No data for {_selected_metric_label}.")
        else:
            # Per-tech mean line data
            _agg_col = f"avg_{_selected_metric}"
            _line = _ty.dropna(subset=[_agg_col]).copy()

            fig2 = go.Figure()
            for tech in TECH_ORDER:
                color = TECH_COLORS.get(tech, "#888")
                # Individual dots (semi-transparent)
                td = _dots[_dots["color_architecture"] == tech]
                if len(td) > 0:
                    fig2.add_trace(go.Scatter(
                        x=td["model_year"],
                        y=td[_selected_metric],
                        mode="markers",
                        marker=dict(color=color, size=8, opacity=0.3),
                        name=tech,
                        legendgroup=tech,
                        hovertext=td["fullname"],
                        hoverinfo="text+y",
                        showlegend=False,
                    ))
                # Mean line
                tl = _line[_line["color_architecture"] == tech].sort_values("model_year")
                if len(tl) > 0:
                    fig2.add_trace(go.Scatter(
                        x=tl["model_year"],
                        y=tl[_agg_col],
                        mode="lines+markers",
                        marker=dict(color=color, size=10),
                        line=dict(color=color, width=3),
                        name=tech,
                        legendgroup=tech,
                    ))
            fig2.update_layout(
                xaxis=dict(
                    title="Model Year",
                    tickmode="array",
                    tickvals=_valid_years,
                    ticktext=[str(y) for y in _valid_years],
                ),
                yaxis=dict(title=_selected_metric_label, range=[0, 10.5]),
                height=480,
                **PL,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2: Pricing Trends
    # ------------------------------------------------------------------
    with tab_price:
        # Chart 3 — Avg $/m² by Technology by Year (grouped bar)
        st.subheader("Average Price per m\u00b2 by Technology & Year")
        _ch3 = _ty.dropna(subset=["avg_price_m2"]).copy()
        if len(_ch3) == 0:
            st.info(f"Not enough priced {PCFG['item_label'].lower()} per technology/year.")
        else:
            _ch3["year_str"] = _ch3["model_year"].astype(str)
            _ch3["label"] = _ch3["avg_price_m2"].apply(lambda v: f"${v:,.0f}")

            # WLED baseline (overall mean across all years)
            _wled_all = tdf[tdf["color_architecture"] == "WLED"]["price_per_m2"].dropna()
            _wled_baseline = float(_wled_all.mean()) if len(_wled_all) > 0 else None

            fig3 = px.bar(
                _ch3,
                x="year_str",
                y="avg_price_m2",
                color="color_architecture",
                barmode="group",
                text="label",
                hover_data={"n": True, "color_architecture": True, "year_str": False},
                color_discrete_map=TECH_COLORS,
                category_orders={
                    "color_architecture": TECH_ORDER,
                    "year_str": [str(y) for y in _valid_years],
                },
                labels={
                    "year_str": "Model Year",
                    "avg_price_m2": "Avg $/m\u00b2",
                    "color_architecture": "Technology",
                    "n": "Sample Size",
                },
            )
            fig3.update_traces(textposition="outside")
            if _wled_baseline:
                fig3.add_hline(
                    y=_wled_baseline,
                    line_dash="dot",
                    line_color=TECH_COLORS["WLED"],
                    opacity=0.5,
                    annotation_text=f"WLED avg ${_wled_baseline:,.0f}",
                    annotation_position="top right",
                    annotation_font_color=TECH_COLORS["WLED"],
                )
            fig3.update_layout(
                yaxis=dict(title="Avg Price per m\u00b2"),
                height=480,
                **PL,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # Chart 4 — Year-over-Year Change Summary (metric cards)
        st.subheader("Year-over-Year Change Summary")
        if _n_valid_years < 2:
            st.info(
                "Need at least two model years with sufficient data to show "
                "year-over-year changes. Currently only "
                f"{_n_valid_years} year(s) available."
            )
        else:
            _latest_yr = _valid_years[-1]
            _prev_yr = _valid_years[-2]
            _prev = _ty[_ty["model_year"] == _prev_yr].set_index("color_architecture")
            _curr = _ty[_ty["model_year"] == _latest_yr].set_index("color_architecture")
            _common_techs = [t for t in TECH_ORDER if t in _prev.index and t in _curr.index]

            if not _common_techs:
                st.info(
                    f"No technologies have enough data in both {_prev_yr} and "
                    f"{_latest_yr} to compare (need n >= {MIN_SAMPLES} each year)."
                )
            else:
                st.caption(f"Comparing {_prev_yr} vs {_latest_yr} model years")
                for tech in _common_techs:
                    color = TECH_COLORS.get(tech, "#888")
                    p_score = _prev.loc[tech, _ps_agg] if _ps_agg in _prev.columns else np.nan
                    c_score = _curr.loc[tech, _ps_agg] if _ps_agg in _curr.columns else np.nan
                    score_delta = c_score - p_score
                    p_n = int(_prev.loc[tech, "n"])
                    c_n = int(_curr.loc[tech, "n"])

                    p_m2 = _prev.loc[tech, "avg_price_m2"]
                    c_m2 = _curr.loc[tech, "avg_price_m2"]

                    st.markdown(
                        f"<span style='color:{color}; font-weight:700; font-size:1.1rem'>"
                        f"{tech}</span> &nbsp; "
                        f"<span style='color:#999; font-size:0.85rem'>"
                        f"n={p_n} \u2192 {c_n}</span>",
                        unsafe_allow_html=True,
                    )
                    cols = st.columns(3)
                    with cols[0]:
                        if pd.notna(c_score):
                            st.metric(
                                _ps_label,
                                f"{c_score:.1f}",
                                delta=f"{score_delta:+.1f}" if pd.notna(score_delta) else None,
                            )
                        else:
                            st.metric(_ps_label, "N/A")
                    with cols[1]:
                        if pd.notna(c_m2):
                            m2_delta = None
                            if pd.notna(p_m2) and p_m2 > 0:
                                pct = (c_m2 - p_m2) / p_m2 * 100
                                m2_delta = f"{pct:+.0f}%"
                            st.metric(
                                "Avg $/m\u00b2",
                                f"${c_m2:,.0f}",
                                delta=m2_delta,
                                delta_color="inverse",
                            )
                        else:
                            st.metric("Avg $/m\u00b2", "N/A")
                    with cols[2]:
                        if pd.notna(c_m2) and c_score > 0:
                            val = c_m2 / c_score
                            prev_val = (p_m2 / p_score) if pd.notna(p_m2) and p_score > 0 else None
                            val_delta = None
                            if prev_val:
                                val_pct = (val - prev_val) / prev_val * 100
                                val_delta = f"{val_pct:+.0f}%"
                            st.metric(
                                "$/m\u00b2 per Point",
                                f"${val:,.0f}",
                                delta=val_delta,
                                delta_color="inverse",
                            )


    # ------------------------------------------------------------------
    # Tab 3: QD Material Trends
    # ------------------------------------------------------------------
    with tab_qd:
        st.subheader("Quantum Dot Material by Model Year")
        st.caption(
            "CdSe vs InP (Cd-Free) classification based on red peak FWHM. "
            "CdSe QDs have narrow red emission (<30nm), InP QDs are wider (>30nm)."
        )
        _qd_df = tdf.dropna(subset=["model_year"]).copy()
        _qd_df = _qd_df[_qd_df["qd_material"].isin(["CdSe", "InP"])]
        if len(_qd_df) == 0:
            st.info("No QD-equipped TVs with model year data.")
        else:
            _qd_df["model_year"] = _qd_df["model_year"].astype(int)
            _qd_counts = (
                _qd_df.groupby(["model_year", "qd_material"])
                .size()
                .reset_index(name="Count")
            )
            _qd_counts["year_str"] = _qd_counts["model_year"].astype(str)
            _qd_colors = {"CdSe": "#e74c3c", "InP": "#2ecc71"}
            fig_qd = px.bar(
                _qd_counts,
                x="year_str",
                y="Count",
                color="qd_material",
                barmode="stack",
                text="Count",
                color_discrete_map=_qd_colors,
                category_orders={
                    "qd_material": ["CdSe", "InP"],
                    "year_str": [str(y) for y in sorted(_qd_counts["model_year"].unique())],
                },
                labels={
                    "year_str": "Model Year",
                    "qd_material": "QD Material",
                },
            )
            fig_qd.update_traces(textposition="inside")
            fig_qd.update_layout(height=420, **PL)
            st.plotly_chart(fig_qd, use_container_width=True)

        # Breakdown table
        st.subheader("QD Material by Brand")
        _qd_brand = tdf[tdf["qd_material"].isin(["CdSe", "InP"])]
        if len(_qd_brand) > 0:
            _qd_pivot = (
                _qd_brand.groupby(["brand", "qd_material"])
                .size()
                .unstack(fill_value=0)
            )
            _qd_pivot["Total"] = _qd_pivot.sum(axis=1)
            _qd_pivot = _qd_pivot.sort_values("Total", ascending=False)
            st.dataframe(_qd_pivot, use_container_width=True)


# ============================================================================
# PAGE: Comparison Tool
# ============================================================================
elif page == "Comparison Tool":
    st.title(f"{PCFG['item_singular']} Comparison Tool")

    all_names = sorted(fdf["fullname"].tolist())
    selected = st.multiselect(
        f"Select {PCFG['item_label'].lower()} to compare (up to 5)", all_names, max_selections=5,
        default=all_names[:2] if len(all_names) >= 2 else all_names[:1],
    )

    if not selected:
        st.info(f"Select at least one {PCFG['item_singular'].lower()} to compare.")
        st.stop()

    comp = fdf[fdf["fullname"].isin(selected)].copy()

    cols = st.columns(len(selected))
    for i, (_, row) in enumerate(comp.iterrows()):
        with cols[i]:
            st.markdown(f"**{row['fullname']}**")
            st.caption(f"{row['brand']} | {row['display_type']}")
            st.markdown(f"**{row['color_architecture']}**")
            if pd.notna(row.get("price_best")):
                st.metric("Price", f"${row['price_best']:,.0f}")
            else:
                st.metric("Price", "N/A")
            _ps = PCFG["primary_score"]
            if _ps in row and pd.notna(row.get(_ps)):
                st.metric(friendly(_ps), f"{row[_ps]:.1f}")

    st.divider()

    st.subheader("Usage Score Comparison")
    score_keys = [c for c in PCFG["score_cols"] if c in comp.columns]
    categories = [friendly(c) for c in score_keys]

    fig = go.Figure()
    for _, row in comp.iterrows():
        values = [row.get(k, 0) for k in score_keys]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=row["fullname"], opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=12)),
            angularaxis=dict(tickfont=dict(size=14, weight=600)),
        ),
        height=520,
        font=dict(family="Inter, sans-serif", size=14),
        legend=dict(font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Specs")
    detail_rows = [
        ("Display Type", "display_type"),
        ("Color Architecture", "color_architecture"),
        ("Backlight", "backlight_type_v2"),
        ("Dimming Zones", "dimming_zone_count"),
        ("QD Material", "qd_material"),
        ("Marketing Label", "marketing_label"),
        ("Panel Type", "panel_type"),
        ("Panel Sub Type", "panel_sub_type"),
        ("Resolution", "resolution"),
        ("Refresh Rate", "native_refresh_rate"),
        ("Price", "price_best"),
        ("Price Size", "price_size"),
        ("Price Source", "channel"),
        ("$/m\u00b2", "price_per_m2"),
    ] + [(friendly(c), c) for c in PCFG["score_cols"]] + [
        ("Native Contrast", "native_contrast"),
        ("HDR Peak (10%)", "hdr_peak_10pct_nits"),
        ("HDR Peak (2%)", "hdr_peak_2pct_nits"),
        ("SDR Peak", "sdr_real_scene_peak_nits"),
        ("BT.2020 Coverage", "hdr_bt2020_coverage_itp_pct"),
        ("DCI-P3 Coverage", "sdr_dci_p3_coverage_pct"),
        (friendly(PCFG["input_lag_col"]), PCFG["input_lag_col"]),
        ("Response Time", "total_response_time_ms"),
    ]

    table_data = {"Spec": [r[0] for r in detail_rows]}
    for _, row in comp.iterrows():
        values = []
        for label, col in detail_rows:
            val = row.get(col)
            if pd.isna(val):
                values.append("\u2014")
            elif col in ("price_best", "price_per_m2"):
                values.append(f"${float(val):,.0f}")
            elif col == "price_size":
                values.append(f'{int(val)}"')
            elif col == "dimming_zone_count":
                values.append(f"{int(val):,}")
            elif isinstance(val, float):
                values.append(f"{val:.1f}")
            else:
                values.append(str(val))
        table_data[row["fullname"]] = values

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True, height=800)


# ============================================================================
# PAGE: TV Profiles
# ============================================================================
elif page == PCFG["profile_page"]:
    st.title(f"{PCFG['item_singular']} Profile")

    selected_tv = st.selectbox(f"Select a {PCFG['item_singular'].lower()}", sorted(fdf["fullname"].tolist()))
    tv = fdf[fdf["fullname"] == selected_tv].iloc[0]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.header(tv["fullname"])
        st.caption(f"{tv['brand']} | Released: {tv['released_at'].strftime('%Y-%m-%d') if pd.notna(tv.get('released_at')) else 'Unknown'}")
    with col2:
        st.metric("Price", f"${tv['price_best']:,.0f}" if pd.notna(tv.get("price_best")) else "N/A")
        if pd.notna(tv.get("price_per_m2")):
            st.metric("$/m\u00b2", f"${tv['price_per_m2']:,.0f}")
    with col3:
        _ps = PCFG["primary_score"]
        _ps_val = tv.get(_ps)
        st.metric(friendly(_ps), f"{_ps_val:.1f}/10" if pd.notna(_ps_val) else "N/A")
        if pd.notna(tv.get("marketing_label")):
            st.markdown(f"*{tv['marketing_label']}*")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Display Technology")
        tech_info = {
            "Display Type": tv.get("display_type"),
            "Color Architecture": tv.get("color_architecture"),
            "Backlight": tv.get("backlight_type_v2"),
            "Dimming Zones": f"{int(tv['dimming_zone_count']):,}" if pd.notna(tv.get("dimming_zone_count")) else "N/A",
            "QD Present": tv.get("qd_present"),
            "QD Material": tv.get("qd_material", "N/A"),
            "SPD Verified": tv.get("spd_verified"),
            "SPD Classification": tv.get("spd_classification"),
            "Marketing Label": tv.get("marketing_label", "N/A"),
            "Panel Type": tv.get("panel_type"),
            "Panel Sub Type": tv.get("panel_sub_type"),
        }
        for k, v in tech_info.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.subheader("Usage Scores")
        scores = {friendly(c): tv.get(c) for c in PCFG["score_cols"]}
        score_df = pd.DataFrame([{"Usage": k, "Score": v} for k, v in scores.items() if pd.notna(v)])
        if len(score_df) > 0:
            fig = px.bar(score_df, x="Score", y="Usage", orientation="h",
                         range_x=[0, 10], text="Score",
                         color_discrete_sequence=["#4B40EB"])
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                              textfont_size=14, textfont_weight=600)
            fig.update_layout(height=250, showlegend=False,
                              margin=dict(l=0, r=50, t=0, b=0), **PL)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        st.markdown("**Brightness**")
        for k, v in {
            "HDR Peak (2%)": f"{tv.get('hdr_peak_2pct_nits', 'N/A')} nits",
            "HDR Peak (10%)": f"{tv.get('hdr_peak_10pct_nits', 'N/A')} nits",
            "SDR Real Scene": f"{tv.get('sdr_real_scene_peak_nits', 'N/A')} nits",
            "Brightness Score": f"{tv.get('brightness_score', 'N/A')}",
        }.items():
            st.markdown(f"**{k}:** {v}")

    with perf_col2:
        st.markdown("**Contrast & Color**")
        for k, v in {
            "Native Contrast": f"{int(tv['native_contrast']):,}:1" if pd.notna(tv.get("native_contrast")) else "N/A",
            "Contrast Score": tv.get("contrast_ratio_score", "N/A"),
            "BT.2020 (ITP)": f"{tv.get('hdr_bt2020_coverage_itp_pct', 'N/A')}%",
            "DCI-P3": f"{tv.get('sdr_dci_p3_coverage_pct', 'N/A')}%",
            "Color Score": tv.get("color_score", "N/A"),
        }.items():
            st.markdown(f"**{k}:** {v}")

    with perf_col3:
        st.markdown("**Gaming & Response**")
        for k, v in {
            "4K Input Lag": f"{tv.get('input_lag_4k_ms', 'N/A')} ms",
            "1080p Input Lag": f"{tv.get('input_lag_1080p_ms', 'N/A')} ms",
            "Response Time": f"{tv.get('total_response_time_ms', 'N/A')} ms",
            "HDMI 2.1": tv.get("hdmi_21_speed", "N/A"),
            "VRR": tv.get("vrr_support", "N/A"),
            "HDMI Ports": tv.get("hdmi_ports", "N/A"),
        }.items():
            st.markdown(f"**{k}:** {v}")

    st.subheader("SPD Spectral Peaks")
    spd_col1, spd_col2, spd_col3 = st.columns(3)
    with spd_col1:
        st.metric("Blue Peak", f"{tv.get('blue_peak_nm', 'N/A')} nm")
        st.metric("Blue FWHM", f"{tv.get('blue_fwhm_nm', 'N/A')} nm")
    with spd_col2:
        st.metric("Green Peak", f"{tv.get('green_peak_nm', 'N/A')} nm")
        st.metric("Green FWHM", f"{tv.get('green_fwhm_nm', 'N/A')} nm")
    with spd_col3:
        if pd.notna(tv.get("red_peak_nm")):
            st.metric("Red Peak", f"{tv['red_peak_nm']} nm")
            st.metric("Red FWHM", f"{tv['red_fwhm_nm']} nm")
        else:
            st.metric("Red Peak", "N/A")
            st.metric("Red FWHM", "N/A")

    if pd.notna(tv.get("price_best")):
        st.subheader("Pricing")
        price_cols = st.columns(4)
        with price_cols[0]:
            st.metric("Best Price", f"${tv['price_best']:,.0f}")
        with price_cols[1]:
            st.metric("Source", tv.get("channel", tv.get("price_source", "N/A")))
        with price_cols[2]:
            st.metric("Size", f"{int(tv['price_size'])}\"" if pd.notna(tv.get("price_size")) else "N/A")
        with price_cols[3]:
            st.metric("$/m\u00b2", f"${tv['price_per_m2']:,.0f}" if pd.notna(tv.get("price_per_m2")) else "N/A")

        if len(prices_df) > 0:
            tv_sizes = prices_df[
                (prices_df["product_id"] == tv["product_id"])
                & prices_df["best_price"].notna()
            ].sort_values("size_inches")
            if len(tv_sizes) > 0:
                st.markdown("**All Available Sizes**")
                tv_sizes_disp = tv_sizes.copy()
                if "price_source" in tv_sizes_disp.columns:
                    tv_sizes_disp["channel"] = tv_sizes_disp["price_source"].replace({
                        "amazon": "Amazon", "amazon_3p": "Amazon",
                        "bestbuy": "Best Buy", "rtings": "RTINGS (affiliate)",
                    })
                size_display = tv_sizes_disp[["size_inches", "best_price", "channel",
                                               "amazon_price", "bestbuy_price", "rtings_price"]].copy()
                size_display.columns = ["Size", "Best Price", "Source", "Amazon", "Best Buy", "RTINGS"]
                size_display["Size"] = size_display["Size"].apply(
                    lambda x: f'{int(x)}"' if pd.notna(x) else "?")
                for col in ["Best Price", "Amazon", "Best Buy", "RTINGS"]:
                    size_display[col] = size_display[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) else "\u2014")
                st.dataframe(size_display, use_container_width=True, hide_index=True)

    if pd.notna(tv.get("review_url")):
        st.markdown(f"[View full review]({tv['review_url']})")


# ============================================================================
# PAGE: Cross-Product Analysis (All Products blended view)
# ============================================================================
elif page == "Cross-Product Analysis":
    st.title("Cross-Product Display Technology Analysis")
    st.caption(f"Combined view: {len(df)} products ({df['product_type'].value_counts().to_dict()}) · "
               "RTINGS-reviewed TVs and Monitors")

    PRODUCT_TYPE_COLORS = {"tv": "#FFC700", "monitor": "#4B40EB"}
    PRODUCT_TYPE_LABELS = {"tv": "TVs", "monitor": "Monitors"}
    df["Product Type"] = df["product_type"].map(PRODUCT_TYPE_LABELS)

    # --- Headline metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", len(df))
    _n_qd = df["qd_present"].eq("Yes").sum()
    c2.metric("QD Products", _n_qd)
    c3.metric("QD Adoption", f"{_n_qd / len(df) * 100:.0f}%")
    _priced = df[df["price_per_m2"].notna()]
    c4.metric("With Pricing", len(_priced))

    st.divider()

    # --- Section 0: Master RTINGS Score by Technology ---
    st.subheader("Master RTINGS Score by Technology")
    st.caption("TV: Mixed Usage score · Monitors: mean of PC Gaming, Console Gaming, Office, Editing · "
               "Master: weighted average by product count")

    # Compute composite score per product
    _mon_score_cols = ["pc_gaming", "console_gaming", "office", "editing"]
    _tv_mask = df["product_type"] == "tv"
    _mon_mask = df["product_type"] == "monitor"

    # TV uses mixed_usage, monitor uses mean of 4 scores
    df["_master_score"] = np.nan
    if "mixed_usage" in df.columns:
        df.loc[_tv_mask, "_master_score"] = df.loc[_tv_mask, "mixed_usage"]
    _available_mon_scores = [c for c in _mon_score_cols if c in df.columns]
    if _available_mon_scores:
        df.loc[_mon_mask, "_master_score"] = df.loc[_mon_mask, _available_mon_scores].mean(axis=1)

    _master_by_tech = (df.dropna(subset=["_master_score"])
                       .groupby("color_architecture", observed=False)
                       .agg(_score=("_master_score", "mean"), _n=("_master_score", "size"))
                       .reset_index())
    _master_by_tech.columns = ["Technology", "Master Score", "n"]

    ms1, ms2 = st.columns(2)
    with ms1:
        st.markdown("**Master Score by Technology**")
        fig = px.bar(_master_by_tech, x="Technology", y="Master Score",
                     color="Technology", color_discrete_map=TECH_COLORS,
                     text=_master_by_tech["Master Score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
                     category_orders={"Technology": TECH_ORDER},
                     hover_data={"n": True})
        fig.update_traces(textposition="outside", textfont_size=14, textfont_weight=600,
                          cliponaxis=False)
        fig.update_layout(showlegend=False, height=400,
                          yaxis=dict(range=[0, 10.5], title="Avg Score"),
                          **PL)
        st.plotly_chart(fig, use_container_width=True)

    with ms2:
        st.markdown("**Score Breakdown: TVs vs Monitors**")
        _score_by_type = (df.dropna(subset=["_master_score"])
                          .groupby(["color_architecture", "Product Type"], observed=False)["_master_score"]
                          .mean().reset_index())
        _score_by_type.columns = ["Technology", "Product Type", "Avg Score"]
        _score_by_type = _score_by_type.dropna(subset=["Avg Score"])
        if len(_score_by_type) > 0:
            fig = px.bar(_score_by_type, x="Technology", y="Avg Score",
                         color="Product Type",
                         color_discrete_map={"TVs": "#FFC700", "Monitors": "#4B40EB"},
                         barmode="group", text=_score_by_type["Avg Score"].apply(lambda x: f"{x:.1f}"),
                         category_orders={"Technology": TECH_ORDER})
            fig.update_traces(textposition="outside", textfont_size=12, textfont_weight=600,
                              cliponaxis=False)
            fig.update_layout(height=400, legend_title_text="",
                              yaxis=dict(range=[0, 10.5], title="Avg Score"), **PL)
            st.plotly_chart(fig, use_container_width=True)

    # Clean up temp column
    df.drop(columns=["_master_score"], inplace=True)

    st.divider()

    # --- Section 1: QD Adoption ---
    st.subheader("Quantum Dot Adoption")
    qd1, qd2, qd3 = st.columns(3)

    with qd1:
        st.markdown("**Overall QD Adoption**")
        _qd_counts = df["qd_present"].value_counts()
        _qd_pie = pd.DataFrame({
            "QD Status": ["QD (QD-LCD + QD-OLED)", "Non-QD"],
            "Count": [_qd_counts.get("Yes", 0), _qd_counts.get("No", 0)],
        })
        fig = px.pie(_qd_pie, names="QD Status", values="Count",
                     color="QD Status",
                     color_discrete_map={
                         "QD (QD-LCD + QD-OLED)": "#FF009F",
                         "Non-QD": "#A8BDD0",
                     },
                     hole=0.4)
        fig.update_traces(textinfo="percent+value", textfont_size=14, textfont_weight=600)
        fig.update_layout(height=350, showlegend=True, legend_title_text="",
                          margin=dict(l=0, r=0, t=10, b=0), **PL)
        st.plotly_chart(fig, use_container_width=True)

    with qd2:
        st.markdown("**QD Adoption by Product Type**")
        _qd_by_type = (df.groupby(["Product Type", "qd_present"], observed=True)
                        .size().reset_index(name="Count"))
        fig = px.bar(_qd_by_type, x="Product Type", y="Count", color="qd_present",
                     color_discrete_map={"Yes": "#FF009F", "No": "#A8BDD0"},
                     barmode="stack", text="Count",
                     labels={"qd_present": "QD Present"})
        fig.update_traces(textposition="inside", textfont_size=13, textfont_weight=600)
        fig.update_layout(height=350, legend_title_text="QD Present",
                          margin=dict(l=0, r=0, t=10, b=0), **PL)
        st.plotly_chart(fig, use_container_width=True)

    with qd3:
        st.markdown("**QD Adoption by Brand**")
        _qd_brands = (df[df["qd_present"] == "Yes"]
                       .groupby("brand").size().reset_index(name="QD Products")
                       .sort_values("QD Products", ascending=True))
        _total_brands = df.groupby("brand").size().reset_index(name="Total")
        _qd_brands = _qd_brands.merge(_total_brands, on="brand")
        _qd_brands["QD %"] = (_qd_brands["QD Products"] / _qd_brands["Total"] * 100).round(0)
        fig = px.bar(_qd_brands, y="brand", x="QD Products", orientation="h",
                     text=_qd_brands.apply(lambda r: f"{int(r['QD Products'])}/{int(r['Total'])} ({int(r['QD %'])}%)", axis=1),
                     color_discrete_sequence=["#FF009F"])
        fig.update_traces(textposition="outside", textfont_size=12, textfont_weight=600,
                          cliponaxis=False)
        fig.update_layout(height=350, showlegend=False, yaxis_title="",
                          margin=dict(l=0, r=80, t=10, b=0), **PL)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Section 2: Technology Distribution Comparison ---
    st.subheader("Technology Distribution: TVs vs Monitors")
    td1, td2 = st.columns(2)

    with td1:
        st.markdown("**Technology Mix by Product Type**")
        _tech_by_type = (df.groupby(["Product Type", "color_architecture"], observed=False)
                         .size().reset_index(name="Count"))
        # Compute percentages within each product type
        _type_totals = _tech_by_type.groupby("Product Type")["Count"].transform("sum")
        _tech_by_type["Pct"] = (_tech_by_type["Count"] / _type_totals * 100).round(1)
        fig = px.bar(_tech_by_type, x="Product Type", y="Pct", color="color_architecture",
                     color_discrete_map=TECH_COLORS,
                     category_orders={"color_architecture": TECH_ORDER},
                     barmode="stack", text="Pct",
                     labels={"Pct": "% of Products", "color_architecture": "Technology"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                          textfont_size=12, textfont_weight=600)
        fig.update_layout(height=420, legend_title_text="Technology",
                          yaxis=dict(range=[0, 105], title="% of Products"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    with td2:
        st.markdown("**Product Count by Technology**")
        _tech_counts = (df.groupby(["color_architecture", "Product Type"], observed=False)
                        .size().reset_index(name="Count"))
        fig = px.bar(_tech_counts, x="color_architecture", y="Count", color="Product Type",
                     color_discrete_map={"TVs": "#FFC700", "Monitors": "#4B40EB"},
                     barmode="group", text="Count",
                     category_orders={"color_architecture": TECH_ORDER},
                     labels={"color_architecture": "Technology"})
        fig.update_traces(textposition="outside", textfont_size=12, textfont_weight=600,
                          cliponaxis=False)
        fig.update_layout(height=420, legend_title_text="",
                          yaxis_title="Products", **PL)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Section 3: $/m² Across Form Factors ---
    st.subheader("Price per m\u00b2: TVs vs Monitors")

    pm1, pm2 = st.columns(2)

    with pm1:
        st.markdown("**Avg $/m\u00b2 by Technology & Product Type**")
        _m2_by = (_priced.groupby(["color_architecture", "Product Type"], observed=False)["price_per_m2"]
                  .mean().reset_index())
        _m2_by.columns = ["Technology", "Product Type", "Avg $/m\u00b2"]
        _m2_by = _m2_by.dropna(subset=["Avg $/m\u00b2"])
        if len(_m2_by) > 0:
            fig = px.bar(_m2_by, x="Technology", y="Avg $/m\u00b2", color="Product Type",
                         color_discrete_map={"TVs": "#FFC700", "Monitors": "#4B40EB"},
                         barmode="group", text="Avg $/m\u00b2",
                         category_orders={"Technology": TECH_ORDER})
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                              textfont_size=11, textfont_weight=600, cliponaxis=False)
            fig.update_layout(height=420, legend_title_text="", yaxis_title="Avg $/m\u00b2", **PL)
            st.plotly_chart(fig, use_container_width=True)

    with pm2:
        st.markdown("**All Products: Price vs $/m\u00b2**")
        _scatter = _priced.dropna(subset=["price_best", "price_per_m2"]).copy()
        if len(_scatter) > 0:
            fig = px.scatter(_scatter, x="price_best", y="price_per_m2",
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             symbol="Product Type",
                             symbol_map={"TVs": "circle", "Monitors": "diamond"},
                             category_orders={"color_architecture": TECH_ORDER},
                             hover_name="fullname",
                             hover_data=["brand", "Product Type"],
                             labels={"price_best": "Price ($)", "price_per_m2": "$/m\u00b2"})
            fig.update_layout(height=420, legend_title_text="",
                              xaxis=dict(range=[0, _scatter["price_best"].max() * 1.1]),
                              yaxis=dict(range=[0, _scatter["price_per_m2"].max() * 1.1]),
                              **PL)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Section 4: Brand Strategy ---
    st.subheader("Brand Technology Strategy")
    st.caption("Which brands use which technologies across TVs and Monitors")

    _brand_tech = (df.groupby(["brand", "color_architecture", "Product Type"], observed=True)
                   .size().reset_index(name="Count"))
    _brand_pivot = _brand_tech.pivot_table(index="brand", columns=["color_architecture", "Product Type"],
                                            values="Count", fill_value=0, observed=True)
    # Flatten column names
    _brand_pivot.columns = [f"{tech}\n({pt})" for tech, pt in _brand_pivot.columns]
    # Only show brands with > 1 product
    _brand_pivot = _brand_pivot[_brand_pivot.sum(axis=1) > 1]
    _brand_pivot = _brand_pivot.loc[:, (_brand_pivot > 0).any()]

    if len(_brand_pivot) > 0:
        fig = px.imshow(_brand_pivot, text_auto=True, color_continuous_scale="Viridis",
                        aspect="auto")
        fig.update_layout(height=max(350, len(_brand_pivot) * 30 + 100),
                          xaxis_title="", yaxis_title="",
                          coloraxis_showscale=False, **PL)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Section 5: FWHM Cross-Product ---
    st.subheader("SPD Fingerprints Across Form Factors")
    st.caption("Same panel technology should produce similar FWHM signatures regardless of product type")

    _fwhm = df.dropna(subset=["green_fwhm_nm", "red_fwhm_nm"]).copy()
    if len(_fwhm) > 0:
        fig = px.scatter(_fwhm, x="green_fwhm_nm", y="red_fwhm_nm",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         symbol="Product Type",
                         symbol_map={"TVs": "circle", "Monitors": "diamond"},
                         category_orders={"color_architecture": TECH_ORDER},
                         hover_name="fullname",
                         labels={"green_fwhm_nm": "Green FWHM (nm)",
                                 "red_fwhm_nm": "Red FWHM (nm)"})
        fig.update_layout(height=500, legend_title_text="",
                          xaxis=dict(range=[0, max(150, _fwhm["green_fwhm_nm"].max() * 1.1)]),
                          yaxis=dict(range=[0, max(60, _fwhm["red_fwhm_nm"].max() * 1.1)]),
                          **PL)
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Circles = TVs, Diamonds = Monitors. Clusters confirm the same underlying "
                   "panel technology is used across form factors.")
