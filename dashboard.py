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
from pathlib import Path

from src.charts import TECH_ORDER, TECH_COLORS, DISPLAY_TYPE_COLORS
from src.data_loader import (
    PRODUCT_CONFIGS, load_data, load_size_prices, load_price_history,
    enrich_history, compute_m2_from_history, get_screen_area_map,
    _enrich_history_core,
)
from views import overview, tech_explorer, price_analyzer, temporal
from views import comparison, profiles, cross_product

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Display Technology Dashboard",
    page_icon="\U0001f4fa",
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
    _login_logo = Path(__file__).parent / "assets" / "logo_white.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if _login_logo.exists():
            st.image(str(_login_logo), use_container_width=True)
            st.markdown("<p style='text-align:center;color:#999;font-size:0.9em;margin-top:-8px'>"
                        "Display Technology Intelligence</p>", unsafe_allow_html=True)
        st.markdown("")
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
html, body, [class*="css"] {
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

    // Only fix the sidebar collapse/expand toggle button, not expanders
    // or other Streamlit components that use Material Symbols correctly.
    function fix() {
        var btn = doc.querySelector('[data-testid="collapsedControl"]');
        if (!btn || btn.getAttribute('data-fixed')) return;
        var span = btn.querySelector('span');
        if (span) {
            var t = span.textContent.trim();
            if (t === 'keyboard_double_arrow_right' || t === 'keyboard_double_arrow_left') {
                btn.setAttribute('data-fixed', '1');
                span.style.fontSize = '0';
                var arrow = doc.createElement('span');
                arrow.textContent = t.indexOf('right') >= 0 ? '\u25B6' : '\u25C0';
                arrow.style.fontSize = '12px';
                arrow.style.color = '#999';
                span.parentElement.appendChild(arrow);
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


# ---------------------------------------------------------------------------
# Sidebar — Branding & Global Filters
# ---------------------------------------------------------------------------
_logo_path = Path(__file__).parent / "assets" / "logo_white.png"
if _logo_path.exists():
    st.sidebar.image(str(_logo_path), use_container_width=True)
    st.sidebar.markdown("<p style='text-align:center;color:#999;font-size:0.85em;margin-top:-8px'>Display Technology Intelligence</p>",
                        unsafe_allow_html=True)

st.sidebar.divider()

# --- Product type selector ---
_product_types = list(PRODUCT_CONFIGS.keys()) + ["All Products"]
product_type = st.sidebar.radio("Product Type", _product_types,
                                 index=0,
                                 key="product_type", horizontal=True)
_is_blended = product_type == "All Products"
PCFG = PRODUCT_CONFIGS.get(product_type, PRODUCT_CONFIGS["TVs"])
_screen_area_map = get_screen_area_map(product_type)
st.sidebar.divider()

# Load data for selected product type
df = load_data(product_type)
if _is_blended:
    if "product_type" not in df.columns:
        df["product_type"] = "tv"
    _tv_hist = load_price_history("TVs")
    if len(_tv_hist) > 0:
        _tv_hist_enriched, _ = enrich_history(_tv_hist, main_df=df[df["product_type"] == "tv"])
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
        history_df, _n_woled_excluded = enrich_history(history_df, main_df=df,
                                                        screen_area_map=_screen_area_map)
    elif len(history_df) > 0:
        history_df = _enrich_history_core(history_df, _screen_area_map)

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
    st.sidebar.caption(" \u00b7 ".join(caveats))

# --- Page selector ---
_PAGE_ICONS = {
    "Overview": ":material/bar_chart:",
    "Technology Explorer": ":material/science:",
    "Price Analyzer": ":material/attach_money:",
    "Temporal Analysis": ":material/calendar_month:",
    "Comparison Tool": ":material/compare:",
    "TV Profiles": ":material/tv:",
    "Monitor Profiles": ":material/desktop_windows:",
    "Cross-Product Analysis": ":material/compare_arrows:",
}

if _is_blended:
    ALL_PAGES = ["Cross-Product Analysis"]
    page = "Cross-Product Analysis"
else:
    ALL_PAGES = ["Overview", "Technology Explorer", "Price Analyzer", "Temporal Analysis", "Comparison Tool", PCFG["profile_page"]]
    qp_page = st.query_params.get("page", None)
    default_idx = ALL_PAGES.index(qp_page) if qp_page in ALL_PAGES else 0
    page = st.sidebar.radio(
        "View", ALL_PAGES, index=default_idx,
        format_func=lambda p: f"{_PAGE_ICONS.get(p, '')} {p}",
    )

# --- Version info (bottom of sidebar) ---
_VERSION = "2.3.0"
_CHANGELOG_TEXT = """\
**v2.3.0** \u2014 2026-03-25
- Monitor-specific QD Advantage metrics: HDR Peak (2%), BT.2020, DCI-P3, Response Time, Color Accuracy, Brightness
- Per-product-type advantage_metrics config in PRODUCT_CONFIGS

**v2.2.1** \u2014 2026-03-25
- Fix Tech Explorer crash on Monitors: skip metrics missing from monitor data (contrast_ratio_score, color_score, black_level_score)

**v2.2.0** \u2014 2026-03-25
- Page navigation icons via :material/ syntax in sidebar radio

**v2.1.3** \u2014 2026-03-25
- Fix garbled expander icons: CSS font override was too broad ([class*="st-"] \u2192 [class*="css"]), overriding Material Symbols icon font

**v2.1.1** \u2014 2026-03-25
- Default to TVs view (shows all 6 page tabs) instead of All Products
- Fix version text invisible on dark theme (color #555 \u2192 #999)

**v2.1.0** \u2014 2026-03-25
- Cross-project harmonization: Nanosys dark theme tokens, semver versioning, expander changelog
- Logo assets standardized to assets/ directory (aligned with SKU Tracker)
- QD export schema contract: SCHEMA_VERSION + EXPORT_COLUMNS validation
- FWHM CdSe/InP threshold safety margin (28\u201334 nm \u2192 Unknown)
- Samsung WOLED detection centralized in silo_config.py
- SPD cache staleness detection via HTTP headers
- Session cookie expiry hardened with unconditional flag writes
- Dashboard refactored: 2,730 \u2192 420 lines (extracted views/, src/)

**v2.0.0** \u2014 2026-03-24
- Monitor support (70 monitors, v2.1.2+)
- "All Products" cross-product analysis view
- Master RTINGS Score by technology
- QD Adoption dashboard with brand breakdown
- $/m\u00b2 comparison across TVs vs Monitors
- Brand strategy heatmap & FWHM cross-validation
- QD SKU Tracker weekly email export
- SPD calibration hardening

**v1.0.0** \u2014 2026-02-15
- Initial TV dashboard with 85+ TVs
- SPD-based technology classification
- Keepa + Best Buy pricing pipeline
- 6 pages: Overview, Tech Explorer, Price Analyzer, Temporal, Comparison, Profiles
"""
st.sidebar.divider()
st.sidebar.markdown(
    f"<p style='text-align:center;color:#999;font-size:0.8em;margin-bottom:2px'>Version {_VERSION}</p>",
    unsafe_allow_html=True,
)
with st.sidebar.expander("What's new?"):
    st.markdown(_CHANGELOG_TEXT)


# ============================================================================
# PAGE ROUTING
# ============================================================================
if page == "Overview":
    overview.render(fdf, PCFG, product_type=product_type, df=df)
elif page == "Technology Explorer":
    tech_explorer.render(fdf, PCFG)
elif page == "Price Analyzer":
    price_analyzer.render(fdf, PCFG, history_df=history_df, prices_df=prices_df,
                          selected_techs=selected_techs)
elif page == "Temporal Analysis":
    temporal.render(tdf, PCFG)
elif page == "Comparison Tool":
    comparison.render(fdf, PCFG)
elif page == PCFG["profile_page"]:
    profiles.render(fdf, PCFG, prices_df=prices_df)
elif page == "Cross-Product Analysis":
    cross_product.render(df)
