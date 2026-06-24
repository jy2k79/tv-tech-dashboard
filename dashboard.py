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

from src.data_loader import (
    load_data, load_size_prices, load_price_history,
    enrich_history, compute_m2_from_history, _enrich_history_core,
)
from src.sidebar import (
    render_sidebar_top, render_report_download,
    render_filters, render_page_selector, render_sidebar_bottom,
)
from views import overview, tech_explorer, price_analyzer, temporal
from views import comparison, profiles, cross_product

ROOT = Path(__file__).parent

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
    _login_logo = ROOT / "assets" / "logo_white.png"
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

# Fix Streamlit sidebar toggle showing raw ligature text
components.html("""
<script>
(function() {
    var doc = window.parent.document;

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

    var colorMap = {
        'QD-OLED': '#FF009F', 'WOLED': '#4B40EB', 'QD-LCD': '#FFC700',
        'RGB MiniLED': '#00A878', 'Pseudo QD': '#FF7E43', 'KSF': '#90BFFF',
        'WLED': '#6E7681', 'OLED': '#4B40EB', 'LCD': '#FFC700'
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

    fix(); addDots();
    setTimeout(function() { fix(); addDots(); }, 300);
    setTimeout(function() { fix(); addDots(); }, 1000);
    setTimeout(function() { fix(); addDots(); }, 2000);
    setTimeout(function() { fix(); addDots(); }, 4000);

    var obs = new MutationObserver(function() { fix(); addDots(); });
    obs.observe(doc.body, { childList: true, subtree: true, characterData: true });
})();
</script>
""", height=0)


# ============================================================================
# SIDEBAR
# ============================================================================
product_type, _is_blended, PCFG, _screen_area_map = render_sidebar_top()

# --- Load & enrich data for selected product type ---
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

_n_8k = 0
if "resolution" in df.columns:
    _n_8k = (df["resolution"] == "8k").sum()
    if _n_8k > 0:
        df = df[df["resolution"] != "8k"].reset_index(drop=True)

if "released_at" in df.columns:
    df["model_year"] = df["released_at"].dt.year

# --- Sidebar: report download, filters, page selector, version ---
render_report_download(ROOT)
filters = render_filters(df, PCFG, _is_blended,
                         n_8k=_n_8k, n_woled_excluded=_n_woled_excluded)
fdf = filters["fdf"]
tdf = filters["tdf"]
selected_techs = filters["selected_techs"]

page = render_page_selector(PCFG, _is_blended)
render_sidebar_bottom()


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
