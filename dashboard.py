#!/usr/bin/env python3
"""
RTINGS TV Display Technology Dashboard
=======================================
Interactive Streamlit dashboard for exploring TV display technologies,
pricing, and performance metrics from RTINGS.com data.

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
    page_title="RTINGS TV Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    // Run repeatedly to catch Streamlit re-renders
    fix();
    setTimeout(fix, 300);
    setTimeout(fix, 1000);
    setTimeout(fix, 2000);
    setTimeout(fix, 4000);

    // Also watch for DOM changes
    var obs = new MutationObserver(function() { fix(); });
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


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "tv_database_with_prices.csv")
    for col in ["first_published_at", "last_updated_at", "released_at", "scraped_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "price_best", "price_per_m2", "price_per_mixed_use",
        "mixed_usage", "home_theater", "gaming", "sports", "bright_room",
        "brightness_score", "contrast_ratio_score", "color_score",
        "black_level_score", "native_contrast_score",
        "hdr_peak_10pct_nits", "hdr_peak_2pct_nits",
        "sdr_real_scene_peak_nits", "native_contrast",
        "dimming_zone_count", "price_size",
        "green_fwhm_nm", "red_fwhm_nm", "blue_fwhm_nm",
        "hdr_bt2020_coverage_itp_pct", "sdr_dci_p3_coverage_pct",
        "input_lag_4k_ms", "first_response_time_ms", "total_response_time_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tech_order = ["QD-OLED", "WOLED", "QD-LCD", "Pseudo QD", "KSF", "WLED"]
    df["color_architecture"] = pd.Categorical(
        df["color_architecture"], categories=tech_order, ordered=True
    )
    return df


@st.cache_data
def load_size_prices():
    path = DATA_DIR / "tv_prices.csv"
    if path.exists():
        df = pd.read_csv(path)
        for col in ["best_price", "amazon_price", "bestbuy_price", "rtings_price",
                     "list_price", "price_per_m2", "size_inches"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data
def load_price_history():
    path = DATA_DIR / "price_history.csv"
    if path.exists():
        hist = pd.read_csv(path)
        hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
        hist["best_price"] = pd.to_numeric(hist["best_price"], errors="coerce")
        return hist
    return pd.DataFrame()


df = load_data()
prices_df = load_size_prices()
history_df = load_price_history()

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
    "total_response_time_ms": "Total Response Time (ms)",
    "first_response_time_ms": "First Response Time (ms)",
    "contrast_ratio_score": "Contrast Ratio Score",
    "black_level_score": "Black Level Score",
    "color_score": "Color Score",
    "brightness_score": "Brightness Score",
    "native_contrast_score": "Native Contrast Score",
    "mixed_usage": "Mixed Usage",
    "home_theater": "Home Theater",
    "gaming": "Gaming",
    "sports": "Sports",
    "bright_room": "Bright Room",
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
# Sidebar — Global Filters (colored checkboxes for discoverability)
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

# --- Technology checkboxes with color dots ---
all_techs = df["color_architecture"].cat.categories.tolist()
st.sidebar.markdown("**Color Architecture**")
tech_all = st.sidebar.checkbox("All technologies", value=True, key="tech_all")
selected_techs = []
for tech in all_techs:
    color = TECH_COLORS.get(tech, "#888")
    dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:4px;vertical-align:middle"></span>'
    # Checkbox label is plain text; color dot is shown above via markdown
    col_dot, col_cb = st.sidebar.columns([0.12, 0.88])
    with col_dot:
        st.markdown(f'<div style="width:12px;height:12px;border-radius:50%;background:{color};margin-top:10px"></div>',
                    unsafe_allow_html=True)
    with col_cb:
        checked = st.checkbox(tech, value=tech_all, key=f"tech_{tech}")
    if checked:
        selected_techs.append(tech)

# --- Display Type checkboxes ---
st.sidebar.markdown("**Display Type**")
all_display_types = sorted(df["display_type"].dropna().unique())
dt_all = st.sidebar.checkbox("All display types", value=True, key="dt_all")
selected_display_types = []
for dt in all_display_types:
    color = DISPLAY_TYPE_COLORS.get(dt, "#888")
    col_dot, col_cb = st.sidebar.columns([0.12, 0.88])
    with col_dot:
        st.markdown(f'<div style="width:12px;height:12px;border-radius:50%;background:{color};margin-top:10px"></div>',
                    unsafe_allow_html=True)
    with col_cb:
        checked = st.checkbox(dt, value=dt_all, key=f"dt_{dt}")
    if checked:
        selected_display_types.append(dt)

# --- Brand checkboxes ---
all_brands = sorted(df["brand"].dropna().unique())
with st.sidebar.expander("**Brands**", expanded=False):
    brand_all = st.checkbox("All brands", value=True, key="brand_all")
    selected_brands = []
    for brand in all_brands:
        checked = st.checkbox(brand, value=brand_all, key=f"brand_{brand}")
        if checked:
            selected_brands.append(brand)

st.sidebar.divider()

# --- Price range ---
price_min = float(df["price_best"].min()) if df["price_best"].notna().any() else 0
price_max = float(df["price_best"].max()) if df["price_best"].notna().any() else 10000
price_range = st.sidebar.slider(
    "Price Range ($)", min_value=0, max_value=int(price_max) + 500,
    value=(0, int(price_max) + 500), step=50,
)
include_unpriced = st.sidebar.checkbox("Include TVs without pricing", value=True)

# --- Build filter mask ---
mask = (
    df["color_architecture"].isin(selected_techs)
    & df["display_type"].isin(selected_display_types)
    & df["brand"].isin(selected_brands)
)
if include_unpriced:
    mask = mask & (df["price_best"].isna() | df["price_best"].between(price_range[0], price_range[1]))
else:
    mask = mask & df["price_best"].between(price_range[0], price_range[1])

fdf = df[mask].copy()
st.sidebar.markdown(f"**Showing {len(fdf)}/{len(df)} TVs**")

page = st.sidebar.radio(
    "View",
    ["Overview", "Technology Explorer", "Price Analyzer", "Comparison Tool", "TV Profiles"],
)


# ============================================================================
# PAGE: Overview
# ============================================================================
if page == "Overview":
    st.title("RTINGS TV Display Technology Dashboard")
    st.caption(f"Database: {len(df)} TVs from RTINGS.com test bench v2.0+")

    priced = fdf[fdf["price_best"].notna()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TVs", len(fdf))
    c2.metric("With Pricing", len(priced))
    c3.metric("Brands", fdf["brand"].nunique())
    c4.metric("Avg Price", f"${priced['price_best'].mean():,.0f}" if len(priced) else "N/A")
    c5.metric("Avg $/m\u00b2", f"${priced['price_per_m2'].mean():,.0f}" if len(priced) else "N/A")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Color Architecture Distribution")
        tech_counts = fdf["color_architecture"].value_counts().reset_index()
        tech_counts.columns = ["Technology", "Count"]
        fig = px.bar(tech_counts, x="Technology", y="Count", color="Technology",
                     color_discrete_map=TECH_COLORS, text="Count")
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
                               labels={"price_best": "Price ($)", "color_architecture": "Technology"})
            fig.update_layout(height=350, barmode="stack", legend_title_text="",
                              xaxis=dict(range=axis_range("price_best")), **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No priced TVs in current filter.")

    with col4:
        st.subheader("Price by Technology")
        if len(priced) > 0:
            fig = px.box(priced, x="color_architecture", y="price_best",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         labels={"price_best": "Price ($)", "color_architecture": "Technology"},
                         points="all")
            fig.update_layout(showlegend=False, height=350,
                              yaxis=dict(range=axis_range("price_best")), **PL)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No priced TVs in current filter.")

    st.subheader("Usage Score Overview")
    score_cols = ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"]
    score_data = fdf[["fullname", "color_architecture"] + score_cols].melt(
        id_vars=["fullname", "color_architecture"], value_vars=score_cols,
        var_name="Usage", value_name="Score"
    )
    score_data["Usage"] = score_data["Usage"].map(friendly)
    fig = px.box(score_data, x="Usage", y="Score", color="color_architecture",
                 color_discrete_map=TECH_COLORS, labels={"color_architecture": "Technology"})
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
                           hover_name="fullname",
                           labels={"green_fwhm_nm": "Green FWHM (nm)", "color_architecture": ""})
            fig.add_hline(y=28, line_dash="dash", line_color="gray",
                          annotation_text="QD-LCD threshold (28nm)",
                          annotation_font_size=12)
            fig.add_hline(y=40, line_dash="dash", line_color="gray",
                          annotation_text="Pseudo QD threshold (40nm)",
                          annotation_font_size=12)
            fig.update_layout(showlegend=False, height=450,
                              yaxis=dict(range=axis_range("green_fwhm_nm")), **PL)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Red Peak FWHM by Technology**")
            valid_red = fdf[fdf["red_fwhm_nm"].notna()]
            fig = px.strip(valid_red, x="color_architecture", y="red_fwhm_nm",
                           color="color_architecture", color_discrete_map=TECH_COLORS,
                           hover_name="fullname",
                           labels={"red_fwhm_nm": "Red FWHM (nm)", "color_architecture": ""})
            fig.add_hline(y=10, line_dash="dash", line_color="gray",
                          annotation_text="KSF narrow (<10nm)",
                          annotation_font_size=12)
            fig.add_hline(y=40, line_dash="dash", line_color="gray",
                          annotation_text="Broad threshold",
                          annotation_font_size=12)
            fig.update_layout(showlegend=False, height=450,
                              yaxis=dict(range=axis_range("red_fwhm_nm")), **PL)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Green vs Red FWHM — Technology Clusters**")
        valid_both = fdf[fdf["green_fwhm_nm"].notna() & fdf["red_fwhm_nm"].notna()]
        fig = px.scatter(valid_both, x="green_fwhm_nm", y="red_fwhm_nm",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         hover_name="fullname", hover_data=["brand", "marketing_label"],
                         labels={"green_fwhm_nm": "Green FWHM (nm)", "red_fwhm_nm": "Red FWHM (nm)"})
        fig.add_shape(type="rect", x0=0, x1=28, y0=0, y1=28,
                       line=dict(color="rgba(255,199,0,0.5)", dash="dash"),
                       fillcolor="rgba(255,199,0,0.05)")
        fig.add_annotation(x=14, y=2, text="QD-LCD zone", showarrow=False,
                           font=dict(color="rgba(255,199,0,0.8)", size=13))
        fig.update_layout(height=500, legend_title_text="Technology",
                          xaxis=dict(range=axis_range("green_fwhm_nm")),
                          yaxis=dict(range=axis_range("red_fwhm_nm")), **PL)
        fig.update_traces(marker=MARKER)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Panel Metrics ---
    with tab3:
        st.subheader("Panel Performance by Technology")

        metric_options = [
            "native_contrast", "hdr_peak_10pct_nits", "sdr_real_scene_peak_nits",
            "hdr_bt2020_coverage_itp_pct", "sdr_dci_p3_coverage_pct",
            "input_lag_4k_ms", "total_response_time_ms",
        ]
        metric = st.selectbox(
            "Metric",
            metric_options,
            format_func=friendly,
        )

        valid = fdf[fdf[metric].notna()]
        fig = px.box(valid, x="color_architecture", y=metric,
                     color="color_architecture", color_discrete_map=TECH_COLORS,
                     points="all", hover_name="fullname",
                     labels={metric: friendly(metric), "color_architecture": ""})
        fig.update_layout(showlegend=False, height=500,
                          yaxis=dict(range=axis_range(metric)), **PL)
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Average Scores by Technology**")
        score_cols = ["mixed_usage", "home_theater", "gaming", "sports", "bright_room",
                      "brightness_score", "contrast_ratio_score", "color_score",
                      "black_level_score", "native_contrast_score"]
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
            ("mixed_usage", "Mixed Usage Score"),
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

    # --- Tab 5: Mixed Usage Drivers ---
    with tab5:
        st.subheader("What Drives Mixed Usage Scores?")
        st.caption("Correlation analysis: which metrics predict overall TV performance")

        corr_metrics = {
            "contrast_ratio_score": "Contrast Ratio Score",
            "black_level_score": "Black Level Score",
            "color_score": "Color Score",
            "hdr_bt2020_coverage_itp_pct": "HDR BT.2020 Coverage",
            "brightness_score": "Brightness Score",
            "sdr_dci_p3_coverage_pct": "DCI-P3 Coverage",
            "native_contrast_score": "Native Contrast Score",
            "hdr_peak_10pct_nits": "HDR Peak Brightness",
            "hdr_peak_2pct_nits": "HDR Peak (2%)",
            "sdr_real_scene_peak_nits": "SDR Peak Brightness",
            "total_response_time_ms": "Response Time",
            "input_lag_4k_ms": "4K Input Lag",
        }
        corr_data = []
        for col, label in corr_metrics.items():
            if col in fdf.columns:
                valid = fdf[["mixed_usage", col]].dropna()
                if len(valid) > 5:
                    r = valid["mixed_usage"].corr(valid[col])
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
            xaxis=dict(range=[-1, 1], title="Pearson Correlation with Mixed Usage"),
            yaxis_title="",
            margin=dict(l=0, r=60, t=10, b=0),
            **PL,
        )
        fig.update_traces(textposition="outside", textfont_size=13, textfont_weight=600,
                          cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        scol1, scol2 = st.columns(2)

        with scol1:
            st.markdown("**Contrast Ratio Score vs Mixed Usage** (r = 0.94)")
            valid = fdf[["contrast_ratio_score", "mixed_usage", "color_architecture", "fullname"]].dropna()
            fig = px.scatter(valid, x="contrast_ratio_score", y="mixed_usage",
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             hover_name="fullname",
                             labels={"contrast_ratio_score": "Contrast Ratio Score",
                                     "mixed_usage": "Mixed Usage"})
            x_arr = valid["contrast_ratio_score"].values
            y_arr = valid["mixed_usage"].values
            m, b = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
            r2 = np.corrcoef(x_arr, y_arr)[0, 1] ** 2
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b, mode="lines",
                name=f"r\u00b2 = {r2:.2f}",
                line=dict(color="rgba(255,255,255,0.5)", dash="dash", width=2),
            ))
            fig.update_layout(height=420, showlegend=True, legend_title_text="",
                              xaxis=dict(range=axis_range("contrast_ratio_score")),
                              yaxis=dict(range=axis_range("mixed_usage")), **PL)
            fig.update_traces(marker=MARKER, selector=dict(mode="markers"))
            st.plotly_chart(fig, use_container_width=True)

        with scol2:
            st.markdown("**Response Time vs Mixed Usage** (r = -0.72)")
            valid = fdf[["total_response_time_ms", "mixed_usage", "color_architecture", "fullname"]].dropna()
            fig = px.scatter(valid, x="total_response_time_ms", y="mixed_usage",
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             hover_name="fullname",
                             labels={"total_response_time_ms": "Response Time (ms)",
                                     "mixed_usage": "Mixed Usage"})
            x_arr = valid["total_response_time_ms"].values
            y_arr = valid["mixed_usage"].values
            m, b = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
            r2 = np.corrcoef(x_arr, y_arr)[0, 1] ** 2
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b, mode="lines",
                name=f"r\u00b2 = {r2:.2f}",
                line=dict(color="rgba(255,255,255,0.5)", dash="dash", width=2),
            ))
            fig.update_layout(height=420, showlegend=True, legend_title_text="",
                              xaxis=dict(range=axis_range("total_response_time_ms")),
                              yaxis=dict(range=axis_range("mixed_usage")), **PL)
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
                size="mixed_usage", size_max=22,
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
            st.warning("No priced TVs match the current filters.")
        else:
            fig = px.box(val_priced, x="color_architecture", y="price_per_mixed_use",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
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
            st.markdown("**Value Frontier: Price vs Performance**")
            st.caption("Dashed line = efficient frontier (best score at each price point)")
            fig = px.scatter(val_priced, x="price_best", y="mixed_usage",
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             hover_name="fullname",
                             hover_data=["price_per_mixed_use", "price_size", "brand"],
                             labels={"price_best": "Price ($)", "mixed_usage": "Mixed Usage Score"})
            sorted_v = val_priced.sort_values("mixed_usage", ascending=False)
            frontier = []
            min_price = float("inf")
            for _, row in sorted_v.iterrows():
                if row["price_best"] <= min_price:
                    frontier.append(row)
                    min_price = row["price_best"]
            if frontier:
                ffront = pd.DataFrame(frontier).sort_values("price_best")
                fig.add_trace(go.Scatter(
                    x=ffront["price_best"], y=ffront["mixed_usage"],
                    mode="lines+markers", name="Value Frontier",
                    line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=2),
                    marker=dict(size=7, color="rgba(255,255,255,0.6)"),
                ))
            fig.update_layout(height=520, legend_title_text="Technology",
                              xaxis=dict(range=axis_range("price_best")),
                              yaxis=dict(range=axis_range("mixed_usage")), **PL)
            fig.update_traces(marker=MARKER, selector=dict(mode="markers"))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("**Top 20 Best Value TVs** (lowest $/mixed usage point)")
            top_val = val_priced.sort_values("price_per_mixed_use").head(20)
            val_table = top_val[[
                "fullname", "color_architecture", "price_best", "mixed_usage",
                "price_per_mixed_use", "hdr_peak_10pct_nits",
                "hdr_bt2020_coverage_itp_pct", "price_size",
            ]].copy()
            val_table.columns = ["TV", "Technology", "Price", "Mixed Usage",
                                 "$/Point", "HDR Brightness", "BT.2020 %", "Size"]
            val_table["Price"] = val_table["Price"].apply(lambda x: f"${x:,.0f}")
            val_table["$/Point"] = val_table["$/Point"].apply(lambda x: f"${x:,.0f}")
            val_table["Mixed Usage"] = val_table["Mixed Usage"].apply(lambda x: f"{x:.1f}")
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
        st.warning("No priced TVs match the current filters.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Value Map", "Price/m\u00b2", "Best Deals", "Price Trends"])

    with tab1:
        score_metric = st.selectbox(
            "Score metric",
            ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"],
            format_func=friendly,
            key="value_score",
        )
        score_label = friendly(score_metric)

        fig = px.scatter(priced, x="price_best", y=score_metric,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         hover_name="fullname",
                         hover_data=["brand", "price_source", "price_size"],
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
                         hover_data=["price_best", "price_size", "brand"],
                         labels={"price_per_m2": "$/m\u00b2", "fullname": ""})
            fig.update_layout(height=500, showlegend=False, xaxis_tickangle=-45,
                              yaxis=dict(range=axis_range("price_per_m2")), **PL)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(priced, x="color_architecture", y="price_per_m2",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         points="all", hover_name="fullname",
                         labels={"price_per_m2": "$/m\u00b2", "color_architecture": ""})
            fig.update_layout(height=500, showlegend=False,
                              yaxis=dict(range=axis_range("price_per_m2")), **PL)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        if len(prices_df) > 0:
            st.subheader("Price by Screen Size")
            sized = prices_df[prices_df["best_price"].notna() & prices_df["size_inches"].notna()]
            if len(sized) > 0:
                fig = px.scatter(sized, x="size_inches", y="best_price",
                                 color="price_source",
                                 hover_data=["amazon_asin", "bestbuy_sku"],
                                 labels={"size_inches": "Screen Size (inches)",
                                         "best_price": "Price ($)"})
                fig.update_layout(height=400, legend_title_text="Source",
                                  xaxis=dict(range=axis_range("size_inches", prices_df)),
                                  yaxis=dict(range=axis_range("best_price", prices_df)), **PL)
                fig.update_traces(marker=MARKER)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Best Value TVs")
        value_metric = st.selectbox(
            "Optimize for",
            ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"],
            format_func=friendly,
            key="deal_metric",
        )
        value_label = friendly(value_metric)

        priced = priced.copy()
        priced["value_index"] = priced[value_metric] / priced["price_best"] * 1000
        priced = priced.sort_values("value_index", ascending=False)

        st.markdown(f"**Top 15 by {value_label} per $1,000**")
        deal_cols = ["fullname", "color_architecture", "price_best", "price_size",
                     value_metric, "value_index", "price_source"]
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
        st.subheader("Average Price by Technology Over Time")
        if len(history_df) == 0:
            st.info("No price history available yet. Run the pricing pipeline "
                    "(ideally every Monday) to accumulate weekly data.")
        else:
            hist_filtered = history_df[history_df["color_architecture"].isin(selected_techs)].copy()
            if len(hist_filtered) == 0:
                st.info("No price history for selected technologies.")
            else:
                # Aggregate to ISO weeks — average all snapshots within the same week
                hist_filtered["iso_year"] = hist_filtered["snapshot_date"].dt.isocalendar().year.astype(int)
                hist_filtered["iso_week"] = hist_filtered["snapshot_date"].dt.isocalendar().week.astype(int)
                weekly = (
                    hist_filtered.groupby(["iso_year", "iso_week", "color_architecture"])["best_price"]
                    .mean().reset_index()
                )

                # Build a sortable week key and a display label
                n_years = weekly["iso_year"].nunique()
                if n_years <= 1:
                    weekly["Week"] = "Wk " + weekly["iso_week"].astype(str)
                    weekly["_sort"] = weekly["iso_week"]
                else:
                    weekly["Week"] = weekly["iso_year"].astype(str) + " Wk " + weekly["iso_week"].astype(str)
                    weekly["_sort"] = weekly["iso_year"] * 100 + weekly["iso_week"]
                weekly = weekly.sort_values("_sort")
                weekly.rename(columns={"color_architecture": "Technology",
                                       "best_price": "Avg Price"}, inplace=True)

                n_weeks = weekly["Week"].nunique()

                fig = px.line(
                    weekly, x="Week", y="Avg Price",
                    color="Technology", color_discrete_map=TECH_COLORS,
                    markers=True,
                    labels={"Avg Price": "Average Price ($)", "Week": ""},
                    category_orders={"Week": weekly["Week"].unique().tolist()},
                )
                fig.update_traces(marker=dict(size=10))
                fig.update_layout(height=500, legend_title_text="Technology",
                                  xaxis=dict(type="category"), **PL)
                if n_weeks == 1:
                    st.caption("Only one week of data. Run the pipeline each Monday to see trends.")
                st.plotly_chart(fig, use_container_width=True)

                # Optional size filter for a second chart
                available_sizes = sorted(hist_filtered["size_inches"].dropna().unique())
                if len(available_sizes) > 1:
                    size_filter = st.selectbox(
                        "Filter by screen size",
                        ["All sizes"] + [f'{int(s)}"' for s in available_sizes],
                        key="history_size_filter",
                    )
                    if size_filter != "All sizes":
                        size_val = int(size_filter.replace('"', ''))
                        hist_sized = hist_filtered[hist_filtered["size_inches"] == size_val]
                        weekly2 = (
                            hist_sized.groupby(["iso_year", "iso_week", "color_architecture"])["best_price"]
                            .mean().reset_index()
                        )
                        if n_years <= 1:
                            weekly2["Week"] = "Wk " + weekly2["iso_week"].astype(str)
                            weekly2["_sort"] = weekly2["iso_week"]
                        else:
                            weekly2["Week"] = weekly2["iso_year"].astype(str) + " Wk " + weekly2["iso_week"].astype(str)
                            weekly2["_sort"] = weekly2["iso_year"] * 100 + weekly2["iso_week"]
                        weekly2 = weekly2.sort_values("_sort")
                        weekly2.rename(columns={"color_architecture": "Technology",
                                                "best_price": "Avg Price"}, inplace=True)
                        fig2 = px.line(
                            weekly2, x="Week", y="Avg Price",
                            color="Technology", color_discrete_map=TECH_COLORS,
                            markers=True,
                            labels={"Avg Price": f'Average Price ($) \u2014 {size_filter}', "Week": ""},
                            category_orders={"Week": weekly2["Week"].unique().tolist()},
                        )
                        fig2.update_traces(marker=dict(size=10))
                        fig2.update_layout(height=450, legend_title_text="Technology",
                                          xaxis=dict(type="category"), **PL)
                        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE: Comparison Tool
# ============================================================================
elif page == "Comparison Tool":
    st.title("TV Comparison Tool")

    all_names = sorted(fdf["fullname"].tolist())
    selected = st.multiselect(
        "Select TVs to compare (up to 5)", all_names, max_selections=5,
        default=all_names[:2] if len(all_names) >= 2 else all_names[:1],
    )

    if not selected:
        st.info("Select at least one TV to compare.")
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
            st.metric("Mixed Usage", f"{row['mixed_usage']:.1f}")

    st.divider()

    st.subheader("Usage Score Comparison")
    categories = ["Mixed Usage", "Home Theater", "Gaming", "Sports", "Bright Room"]
    score_keys = ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"]

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
        ("Price Source", "price_source"),
        ("$/m\u00b2", "price_per_m2"),
        ("Mixed Usage", "mixed_usage"),
        ("Home Theater", "home_theater"),
        ("Gaming", "gaming"),
        ("Sports", "sports"),
        ("Bright Room", "bright_room"),
        ("Native Contrast", "native_contrast"),
        ("HDR Peak (10%)", "hdr_peak_10pct_nits"),
        ("HDR Peak (2%)", "hdr_peak_2pct_nits"),
        ("SDR Peak", "sdr_real_scene_peak_nits"),
        ("BT.2020 Coverage", "hdr_bt2020_coverage_itp_pct"),
        ("DCI-P3 Coverage", "sdr_dci_p3_coverage_pct"),
        ("4K Input Lag", "input_lag_4k_ms"),
        ("Response Time", "total_response_time_ms"),
        ("HDMI 2.1", "hdmi_21_speed"),
        ("HDMI Ports", "hdmi_ports"),
        ("VRR", "vrr_support"),
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
elif page == "TV Profiles":
    st.title("TV Profile")

    selected_tv = st.selectbox("Select a TV", sorted(fdf["fullname"].tolist()))
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
        st.metric("Mixed Usage", f"{tv['mixed_usage']:.1f}/10")
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
        scores = {
            "Mixed Usage": tv.get("mixed_usage"),
            "Home Theater": tv.get("home_theater"),
            "Gaming": tv.get("gaming"),
            "Sports": tv.get("sports"),
            "Bright Room": tv.get("bright_room"),
        }
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
            st.metric("Source", tv.get("price_source", "N/A"))
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
                size_display = tv_sizes[["size_inches", "best_price", "price_source",
                                          "amazon_price", "bestbuy_price", "rtings_price"]].copy()
                size_display.columns = ["Size", "Best Price", "Source", "Amazon", "Best Buy", "RTINGS"]
                size_display["Size"] = size_display["Size"].apply(
                    lambda x: f'{int(x)}"' if pd.notna(x) else "?")
                for col in ["Best Price", "Amazon", "Best Buy", "RTINGS"]:
                    size_display[col] = size_display[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) else "\u2014")
                st.dataframe(size_display, use_container_width=True, hide_index=True)

    if pd.notna(tv.get("review_url")):
        st.markdown(f"[View full review on RTINGS.com]({tv['review_url']})")
