"""
Shared sidebar layout for the Display Technology Dashboard.
Split into top (logo, product selector) / filters / page selector / bottom
(version, changelog) so dashboard.py controls the flow between them.

Mirrors the SKU Tracker's src/sidebar.py render_sidebar_top / render_sidebar_bottom
pattern, extended with filter and page selector helpers.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import PRODUCT_CONFIGS, get_screen_area_map

LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo_white.png"

VERSION = "2.4.7"
CHANGELOG = """\
**v2.4.7** \u2014 2026-04-23
- Fix price-history SKU aggregation: snapshots now take the cheapest SKU per (date, product, size), matching how the current-price bar chart works. Previously `keep='last'` picked arbitrarily, making the line chart overstate prices for multi-SKU products like Apple Studio Display XDR (4 SKUs at 27", $2,849\u2013$3,600).
- Backfilled 7 historical monitor snapshots where today's data shows the dedup bug picked a non-min SKU with <30% price spread (stable price pattern). Skipped 5 candidates with >30% spread (indistinguishable from real sales).

**v2.4.6** \u2014 2026-04-22
- SPD analyzer: pick the integrated-area-dominant peak per band (intensity \u00d7 FWHM) instead of max-intensity. Fixes KSF monitors whose "green" was picked as a 547nm JPEG gridline artifact (10nm FWHM, intensity 0.73) beating the real \u03b2-SiAlON phosphor (45nm FWHM, intensity 0.55). Physically correct metric; red stays intensity-based so narrow KSF Mn\u2074\u207a lines win.
- Monitor KSF green FWHM now 24\u201364nm (was 3\u201364nm with bimodal artifact cluster at 3\u201312nm). Dashboard "Green Peak FWHM by Technology" shows correct distribution.

**v2.4.5** \u2014 2026-04-22
- SPD analyzer: drop sub-2nm FWHM artifacts (JPEG gridlines/compression spikes that appeared as 0\u20131nm "peaks" in verification plots). Real LED/QD peaks are all >10nm; no classifications change.

**v2.4.4** \u2014 2026-04-22
- Guard Tech Explorer correlation chart against empty data (crashed with KeyError when scores were all-null, e.g. during cookie expiry)

**v2.4.3** \u2014 2026-04-21
- Fix SPD analyzer timeout on monitor silo (#48): skip matplotlib plot_verification in pipeline runs via SKIP_SPD_PLOTS env var; raise script timeout 20\u201440 min
- Monitor SPD reclassification after fresh analysis: 40 KSF (new), 20 QD-OLED, 18 WOLED, 14 QD-LCD, 3 Pseudo QD, 2 WLED (previously stuck on stale labels with 0 KSF / 0 WLED)

**v2.4.2** \u2014 2026-04-21
- Monitor data refresh with restored session: 97 monitors, pc_gaming/console_gaming/office/editing scores unblurred (were all NULL), 79 priced

**v2.4.1** \u2014 2026-04-21
- Data refresh with restored RTINGS session: 90 TVs, 1 new, 110 score/classification changes unblurred, 78 priced
- Safari-specific cookie refresh playbook in CLAUDE.md + expiry notification email (HttpOnly caveat documented)

**v2.4.0** \u2014 2026-03-26
- Extract sidebar into src/sidebar.py (aligned with SKU Tracker pattern). Closes #42

**v2.3.1** \u2014 2026-03-26
- Fix stale WLED in monitor price history (Acer Nitro XV275K P3, reclassified to QD-LCD in cec36d7)
- Both pricing pipelines now refresh stale color_architecture labels on every run

**v2.3.0** \u2014 2026-03-25
- Monitor-specific QD Advantage metrics: HDR Peak (10%), BT.2020, Response Time, Brightness, Input Lag, Console Gaming
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


def render_sidebar_top():
    """Logo, tagline, product type selector.

    Returns (product_type, is_blended, pcfg, screen_area_map).
    """
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
        st.sidebar.markdown(
            "<p style='text-align:center;color:#999;font-size:0.85em;margin-top:-8px'>"
            "Display Technology Intelligence</p>",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    _product_types = list(PRODUCT_CONFIGS.keys()) + ["All Products"]
    product_type = st.sidebar.radio(
        "Product Type", _product_types, index=0,
        key="product_type", horizontal=True,
    )
    is_blended = product_type == "All Products"
    pcfg = PRODUCT_CONFIGS.get(product_type, PRODUCT_CONFIGS["TVs"])
    screen_area_map = get_screen_area_map(product_type)
    st.sidebar.divider()

    return product_type, is_blended, pcfg, screen_area_map


def render_report_download(root):
    """Monthly report PDF download button (if any reports exist)."""
    reports_dir = root / "data" / "reports"
    report_files = sorted(reports_dir.glob("display_intelligence_*.pdf"), reverse=True) if reports_dir.exists() else []
    if report_files:
        latest = report_files[0]
        label = latest.stem.replace("display_intelligence_", "").replace("_", " ")
        with open(latest, "rb") as f:
            st.sidebar.download_button(
                f"Monthly Report ({label})",
                f.read(),
                file_name=latest.name,
                mime="application/pdf",
                use_container_width=True,
            )
        st.sidebar.divider()


def render_filters(df, pcfg, is_blended, *, n_8k=0, n_woled_excluded=0):
    """All filter controls: technology, display type, brand, price, model year.

    Returns dict with keys: fdf, tdf, selected_techs.
    """
    st.sidebar.title("Filters")

    # --- Technology checkboxes ---
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
            if st.sidebar.checkbox(brand, value=brand_all, key=f"brand_{brand}"):
                selected_brands.append(brand)
    else:
        selected_brands = list(all_brands) if brand_all else []

    st.sidebar.divider()

    # --- Price range ---
    price_max = float(df["price_best"].max()) if df["price_best"].notna().any() else 10000
    price_range = st.sidebar.slider(
        "Price Range ($)", min_value=0, max_value=int(price_max) + 500,
        value=(0, int(price_max) + 500), step=50,
    )
    _unpriced_label = "products" if is_blended else pcfg["item_label"].lower()
    include_unpriced = st.sidebar.checkbox(
        f"Include {_unpriced_label} without pricing", value=True,
    )

    # --- Model Year checkboxes ---
    selected_years = []
    if pcfg["has_model_year"] and "model_year" in df.columns:
        available_years = sorted(
            df["model_year"].dropna().unique().astype(int).tolist(), reverse=True,
        )
        if available_years:
            st.sidebar.markdown("**Model Year**")
            year_all = st.sidebar.checkbox("All years", value=True, key="year_all")
            for yr in available_years:
                if st.sidebar.checkbox(str(yr), value=year_all, key=f"year_{yr}"):
                    selected_years.append(yr)

    # --- Build filter masks ---
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
    _filter_label = "Products" if is_blended else pcfg["item_label"]
    st.sidebar.markdown(f"**Showing {len(fdf)}/{len(df)} {_filter_label}**")

    # Temporal dataframe — all filters EXCEPT year, for year-over-year analysis
    temporal_mask = (
        df["color_architecture"].isin(selected_techs)
        & df["display_type"].isin(selected_display_types)
        & df["brand"].isin(selected_brands)
    )
    if include_unpriced:
        temporal_mask = temporal_mask & (
            df["price_best"].isna() | df["price_best"].between(price_range[0], price_range[1])
        )
    else:
        temporal_mask = temporal_mask & df["price_best"].between(price_range[0], price_range[1])
    tdf = df[temporal_mask].copy()

    # Caveats
    caveats = []
    if n_8k > 0:
        caveats.append(f"{n_8k} 8K sets excluded from all metrics")
    if n_woled_excluded > 0:
        caveats.append(f"{n_woled_excluded} Samsung WOLED-panel SKUs excluded from QD-OLED pricing")
    if caveats:
        st.sidebar.caption(" \u00b7 ".join(caveats))

    return {"fdf": fdf, "tdf": tdf, "selected_techs": selected_techs}


def render_page_selector(pcfg, is_blended):
    """Page navigation radio. Returns the selected page name."""
    if is_blended:
        return "Cross-Product Analysis"

    all_pages = [
        "Overview", "Technology Explorer", "Price Analyzer",
        "Temporal Analysis", "Comparison Tool", pcfg["profile_page"],
    ]
    qp_page = st.query_params.get("page", None)
    default_idx = all_pages.index(qp_page) if qp_page in all_pages else 0
    return st.sidebar.radio(
        "View", all_pages, index=default_idx,
        format_func=lambda p: f"{_PAGE_ICONS.get(p, '')} {p}",
    )


def render_sidebar_bottom():
    """Version and changelog — call last, renders at bottom of sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown(
        f"<p style='text-align:center;color:#999;font-size:0.8em;margin-bottom:2px'>"
        f"Version {VERSION}</p>",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("What's new?"):
        st.markdown(CHANGELOG)
