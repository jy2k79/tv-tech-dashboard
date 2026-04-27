"""Overview page — high-level KPIs, technology distribution, and pricing."""

from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from src.charts import TECH_ORDER, TECH_COLORS, DISPLAY_TYPE_COLORS, friendly, axis_range, PL

_BEST_OF_CSV = Path(__file__).parent.parent / "data" / "rtings_best_of_tvs.csv"
_BEST_OF_HISTORY_CSV = Path(__file__).parent.parent / "data" / "rtings_best_of_history.csv"


def _load_best_of(df_full: pd.DataFrame) -> pd.DataFrame | None:
    """Load the latest RTINGS Best-Of TV snapshot and join in our
    classifications. Tries product_id first (stable), falls back to
    url_part and then normalized fullname so notable mentions (which
    don't carry product_id in the source page) still resolve."""
    if not _BEST_OF_CSV.exists():
        return None
    best = pd.read_csv(_BEST_OF_CSV)
    if best.empty:
        return None

    db = df_full[["product_id", "url_part", "fullname",
                  "color_architecture", "panel_type"]].copy()
    db["product_id"] = pd.to_numeric(db["product_id"], errors="coerce")
    best["product_id"] = pd.to_numeric(best["product_id"], errors="coerce")

    # Stage 1: join by product_id where available
    by_pid = best.merge(db, on="product_id", how="left",
                        suffixes=("", "_db"))

    # Stage 2: for rows still unresolved, fall back to url_part
    unresolved = by_pid["color_architecture"].isna()
    if unresolved.any():
        fallback = best[unresolved][["url_part"]].merge(
            db.dropna(subset=["url_part"])[["url_part",
                                            "color_architecture",
                                            "panel_type"]],
            on="url_part", how="left",
        )
        by_pid.loc[unresolved, "color_architecture"] = fallback["color_architecture"].values
        by_pid.loc[unresolved, "panel_type"] = fallback["panel_type"].values

    # Stage 3: last-resort fullname match (strip case/spacing)
    unresolved = by_pid["color_architecture"].isna()
    if unresolved.any():
        norm = lambda s: str(s).lower().strip().replace("  ", " ")
        db_norm = db.dropna(subset=["fullname"]).copy()
        db_norm["fn_key"] = db_norm["fullname"].map(norm)
        best_norm_keys = by_pid.loc[unresolved, "fullname"].map(norm)
        lookup = dict(zip(db_norm["fn_key"], db_norm["color_architecture"]))
        plookup = dict(zip(db_norm["fn_key"], db_norm["panel_type"]))
        by_pid.loc[unresolved, "color_architecture"] = best_norm_keys.map(lookup)
        by_pid.loc[unresolved, "panel_type"] = best_norm_keys.map(plookup)

    return by_pid


def render(fdf, pcfg, *, product_type=None, df=None):
    """Render the Overview page.

    Parameters
    ----------
    fdf : pd.DataFrame
        Filtered DataFrame (after sidebar filters).
    pcfg : dict
        Product-type config (item_label, score_cols, etc.).
    product_type : str | None
        Active product type ("TVs", "Monitors", "All Products").
    df : pd.DataFrame | None
        Full (unfiltered) DataFrame, used for total count in subtitle.
    """
    st.title(f"{pcfg['item_singular']} Display Technology Dashboard")
    _bench_label = "v2.0+" if product_type == "TVs" else "v2.1.2+"
    _total = len(df) if df is not None else len(fdf)
    st.caption(f"Database: {_total} {pcfg['item_label']} — test bench {_bench_label} · "
               f"Data covers RTINGS-reviewed models only, not the full {pcfg['item_singular'].lower()} market")

    priced = fdf[fdf["price_best"].notna()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(pcfg["item_label"], len(fdf))
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
            st.info(f"No priced {pcfg['item_label'].lower()} in current filter.")

    with hero2:
        _primary = pcfg["primary_score"]
        _primary_label = friendly(_primary)
        st.subheader(f"{_primary_label} Score by Technology")
        if _primary in fdf.columns:
            fig = px.box(fdf, x="color_architecture", y=_primary,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         labels={_primary: f"{_primary_label} Score", "color_architecture": ""},
                         points="all")
            fig.update_layout(showlegend=False, height=370,
                              yaxis=dict(range=axis_range(_primary, fdf)), **PL)
            fig.update_traces(marker=dict(size=7))
            st.plotly_chart(fig, use_container_width=True)

    if product_type == "TVs":
        _render_best_of_section(df if df is not None else fdf)

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
                              xaxis=dict(range=axis_range("price_best", fdf)), **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No priced {pcfg['item_label'].lower()} in current filter.")

    with col4:
        st.subheader("Price by Technology")
        if len(priced) > 0:
            fig = px.box(priced, x="color_architecture", y="price_best",
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         labels={"price_best": "Price ($)", "color_architecture": "Technology"},
                         hover_name="fullname", points="all")
            fig.update_layout(showlegend=False, height=350,
                              yaxis=dict(range=axis_range("price_best", priced)), **PL)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No priced {pcfg['item_label'].lower()} in current filter.")

    st.subheader("Usage Score Overview")
    score_cols = [c for c in pcfg["score_cols"] if c in fdf.columns]
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
                      yaxis=dict(range=axis_range("mixed_usage", fdf)), **PL)
    fig.update_traces(marker=dict(size=7))
    st.plotly_chart(fig, use_container_width=True)


def _tech_share_pie(subset: pd.DataFrame, height: int = 360) -> None:
    """Donut/pie of color_architecture counts in `subset`."""
    counts = (subset["color_architecture"]
              .fillna("Unknown")
              .value_counts()
              .reindex(TECH_ORDER + ["Unknown"])
              .dropna()
              .reset_index())
    counts.columns = ["Technology", "Count"]
    counts = counts[counts["Count"] > 0]
    if counts.empty:
        st.info("No classified picks in this group.")
        return
    colors = {**TECH_COLORS, "Unknown": "#555555"}
    fig = px.pie(counts, names="Technology", values="Count",
                 color="Technology", color_discrete_map=colors,
                 category_orders={"Technology": TECH_ORDER + ["Unknown"]},
                 hole=0.45)
    fig.update_traces(textposition="inside", textinfo="label+percent",
                      insidetextorientation="horizontal",
                      textfont=dict(size=13, color="#0E1117", weight=600))
    fig.update_layout(height=height,
                      legend_title_text="Technology",
                      margin=dict(l=10, r=10, t=10, b=10),
                      **PL)
    st.plotly_chart(fig, use_container_width=True)


def _qd_share_caption(subset: pd.DataFrame) -> None:
    n_qd = int(subset["color_architecture"]
               .isin(["QD-OLED", "QD-LCD", "Pseudo QD"]).sum())
    n_total = int(subset["color_architecture"].notna().sum())
    if n_total:
        pct = round(100 * n_qd / n_total)
        st.caption(
            f"**{n_qd} of {n_total}** classified picks ({pct}%) use a "
            "quantum-dot or pseudo-QD enhancement layer."
        )


def _render_history_chart(df_full: pd.DataFrame) -> None:
    """Stacked step-area showing tech composition of the main 6 picks over
    time. Each snapshot defines the picks until the next snapshot, so we
    use `line_shape='hv'` (horizontal-then-vertical step) — the value
    held by a snapshot persists across the gap until RTINGS publishes
    the next one."""
    if not _BEST_OF_HISTORY_CSV.exists():
        return
    hist = pd.read_csv(_BEST_OF_HISTORY_CSV)
    main = hist[~hist["is_mention"]].copy()
    if main.empty:
        return

    # Join classifications by url_part (works for backfilled rows too)
    db = df_full[["url_part", "color_architecture"]].dropna(subset=["url_part"])
    main = main.merge(db, on="url_part", how="left")
    main["color_architecture"] = main["color_architecture"].fillna("Unknown")

    cat_order = TECH_ORDER + ["Unknown"]
    # Counts per (snapshot_date, color_architecture). Reindex to ensure
    # every (date, tech) pair is present (zeros where absent) so the
    # stacked area stays numerically stable across snapshots.
    snapshot_dates = sorted(main["snapshot_date"].unique())
    grouped = (main.groupby(["snapshot_date", "color_architecture"]).size()
               .unstack(fill_value=0)
               .reindex(columns=cat_order, fill_value=0)
               .reindex(index=snapshot_dates, fill_value=0)
               .reset_index()
               .melt(id_vars="snapshot_date",
                     var_name="color_architecture",
                     value_name="Count"))
    grouped["snapshot_date"] = pd.to_datetime(grouped["snapshot_date"])

    colors = {**TECH_COLORS, "Unknown": "#555555"}
    fig = px.area(grouped, x="snapshot_date", y="Count",
                  color="color_architecture",
                  color_discrete_map=colors,
                  category_orders={"color_architecture": cat_order},
                  line_shape="hv",
                  labels={"snapshot_date": "", "Count": "Picks"})
    fig.update_layout(height=320,
                      legend_title_text="Technology",
                      margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(range=[0, 6.2]),
                      **PL)
    st.plotly_chart(fig, use_container_width=True)


def _render_best_of_section(df_full: pd.DataFrame) -> None:
    """RTINGS Best-Of TVs picks + tech-share visualization.

    TV-only. Reads the latest snapshot scraped from
    rtings.com/tv/reviews/best/tvs-on-the-market.
    """
    best = _load_best_of(df_full)
    if best is None or best.empty:
        return

    st.divider()
    st.subheader("RTINGS Best-Of TVs")
    snapshot_date = best["snapshot_date"].iloc[0]
    main_count = int((~best["is_mention"]).sum())
    mention_count = int(best["is_mention"].sum())
    st.caption(
        f"As of {snapshot_date} \u00b7 "
        f"[rtings.com/tv/reviews/best/tvs-on-the-market](https://www.rtings.com/tv/reviews/best/tvs-on-the-market) "
        f"\u00b7 {main_count} category picks + {mention_count} notable mentions"
    )

    main_picks = best[~best["is_mention"]].copy()
    mentions = best[best["is_mention"]].copy()

    # --- Tables on left, single pie on right ---
    list_col, chart_col = st.columns([3, 2])

    with list_col:
        st.markdown("**Category picks**")
        display = main_picks.rename(columns={"fullname": "TV",
                                             "category": "Category"})
        display["Tech"] = display["color_architecture"].fillna("\u2014")
        st.dataframe(display[["Category", "TV", "Tech"]],
                     use_container_width=True, hide_index=True, height=240)

        if not mentions.empty:
            with st.expander("Notable mentions", expanded=False):
                mdisplay = mentions.rename(columns={"fullname": "TV"})
                mdisplay["Tech"] = mdisplay["color_architecture"].fillna("\u2014")
                mdisplay["#"] = mdisplay["rank"].astype(int)
                st.dataframe(mdisplay[["#", "TV", "Tech"]],
                             use_container_width=True, hide_index=True,
                             height=240)

    with chart_col:
        st.markdown("**Tech share**")
        include_mentions = st.toggle(
            "Include notable mentions",
            value=False,
            help="Toggle to include the 6 notable mentions alongside "
                 "the 6 category picks in the tech-share pie.",
            key="best_of_include_mentions",
        )
        scope = best if include_mentions else main_picks
        _tech_share_pie(scope, height=360)
        _qd_share_caption(scope)

    # --- Composition over time (uses backfilled history) ---
    if _BEST_OF_HISTORY_CSV.exists():
        with st.expander("Composition over time", expanded=False):
            st.caption(
                "Tech composition of the main 6 category picks across "
                "RTINGS' confirmation dates. The Mid-Range slot flipped "
                "from WOLED (LG B4) to QD-LCD (TCL QM8K) on 2025-10-27, "
                "after which all 6 picks have been QD-family."
            )
            _render_history_chart(df_full)
