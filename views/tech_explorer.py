"""Technology Explorer page — SPD analysis, panel metrics, QD advantage, and value."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import TECH_ORDER, TECH_COLORS, friendly, axis_range, PL, MARKER


def render(fdf, pcfg):
    """Render the Technology Explorer page."""
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

        st.markdown("**Narrow Green, Wider Color \u2014 Green FWHM vs Rec.2020 Coverage**")
        st.caption(
            f"Every {pcfg['item_singular'].lower()} in the dataset. Narrower green emission "
            "(X, left side) maps to wider color gamut coverage (Y, top). Quantum-dot techs "
            "cluster in their own corner."
        )
        _gamut = fdf[fdf["green_fwhm_nm"].notna()
                     & fdf["hdr_bt2020_coverage_itp_pct"].notna()]
        _n_wide = int((_gamut["green_fwhm_nm"] > 80).sum())
        _gamut_plot = _gamut[_gamut["green_fwhm_nm"] <= 80]
        fig = px.scatter(
            _gamut_plot, x="green_fwhm_nm", y="hdr_bt2020_coverage_itp_pct",
            color="color_architecture", color_discrete_map=TECH_COLORS,
            category_orders={"color_architecture": TECH_ORDER},
            hover_name="fullname", hover_data=["brand", "marketing_label"],
            labels={"green_fwhm_nm": "Green FWHM (nm)",
                    "hdr_bt2020_coverage_itp_pct": "Rec.2020 Coverage (%)"},
        )
        fig.add_shape(type="rect", x0=20, x1=30, y0=40, y1=75,
                      line=dict(color="rgba(255,0,159,0.5)", dash="dash"),
                      fillcolor="rgba(255,0,159,0.06)")
        fig.add_annotation(x=25, y=1.18, xref="x", yref="paper",
                           text="<b>QD ZONE</b>", showarrow=False,
                           font=dict(color="rgba(255,0,159,0.9)", size=14))
        fig.add_annotation(x=25, y=1.10, xref="x", yref="paper",
                           text="narrow emission \u00b7 wide gamut",
                           showarrow=False,
                           font=dict(color="rgba(255,0,159,0.7)", size=11))
        fig.update_traces(marker=MARKER)
        fig.update_layout(height=500, legend_title_text="Technology",
                          xaxis=dict(range=[20, 80]),
                          yaxis=dict(range=[15, 75]),
                          margin=dict(t=110),
                          **PL)
        st.plotly_chart(fig, use_container_width=True)
        if _n_wide:
            _label = pcfg["item_label"].lower()
            st.caption(
                f"{_n_wide} WOLED {_label.rstrip('s')}(s) with green FWHM > 80 nm not shown."
            )

        st.divider()

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
            pcfg["input_lag_col"], "total_response_time_ms",
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
            y_range = axis_range(metric, finite)
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
                fig.add_annotation(text="\u221e", x=xi, y=ymax * 0.95,
                                   showarrow=False,
                                   font=dict(size=18, color="white", family="Inter"))
        else:
            fig = px.box(valid, x="color_architecture", y=metric,
                         color="color_architecture", color_discrete_map=TECH_COLORS,
                         category_orders={"color_architecture": TECH_ORDER},
                         points="all", hover_name="fullname",
                         labels={metric: friendly(metric), "color_architecture": ""})
            fig.update_layout(showlegend=False, height=500,
                              yaxis=dict(range=axis_range(metric, fdf)), **PL)
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Average Scores by Technology**")
        score_cols = pcfg["score_cols"] + pcfg["extra_score_cols"]
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
            (col, label) for col, label in pcfg.get("advantage_metrics", [])
            if col in fdf.columns
        ]

        for row_start in range(0, len(advantage_metrics), 3):
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
            (col, label) for col, label in [
                ("total_response_time_ms", "Total Response Time (ms)"),
                ("first_response_time_ms", "First Response Time (ms)"),
            ] if col in fdf.columns
        ]
        if not resp_metrics:
            st.info("Response time data not available for this product type.")
        rcols = st.columns(max(len(resp_metrics), 1))
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
                                  yaxis=dict(range=axis_range(resp_col, fdf)), **PL)
                fig.update_traces(marker=dict(size=9))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Technology radar overlay
        st.markdown("**Technology Profile Radar**")
        st.caption("Metrics normalized 0\u20131 across technologies (higher = better on all axes)")
        radar_metrics = [
            (col, label) for col, label in [
                ("brightness_score", "Brightness"),
                ("contrast_ratio_score", "Contrast"),
                ("color_score", "Color"),
                ("color_accuracy", "Color Accuracy"),
                ("black_level_score", "Black Level"),
                ("hdr_bt2020_coverage_itp_pct", "HDR Gamut"),
            ] if col in fdf.columns
        ]
        tech_means = fdf.groupby("color_architecture").agg(
            {m[0]: "mean" for m in radar_metrics}
        )
        # Response speed: invert so higher = faster (better)
        _resp_col = "total_response_time_ms" if "total_response_time_ms" in fdf.columns else None
        if _resp_col:
            resp_mean = fdf.groupby("color_architecture")[_resp_col].mean()
            max_resp = resp_mean.max()
            tech_means["response_speed"] = max_resp - resp_mean

        radar_labels = [m[1] for m in radar_metrics] + (["Response Speed"] if _resp_col else [])
        radar_cols = [m[0] for m in radar_metrics] + (["response_speed"] if _resp_col else [])

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
            (pcfg["primary_score"], friendly(pcfg["primary_score"]) + " Score"),
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
        _ps = pcfg["primary_score"]
        _ps_label = friendly(_ps)
        st.subheader(f"What Drives {_ps_label} Scores?")
        st.caption(f"Correlation analysis: which metrics predict overall {pcfg['item_singular'].lower()} performance")

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
            pcfg["input_lag_col"]: friendly(pcfg["input_lag_col"]),
        }
        corr_metrics = {k: v for k, v in _all_corr_metrics.items() if k in fdf.columns}
        corr_data = []
        for col, label in corr_metrics.items():
            if col in fdf.columns and _ps in fdf.columns:
                valid = fdf[[_ps, col]].dropna()
                if len(valid) > 5:
                    r = valid[_ps].corr(valid[col])
                    corr_data.append({"Metric": label, "col": col, "Correlation": r})

        if not corr_data:
            st.info(
                f"Not enough data to compute correlations with {_ps_label}. "
                f"This usually means {pcfg['item_label'].lower()} scores are missing — "
                "check that the RTINGS session cookie is fresh and the pipeline has run."
            )
        else:
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
                                          xaxis=dict(range=axis_range(_dx, fdf)),
                                          yaxis=dict(range=axis_range(_ps, fdf)), **PL)
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
                size=pcfg["primary_score"] if pcfg["primary_score"] in pos_valid.columns else None,
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
                              xaxis=dict(range=axis_range("hdr_peak_10pct_nits", fdf)),
                              yaxis=dict(range=axis_range("hdr_bt2020_coverage_itp_pct", fdf)), **PL)
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 6: Value Analysis ---
    with tab6:
        st.subheader("Value Analysis: Cost per Performance Point")
        st.caption("Lower $/point = more performance for your money")

        if pcfg["has_price_per_score"] and pcfg["price_per_score_col"] in fdf.columns:
            val_priced = fdf[fdf[pcfg["price_per_score_col"]].notna()].copy()
        else:
            val_priced = fdf[fdf["price_best"].notna()].copy()

        if len(val_priced) == 0:
            st.warning(f"No priced {pcfg['item_label'].lower()} match the current filters.")
        else:
            if pcfg["has_price_per_score"] and pcfg["price_per_score_col"] in val_priced.columns:
                _pps_col = pcfg["price_per_score_col"]
                fig = px.box(val_priced, x="color_architecture", y=_pps_col,
                             color="color_architecture", color_discrete_map=TECH_COLORS,
                             category_orders={"color_architecture": TECH_ORDER},
                             points="all", hover_name="fullname",
                             labels={_pps_col: "$ per Mixed Usage Point",
                                     "color_architecture": ""})
                fig.update_layout(showlegend=False, height=440,
                                  yaxis=dict(range=axis_range(_pps_col, fdf)), **PL)
                fig.update_traces(marker=dict(size=9))
                fig.add_annotation(
                    x=0.5, y=-0.12, xref="paper", yref="paper",
                    text="Lower = better value", showarrow=False,
                    font=dict(size=13, color="rgba(255,255,255,0.5)"),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            _ps = pcfg["primary_score"]
            _ps_label = friendly(_ps)
            st.markdown("**Value Frontier: Price vs Performance**")
            st.caption(f"The dashed line traces the \"efficient frontier\" — {pcfg['item_label'].lower()} that offer "
                       f"the highest {_ps_label} score for their price. Any {pcfg['item_singular'].lower()} on or near "
                       f"the line is the best performance you can buy at that budget. "
                       f"{pcfg['item_label']} far below the line are overpriced for what they deliver.")
            _hover = ["price_size", "brand"]
            if pcfg["price_per_score_col"] and pcfg["price_per_score_col"] in val_priced.columns:
                _hover.insert(0, pcfg["price_per_score_col"])
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
                              xaxis=dict(range=axis_range("price_best", fdf)),
                              yaxis=dict(range=axis_range(_ps, fdf)), **PL)
            fig.update_traces(marker=MARKER, selector=dict(mode="markers"))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            if pcfg["has_price_per_score"] and pcfg["price_per_score_col"] in val_priced.columns:
                _pps = pcfg["price_per_score_col"]
                st.markdown(f"**Top 20 Best Value {pcfg['item_label']}** (lowest {pcfg['price_per_score_label']})")
                top_val = val_priced.sort_values(_pps).head(20)
            else:
                st.markdown(f"**Top 20 Best Value {pcfg['item_label']}** (lowest price)")
                top_val = val_priced.sort_values("price_best").head(20)
                _pps = None
            _val_cols = ["fullname", "color_architecture", "price_best", _ps]
            _val_headers = [pcfg["item_singular"], "Technology", "Price", _ps_label]
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
            if "HDR Brightness" in val_table.columns:
                val_table["HDR Brightness"] = val_table["HDR Brightness"].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "\u2014")
            if "BT.2020 %" in val_table.columns:
                val_table["BT.2020 %"] = val_table["BT.2020 %"].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "\u2014")
            if "Size" in val_table.columns:
                val_table["Size"] = val_table["Size"].apply(
                    lambda x: f'{int(x)}"' if pd.notna(x) else "?")
            st.dataframe(val_table, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Average $/Point by Technology**")
            if pcfg["has_price_per_score"] and pcfg["price_per_score_col"] in val_priced.columns:
                _avg_col = pcfg["price_per_score_col"]
            else:
                _avg_col = "price_best"
            overall_avg = val_priced[_avg_col].mean()
            tech_means_val = val_priced.groupby("color_architecture")[_avg_col].mean()
            mcols = st.columns(len(tech_means_val))
            for i, (tech, mean_val) in enumerate(tech_means_val.sort_values().items()):
                delta = mean_val - overall_avg
                with mcols[i]:
                    st.metric(str(tech), f"${mean_val:,.0f}/pt",
                              delta=f"${delta:+,.0f} vs avg", delta_color="inverse")
