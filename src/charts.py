"""Shared chart constants and utilities for the Display Technology Dashboard."""

import numpy as np
import pandas as pd

# Canonical technology ordering (left-to-right: low → high QD content)
TECH_ORDER = ["WLED", "KSF", "Pseudo QD", "QD-LCD", "RGB MiniLED", "WOLED", "QD-OLED"]

# Nanosys brand color palette (colorblind-safe selections)
TECH_COLORS = {
    "QD-OLED": "#FF009F",    # Nanosys magenta
    "WOLED": "#4B40EB",      # Nanosys violet
    "QD-LCD": "#FFC700",     # Nanosys gold
    "RGB MiniLED": "#00A878", # teal — direct RGB-LED backlight (non-QD, spectrally QD-like)
    "Pseudo QD": "#FF7E43",  # Nanosys orange
    "KSF": "#90BFFF",        # Nanosys sky blue
    "WLED": "#6E7681",       # neutral gray — baseline tech, no spectral enhancement
}

DISPLAY_TYPE_COLORS = {
    "OLED": "#4B40EB",
    "LCD": "#FFC700",
}

# Plotly defaults for consistent styling
PL = dict(font=dict(family="Inter, sans-serif", size=14))
MARKER = dict(size=11)

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


def axis_range(col, data_df):
    """Return [min, max] range for a column based on the provided data.

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
    if col not in data_df.columns:
        return None
    vals = pd.to_numeric(data_df[col], errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    price_cols = {"price_best", "price_per_m2", "price_per_mixed_use", "best_price"}
    if col in price_cols:
        return [0, vmax * 1.1]
    pad = (vmax - vmin) * 0.05 if vmax > vmin else 1
    return [max(0, vmin - pad), vmax + pad]
