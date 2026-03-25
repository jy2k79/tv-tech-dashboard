"""
Silo Configuration — per-product-type settings for the RTINGS pipeline.
=======================================================================

Each silo dict contains API parameters, test IDs, score IDs, file paths,
and normalization rules. Pipeline scripts import the config for their
target silo and use it to parameterize API calls, output paths, etc.

Adding a new product type (e.g., laptops) requires only adding a new
entry here — shared pipeline code (scraper, SPD analyzer) picks it up.
"""

from pathlib import Path

DATA_DIR = Path("data")
SPD_DIR = Path("spd_images")

# =============================================================================
# TV CONFIGURATION
# =============================================================================
TV = {
    "name": "tv",
    "display_name": "TVs",
    "silo_id": 1,
    "silo_url_part": "tv",

    # Minimum test bench version to include (inclusive)
    # TV: v2.0 and above
    "min_bench_version": (2, 0, 0),

    # Fallback bench IDs if dynamic discovery fails
    # 227 = v2.2, 210 = v2.1, 197 = v2.0.1
    "fallback_bench_ids": ["227", "210", "197"],

    # ---- RTINGS API test IDs (original_id → column name) ----
    "test_ids": {
        # Identity / panel info
        "217": "panel_type",
        "216": "panel_sub_type",
        "215": "backlight_type",
        "208": "resolution",
        "219": "native_refresh_rate",

        # Brightness — HDR
        "141": "hdr_peak_2pct_nits",
        "461": "hdr_peak_10pct_nits",
        "462": "hdr_peak_25pct_nits",
        "96":  "hdr_peak_50pct_nits",
        "463": "hdr_peak_100pct_nits",

        # Brightness — SDR
        "609": "sdr_peak_2pct_nits",
        "610": "sdr_peak_10pct_nits",
        "619": "sdr_real_scene_peak_nits",

        # Contrast / black level
        "11":  "native_contrast",
        "647": "contrast_ratio",

        # Dimming
        "16981": "dimming_zone_count",

        # Color
        "28334": "sdr_dci_p3_coverage_pct",
        "28336": "sdr_bt2020_coverage_pct",
        "927":   "hdr_bt2020_coverage_itp_pct",

        # Response time
        "30862": "first_response_time_ms",
        "30860": "total_response_time_ms",

        # Input lag
        "12237": "input_lag_1080p_ms",
        "12239": "input_lag_4k_ms",

        # Connectivity
        "487":  "hdmi_ports",
        "2684": "hdmi_21_speed",

        # VRR
        "2190": "vrr_support",
        "5039": "hdmi_forum_vrr",

        # SPD image (picture type — returns asset_url)
        "26239": "spd_image",

        # Backlight chart image
        "133": "backlight_chart",
    },

    # ---- Usage/performance score IDs ----
    "usage_ids": {
        "1":     "mixed_usage",
        "12":    "home_theater",
        "32477": "bright_room",
        "9":     "sports",
        "10":    "gaming",
        "32475": "brightness_score",
        "32565": "black_level_score",
        "32566": "color_score",
        "32496": "game_mode_responsiveness",
    },

    # ---- Panel type normalization ----
    # TV panel_type values are already clean (LCD, OLED)
    "panel_type_normalization": {},

    # ---- File paths ----
    "paths": {
        "scraped_csv": DATA_DIR / "rtings_tv_data.csv",
        "scraped_json": DATA_DIR / "rtings_tv_data.json",
        "scraped_xlsx": DATA_DIR / "rtings_tv_data.xlsx",
        "spd_results": DATA_DIR / "spd_analysis_results.csv",
        "database": DATA_DIR / "tv_database.csv",
        "database_xlsx": DATA_DIR / "tv_database.xlsx",
        "database_with_prices": DATA_DIR / "tv_database_with_prices.csv",
        "price_history": DATA_DIR / "price_history.csv",
        "registry": DATA_DIR / "tv_registry.csv",
        "changelog": DATA_DIR / "changelog.csv",
        "spd_images": SPD_DIR,
        "raw_api": DATA_DIR / "raw",
        "session_flag": DATA_DIR / ".session_ok",
    },
}


# =============================================================================
# MONITOR CONFIGURATION
# =============================================================================
MONITOR = {
    "name": "monitor",
    "display_name": "Monitors",
    "silo_id": 5,
    "silo_url_part": "monitor",

    # Minimum test bench version to include (inclusive)
    # Monitor: v2.1.2 and above (when SPD was added)
    "min_bench_version": (2, 1, 2),

    # Fallback bench IDs if dynamic discovery fails
    # 238 = v2.1.2
    "fallback_bench_ids": ["238"],

    # ---- RTINGS API test IDs (original_id → column name) ----
    # Column names match TV where concepts overlap (for future master dashboard).
    "test_ids": {
        # Identity / panel info
        "1419":  "panel_type",
        "40809": "panel_sub_type",
        "1390":  "backlight_type",
        "1532":  "resolution",
        "4448":  "native_refresh_rate",

        # Display geometry (monitor-specific)
        "1602": "display_size",
        "1533": "aspect_ratio",
        "1558": "pixel_density",

        # Brightness — HDR
        "1562": "hdr_peak_2pct_nits",
        "1563": "hdr_peak_10pct_nits",

        # Contrast / black level
        "1385": "native_contrast",

        # Dimming
        "1388": "local_dimming",

        # Color
        "9332": "sdr_dci_p3_coverage_pct",
        "1500": "sdr_bt2020_coverage_pct",
        "1547": "hdr_bt2020_coverage_itp_pct",

        # Response time
        "1428": "first_response_time_ms",
        "1429": "total_response_time_ms",

        # Input lag
        "1436": "input_lag_native_ms",

        # SPD image (picture type — returns asset_url)
        "40810": "spd_image",
    },

    # ---- Usage/performance score IDs ----
    "usage_ids": {
        "4112":  "pc_gaming",
        "17021": "console_gaming",
        "4113":  "office",
        "4114":  "editing",
        "28295": "brightness_score",
        "28299": "color_accuracy",
    },

    # ---- Panel type normalization ----
    # Monitors report IPS/VA/TN as panel_type; normalize to LCD for display_type
    "panel_type_normalization": {
        "IPS": "LCD",
        "VA": "LCD",
        "TN": "LCD",
    },

    # ---- File paths ----
    "paths": {
        "scraped_csv": DATA_DIR / "rtings_monitor_data.csv",
        "scraped_json": DATA_DIR / "rtings_monitor_data.json",
        "scraped_xlsx": DATA_DIR / "rtings_monitor_data.xlsx",
        "spd_results": DATA_DIR / "spd_monitor_analysis_results.csv",
        "database": DATA_DIR / "monitor_database.csv",
        "database_xlsx": DATA_DIR / "monitor_database.xlsx",
        "database_with_prices": DATA_DIR / "monitor_database_with_prices.csv",
        "price_history": DATA_DIR / "monitor_price_history.csv",
        "registry": DATA_DIR / "monitor_registry.csv",
        "changelog": DATA_DIR / "monitor_changelog.csv",
        "spd_images": SPD_DIR / "monitors",
        "raw_api": DATA_DIR / "raw_monitors",
        "session_flag": DATA_DIR / ".monitor_session_ok",
    },
}


# =============================================================================
# SILO REGISTRY
# =============================================================================
# All configured silos, keyed by name. Pipeline scripts use this to
# iterate over silos or look up a specific one.
SILOS = {
    "tv": TV,
    "monitor": MONITOR,
}


def get_silo(name: str) -> dict:
    """Get a silo config by name. Raises KeyError if not found."""
    if name not in SILOS:
        valid = ", ".join(SILOS.keys())
        raise KeyError(f"Unknown silo '{name}'. Valid silos: {valid}")
    return SILOS[name]


# =============================================================================
# SAMSUNG WOLED PANEL SIZE MAPPING
# =============================================================================
# Samsung OLED sizes that use LG WOLED panels instead of QD-OLED.
# Samsung Display only makes QD-OLED at 55", 65", 77". All other sizes are WOLED.
# Updated for 2024-2026 lineup (S90D/F/H, S95D/F/H, S85D/F/H, S99H, S82H, S83H).
#
# Format: model_substring → set of sizes that are WOLED.
# If a model is all-WOLED, use _ALL_WOLED sentinel.
_ALL_WOLED = {42, 48, 55, 65, 77, 83, 85, 98, 100}  # every plausible TV size
SAMSUNG_WOLED_SIZES = {
    # 2024-2025 models
    "S90": {42, 48, 83},          # S90D/F/H: 42/48/83 WOLED, 55/65/77 QD-OLED
    "S95": {48, 83},              # S95F: 83 WOLED. S95H: 48+83 WOLED. 55/65/77 QD-OLED.
    "S99": {83},                  # S99H: 83 WOLED (no 83" QD-OLED panel exists). 55/65/77 QD-OLED.
    # All-WOLED models (no QD-OLED at any size)
    "S85H": _ALL_WOLED,           # S85H: all WOLED (regression from S85F)
    "S85D": _ALL_WOLED,           # S85D: all WOLED
    "S82": _ALL_WOLED,            # S82H: budget OLED, all WOLED
    "S83": _ALL_WOLED,            # S83H: budget OLED, all WOLED
    # S85F is special: 55/65 are QD-OLED, 77/83 are WOLED
    "S85F": {77, 83},
}


def is_samsung_woled_sku(fullname: str, color_arch: str, size: int | None) -> bool:
    """Return True if this SKU is a Samsung OLED size that uses a WOLED panel
    despite the product being classified as QD-OLED."""
    if color_arch != "QD-OLED" or not size or "Samsung" not in fullname:
        return False
    for model_substr, woled_sizes in SAMSUNG_WOLED_SIZES.items():
        if model_substr in fullname and size in woled_sizes:
            return True
    return False
