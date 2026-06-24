#!/usr/bin/env python3
"""
Build the 7-column display technology schema for all TVs.

Reads:
    - data/rtings_tv_data.csv (scraper output)
    - data/spd_analysis_results.csv (SPD classification results)

Produces:
    - data/tv_database.csv (final enriched dataset with all 7 schema columns)

Schema columns (v1.0, locked):
    1. display_type:       OLED | LCD | MicroLED
    2. backlight_type:     Edge-lit | Direct-lit | RGB Mini-LED (LCD only)
    3. dimming_zone_count: integer (nullable)
    4. color_architecture: WLED | KSF | Pseudo QD | QD-LCD | QD-OLED | WOLED
    5. qd_present:         Yes | No
    6. qd_material:        CdSe | InP | Perovskite | Unknown (if qd_present=Yes)
    7. spd_verified:       Yes | No | Pending
    8. marketing_label:    free text (manufacturer's marketing term)
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")


def map_backlight_type(rtings_value, panel_type):
    """Map RTINGS backlight_type to schema backlight_type."""
    if panel_type == 'OLED':
        return None  # OLEDs don't have backlights
    mapping = {
        'Full-Array': 'Direct-lit',
        'Direct': 'Direct-lit',
        'Edge': 'Edge-lit',
        'No Backlight': None,
    }
    return mapping.get(rtings_value, rtings_value)


def determine_qd_material(row):
    """Determine quantum dot material from SPD red peak FWHM.

    Two clean clusters in the data with a gap at ~28-34nm:
      CdSe:        red FWHM < 28nm (narrow)
      InP (Cd-Free): red FWHM > 34nm (wider)
      Unknown:     28-34nm (ambiguous — between clusters)
    """
    color_arch = row['color_architecture']
    if color_arch not in ('QD-LCD', 'QD-OLED'):
        return None

    # QD-OLED is always InP (Samsung/Sony use InP for OLED QD conversion)
    if color_arch == 'QD-OLED':
        return 'InP'

    # QD-LCD: classify from red FWHM — bimodal split with gap at 28-34nm
    CDSE_UPPER = 28   # nm — CdSe cluster tops out here
    INP_LOWER = 34    # nm — InP cluster starts here
    try:
        r_fwhm = float(row.get('red_fwhm_nm', ''))
        if r_fwhm < CDSE_UPPER:
            return 'CdSe'
        elif r_fwhm > INP_LOWER:
            return 'InP'
        else:
            return 'Unknown'
    except (ValueError, TypeError):
        return 'Unknown'


def determine_marketing_label(row):
    """Derive marketing label from product name and brand conventions.

    Not exhaustive — captures the main marketing terms used by manufacturers.
    """
    name = row['fullname']
    brand = row['brand']
    color_arch = row['color_architecture']

    # Samsung marketing labels
    if brand == 'Samsung':
        if 'QN9' in name or 'QN8' in name or 'QN7' in name:
            return 'Neo QLED'
        if name.startswith('Samsung Q') and 'OLED' not in name:
            return 'QLED'
        if 'S95' in name or 'S90' in name:
            return 'QD-OLED'
        if 'S85' in name and 'OLED' in name:
            if color_arch == 'QD-OLED':
                return 'QD-OLED'
            return 'Samsung OLED'
        if 'Frame' in name:
            if 'Pro' in name:
                return 'Neo QLED'
            return 'QLED'
        if 'DU' in name or 'U7' in name or 'U8' in name:
            return 'Crystal UHD'

    # LG marketing labels
    if brand == 'LG':
        if 'OLED' in name:
            if 'G4' in name or 'G5' in name:
                return 'OLED evo'
            return 'OLED'
        if 'QNED' in name:
            return 'QNED'
        if 'UT' in name or 'UA' in name:
            return 'UHD'

    # Sony marketing labels
    if brand == 'Sony':
        if color_arch == 'QD-OLED':
            return 'QD-OLED'
        if 'OLED' in name:
            return 'OLED'
        return 'BRAVIA'

    # Hisense marketing labels
    if brand == 'Hisense':
        if color_arch in ('QD-LCD', 'KSF', 'Pseudo QD'):
            if 'U8' in name or 'U9' in name:
                return 'ULED'
            return 'QLED'
        if 'Canvas' in name:
            return 'CanvasTV'
        return 'ULED'

    # TCL marketing labels
    if brand == 'TCL':
        if 'QM' in name or 'Q7' in name or 'Q6' in name:
            return 'QLED'
        return ''

    # Panasonic
    if brand == 'Panasonic':
        if 'OLED' in name:
            return 'OLED'
        return ''

    # Roku
    if brand == 'Roku':
        return ''

    # Sharp
    if brand == 'Sharp':
        if 'XLED' in name:
            return 'XLED'
        return ''

    # Philips
    if brand == 'Philips':
        if 'OLED' in name:
            return 'OLED'
        return ''

    return ''


def build_schema():
    """Build the complete 7-column display technology schema."""
    # Load data
    tv_df = pd.read_csv(DATA_DIR / "rtings_tv_data.csv")
    spd_df = pd.read_csv(DATA_DIR / "spd_analysis_results.csv")

    print(f"Loaded {len(tv_df)} TVs from scraper data")
    print(f"Loaded {len(spd_df)} SPD analysis results")

    # Merge SPD results into TV data
    spd_cols = ['product_id', 'spd_classification', 'spd_confidence',
                'ground_truth_tech', 'ground_truth_qd_type', 'match_status',
                'blue_peak_nm', 'blue_fwhm_nm', 'green_peak_nm', 'green_fwhm_nm',
                'red_peak_nm', 'red_fwhm_nm', 'num_peaks']
    merged = tv_df.merge(spd_df[spd_cols], on='product_id', how='left')

    # Normalize panel_sub_type: collapse variants like "VA (except 75")" → "VA"
    if 'panel_sub_type' in merged.columns:
        merged['panel_sub_type'] = merged['panel_sub_type'].str.extract(r'^([\w-]+)', expand=False)

    # =========================================================================
    # Column 1: display_type
    # =========================================================================
    merged['display_type'] = merged['panel_type'].map({
        'OLED': 'OLED',
        'LCD': 'LCD',
    }).fillna('LCD')

    # =========================================================================
    # Column 2: backlight_type (LCD only)
    # =========================================================================
    merged['backlight_type_schema'] = merged.apply(
        lambda r: map_backlight_type(r['backlight_type'], r['panel_type']), axis=1
    )

    # =========================================================================
    # Column 2b: dimming_zone_count
    # =========================================================================
    merged['dimming_zone_count'] = pd.to_numeric(
        merged['dimming_zone_count'], errors='coerce'
    ).astype('Int64')  # nullable integer

    # =========================================================================
    # Column 2b fix: set dimming_zone_count to null for OLEDs
    # RTINGS reports pixel count (8,294,400) for OLEDs which isn't meaningful here
    # =========================================================================
    merged.loc[merged['display_type'] == 'OLED', 'dimming_zone_count'] = pd.NA

    # =========================================================================
    # Column 3: color_architecture
    # =========================================================================
    merged['color_architecture'] = merged['spd_classification']

    # =========================================================================
    # Column 3b: Override OLED color_architecture from panel_sub_type
    # RTINGS API test 216 returns QD-OLED or WOLED — higher confidence than SPD
    # =========================================================================
    if 'panel_sub_type' in merged.columns:
        oled_mask = merged['display_type'] == 'OLED'
        has_sub = oled_mask & merged['panel_sub_type'].isin(['QD-OLED', 'WOLED'])
        override_count = has_sub.sum()
        if override_count > 0:
            # Check for mismatches before overriding
            mismatches = has_sub & (merged['color_architecture'] != merged['panel_sub_type'])
            if mismatches.any():
                print(f"\nOLED classification overrides (panel_sub_type vs SPD):")
                for _, row in merged[mismatches].iterrows():
                    print(f"  {row['fullname']:45s} SPD={row['color_architecture']!r:10s} → API={row['panel_sub_type']!r}")
            merged.loc[has_sub, 'color_architecture'] = merged.loc[has_sub, 'panel_sub_type']
            merged.loc[has_sub, 'spd_confidence'] = 'high'
            print(f"\nApplied panel_sub_type override for {override_count} OLEDs")

    # =========================================================================
    # Column 7 (early): marketing_label — needed for KSF/Pseudo QD reclassification
    # =========================================================================
    merged['marketing_label'] = merged.apply(determine_marketing_label, axis=1)

    # =========================================================================
    # Post-SPD reclassification: KSF → Pseudo QD for QLED-marketed sets
    # Per project brief: Pseudo QD = "Marketed as QLED but SPD shows broad peaks
    # inconsistent with true QD nanocrystal emission. Includes sets with trace
    # amounts of QD material that don't meaningfully contribute."
    # KSF-dominant SPD + QLED/QNED marketing = Pseudo QD.
    # =========================================================================
    qled_marketing = ['QLED', 'QNED', 'ULED']
    ksf_mask = merged['color_architecture'] == 'KSF'
    marketed_as_qd = merged['marketing_label'].isin(qled_marketing)
    reclass_mask = ksf_mask & marketed_as_qd
    reclass_count = reclass_mask.sum()
    if reclass_count > 0:
        merged.loc[reclass_mask, 'color_architecture'] = 'Pseudo QD'
        merged.loc[reclass_mask, 'spd_confidence'] = 'medium'
        print(f"\nReclassified {reclass_count} KSF → Pseudo QD (marketed as QLED/QNED/ULED):")
        for _, row in merged[reclass_mask].iterrows():
            print(f"  {row['fullname']:45s} marketing={row['marketing_label']!r}")

    # =========================================================================
    # Post-SPD override: RGB MiniLED (curated model list)
    # RGB MiniLED backlights use discrete red/green/blue LED emitters and are
    # spectrally indistinguishable from QD (confirmed 2026-06: the SPD-based
    # classifier tags them QD-LCD; a green-position/asymmetry signal false-
    # positived on genuine QD sets). RTINGS exposes no "RGB MiniLED" field, so
    # we override by reviewed model. Non-QD: qd_present="No" and qd_material
    # stays blank (handled by the QD-only branches below).
    # =========================================================================
    RGB_MINILED_MODELS = ["R95H", "UR9SG"]
    rgb_pattern = r"\b(?:" + "|".join(RGB_MINILED_MODELS) + r")\b"
    rgb_mask = merged["fullname"].astype(str).str.contains(
        rgb_pattern, case=False, regex=True, na=False)
    rgb_count = int(rgb_mask.sum())
    if rgb_count > 0:
        merged.loc[rgb_mask, "color_architecture"] = "RGB MiniLED"
        merged.loc[rgb_mask, "spd_confidence"] = "high"
        print(f"\nApplied RGB MiniLED override for {rgb_count} sets (curated model list):")
        for _, row in merged[rgb_mask].iterrows():
            print(f"  {row['fullname']}")

    # =========================================================================
    # Column 4: qd_present
    # =========================================================================
    merged['qd_present'] = merged['color_architecture'].map(
        lambda x: 'Yes' if x in ('QD-LCD', 'QD-OLED') else 'No'
    )

    # =========================================================================
    # Column 5: qd_material
    # =========================================================================
    merged['qd_material'] = merged.apply(determine_qd_material, axis=1)

    # =========================================================================
    # Column 6: spd_verified
    # =========================================================================
    merged['spd_verified'] = merged['spd_classification'].map(
        lambda x: 'Yes' if x and x not in ('NO_SPD_IMAGE', '') and not str(x).startswith('ERROR') else 'No'
    )

    # (marketing_label already computed above for KSF/Pseudo QD reclassification)

    # =========================================================================
    # product_type — for future master dashboard compatibility
    # =========================================================================
    merged['product_type'] = 'tv'

    # =========================================================================
    # Select and order output columns
    # =========================================================================
    schema_cols = [
        # Identity
        'product_id', 'fullname', 'brand', 'url_part', 'review_url',
        'test_bench_id', 'test_bench_version', 'released_at', 'first_published_at', 'last_updated_at',
        'sizes_available', 'product_type',

        # Display Technology Schema (7 columns)
        'display_type',
        'backlight_type_schema',
        'dimming_zone_count',
        'color_architecture',
        'qd_present',
        'qd_material',
        'spd_verified',
        'marketing_label',

        # RTINGS panel metadata (for reference)
        'panel_type', 'panel_sub_type', 'backlight_type',

        # SPD analysis details
        'spd_classification', 'spd_confidence',
        'blue_peak_nm', 'blue_fwhm_nm',
        'green_peak_nm', 'green_fwhm_nm',
        'red_peak_nm', 'red_fwhm_nm',

        # Scores
        'mixed_usage', 'home_theater', 'gaming', 'sports', 'bright_room',

        # Picture quality
        'color_score', 'brightness_score', 'black_level_score',
        'contrast_ratio_score', 'native_contrast_score',

        # Color measurements
        'sdr_dci_p3_coverage_pct', 'sdr_bt2020_coverage_pct',
        'hdr_bt2020_coverage_itp_pct',

        # Brightness measurements
        'sdr_real_scene_peak_nits', 'sdr_peak_10pct_nits', 'sdr_peak_2pct_nits',
        'hdr_peak_100pct_nits', 'hdr_peak_50pct_nits', 'hdr_peak_25pct_nits',
        'hdr_peak_10pct_nits', 'hdr_peak_2pct_nits',

        # Contrast
        'native_contrast', 'contrast_ratio',

        # Response time
        'first_response_time_ms', 'total_response_time_ms',
        'input_lag_1080p_ms', 'input_lag_4k_ms',

        # Gaming
        'native_refresh_rate', 'vrr_support', 'hdmi_forum_vrr',
        'hdmi_ports', 'hdmi_21_speed', 'resolution',

        # Ground truth comparison
        'ground_truth_tech', 'ground_truth_qd_type', 'match_status',

        # Metadata
        'scraped_at', 'spd_image', 'spd_image_local',
    ]

    # Only include columns that exist
    available_cols = [c for c in schema_cols if c in merged.columns]
    output = merged[available_cols].copy()

    # Rename for clarity
    output = output.rename(columns={
        'backlight_type_schema': 'backlight_type_v2',
        'backlight_type': 'backlight_type_rtings',
    })

    # Save
    csv_out = DATA_DIR / "tv_database.csv"
    output.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    xlsx_out = DATA_DIR / "tv_database.xlsx"
    output.to_excel(xlsx_out, index=False, sheet_name="TV Database")
    print(f"Saved: {xlsx_out}")

    # Print schema summary
    print("\n" + "=" * 70)
    print("DISPLAY TECHNOLOGY SCHEMA SUMMARY")
    print("=" * 70)

    print(f"\nTotal TVs: {len(output)}")

    print(f"\n1. display_type:")
    for val, count in output['display_type'].value_counts().items():
        print(f"     {val}: {count}")

    print(f"\n2. backlight_type_v2 (LCD only):")
    lcd = output[output['display_type'] == 'LCD']
    for val, count in lcd['backlight_type_v2'].value_counts(dropna=False).items():
        print(f"     {val}: {count}")

    print(f"\n2b. dimming_zone_count: {output['dimming_zone_count'].notna().sum()} TVs with data")
    print(f"     Range: {output['dimming_zone_count'].min()} - {output['dimming_zone_count'].max()}")

    print(f"\n3. color_architecture:")
    for val, count in output['color_architecture'].value_counts().items():
        print(f"     {val}: {count}")

    print(f"\n4. qd_present:")
    for val, count in output['qd_present'].value_counts().items():
        print(f"     {val}: {count}")

    print(f"\n5. qd_material (where qd_present=Yes):")
    qd_yes = output[output['qd_present'] == 'Yes']
    for val, count in qd_yes['qd_material'].value_counts(dropna=False).items():
        print(f"     {val}: {count}")

    print(f"\n6. spd_verified:")
    for val, count in output['spd_verified'].value_counts().items():
        print(f"     {val}: {count}")

    print(f"\n7. marketing_label:")
    for val, count in output['marketing_label'].value_counts().head(15).items():
        print(f"     {val!r}: {count}")

    # Show full technology breakdown
    print(f"\n{'='*70}")
    print("COLOR ARCHITECTURE BY BRAND")
    print("=" * 70)
    pivot = pd.crosstab(output['brand'], output['color_architecture'])
    print(pivot.to_string())

    # Flag KSF/Pseudo QD ambiguities
    ksf_tvs = output[output['color_architecture'] == 'KSF']
    print(f"\n{'='*70}")
    print(f"KSF CLASSIFICATIONS ({len(ksf_tvs)}) — may include Pseudo QD sets")
    print("(SPD cannot distinguish KSF from Pseudo QD with KSF-dominant emission)")
    print("=" * 70)
    for _, row in ksf_tvs.iterrows():
        mkt = row.get('marketing_label', '')
        print(f"  {row['fullname']:45s} marketing={mkt!r:15s}")

    return output


if __name__ == '__main__':
    build_schema()
