#!/usr/bin/env python3
"""
SPD Analyzer for RTINGS TV Display Technology Classification
=============================================================
Extracts spectral curves from RTINGS SPD chart images, analyzes peak
positions and widths (FWHM), and classifies each TV's color architecture.

Now reads from scraped data (data/rtings_tv_data.csv) and uses
auto-calibration to detect plot boundaries from gridlines.

Requirements:
    pip install Pillow numpy scipy pandas openpyxl matplotlib

Usage:
    python spd_analyzer.py                    # Analyze all 85 scraped TVs
    python spd_analyzer.py --validate-only    # Only validate against ground truth

Output:
    - spd_extracted_curves/ folder with verification plots
    - data/spd_analysis_results.csv with full classification results
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CALIBRATED CHART PARAMETERS
# ============================================================================
# Detected from gridline analysis on 3840x2160 RTINGS SPD charts.
# All RTINGS SPD charts use identical layout: 350-750nm x-axis, 0-0.008 y-axis.
# The auto-calibrate function below adapts these to any image size.

# Reference gridline positions (for 3840x2160 images)
REF_WIDTH = 3840
REF_HEIGHT = 2160
# Vertical gridlines at 350, 400, 450, ..., 750nm
REF_VGRID_X = [231, 672, 1114, 1555, 1997, 2439, 2881, 3322, 3764]
REF_VGRID_WL = [350, 400, 450, 500, 550, 600, 650, 700, 750]
# Horizontal gridlines at 0.008, 0.006, 0.004 (0.002 and 0.000 derived)
REF_HGRID_Y = [132, 605, 1078]  # measured from image
REF_HGRID_VAL = [0.008, 0.006, 0.004]  # corresponding values

WL_MIN = 350
WL_MAX = 750

# Peak classification thresholds (nm FWHM)
# Calibrated from measured FWHM on 55 ground-truth TVs:
#   QD-OLED:   B:20-21  G:26-28  R:36-37
#   QD-LCD:    B:14-18  G:22-45  R:9-44  (CdSe narrow, InP broader)
#   KSF:       B:15-17  G:31-66  R:9 (narrow red, broad green)
#   Pseudo QD: B:15-17  G:39-43  R:9-42
#   WOLED:     B:22-25  G:58-75  R:19-21
#   WLED:      B:17-18  G:62-64  R:30-31
NARROW_FWHM = 40     # QD peaks are < 40nm (raised from 35 to capture InP QD + QD-OLED red)
BROAD_FWHM = 55       # WOLED/WLED phosphor humps > 55nm

# Output directories
CURVES_DIR = Path("spd_extracted_curves")
DATA_DIR = Path("data")


# ============================================================================
# AUTO-CALIBRATION
# ============================================================================
def calibrate_chart(img_array):
    """
    Detect plot area boundaries by finding gridlines in the image.
    Returns pixel coordinates for the plot area and wavelength mapping.

    Works for both large (3840x2160) and small (960x540) SPD images.
    """
    h, w, _ = img_array.shape
    gray = np.mean(img_array, axis=2)

    # Strategy: sample a row near the top of the chart (where there's no data,
    # just white background + gridlines) and find the gray dips = gridlines.
    # Then use known wavelength mapping (350-750nm, 9 gridlines) to calibrate.

    # Find vertical gridlines by sampling at ~15% height
    sample_y = int(h * 0.15)
    row = gray[sample_y, :]
    inverted = 255.0 - row

    # Find dips (gridlines are gray ~150-180 on white ~255 background)
    vpeaks, _ = find_peaks(inverted, height=15, distance=max(w // 20, 10), prominence=8)

    if len(vpeaks) >= 8:
        # Good detection — use the 9 gridlines directly
        # Take the ones most evenly spaced (should be exactly 9)
        if len(vpeaks) > 9:
            # Pick the 9 most evenly-spaced peaks
            # Simple heuristic: take first and last, interpolate
            vpeaks = np.array(sorted(vpeaks))
            # Choose subset closest to uniform spacing
            best = vpeaks[:9]  # fallback
            if len(vpeaks) >= 9:
                best_score = float('inf')
                for start in range(len(vpeaks) - 8):
                    subset = vpeaks[start:start + 9]
                    diffs = np.diff(subset)
                    score = np.std(diffs)
                    if score < best_score:
                        best_score = score
                        best = subset
                vpeaks = best

        grid_x = np.array(sorted(vpeaks[:9]))
    else:
        # Fallback: scale from reference positions
        scale_x = w / REF_WIDTH
        grid_x = np.array([int(x * scale_x) for x in REF_VGRID_X])

    # Linear mapping: wavelength -> pixel
    # grid_x positions correspond to 350, 400, 450, ..., 750
    wl_arr = np.array(REF_VGRID_WL[:len(grid_x)])
    px_fit = np.polyfit(wl_arr, grid_x, 1)  # pixel = a*nm + b

    # Inverse: nm = (pixel - b) / a
    plot_x_left = int(np.polyval(px_fit, WL_MIN))
    plot_x_right = int(np.polyval(px_fit, WL_MAX))

    # Find horizontal gridlines for y-axis calibration
    sample_x = int(w * 0.5)
    col = gray[:, sample_x]
    inverted_col = 255.0 - col

    hpeaks, _ = find_peaks(inverted_col, height=15, distance=max(h // 15, 10), prominence=8)
    # Filter: only keep peaks in the upper 95% of image, above the data area
    # The true gridlines are at consistent gray brightness, not data curve crossings
    hgrid = []
    for hp in hpeaks:
        # Gridlines have moderate darkness (~150-200); curve crossings are very dark (<50)
        if gray[hp, sample_x] > 100 and hp < h * 0.95:
            hgrid.append(hp)

    if len(hgrid) >= 3:
        # First 3 horizontal gridlines = 0.008, 0.006, 0.004
        hgrid = sorted(hgrid)[:3]
        # Spacing per 0.002 unit
        spacing = np.mean(np.diff(hgrid))
        # 0.000 level = hgrid[2] + 2 * spacing
        plot_y_top = hgrid[0]
        plot_y_bottom = int(hgrid[2] + 2 * spacing)
    else:
        # Fallback: scale from reference
        scale_y = h / REF_HEIGHT
        plot_y_top = int(REF_HGRID_Y[0] * scale_y)
        spacing = np.mean(np.diff(REF_HGRID_Y)) * scale_y
        plot_y_bottom = int(REF_HGRID_Y[2] * scale_y + 2 * spacing)

    return {
        'plot_x_left': max(0, plot_x_left),
        'plot_x_right': min(w - 1, plot_x_right),
        'plot_y_top': max(0, plot_y_top),
        'plot_y_bottom': min(h - 1, plot_y_bottom),
        'wl_fit': px_fit,  # pixel = a*nm + b
        'grid_x': grid_x,
    }


# ============================================================================
# SPD CURVE EXTRACTION
# ============================================================================
def extract_spd_curve(image_path):
    """
    Extract the spectral curve from an RTINGS SPD chart image.

    Strategy: For each pixel column, scan top-to-bottom looking for the
    transition from white background to the colored curve/fill area.
    After extraction, remove gridline spike artifacts by masking gridline
    positions and interpolating through them.

    Returns:
        wavelengths: numpy array (nm)
        intensities: numpy array (0-1 normalized)
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    h, w, _ = img_array.shape

    # Auto-calibrate plot boundaries
    cal = calibrate_chart(img_array)
    x_left = cal['plot_x_left']
    x_right = cal['plot_x_right']
    y_top = cal['plot_y_top']
    y_bottom = cal['plot_y_bottom']
    grid_x = cal['grid_x']
    plot_h = y_bottom - y_top

    # Build set of pixel columns to mask (gridline positions +/- margin)
    gridline_margin = max(3, int(w / 800))  # ~5px at 3840, ~2px at 960
    gridline_cols = set()
    for gx in grid_x:
        for offset in range(-gridline_margin, gridline_margin + 1):
            col_idx = gx - x_left + offset
            if 0 <= col_idx < (x_right - x_left):
                gridline_cols.add(col_idx)

    # Extract intensity profile for each column in the plot area
    gray = np.mean(img_array, axis=2)
    n_cols = x_right - x_left
    raw_intensities = np.zeros(n_cols)
    is_gridline = np.zeros(n_cols, dtype=bool)

    for i in range(n_cols):
        if i in gridline_cols:
            is_gridline[i] = True
            continue  # skip gridline columns; interpolate later

        x = x_left + i
        col = gray[y_top:y_bottom, x]

        # Scan from top to find where white background ends and curve begins.
        # The curve boundary is a thin dark line (~2px). Below it is colored fill.
        # Above it is white background (brightness > 240).
        # Strategy: find the first sustained dark region from the top.
        curve_y = plot_h - 1  # default: bottom (zero intensity)

        for row in range(len(col)):
            if col[row] < 220:
                # Verify this isn't a horizontal gridline: check if neighbors
                # to the left and right are also dark at this y (gridline = wide)
                # vs. curve (only dark at this specific column's curve position)
                if row + 5 < len(col):
                    ahead = col[row:row + 6]
                    if np.mean(ahead) < 215:
                        curve_y = row
                        break
                else:
                    curve_y = row
                    break

        raw_intensities[i] = 1.0 - (curve_y / plot_h)

    # Interpolate through gridline positions
    valid_idx = np.where(~is_gridline)[0]
    gridline_idx = np.where(is_gridline)[0]
    if len(valid_idx) > 0 and len(gridline_idx) > 0:
        raw_intensities[gridline_idx] = np.interp(
            gridline_idx, valid_idx, raw_intensities[valid_idx]
        )

    # Map columns to wavelengths
    wavelengths = np.linspace(WL_MIN, WL_MAX, n_cols)

    # Smooth to reduce JPEG artifacts
    intensities = gaussian_filter1d(raw_intensities, sigma=3)

    # Normalize to 0-1
    max_val = np.max(intensities)
    if max_val > 0:
        intensities = intensities / max_val

    return wavelengths, intensities


# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================
def _measure_fwhm_robust(wavelengths, intensities, peak_idx, all_peaks_idx,
                         samples_per_nm):
    """
    Overlap-aware FWHM measurement.

    For well-separated peaks, uses standard scipy peak_widths (prominence-based).
    For overlapping peaks (e.g. WLED green/red sharing a broad phosphor hump),
    measures the half-width on the uncontaminated side and doubles it (HWHM
    mirroring).  This avoids the artifact where prominence-based measurement
    gives artificially narrow FWHM when the valley between peaks is high.

    Returns (fwhm_nm, method_str).
    """
    peak_height = intensities[peak_idx]
    half_max_abs = peak_height / 2.0

    # --- Standard prominence-based FWHM (baseline) ---
    w = peak_widths(intensities, [peak_idx], rel_height=0.5)
    standard_fwhm = w[0][0] / samples_per_nm

    # --- Detect overlap on each side ---
    overlap_left = False
    overlap_right = False
    OVERLAP_THRESH = 0.50  # valley > 50% of shorter peak → overlapping

    for nb in all_peaks_idx:
        if nb == peak_idx:
            continue
        if nb < peak_idx:
            valley = np.min(intensities[nb:peak_idx + 1])
            shorter = min(peak_height, intensities[nb])
            if shorter > 0 and valley / shorter > OVERLAP_THRESH:
                overlap_left = True
        else:
            valley = np.min(intensities[peak_idx:nb + 1])
            shorter = min(peak_height, intensities[nb])
            if shorter > 0 and valley / shorter > OVERLAP_THRESH:
                overlap_right = True

    if not overlap_left and not overlap_right:
        return standard_fwhm, 'standard'

    # --- HWHM mirroring from the clean side ---
    # Compute HWHM from the uncontaminated side and double it.
    # Then take the MAX of standard and HWHM to handle asymmetric peaks:
    #   - Green phosphor peak: left side is steep, standard is wider → keeps standard
    #   - Red phosphor peak: standard is artificially narrow → HWHM wins
    def _right_hwhm():
        for i in range(peak_idx + 1, len(intensities)):
            if intensities[i] <= half_max_abs:
                frac = ((intensities[i - 1] - half_max_abs)
                        / (intensities[i - 1] - intensities[i]))
                return (i - 1 + frac - peak_idx) / samples_per_nm
        return None

    def _left_hwhm():
        for i in range(peak_idx - 1, -1, -1):
            if intensities[i] <= half_max_abs:
                frac = ((intensities[i + 1] - half_max_abs)
                        / (intensities[i + 1] - intensities[i]))
                return (peak_idx - (i + 1 - frac)) / samples_per_nm
        return None

    hwhm_fwhm = None
    hwhm_side = None

    if overlap_left and not overlap_right:
        hw = _right_hwhm()
        if hw is not None:
            hwhm_fwhm = 2.0 * hw
            hwhm_side = 'right'
    elif overlap_right and not overlap_left:
        hw = _left_hwhm()
        if hw is not None:
            hwhm_fwhm = 2.0 * hw
            hwhm_side = 'left'
    else:
        # Both sides overlap — try right first (red tail is usually clean)
        hw = _right_hwhm()
        if hw is not None:
            hwhm_fwhm = 2.0 * hw
            hwhm_side = 'right'
        else:
            hw = _left_hwhm()
            if hw is not None:
                hwhm_fwhm = 2.0 * hw
                hwhm_side = 'left'

    if hwhm_fwhm is not None and hwhm_fwhm > standard_fwhm:
        return hwhm_fwhm, f'hwhm_{hwhm_side}'

    return standard_fwhm, 'standard'


def analyze_spd(wavelengths, intensities, panel_type='', panel_sub_type=''):
    """
    Analyze an extracted SPD curve. Detects peaks, measures FWHM,
    and classifies the display's color architecture.
    panel_type: 'OLED' or 'LCD' from RTINGS data.
    panel_sub_type: 'QD-OLED', 'WOLED', 'VA', 'IPS' from RTINGS data.
    """
    samples_per_nm = len(wavelengths) / (WL_MAX - WL_MIN)

    # Find peaks
    peaks_idx, properties = find_peaks(
        intensities,
        height=0.10,
        prominence=0.05,
        distance=int(12 * samples_per_nm),  # min 12nm between peaks
        width=2
    )

    if len(peaks_idx) == 0:
        return {
            'peaks': [], 'blue_peak': None, 'green_peak': None, 'red_peak': None,
            'classification': 'UNKNOWN', 'confidence': 'low',
            'notes': 'No peaks detected'
        }

    # Calculate FWHM with overlap-aware measurement + asymmetry
    peak_data = []
    for idx in peaks_idx:
        wl = wavelengths[idx]
        fwhm_nm, method = _measure_fwhm_robust(
            wavelengths, intensities, idx, peaks_idx, samples_per_nm
        )
        # Measure left/right HWHM for asymmetry detection
        # Organic OLED emitters have vibronic sidebands → long red tails (ratio >> 1)
        # QD emitters are symmetric Gaussians → ratio ≈ 1.0
        half_max = intensities[idx] / 2.0
        left_hwhm = None
        for i in range(idx - 1, -1, -1):
            if intensities[i] <= half_max:
                frac = (intensities[i + 1] - half_max) / (intensities[i + 1] - intensities[i])
                left_hwhm = (idx - (i + 1 - frac)) / samples_per_nm
                break
        right_hwhm = None
        for i in range(idx + 1, len(intensities)):
            if intensities[i] <= half_max:
                frac = (intensities[i - 1] - half_max) / (intensities[i - 1] - intensities[i])
                right_hwhm = (i - 1 + frac - idx) / samples_per_nm
                break
        if left_hwhm and right_hwhm and left_hwhm > 0:
            asymmetry = right_hwhm / left_hwhm
        else:
            asymmetry = None

        peak_data.append({
            'wavelength': wl,
            'intensity': intensities[idx],
            'fwhm_nm': fwhm_nm,
            'fwhm_method': method,
            'left_hwhm': left_hwhm,
            'right_hwhm': right_hwhm,
            'asymmetry': asymmetry,
            'index': idx,
        })

    # Categorize by spectral band
    blue_peaks = [p for p in peak_data if 420 <= p['wavelength'] <= 490]
    green_peaks = [p for p in peak_data if 500 <= p['wavelength'] <= 575]
    red_peaks = [p for p in peak_data if 590 <= p['wavelength'] <= 680]

    blue = max(blue_peaks, key=lambda p: p['intensity']) if blue_peaks else None
    green = max(green_peaks, key=lambda p: p['intensity']) if green_peaks else None
    red = max(red_peaks, key=lambda p: p['intensity']) if red_peaks else None

    classification, confidence, notes = classify_spd(
        peak_data, blue, green, red, panel_type, panel_sub_type
    )

    return {
        'peaks': peak_data,
        'blue_peak': blue,
        'green_peak': green,
        'red_peak': red,
        'classification': classification,
        'confidence': confidence,
        'notes': notes,
    }


def classify_spd(all_peaks, blue, green, red, panel_type='', panel_sub_type=''):
    """
    Classify color architecture from SPD + RTINGS panel metadata.

    Uses panel_sub_type (QD-OLED/WOLED/VA/IPS) as ground truth for OLEDs.
    For LCDs, uses SPD peak analysis to determine KSF/QD-LCD/Pseudo QD/WLED.
    """
    notes = []
    green_fwhm = green['fwhm_nm'] if green else None
    red_fwhm = red['fwhm_nm'] if red else None
    blue_fwhm = blue['fwhm_nm'] if blue else None
    blue_wl = blue['wavelength'] if blue else None
    green_wl = green['wavelength'] if green else None
    red_wl = red['wavelength'] if red else None

    notes.append(f"B:{blue_wl:.0f}nm/{blue_fwhm:.0f}nm" if blue else "No blue")
    notes.append(f"G:{green_wl:.0f}nm/{green_fwhm:.0f}nm" if green else "No green")
    notes.append(f"R:{red_wl:.0f}nm/{red_fwhm:.0f}nm" if red else "No red")

    # Note any HWHM-mirrored measurements for transparency
    for label, peak in [('G', green), ('R', red)]:
        if peak and peak.get('fwhm_method', 'standard') != 'standard':
            notes.append(f"{label}:fwhm_via_{peak['fwhm_method']}")

    sub = panel_sub_type.upper().strip()
    is_oled = panel_type.upper() == 'OLED'

    # =========================================================
    # OLED PATH: Use panel_sub_type directly (most reliable)
    # =========================================================
    if is_oled:
        if sub == 'QD-OLED':
            return 'QD-OLED', 'high', '; '.join(notes) + '; panel_sub_type=QD-OLED'
        if sub == 'WOLED':
            return 'WOLED', 'high', '; '.join(notes) + '; panel_sub_type=WOLED'

        # Fallback SPD-based OLED classification
        if (blue and green and red
                and green_fwhm < NARROW_FWHM and red_fwhm < NARROW_FWHM):
            # Narrow peaks — could be QD-OLED or RGB Tandem WOLED.
            # Discriminate via red peak asymmetry and position:
            #   QD-OLED:       red ~638nm, symmetric (asymmetry ~1.0)
            #   Tandem WOLED:  red ~626nm, asymmetric (long red tail, asymmetry >1.25)
            red_asym = red.get('asymmetry')
            if red_asym is not None and red_asym > 1.25 and red_wl < 632:
                notes.append(f'red_asymmetry={red_asym:.2f}')
                return 'WOLED', 'medium', '; '.join(notes) + '; narrow but asymmetric red → RGB Tandem WOLED'
            notes.append(f'red_asymmetry={red_asym:.2f}' if red_asym else 'red_asymmetry=N/A')
            return 'QD-OLED', 'medium', '; '.join(notes) + '; narrow symmetric peaks suggest QD-OLED'
        if green and green_fwhm >= BROAD_FWHM:
            return 'WOLED', 'medium', '; '.join(notes) + '; broad green suggests WOLED'
        return 'WOLED', 'low', '; '.join(notes) + '; OLED fallback'

    # =========================================================
    # LCD PATH: QD-LCD, KSF, Pseudo QD, WLED
    # =========================================================

    # --- KSF ---
    # Key signature: VERY narrow red peak (FWHM < 15nm) from KSF phosphor emission.
    # Green is broad (standard phosphor, >35nm FWHM).
    # The narrow red is the most reliable KSF discriminator.
    if red and red_fwhm < 15:
        if green and green_fwhm >= 35:
            return 'KSF', 'high', '; '.join(notes)
        # Narrow red + narrower green = might be KSF with unusual green
        return 'KSF', 'medium', '; '.join(notes) + '; narrow red suggests KSF'

    # --- QD-LCD ---
    # Narrow green AND narrow red (both < 40nm).
    # CdSe QDs: green 22-25nm, red 19-27nm (very narrow)
    # InP QDs: green 33-36nm, red 35-37nm (wider but still < 40nm)
    if (green and red
            and green_fwhm < NARROW_FWHM
            and red_fwhm < NARROW_FWHM):
        return 'QD-LCD', 'high', '; '.join(notes)

    # QD-LCD asymmetric: one clearly narrow peak + the other borderline.
    # Catches InP QD sets where red can be wider (e.g., LG QNED90T: G:34.7, R:41.2).
    if (green and red
            and ((green_fwhm < 36 and red_fwhm < 45)
                 or (red_fwhm < 36 and green_fwhm < 45))):
        return 'QD-LCD', 'medium', '; '.join(notes) + '; asymmetric QD peaks'

    # --- Pseudo QD ---
    # Moderate green and/or red peaks (not narrow enough for true QD, not broad for WLED).
    # Includes Samsung Q60D/Q70D/Frame 2024, and sets marketed as QLED with InP material
    # that doesn't produce narrow QD emission.
    if (green and red
            and green_fwhm < BROAD_FWHM
            and red_fwhm < BROAD_FWHM):
        return 'Pseudo QD', 'medium', '; '.join(notes)

    # Also Pseudo QD if green is moderate (no distinct narrow red)
    if green and green_fwhm < BROAD_FWHM:
        return 'Pseudo QD', 'low', '; '.join(notes) + '; moderate green, weak red'

    # --- WLED ---
    # Very broad green phosphor (>55nm), often no distinct red peak.
    if green and green_fwhm >= BROAD_FWHM:
        return 'WLED', 'high', '; '.join(notes)

    # WLED if only 1-2 peaks (blue + one broad phosphor hump)
    if len(all_peaks) <= 2 and blue:
        return 'WLED', 'medium', '; '.join(notes) + '; few peaks'

    # Fallback
    return 'UNKNOWN', 'low', '; '.join(notes)


# ============================================================================
# VISUALIZATION
# ============================================================================
def wavelength_to_rgb(wl):
    """Convert wavelength (nm) to approximate RGB for visualization."""
    if wl < 380 or wl > 780:
        return (0.5, 0.5, 0.5)
    if wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    return (r, g, b)


def plot_verification(wavelengths, intensities, analysis, name, label, output_path):
    """Generate verification plot showing extracted curve + peak analysis."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(wavelengths, intensities, 'k-', linewidth=1.2, label='Extracted SPD')
    for i in range(len(wavelengths) - 1):
        color = wavelength_to_rgb(wavelengths[i])
        ax.fill_between(wavelengths[i:i+2], intensities[i:i+2], alpha=0.3, color=color)

    # Annotate peaks
    for p in analysis['peaks']:
        color = 'red' if p['fwhm_nm'] < NARROW_FWHM else 'orange' if p['fwhm_nm'] < BROAD_FWHM else 'gray'
        ax.axvline(x=p['wavelength'], color=color, linestyle='--', alpha=0.4)
        ax.annotate(
            f"{p['wavelength']:.0f}nm\n{p['fwhm_nm']:.0f}nm FWHM",
            xy=(p['wavelength'], p['intensity']),
            xytext=(0, 20), textcoords='offset points', ha='center', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )

    cls = analysis['classification']
    conf = analysis['confidence']
    match = "MATCH" if label and cls == label else "MISMATCH" if label else ""
    match_color = 'green' if match == "MATCH" else 'red' if match == "MISMATCH" else 'black'

    title = f"{name}\nAuto: {cls} ({conf})"
    if label:
        title += f"  |  Ground truth: {label}  [{match}]"

    ax.set_title(title, fontsize=13, color=match_color if match == "MISMATCH" else 'black')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative Intensity')
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Notes in bottom-left
    ax.text(0.02, 0.02, analysis['notes'], transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================================
# GROUND TRUTH LOADING
# ============================================================================
def load_ground_truth():
    """Load the 55 hand-classified TVs from the Excel ground truth file."""
    xlsx_path = Path("RTINGS Scoring Analysis 1.11 vs 2.0v2.xlsx")
    if not xlsx_path.exists():
        return {}

    gt = pd.read_excel(xlsx_path, sheet_name="RTINGS 2.0 Analysis",
                        header=None, skiprows=18, nrows=55)
    gt = gt.iloc[:, :6]
    gt.columns = ["brand", "model", "price_65", "tech", "qd_type", "release_year"]
    gt = gt.dropna(subset=["brand"])

    # Map QDEF -> QD-LCD (schema v1.0)
    gt["tech"] = gt["tech"].replace({"QDEF": "QD-LCD"})

    # Build lookup by (brand, model_keyword) for fuzzy matching
    lookup = {}
    for _, row in gt.iterrows():
        brand = str(row["brand"]).strip()
        model = str(row["model"]).strip()
        tech = str(row["tech"]).strip()
        qd_type = str(row["qd_type"]).strip() if pd.notna(row["qd_type"]) else ""
        # Store by multiple keys for matching
        lookup[f"{brand}|{model}"] = {"tech": tech, "qd_type": qd_type}

    return lookup


def match_ground_truth(fullname, brand, lookup):
    """Try to match a scraped TV to ground truth using bidirectional token matching.

    Uses Jaccard-like scoring: all GT tokens must appear in the fullname,
    and extra tokens in the fullname penalize the score. This prevents
    e.g. GT "BRAVIA 8" (WOLED) from matching "BRAVIA 8 II OLED" (QD-OLED).
    """
    fn_lower = fullname.lower()
    best_match = None
    best_score = 0

    # Marketing/type suffixes to strip from both sides
    strip_words = ["qled", "oled", "series", "mini-led", "mini", "led"]

    def _is_covered(fn_tok, gt_clean_str, gt_tok_list):
        """Check if an fn_token is 'covered' by the GT (substring either direction)."""
        if fn_tok in gt_clean_str:
            return True
        for gt_t in gt_tok_list:
            if gt_t in fn_tok or fn_tok in gt_t:
                return True
        return False

    for key, val in lookup.items():
        gt_brand, gt_model = key.split("|", 1)
        if gt_brand.lower() != brand.lower():
            continue

        gt_lower = gt_model.lower()
        gt_clean = gt_lower
        fn_clean = fn_lower.replace(brand.lower(), "", 1)  # remove brand from fullname
        for w in strip_words:
            gt_clean = gt_clean.replace(w, "")
            fn_clean = fn_clean.replace(w, "")

        # Split on whitespace and slashes for tokenization
        gt_tokens = [t for t in gt_clean.replace("/", " ").split() if t]
        fn_tokens = [t for t in fn_clean.replace("/", " ").split() if t]

        if not gt_tokens:
            continue

        # All GT tokens must appear in fullname (substring match)
        matched_gt = sum(1 for t in gt_tokens if t in fn_clean)
        if matched_gt < len(gt_tokens):
            continue

        # Bidirectional: count fn_tokens NOT covered by GT
        extra_fn = sum(1 for t in fn_tokens if not _is_covered(t, gt_clean, gt_tokens))

        # Jaccard-like: matched / (matched + extras)
        score = matched_gt / (matched_gt + extra_fn) if (matched_gt + extra_fn) > 0 else 0

        # Require minimum similarity score to accept match.
        # This prevents "BRAVIA 8" from matching "BRAVIA 8 II" (score 0.67).
        if score < 0.7:
            continue

        if score > best_score:
            best_score = score
            best_match = val

    return best_match


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
IN_CI = bool(os.environ.get("GITHUB_ACTIONS"))


def analyze_all_tvs():
    """Run SPD analysis on all scraped TVs."""
    # Load scraped data
    csv_path = DATA_DIR / "rtings_tv_data.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run rtings_scraper.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} TVs from scraped data")

    # Load ground truth
    gt_lookup = load_ground_truth()
    print(f"Ground truth: {len(gt_lookup)} hand-classified TVs")

    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    match_count = 0
    mismatch_count = 0
    total_with_gt = 0

    for idx, row in df.iterrows():
        fullname = row['fullname']
        brand = row['brand']
        spd_path = row.get('spd_image_local', '')
        panel_type = row.get('panel_type', '')

        if not spd_path or not os.path.exists(spd_path):
            results.append(_empty_result(row, 'NO_SPD_IMAGE'))
            continue

        print(f"  [{idx+1}/{len(df)}] {fullname}...", end='')

        try:
            wavelengths, intensities = extract_spd_curve(spd_path)
            panel_sub = row.get('panel_sub_type', '')
            analysis = analyze_spd(wavelengths, intensities,
                                   panel_type=panel_type,
                                   panel_sub_type=str(panel_sub) if pd.notna(panel_sub) else '')

            # Match to ground truth
            gt = match_ground_truth(fullname, brand, gt_lookup)
            gt_tech = gt['tech'] if gt else None
            gt_qd = gt['qd_type'] if gt else None

            # Check match
            if gt_tech:
                total_with_gt += 1
                if analysis['classification'] == gt_tech:
                    match_status = 'MATCH'
                    match_count += 1
                else:
                    match_status = 'MISMATCH'
                    mismatch_count += 1
            else:
                match_status = 'NO_GT'

            print(f" -> {analysis['classification']} ({analysis['confidence']})"
                  f"{' [' + match_status + ']' if gt_tech else ''}")

            # Generate verification plot (skip on CI to save time)
            if not IN_CI:
                safe_name = row['url_part'].replace('/', '-')
                plot_path = CURVES_DIR / f"{safe_name}_spd_analysis.png"
                plot_verification(wavelengths, intensities, analysis, fullname,
                                gt_tech, str(plot_path))

            results.append({
                'product_id': row['product_id'],
                'fullname': fullname,
                'brand': brand,
                'url_part': row['url_part'],
                'panel_type': panel_type,
                'spd_classification': analysis['classification'],
                'spd_confidence': analysis['confidence'],
                'ground_truth_tech': gt_tech or '',
                'ground_truth_qd_type': gt_qd or '',
                'match_status': match_status,
                'blue_peak_nm': f"{analysis['blue_peak']['wavelength']:.1f}" if analysis['blue_peak'] else '',
                'blue_fwhm_nm': f"{analysis['blue_peak']['fwhm_nm']:.1f}" if analysis['blue_peak'] else '',
                'green_peak_nm': f"{analysis['green_peak']['wavelength']:.1f}" if analysis['green_peak'] else '',
                'green_fwhm_nm': f"{analysis['green_peak']['fwhm_nm']:.1f}" if analysis['green_peak'] else '',
                'red_peak_nm': f"{analysis['red_peak']['wavelength']:.1f}" if analysis['red_peak'] else '',
                'red_fwhm_nm': f"{analysis['red_peak']['fwhm_nm']:.1f}" if analysis['red_peak'] else '',
                'num_peaks': len(analysis['peaks']),
                'notes': analysis['notes'],
                'spd_image_local': spd_path,
            })

        except Exception as e:
            print(f" ERROR: {e}")
            results.append(_empty_result(row, f'ERROR: {e}'))

    # Save results
    results_df = pd.DataFrame(results)
    csv_out = DATA_DIR / "spd_analysis_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"\nResults saved: {csv_out}")

    # Print summary
    print_summary(results_df, match_count, mismatch_count, total_with_gt)

    return results_df


def _empty_result(row, reason):
    return {
        'product_id': row['product_id'],
        'fullname': row['fullname'],
        'brand': row['brand'],
        'url_part': row['url_part'],
        'panel_type': row.get('panel_type', ''),
        'spd_classification': reason,
        'spd_confidence': '',
        'ground_truth_tech': '',
        'ground_truth_qd_type': '',
        'match_status': '',
        'blue_peak_nm': '', 'blue_fwhm_nm': '',
        'green_peak_nm': '', 'green_fwhm_nm': '',
        'red_peak_nm': '', 'red_fwhm_nm': '',
        'num_peaks': 0,
        'notes': reason,
        'spd_image_local': '',
    }


def print_summary(df, matches, mismatches, total_gt):
    """Print classification summary."""
    print("\n" + "=" * 70)
    print("SPD ANALYSIS SUMMARY")
    print("=" * 70)

    analyzed = df[~df['spd_classification'].isin(['NO_SPD_IMAGE', '']) &
                  ~df['spd_classification'].str.startswith('ERROR')]

    print(f"\n  Total TVs: {len(df)}")
    print(f"  Analyzed: {len(analyzed)}")

    if total_gt > 0:
        accuracy = matches / total_gt * 100
        print(f"\n  Ground truth validation:")
        print(f"    Matched: {matches}/{total_gt} ({accuracy:.0f}%)")
        print(f"    Mismatched: {mismatches}/{total_gt}")

    print(f"\n  Classification breakdown:")
    for cls, count in analyzed['spd_classification'].value_counts().items():
        print(f"    {cls}: {count}")

    # Show mismatches
    mm = df[df['match_status'] == 'MISMATCH']
    if len(mm) > 0:
        print(f"\n  MISMATCHES ({len(mm)}):")
        for _, row in mm.iterrows():
            print(f"    {row['fullname']:40s} auto={row['spd_classification']:12s} "
                  f"gt={row['ground_truth_tech']:12s} "
                  f"B:{row['blue_fwhm_nm']:>5s} G:{row['green_fwhm_nm']:>5s} R:{row['red_fwhm_nm']:>5s}")

    print(f"\n  Verification plots: {CURVES_DIR}/")


# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("RTINGS SPD Analyzer v2 — Calibrated")
    print("=" * 70)
    analyze_all_tvs()
