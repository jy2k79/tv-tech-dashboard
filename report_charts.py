#!/usr/bin/env python3
"""
Report Chart Generators
========================
Matplotlib chart functions for the monthly intelligence report.
Dark-themed static PNGs matching the dashboard aesthetic.

Each function takes a DataFrame + output_path and returns the Path
on success, or None if insufficient data.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Styling constants (match dashboard.py TECH_COLORS)
# ---------------------------------------------------------------------------
TECH_COLORS = {
    "QD-OLED": "#FF009F",
    "WOLED": "#4B40EB",
    "QD-LCD": "#FFC700",
    "Pseudo QD": "#FF7E43",
    "KSF": "#90BFFF",
    "WLED": "#A8BDD0",
}

TECH_ORDER = ["WLED", "KSF", "Pseudo QD", "QD-LCD", "WOLED", "QD-OLED"]

DARK_BG = '#1a1a2e'
CHART_BG = '#12122a'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#333355'
ACCENT_DIM = 'rgba(255,255,255,0.3)'

FIG_SIZE = (8, 5)
DPI = 150


def _setup_style():
    """Configure matplotlib for dark-themed report charts."""
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': CHART_BG,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.grid': True,
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.3,
        'axes.edgecolor': GRID_COLOR,
        'axes.linewidth': 0.5,
    })


def _tech_color(tech):
    """Get color for a technology, with fallback."""
    return TECH_COLORS.get(tech, '#888888')


def _filter_tech_order(techs_present):
    """Return TECH_ORDER filtered to only technologies present in data."""
    return [t for t in TECH_ORDER if t in techs_present]


def _save(fig, output_path):
    """Save figure and close."""
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return Path(output_path)


# ---------------------------------------------------------------------------
# Chart 1: Technology Distribution (donut)
# ---------------------------------------------------------------------------
def chart_tech_distribution(db, output_path):
    """Donut chart of TV count by technology. Used on cover page."""
    _setup_style()

    counts = db['color_architecture'].value_counts()
    ordered = _filter_tech_order(set(counts.index))
    if not ordered:
        return None

    values = [counts.get(t, 0) for t in ordered]
    colors = [_tech_color(t) for t in ordered]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=DARK_BG)
    wedges, texts, autotexts = ax.pie(
        values, labels=ordered, colors=colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.4, edgecolor=DARK_BG, linewidth=2),
    )
    for t in texts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(11)
    for t in autotexts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(10)
        t.set_fontweight('bold')

    ax.set_title(f'{sum(values)} TVs Tracked', color=TEXT_COLOR,
                 fontsize=14, fontweight='bold', pad=20)
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 2: Score Distribution (box plot)
# ---------------------------------------------------------------------------
def chart_score_distribution(db, output_path):
    """Box plot of mixed_usage scores by technology."""
    _setup_style()

    valid = db.dropna(subset=['mixed_usage', 'color_architecture'])
    if len(valid) < 5:
        return None

    ordered = _filter_tech_order(set(valid['color_architecture'].dropna()))
    if not ordered:
        return None

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)

    data_groups = [valid[valid['color_architecture'] == t]['mixed_usage'].values
                   for t in ordered]
    colors = [_tech_color(t) for t in ordered]

    bp = ax.boxplot(data_groups, tick_labels=ordered, patch_artist=True,
                    widths=0.6, showfliers=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(marker='o', markerfacecolor='#666',
                                    markeredgecolor='none', markersize=4))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(TEXT_COLOR)

    ax.set_ylabel('Mixed Usage Score')
    ax.set_ylim(0, 10.5)
    ax.set_title('Score Distribution by Technology', color=TEXT_COLOR,
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 3: FWHM by Technology (grouped bar)
# ---------------------------------------------------------------------------
def chart_fwhm_by_tech(db, output_path):
    """Grouped bar chart of green and red FWHM by technology."""
    _setup_style()

    valid = db.dropna(subset=['green_fwhm_nm', 'red_fwhm_nm', 'color_architecture'])
    if len(valid) < 3:
        return None

    ordered = _filter_tech_order(set(valid['color_architecture'].dropna()))
    if not ordered:
        return None

    green_means = [valid[valid['color_architecture'] == t]['green_fwhm_nm'].mean()
                   for t in ordered]
    red_means = [valid[valid['color_architecture'] == t]['red_fwhm_nm'].mean()
                 for t in ordered]

    x = np.arange(len(ordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)
    bars_g = ax.bar(x - width/2, green_means, width, label='Green FWHM',
                    color='#4ade80', alpha=0.8, edgecolor='none')
    bars_r = ax.bar(x + width/2, red_means, width, label='Red FWHM',
                    color='#f87171', alpha=0.8, edgecolor='none')

    # Value labels
    for bar in bars_g:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.0f}', ha='center', va='bottom',
                    color='#4ade80', fontsize=9, fontweight='bold')
    for bar in bars_r:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.0f}', ha='center', va='bottom',
                    color='#f87171', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=15)
    ax.set_ylabel('FWHM (nm)')
    ax.set_title('Peak Width by Technology (narrower = purer color)',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 4: New Model Scorecard (horizontal bar)
# ---------------------------------------------------------------------------
def chart_new_model_scorecard(new_tvs, db, output_path):
    """Horizontal bar chart of new models this month with their scores.

    Args:
        new_tvs: list of dicts with 'fullname', 'color_architecture', 'mixed_usage'
                 (or DataFrame with those columns)
        db: full database (unused, available for enrichment)
        output_path: where to save PNG

    Returns None if no new TVs.
    """
    _setup_style()

    if isinstance(new_tvs, pd.DataFrame):
        if len(new_tvs) == 0:
            return None
        items = new_tvs.to_dict('records')
    elif isinstance(new_tvs, list):
        if not new_tvs:
            return None
        items = new_tvs
    else:
        return None

    # Filter to those with scores
    items = [i for i in items if pd.notna(i.get('mixed_usage'))]
    if not items:
        return None

    # Sort by score descending, limit to 15
    items = sorted(items, key=lambda x: x.get('mixed_usage', 0), reverse=True)[:15]

    names = [i.get('fullname', '?')[:35] for i in items]
    scores = [float(i.get('mixed_usage', 0)) for i in items]
    colors = [_tech_color(i.get('color_architecture', '')) for i in items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(items) * 0.5)),
                           facecolor=DARK_BG)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.85, edgecolor='none')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', color=TEXT_COLOR, fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel('Mixed Usage Score')
    ax.set_title('New Models This Month', color=TEXT_COLOR,
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 5: Average Price by Technology (bar)
# ---------------------------------------------------------------------------
def chart_price_by_tech(db, output_path):
    """Bar chart of average $/m² by technology with WLED premium annotations."""
    _setup_style()

    priced = db.dropna(subset=['price_per_m2', 'color_architecture'])
    if len(priced) < 3:
        return None

    ordered = _filter_tech_order(set(priced['color_architecture'].dropna()))
    if not ordered:
        return None

    averages = [priced[priced['color_architecture'] == t]['price_per_m2'].mean()
                for t in ordered]
    colors = [_tech_color(t) for t in ordered]

    # Compute premium over WLED baseline
    wled_baseline = None
    if 'WLED' in ordered:
        wled_baseline = averages[ordered.index('WLED')]

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)
    bars = ax.bar(ordered, averages, color=colors, alpha=0.85, edgecolor='none')

    for bar, val, tech in zip(bars, averages, ordered):
        label = f'${val:,.0f}'
        if wled_baseline and tech != 'WLED':
            pct = (val - wled_baseline) / wled_baseline * 100
            label += f'\n+{pct:.0f}%'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                label, ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=10, fontweight='bold')

    ax.set_ylabel('Average Price per m\u00b2')
    ax.set_title('Technology Cost per m\u00b2 (size-normalized)',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f'${x:,.0f}'))
    ax.tick_params(axis='x', rotation=15)

    # Add subtle WLED baseline reference line
    if wled_baseline:
        ax.axhline(y=wled_baseline, color=_tech_color('WLED'),
                    linestyle=':', alpha=0.4, linewidth=1)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 6: Price Trends (line)
# ---------------------------------------------------------------------------
def chart_price_trends(price_history, output_path):
    """Line chart of average $/m² over time by technology.
    Returns None if fewer than 2 weekly snapshots.
    """
    _setup_style()

    if price_history is None or len(price_history) == 0:
        return None

    n_snapshots = price_history['snapshot_date'].nunique()
    if n_snapshots < 2:
        return None

    # Screen area lookup (diagonal inches → m²) for $/m² calculation
    screen_area_m2 = {
        32: 0.22, 40: 0.34, 42: 0.38, 43: 0.40, 48: 0.50,
        50: 0.54, 55: 0.65, 58: 0.72, 60: 0.77, 65: 0.91,
        70: 1.06, 75: 1.21, 77: 1.28, 80: 1.38, 83: 1.49,
        85: 1.56, 86: 1.59, 98: 2.07, 100: 2.15,
    }

    hist = price_history.copy()
    hist['screen_area_m2'] = hist['size_inches'].map(screen_area_m2)
    hist['price_per_m2'] = hist['best_price'] / hist['screen_area_m2']
    hist = hist.dropna(subset=['price_per_m2'])

    if len(hist) == 0:
        return None

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)

    for tech in TECH_ORDER:
        tech_data = hist[hist['color_architecture'] == tech]
        if len(tech_data) == 0:
            continue
        weekly_avg = (tech_data.groupby('snapshot_date')['price_per_m2']
                      .mean().sort_index())
        ax.plot(weekly_avg.index, weekly_avg.values,
                color=_tech_color(tech), linewidth=2, marker='o',
                markersize=5, label=tech)

    ax.set_ylabel('Average $/m\u00b2')
    ax.set_title('Price Trends by Technology ($/m\u00b2)', color=TEXT_COLOR,
                 fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f'${x:,.0f}'))
    ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9)
    fig.autofmt_xdate(rotation=30)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 7: Price vs Performance with Value Frontier
# ---------------------------------------------------------------------------
def chart_price_performance(db, output_path):
    """Scatter of price vs mixed_usage with value frontier line."""
    _setup_style()

    valid = db.dropna(subset=['price_best', 'mixed_usage', 'color_architecture'])
    if len(valid) < 5:
        return None

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)

    for tech in TECH_ORDER:
        subset = valid[valid['color_architecture'] == tech]
        if len(subset) == 0:
            continue
        ax.scatter(subset['price_best'], subset['mixed_usage'],
                   c=_tech_color(tech), label=tech, s=50, alpha=0.7,
                   edgecolors='none', zorder=3)

    # Value frontier (same algorithm as dashboard.py lines 941-955)
    sorted_v = valid.sort_values('mixed_usage', ascending=False)
    frontier = []
    min_price = float('inf')
    for _, row in sorted_v.iterrows():
        if row['price_best'] <= min_price:
            frontier.append(row)
            min_price = row['price_best']
    if frontier:
        ffront = pd.DataFrame(frontier).sort_values('price_best')
        ax.plot(ffront['price_best'], ffront['mixed_usage'],
                color=(1, 1, 1, 0.4), linestyle='--', linewidth=2,
                marker='D', markersize=5, markerfacecolor='white',
                markeredgecolor='none', label='Value Frontier', zorder=4)

    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Mixed Usage Score')
    ax.set_title('Price vs Performance', color=TEXT_COLOR,
                 fontsize=13, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f'${x:,.0f}'))
    ax.set_ylim(0, 10.5)
    ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=9, loc='lower right')

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 8: Temporal Scores (grouped bar — two most recent model years)
# ---------------------------------------------------------------------------
def chart_temporal_scores(db, output_path):
    """Grouped bar of avg mixed_usage by tech for the two most recent model years.
    Returns None if fewer than 2 years each have >= 5 TVs.
    """
    _setup_style()

    if 'released_at' not in db.columns:
        return None

    db = db.copy()
    db['model_year'] = pd.to_datetime(db['released_at'], errors='coerce').dt.year
    year_counts = db['model_year'].dropna().value_counts()
    valid_years = sorted([int(y) for y, n in year_counts.items() if n >= 5])
    if len(valid_years) < 2:
        return None

    yr_prev, yr_curr = valid_years[-2], valid_years[-1]
    subset = db[db['model_year'].isin([yr_prev, yr_curr])].dropna(
        subset=['mixed_usage', 'color_architecture'])
    if len(subset) < 5:
        return None

    techs_present = set(subset['color_architecture'].dropna())
    ordered = _filter_tech_order(techs_present)
    if not ordered:
        return None

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)
    x = np.arange(len(ordered))
    width = 0.35

    for i, year in enumerate([yr_prev, yr_curr]):
        means = []
        for t in ordered:
            vals = subset[(subset['color_architecture'] == t)
                          & (subset['model_year'] == year)]['mixed_usage']
            means.append(float(vals.mean()) if len(vals) >= 2 else float('nan'))
        offset = -width / 2 + i * width
        alpha = 0.55 if i == 0 else 0.85
        bars = ax.bar(x + offset, means, width, label=str(int(year)),
                       color=[_tech_color(t) for t in ordered],
                       alpha=alpha, edgecolor='none')
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                        f'{val:.1f}', ha='center', va='bottom',
                        color=TEXT_COLOR, fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=15)
    ax.set_ylabel('Avg Mixed Usage Score')
    ax.set_ylim(0, 10.5)
    ax.set_title(f'Score Comparison: {yr_prev} vs {yr_curr} Models',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 9: Temporal Pricing (grouped bar — two most recent model years)
# ---------------------------------------------------------------------------
def chart_temporal_pricing(db, output_path):
    """Grouped bar of avg $/m² by tech for the two most recent model years.
    Returns None if insufficient priced data across 2+ years.
    """
    _setup_style()

    if 'released_at' not in db.columns:
        return None

    db = db.copy()
    db['model_year'] = pd.to_datetime(db['released_at'], errors='coerce').dt.year
    priced = db.dropna(subset=['price_per_m2', 'color_architecture', 'model_year'])
    year_counts = priced['model_year'].value_counts()
    valid_years = sorted([int(y) for y, n in year_counts.items() if n >= 5])
    if len(valid_years) < 2:
        return None

    yr_prev, yr_curr = valid_years[-2], valid_years[-1]
    subset = priced[priced['model_year'].isin([yr_prev, yr_curr])]

    techs_present = set(subset['color_architecture'].dropna())
    ordered = _filter_tech_order(techs_present)
    if not ordered:
        return None

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=DARK_BG)
    x = np.arange(len(ordered))
    width = 0.35

    for i, year in enumerate([yr_prev, yr_curr]):
        means = []
        for t in ordered:
            vals = subset[(subset['color_architecture'] == t)
                          & (subset['model_year'] == year)]['price_per_m2']
            means.append(float(vals.mean()) if len(vals) >= 2 else float('nan'))
        offset = -width / 2 + i * width
        alpha = 0.55 if i == 0 else 0.85
        bars = ax.bar(x + offset, means, width, label=str(int(year)),
                       color=[_tech_color(t) for t in ordered],
                       alpha=alpha, edgecolor='none')
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        f'${val:,.0f}', ha='center', va='bottom',
                        color=TEXT_COLOR, fontsize=9, fontweight='bold')

    # WLED baseline
    wled_overall = db.dropna(subset=['price_per_m2'])
    wled_vals = wled_overall[wled_overall['color_architecture'] == 'WLED']['price_per_m2']
    if len(wled_vals) > 0:
        baseline = float(wled_vals.mean())
        ax.axhline(y=baseline, color=_tech_color('WLED'),
                    linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=15)
    ax.set_ylabel('Avg Price per m\u00b2')
    ax.set_title(f'Pricing Comparison: {yr_prev} vs {yr_curr} Models',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: f'${x:,.0f}'))
    ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from pathlib import Path

    DATA = Path(__file__).parent / 'data'
    OUT = Path(__file__).parent / 'data' / 'reports' / 'charts_test'
    OUT.mkdir(parents=True, exist_ok=True)

    db = pd.read_csv(DATA / 'tv_database_with_prices.csv')
    for col in ['price_best', 'mixed_usage', 'green_fwhm_nm', 'red_fwhm_nm']:
        if col in db.columns:
            db[col] = pd.to_numeric(db[col], errors='coerce')
    tech_order = ["WLED", "KSF", "Pseudo QD", "QD-LCD", "WOLED", "QD-OLED"]
    db['color_architecture'] = pd.Categorical(
        db['color_architecture'], categories=tech_order, ordered=True)

    hist_path = DATA / 'price_history.csv'
    hist = pd.DataFrame()
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist['snapshot_date'] = pd.to_datetime(hist['snapshot_date'], errors='coerce')
        hist['best_price'] = pd.to_numeric(hist['best_price'], errors='coerce')

    # Parse released_at for temporal charts
    if 'released_at' in db.columns:
        db['released_at'] = pd.to_datetime(db['released_at'], errors='coerce')

    charts = [
        ('tech_dist', chart_tech_distribution(db, OUT / 'tech_dist.png')),
        ('scores', chart_score_distribution(db, OUT / 'scores.png')),
        ('fwhm', chart_fwhm_by_tech(db, OUT / 'fwhm.png')),
        ('price_tech', chart_price_by_tech(db, OUT / 'price_tech.png')),
        ('price_trend', chart_price_trends(hist, OUT / 'price_trend.png')),
        ('price_perf', chart_price_performance(db, OUT / 'price_perf.png')),
        ('temporal_scores', chart_temporal_scores(db, OUT / 'temporal_scores.png')),
        ('temporal_pricing', chart_temporal_pricing(db, OUT / 'temporal_pricing.png')),
    ]

    for name, path in charts:
        status = f'OK -> {path}' if path else 'SKIPPED (insufficient data)'
        print(f'  {name}: {status}')
