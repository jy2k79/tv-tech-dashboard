#!/usr/bin/env python3
"""
Monthly Display Technology Intelligence Report
================================================
Generates a PDF analyst report with three sections:
1. Display Technology Overview
2. Pricing Intelligence
3. Macro & Industry Context

Runs on the first Monday of each month after the weekly data pull.
Uses Claude API for narrative generation and matplotlib for charts.

Usage:
    python monthly_report.py          # Generate and send report
    python monthly_report.py --dry    # Show what would run
    python monthly_report.py --force  # Run even if not first Monday
"""

import json
import os
import shutil
import smtplib
import sys
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
REPORTS = DATA / "reports"

TODAY = datetime.now()
REPORT_MONTH = TODAY.strftime("%B %Y")
REPORT_TAG = TODAY.strftime("%Y_%m")

IN_CI = bool(os.environ.get("GITHUB_ACTIONS"))

TECH_ORDER = ["WLED", "KSF", "Pseudo QD", "QD-LCD", "WOLED", "QD-OLED"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level}: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tv_database():
    path = DATA / "tv_database_with_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"TV database not found: {path}")
    df = pd.read_csv(path)
    numeric_cols = [
        "price_best", "price_per_m2", "price_per_mixed_use",
        "mixed_usage", "home_theater", "gaming", "sports", "bright_room",
        "brightness_score", "contrast_ratio_score", "color_score",
        "black_level_score", "native_contrast_score",
        "green_fwhm_nm", "red_fwhm_nm", "blue_fwhm_nm",
        "hdr_bt2020_coverage_itp_pct", "sdr_dci_p3_coverage_pct",
        "hdr_peak_10pct_nits",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["color_architecture"] = pd.Categorical(
        df["color_architecture"], categories=TECH_ORDER, ordered=True)
    for col in ["released_at", "first_published_at", "last_updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def load_price_history():
    path = DATA / "price_history.csv"
    if not path.exists():
        return pd.DataFrame()
    hist = pd.read_csv(path)
    hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    hist["best_price"] = pd.to_numeric(hist["best_price"], errors="coerce")
    return hist


def load_changelog():
    path = DATA / "changelog.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "date", "product_id", "fullname", "change_type",
            "field", "old_value", "new_value"])
    return pd.read_csv(path)


def load_registry():
    path = DATA / "tv_registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "product_id", "fullname", "first_seen_date",
            "last_seen_date", "status"])
    return pd.read_csv(path)


def load_spd_data():
    path = DATA / "spd_analysis_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["green_fwhm_nm", "red_fwhm_nm", "blue_fwhm_nm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_tech_metrics(db, changelog, registry, spd):
    """Compute Section 1 metrics."""
    month_start = TODAY.replace(day=1)

    # New and removed TVs this month
    new_tvs = []
    removed_tvs = []

    if len(changelog) > 0:
        cl = changelog.copy()
        cl["date"] = pd.to_datetime(cl["date"], errors="coerce")
        this_month = cl[cl["date"] >= month_start]
        new_from_cl = this_month[this_month["change_type"] == "new_tv"]
        removed_from_cl = this_month[this_month["change_type"] == "removed_tv"]
        for _, row in new_from_cl.iterrows():
            match = db[db["product_id"].astype(str) == str(row["product_id"])]
            score = float(match["mixed_usage"].iloc[0]) if len(match) > 0 and pd.notna(match["mixed_usage"].iloc[0]) else None
            tech = row.get("new_value", "")
            new_tvs.append({
                "fullname": row["fullname"],
                "color_architecture": tech,
                "mixed_usage": score,
            })
        for _, row in removed_from_cl.iterrows():
            removed_tvs.append({"fullname": row["fullname"],
                                "color_architecture": row.get("old_value", "")})
    elif len(registry) > 0:
        # Fallback: use registry first_seen_date
        reg = registry.copy()
        reg["first_seen_date"] = pd.to_datetime(reg["first_seen_date"], errors="coerce")
        new_reg = reg[reg["first_seen_date"] >= month_start]
        for _, row in new_reg.iterrows():
            match = db[db["product_id"].astype(str) == str(row["product_id"])]
            if len(match) > 0:
                new_tvs.append({
                    "fullname": row["fullname"],
                    "color_architecture": str(match["color_architecture"].iloc[0]),
                    "mixed_usage": float(match["mixed_usage"].iloc[0]) if pd.notna(match["mixed_usage"].iloc[0]) else None,
                })

    # TV count by technology
    tv_count = db["color_architecture"].value_counts().to_dict()
    tv_count = {str(k): int(v) for k, v in tv_count.items()}

    # Score summary by technology
    score_cols = ["mixed_usage", "home_theater", "gaming"]
    score_summary = {}
    for tech in TECH_ORDER:
        subset = db[db["color_architecture"] == tech]
        if len(subset) == 0:
            continue
        summary = {}
        for col in score_cols:
            vals = subset[col].dropna()
            if len(vals) > 0:
                summary[col] = {"mean": round(float(vals.mean()), 1),
                                "min": round(float(vals.min()), 1),
                                "max": round(float(vals.max()), 1)}
        if summary:
            score_summary[tech] = summary

    # FWHM by technology
    fwhm_by_tech = {}
    for tech in TECH_ORDER:
        subset = db[db["color_architecture"] == tech]
        g = subset["green_fwhm_nm"].dropna()
        r = subset["red_fwhm_nm"].dropna()
        if len(g) > 0 or len(r) > 0:
            fwhm_by_tech[tech] = {
                "green_mean": round(float(g.mean()), 1) if len(g) > 0 else None,
                "red_mean": round(float(r.mean()), 1) if len(r) > 0 else None,
            }

    # Model year comparison
    model_year = {}
    if "released_at" in db.columns:
        db_year = db.copy()
        db_year["year"] = db_year["released_at"].dt.year
        for year in [2025, 2026]:
            yr_data = db_year[db_year["year"] == year]
            if len(yr_data) >= 3:
                model_year[str(year)] = {
                    "count": int(len(yr_data)),
                    "avg_mixed": round(float(yr_data["mixed_usage"].mean()), 1),
                    "avg_gaming": round(float(yr_data["gaming"].dropna().mean()), 1) if len(yr_data["gaming"].dropna()) > 0 else None,
                }

    return {
        "total_tvs": len(db),
        "new_tvs": new_tvs,
        "removed_tvs": removed_tvs,
        "tv_count_by_tech": tv_count,
        "score_summary": score_summary,
        "fwhm_by_tech": fwhm_by_tech,
        "model_year": model_year,
        "report_month": REPORT_MONTH,
    }


def compute_pricing_metrics(db, price_history):
    """Compute Section 2 metrics."""
    priced = db.dropna(subset=["price_best"])
    n_priced = len(priced)

    # Average $/m² by tech (size-normalized, most meaningful for QD suppliers)
    price_per_m2 = {}
    avg_price = {}
    for tech in TECH_ORDER:
        subset = priced[priced["color_architecture"] == tech]
        if len(subset) > 0:
            avg_price[tech] = round(float(subset["price_best"].mean()), 0)
            m2_vals = subset["price_per_m2"].dropna()
            if len(m2_vals) > 0:
                price_per_m2[tech] = round(float(m2_vals.mean()), 0)

    # QD premium over WLED baseline
    wled_m2 = price_per_m2.get("WLED")
    tech_premium = {}
    if wled_m2 and wled_m2 > 0:
        for tech, m2 in price_per_m2.items():
            tech_premium[tech] = {
                "per_m2": m2,
                "premium_per_m2": round(m2 - wled_m2, 0),
                "premium_pct": round((m2 - wled_m2) / wled_m2 * 100, 0),
            }

    # Top value TVs (lowest $/mixed_usage point)
    top_value = []
    val = priced.dropna(subset=["price_per_mixed_use"]).sort_values("price_per_mixed_use")
    for _, row in val.head(5).iterrows():
        top_value.append({
            "fullname": row["fullname"],
            "color_architecture": str(row["color_architecture"]),
            "price": round(float(row["price_best"]), 0),
            "mixed_usage": round(float(row["mixed_usage"]), 1),
            "dollar_per_point": round(float(row["price_per_mixed_use"]), 0),
        })

    # Price history depth
    n_snapshots = 0
    price_trends = None
    if len(price_history) > 0:
        n_snapshots = price_history["snapshot_date"].nunique()
        if n_snapshots >= 2:
            dates = sorted(price_history["snapshot_date"].dropna().unique())
            latest = dates[-1]
            previous = dates[-2]
            trends = {}
            for tech in TECH_ORDER:
                curr = price_history[
                    (price_history["snapshot_date"] == latest) &
                    (price_history["color_architecture"] == tech)
                ]["best_price"].mean()
                prev = price_history[
                    (price_history["snapshot_date"] == previous) &
                    (price_history["color_architecture"] == tech)
                ]["best_price"].mean()
                if pd.notna(curr) and pd.notna(prev) and prev > 0:
                    trends[tech] = {
                        "current": round(float(curr), 0),
                        "previous": round(float(prev), 0),
                        "pct_change": round(float((curr - prev) / prev * 100), 1),
                    }
            if trends:
                price_trends = trends

    trend_caveat = None
    if n_snapshots < 2:
        trend_caveat = "Baseline month -- price trends will appear in future reports as data accumulates."
    elif n_snapshots < 4:
        trend_caveat = f"Based on {n_snapshots} weeks of data. Trends will become more reliable over time."

    return {
        "n_priced": n_priced,
        "avg_price_by_tech": avg_price,
        "price_per_m2_by_tech": price_per_m2,
        "tech_premium_vs_wled": tech_premium,
        "top_value_tvs": top_value,
        "data_depth_weeks": n_snapshots,
        "price_trends": price_trends,
        "trend_caveat": trend_caveat,
        "report_month": REPORT_MONTH,
    }


def compute_temporal_metrics(db):
    """Compare the two most recent model years. Returns dict with per-tech
    score deltas and price deltas, or None if < 2 valid years."""
    if "released_at" not in db.columns:
        return None

    db_t = db.copy()
    db_t["model_year"] = db_t["released_at"].dt.year
    year_counts = db_t["model_year"].dropna().value_counts()
    valid_years = sorted([int(y) for y, n in year_counts.items() if n >= 5])
    if len(valid_years) < 2:
        return None

    yr_prev, yr_curr = valid_years[-2], valid_years[-1]
    changes = {}
    for tech in TECH_ORDER:
        prev = db_t[(db_t["color_architecture"] == tech)
                     & (db_t["model_year"] == yr_prev)]
        curr = db_t[(db_t["color_architecture"] == tech)
                     & (db_t["model_year"] == yr_curr)]
        if len(prev) < 2 or len(curr) < 2:
            continue
        entry = {
            "n_prev": int(len(prev)),
            "n_curr": int(len(curr)),
        }
        for col, label in [("mixed_usage", "score")]:
            p = prev[col].dropna()
            c = curr[col].dropna()
            if len(p) > 0 and len(c) > 0:
                entry[f"{label}_prev"] = round(float(p.mean()), 1)
                entry[f"{label}_curr"] = round(float(c.mean()), 1)
                entry[f"{label}_delta"] = round(float(c.mean() - p.mean()), 1)
        p_m2 = prev["price_per_m2"].dropna()
        c_m2 = curr["price_per_m2"].dropna()
        if len(p_m2) > 0 and len(c_m2) > 0:
            pm, cm = float(p_m2.mean()), float(c_m2.mean())
            entry["price_m2_prev"] = round(pm, 0)
            entry["price_m2_curr"] = round(cm, 0)
            entry["price_m2_delta_pct"] = round((cm - pm) / pm * 100, 1) if pm > 0 else None
        if entry:
            changes[tech] = entry

    if not changes:
        return None

    return {
        "year_prev": yr_prev,
        "year_curr": yr_curr,
        "changes": changes,
        "report_month": REPORT_MONTH,
    }


def _should_include_temporal(metrics):
    """Return True if temporal section warrants inclusion in the report."""
    if metrics is None:
        return False
    for tech, data in metrics["changes"].items():
        if abs(data.get("score_delta", 0)) >= 0.3:
            return True
        if data.get("price_m2_delta_pct") is not None and abs(data["price_m2_delta_pct"]) >= 10:
            return True
    return False


def prepare_section3_context(db):
    """Build context string for Claude API Section 3."""
    tech_counts = db["color_architecture"].value_counts().to_dict()
    brands = db["brand"].value_counts().head(10).to_dict() if "brand" in db.columns else {}

    return (
        f"Our TV database tracks {len(db)} televisions across 6 display technologies: "
        f"{', '.join(f'{k}: {v}' for k, v in sorted(tech_counts.items(), key=lambda x: -x[1]))}. "
        f"Top brands by coverage: {', '.join(f'{k} ({v})' for k, v in brands.items())}. "
        f"Technologies tracked: WLED (basic phosphor), KSF (narrow red phosphor), "
        f"Pseudo QD (KSF marketed as quantum), QD-LCD (true quantum dot LCD), "
        f"WOLED (white OLED with color filters), QD-OLED (quantum dot OLED). "
        f"Report month: {REPORT_MONTH}. "
        f"Data sources: RTINGS.com reviews, Amazon/Best Buy pricing."
    )


# ---------------------------------------------------------------------------
# Claude API Integration
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_SECTION_TOKENS = 1500

SYSTEM_PROMPT = """You are a display technology analyst writing a monthly intelligence report for a TV industry team. Your voice is factual but compelling -- think Wired magazine meets an equity research note. You cite specific numbers from the data provided. You avoid hedging language like "it's worth noting" or "interestingly." Instead, make direct, confident observations. Write in present tense when describing current state, past tense for changes.

Rules:
- Reference specific TV models, scores, and prices from the data
- Use 2-3 short paragraphs per subsection
- Bold key numbers and model names using **markdown**
- Each section should be 400-600 words
- Do NOT use bullet points -- write in flowing prose paragraphs
- End each section with a single forward-looking "so what" sentence"""


def call_claude(user_prompt, tools=None):
    """Call Claude API and return text response. Handles tool-use loop for web search."""
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_SECTION_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if tools:
        kwargs["tools"] = tools

    response = client.messages.create(**kwargs)
    log(f"Claude API: {response.usage.input_tokens} in / "
        f"{response.usage.output_tokens} out")

    # Extract text from response
    text_parts = [b.text for b in response.content if hasattr(b, 'text')]
    return "\n".join(text_parts) if text_parts else ""


def generate_section1(metrics, chart_names):
    """Generate Section 1: Display Technology Overview narrative."""
    prompt = f"""Write Section 1: "Display Technology Overview" for the {REPORT_MONTH} report.

DATA SUMMARY:
{json.dumps(metrics, indent=2, default=str)}

CHARTS INCLUDED IN THIS SECTION: {', '.join(chart_names)}

Cover these topics in order:
1. New & removed devices this month (if any; if none, note stability)
2. Performance snapshot: which technology leads on mixed_usage, gaming, home_theater scores and by how much
3. FWHM trends: what the spectral data reveals about color technology (green and red peak widths by technology)
4. Model year comparison: if 2026 models are present vs 2025, how do they compare

If any data is insufficient (e.g., no changelog yet, few 2026 models), acknowledge this briefly and focus on what IS available."""

    return call_claude(prompt)


def generate_section2(metrics, chart_names):
    """Generate Section 2: Pricing Intelligence narrative."""
    prompt = f"""Write Section 2: "Pricing Intelligence" for the {REPORT_MONTH} report.

DATA SUMMARY:
{json.dumps(metrics, indent=2, default=str)}

CHARTS INCLUDED IN THIS SECTION: {', '.join(chart_names)}

Cover these topics:
1. Current pricing landscape: average price by technology, cheapest and most expensive categories
2. Price trends (if multiple weeks of data exist; if not, establish the baseline)
3. Value analysis: which technologies deliver the best $/performance point, the top value TVs
4. Notable pricing gaps between technologies

Handle sparse data gracefully -- if only 1-2 weeks of price history exist, focus on the current snapshot and frame it as a baseline."""

    return call_claude(prompt)


def generate_section3(context):
    """Generate Section 3: Macro & Industry Context narrative."""
    prompt = f"""Write Section 3: "Macro & Industry Context" for the {REPORT_MONTH} report.

CURRENT DATABASE CONTEXT:
{context}

Write 3-4 paragraphs covering:
1. The current state of the TV display technology landscape based on our database
2. Key technology differentiation: what separates QD-OLED, WOLED, QD-LCD, and budget technologies in practice
3. Where the industry is heading: emerging technologies like tandem OLED, microLED, perovskite quantum dots
4. A "so what" editorial takeaway for a display technology team

Since you cannot search the web, focus on synthesizing insights from our data context and your knowledge of the display industry. Be specific about the technology trends our data reveals."""

    return call_claude(prompt)


def generate_temporal_section(metrics, chart_names):
    """Generate Section 4: Temporal Analysis narrative."""
    prompt = f"""Write Section 4: "Year-over-Year Technology Trends" for the {REPORT_MONTH} report.

DATA SUMMARY:
{json.dumps(metrics, indent=2, default=str)}

CHARTS INCLUDED IN THIS SECTION: {', '.join(chart_names)}

Cover these topics:
1. Which technologies improved most in mixed_usage score between {metrics['year_prev']} and {metrics['year_curr']} models
2. Biggest pricing shifts ($/m² changes) — who's getting cheaper, who's maintaining premium
3. Competitive positioning implications — is the gap between QD and non-QD narrowing or widening
4. A forward-looking "so what" about technology trajectories"""

    return call_claude(prompt)


def generate_section_safe(section_num, generator_func, *args):
    """Wrapper that catches API errors and returns fallback text."""
    try:
        return generator_func(*args)
    except Exception as e:
        log(f"Claude API error for Section {section_num}: {e}", "ERROR")
        return f"[Section {section_num} narrative generation failed: {e}]\n\nRaw metrics are included in the charts for this section."


# ---------------------------------------------------------------------------
# PDF Assembly
# ---------------------------------------------------------------------------

def build_pdf(sections, cover_chart, stats, output_path):
    """Assemble the complete PDF report.

    Args:
        sections: list of (title, narrative_text, [chart_paths])
        cover_chart: Path to tech distribution chart for cover
        stats: dict with total_tvs, n_priced, etc.
        output_path: where to save PDF
    """
    from fpdf import FPDF

    # Dark theme constants
    BG = (26, 26, 46)       # #1a1a2e
    TEXT = (224, 224, 224)   # #e0e0e0
    TEXT_DIM = (150, 150, 170)
    ACCENT_BLUE = (144, 191, 255)

    logo_white = ROOT / "logos" / "Nanosys Logo White Text 4X.png"

    class ReportPDF(FPDF):
        _in_cover = False

        def _dark_bg(self):
            """Fill current page with dark background."""
            self.set_fill_color(*BG)
            self.rect(0, 0, 210, 297, "F")

        def header(self):
            if self._in_cover or self.page_no() <= 1:
                return
            self._dark_bg()
            # Small logo top-left
            if logo_white.exists():
                self.image(str(logo_white), x=10, y=8, w=28)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*TEXT_DIM)
            self.set_y(10)
            self.cell(0, 8, "Display Technology Intelligence Report", align="C")
            self.cell(0, 8, REPORT_MONTH, align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(6)

        def footer(self):
            if self._in_cover:
                return
            self.set_y(-15)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*TEXT_DIM)
            self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    # --- Cover page ---
    pdf._in_cover = True
    pdf.add_page()
    pdf._dark_bg()

    # Logo
    if logo_white.exists():
        x = (210 - 70) / 2
        pdf.image(str(logo_white), x=x, y=20, w=70)

    # Title
    pdf.set_y(55)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 14, "Display Technology", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Month
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(*ACCENT_BLUE)
    pdf.cell(0, 10, REPORT_MONTH, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)

    # Stats line
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*TEXT_DIM)
    n_priced = stats.get("n_priced", 0)
    n_8k = stats.get("n_8k_excluded", 0)
    stat_line = f"{stats.get('total_tvs', 0)} 4K TVs tracked  |  {n_priced} with pricing  |  6 technologies"
    if n_8k > 0:
        stat_line += f"  |  {n_8k} 8K excluded"
    pdf.cell(0, 8, stat_line, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Cover chart
    if cover_chart and cover_chart.exists():
        x = (210 - 100) / 2
        pdf.image(str(cover_chart), x=x, w=100)
    pdf._in_cover = False

    # --- Section pages ---
    section_colors = [
        (74, 222, 128),   # Green for tech overview
        (255, 199, 0),    # Gold for pricing
        (144, 191, 255),  # Blue for macro
        (255, 126, 67),   # Orange for temporal
    ]

    for i, (title, narrative, chart_paths) in enumerate(sections):
        pdf.add_page()
        r, g, b = section_colors[i] if i < len(section_colors) else (200, 200, 200)

        # Section header bar (pushed below header area)
        pdf.set_fill_color(r, g, b)
        pdf.rect(10, 28, 190, 12, "F")
        pdf.set_xy(14, 28)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*BG)
        pdf.cell(0, 12, f"  {i + 1}. {title}")
        pdf.set_y(46)

        # Narrative text
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*TEXT)

        # Sanitize for Helvetica (latin-1 only) and strip markdown bold
        clean_text = narrative.replace("**", "")
        clean_text = (clean_text
                      .replace("\u2014", " -- ")   # em dash
                      .replace("\u2013", " - ")    # en dash
                      .replace("\u2018", "'")       # left single quote
                      .replace("\u2019", "'")       # right single quote
                      .replace("\u201c", '"')       # left double quote
                      .replace("\u201d", '"')       # right double quote
                      .replace("\u2026", "...")     # ellipsis
                      .replace("\u00b2", "2")       # superscript 2
                      .encode("latin-1", errors="replace").decode("latin-1"))
        for paragraph in clean_text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(4)

        # Embed charts
        valid_charts = [p for p in chart_paths if p and p.exists()]
        for chart_path in valid_charts:
            # Check if we need a new page
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.ln(4)
            pdf.image(str(chart_path), x=15, w=180)
            pdf.ln(6)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    log(f"PDF saved: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Email Delivery
# ---------------------------------------------------------------------------

def send_report_email(pdf_path, excerpt_html, report_date):
    """Send email with PDF attachment and HTML body excerpt."""
    sender = os.environ.get("GMAIL_ADDRESS")
    app_pw = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("NOTIFY_EMAIL")

    if not all([sender, app_pw, recipient]):
        log("Email secrets not configured -- skipping email", "WARN")
        return False

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Display Technology Intelligence Report -- {report_date}"
    msg["From"] = f"TV Dashboard <{sender}>"
    msg["To"] = recipient

    # HTML body
    msg.attach(MIMEText(excerpt_html, "html"))

    # PDF attachment
    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "pdf")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={pdf_path.name}")
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_pw)
            server.sendmail(sender, recipient, msg.as_string())
        log(f"Report emailed to {recipient}")
        return True
    except Exception as e:
        log(f"Email failed: {e}", "ERROR")
        return False


def build_email_excerpt(tech_metrics, pricing_metrics):
    """Build HTML excerpt for the email body."""
    new_count = len(tech_metrics.get("new_tvs", []))
    removed_count = len(tech_metrics.get("removed_tvs", []))
    total = tech_metrics.get("total_tvs", 0)
    n_priced = pricing_metrics.get("n_priced", 0)

    html = f"""
    <div style="font-family:Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;
                color:#e0e0e0;background:#1a1a2e;padding:24px;border-radius:8px">
      <h2 style="color:#fff;margin-top:0">Display Technology Intelligence Report</h2>
      <p style="color:#90bfff;font-size:14px">{REPORT_MONTH}</p>
      <hr style="border:1px solid #333">

      <table style="width:100%;font-size:15px;border-collapse:collapse">
        <tr><td style="padding:6px 0;color:#999">Total TVs</td>
            <td style="padding:6px 0;text-align:right;color:#fff"><b>{total}</b></td></tr>
        <tr><td style="padding:6px 0;color:#999">TVs with pricing</td>
            <td style="padding:6px 0;text-align:right;color:#fff">{n_priced}</td></tr>
        <tr><td style="padding:6px 0;color:#999">New this month</td>
            <td style="padding:6px 0;text-align:right;color:#4ade80">{new_count}</td></tr>
        <tr><td style="padding:6px 0;color:#999">Removed this month</td>
            <td style="padding:6px 0;text-align:right;color:#f87171">{removed_count}</td></tr>
      </table>

      <hr style="border:1px solid #333">
      <p style="color:#999;font-size:13px">Full report attached as PDF.</p>
      <p style="color:#666;font-size:12px;margin-bottom:0">
        Automated report from TV Display Technology Dashboard</p>
    </div>"""
    return html


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MONTHLY DISPLAY TECHNOLOGY INTELLIGENCE REPORT")
    print(f"Date: {TODAY.strftime('%Y-%m-%d')}  |  Month: {REPORT_MONTH}")
    print(f"Environment: {'GitHub Actions' if IN_CI else 'Local'}")
    print("=" * 70)

    # --- Check flags ---
    dry_run = "--dry" in sys.argv
    force = "--force" in sys.argv

    if dry_run:
        log("DRY RUN -- showing execution plan")
        log("  1. Load data (tv_database, price_history, changelog, registry, spd)")
        log("  2. Compute metrics (tech overview, pricing, macro context)")
        log("  3. Generate 7 matplotlib charts")
        log("  4. Call Claude API x3 (Sonnet) for section narratives")
        log("  5. Assemble PDF with fpdf2")
        log("  6. Email PDF attachment + HTML excerpt")
        return

    # First Monday check
    if not force and TODAY.day > 7:
        log(f"Not first Monday of month (day={TODAY.day}). Use --force to override.")
        return

    errors = []

    # --- Step 1: Load data ---
    log("Loading data...")
    try:
        db = load_tv_database()
        # Exclude 8K TVs — tiny market segment that skews both scores and pricing
        n_8k = 0
        if "resolution" in db.columns:
            n_8k = (db["resolution"] == "8k").sum()
            db = db[db["resolution"] != "8k"].reset_index(drop=True)
        log(f"  TV database: {len(db)} TVs ({n_8k} 8K excluded)")
    except Exception as e:
        log(f"FATAL: Cannot load TV database: {e}", "ERROR")
        sys.exit(1)

    try:
        price_history = load_price_history()
        log(f"  Price history: {len(price_history)} rows, "
            f"{price_history['snapshot_date'].nunique() if len(price_history) > 0 else 0} snapshots")
    except Exception:
        price_history = pd.DataFrame()
        errors.append("Price history unavailable")

    changelog = load_changelog()
    log(f"  Changelog: {len(changelog)} entries")

    registry = load_registry()
    log(f"  Registry: {len(registry)} TVs")

    spd = load_spd_data()
    log(f"  SPD data: {len(spd)} results")

    # --- Step 2: Compute metrics ---
    log("Computing metrics...")
    tech_metrics = compute_tech_metrics(db, changelog, registry, spd)
    tech_metrics["n_8k_excluded"] = n_8k
    pricing_metrics = compute_pricing_metrics(db, price_history)
    section3_context = prepare_section3_context(db)
    temporal_metrics = compute_temporal_metrics(db)
    include_temporal = _should_include_temporal(temporal_metrics)

    log(f"  New TVs: {len(tech_metrics['new_tvs'])}")
    log(f"  Removed TVs: {len(tech_metrics['removed_tvs'])}")
    log(f"  Priced: {pricing_metrics['n_priced']}")
    log(f"  Price data depth: {pricing_metrics['data_depth_weeks']} weeks")
    log(f"  Temporal section: {'included' if include_temporal else 'skipped (thresholds not met)'}")

    # --- Step 3: Generate charts ---
    log("Generating charts...")
    REPORTS.mkdir(parents=True, exist_ok=True)
    chart_dir = REPORTS / "charts_tmp"
    chart_dir.mkdir(exist_ok=True)

    from report_charts import (
        chart_tech_distribution, chart_score_distribution, chart_fwhm_by_tech,
        chart_new_model_scorecard, chart_price_by_tech, chart_price_trends,
        chart_price_performance, chart_temporal_scores, chart_temporal_pricing,
    )

    charts = {}
    charts["tech_dist"] = chart_tech_distribution(db, chart_dir / "tech_dist.png")
    charts["scores"] = chart_score_distribution(db, chart_dir / "scores.png")
    charts["fwhm"] = chart_fwhm_by_tech(db, chart_dir / "fwhm.png")
    charts["new_models"] = chart_new_model_scorecard(
        tech_metrics["new_tvs"], db, chart_dir / "new_models.png")
    charts["price_tech"] = chart_price_by_tech(db, chart_dir / "price_tech.png")
    charts["price_trend"] = chart_price_trends(
        price_history, chart_dir / "price_trend.png")
    charts["price_perf"] = chart_price_performance(db, chart_dir / "price_perf.png")
    if include_temporal:
        charts["temporal_scores"] = chart_temporal_scores(
            db, chart_dir / "temporal_scores.png")
        charts["temporal_pricing"] = chart_temporal_pricing(
            db, chart_dir / "temporal_pricing.png")

    generated = {k: v for k, v in charts.items() if v is not None}
    skipped = {k for k, v in charts.items() if v is None}
    log(f"  Generated: {len(generated)} charts")
    if skipped:
        log(f"  Skipped (insufficient data): {skipped}")

    # --- Step 4: Generate narratives via Claude API ---
    log("Generating narratives via Claude API...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log("ANTHROPIC_API_KEY not set -- using placeholder text", "WARN")
        section1_text = "[Claude API key not configured. Section 1 narrative would appear here with analysis of display technology trends, new models, and performance metrics.]"
        section2_text = "[Claude API key not configured. Section 2 narrative would appear here with pricing intelligence, value analysis, and trend data.]"
        section3_text = "[Claude API key not configured. Section 3 narrative would appear here with industry context and macro analysis.]"
        section4_text = "[Claude API key not configured. Section 4 narrative would appear here with year-over-year technology trends.]" if include_temporal else None
        errors.append("No ANTHROPIC_API_KEY -- used placeholder narratives")
    else:
        s1_charts = [k for k in ["scores", "fwhm", "new_models", "tech_dist"]
                     if k in generated]
        s2_charts = [k for k in ["price_tech", "price_trend", "price_perf"]
                     if k in generated]

        section1_text = generate_section_safe(1, generate_section1, tech_metrics, s1_charts)
        section2_text = generate_section_safe(2, generate_section2, pricing_metrics, s2_charts)
        section3_text = generate_section_safe(3, generate_section3, section3_context)

        section4_text = None
        if include_temporal:
            s4_charts = [k for k in ["temporal_scores", "temporal_pricing"]
                         if k in generated]
            section4_text = generate_section_safe(
                4, generate_temporal_section, temporal_metrics, s4_charts)

    # --- Step 5: Assemble PDF ---
    log("Assembling PDF...")
    pdf_path = REPORTS / f"display_intelligence_{REPORT_TAG}.pdf"

    sections = [
        ("Display Technology Overview", section1_text,
         [charts.get("scores"), charts.get("fwhm"), charts.get("new_models")]),
        ("Pricing Intelligence", section2_text,
         [charts.get("price_tech"), charts.get("price_trend"),
          charts.get("price_perf")]),
        ("Macro & Industry Context", section3_text, []),
    ]
    if include_temporal and section4_text:
        sections.append((
            "Year-over-Year Technology Trends", section4_text,
            [charts.get("temporal_scores"), charts.get("temporal_pricing")],
        ))

    n_sections = len(sections)

    build_pdf(sections, charts.get("tech_dist"), {
        "total_tvs": tech_metrics["total_tvs"],
        "n_priced": pricing_metrics["n_priced"],
        "n_8k_excluded": n_8k,
    }, pdf_path)

    # --- Step 6: Email report ---
    log("Sending email...")
    excerpt = build_email_excerpt(tech_metrics, pricing_metrics)
    send_report_email(pdf_path, excerpt, REPORT_MONTH)

    # --- Step 7: Cleanup ---
    shutil.rmtree(chart_dir, ignore_errors=True)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"REPORT COMPLETE: {pdf_path}")
    print(f"  Sections: {n_sections}  |  Charts: {len(generated)}  |  Pages: ~{4 + len(generated)}")
    if errors:
        print(f"\nWarnings:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
