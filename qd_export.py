#!/usr/bin/env python3
"""
QD SKU Tracker Export
=====================
Builds a CSV of all SPD-confirmed quantum dot displays (TVs + Monitors)
and emails it to qdskutracker@gmail.com for ingestion by the QD SKU
Tracker pipeline.

Runs as a post-pipeline step in weekly_update.py.

Usage:
    python qd_export.py              # Build CSV + email
    python qd_export.py --csv-only   # Build CSV without emailing
"""

import os
import re
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "rtings_qd_verified.csv"

QD_RECIPIENT = "qdskutracker@gmail.com"
EMAIL_SUBJECT = "RTINGS SPD Verified QD Displays"

# QD color architectures to include
QD_ARCHITECTURES = {"QD-LCD", "QD-OLED", "Pseudo QD"}

# Backlight type mapping (RTINGS → export schema)
BACKLIGHT_MAP = {
    "Full-Array": "Direct",
    "Direct": "Direct",
    "Edge": "Edge",
    "No Backlight": "",
}

# Panel type mapping (panel_sub_type → panel_type for export)
PANEL_TYPE_MAP = {
    "QD-OLED": "OLED",
    "WOLED": "OLED",
    "IPS": "IPS",
    "VA": "VA",
}


def extract_model(fullname: str, brand: str) -> str:
    """Strip brand prefix from fullname to get model name."""
    if fullname.startswith(brand):
        model = fullname[len(brand):].strip()
        return model
    return fullname


def build_spd_summary(row: pd.Series) -> str:
    """Build a human-readable SPD classification summary."""
    arch = row.get("color_architecture", "")
    classification = row.get("spd_classification", "")
    confidence = row.get("spd_confidence", "")
    green_fwhm = row.get("green_fwhm_nm", "")
    red_fwhm = row.get("red_fwhm_nm", "")
    qd_mat = row.get("qd_material", "")

    if arch == "QD-OLED":
        return (f"Clear QD-OLED emission peaks "
                f"(G:{green_fwhm}nm R:{red_fwhm}nm FWHM, {qd_mat})")
    elif arch == "QD-LCD":
        if qd_mat == "CdSe":
            return (f"Standard QDEF with CdSe "
                    f"(G:{green_fwhm}nm R:{red_fwhm}nm FWHM, narrow)")
        else:
            return (f"Standard QDEF with {qd_mat} "
                    f"(G:{green_fwhm}nm R:{red_fwhm}nm FWHM)")
    elif arch == "Pseudo QD":
        return (f"Weak QD signal, likely pseudo-QD "
                f"(G:{green_fwhm}nm R:{red_fwhm}nm FWHM, KSF-dominant)")
    else:
        return classification


def build_qd_export() -> pd.DataFrame:
    """Build the QD verified export CSV from TV + Monitor databases."""
    frames = []

    # Load TV data
    tv_path = DATA_DIR / "tv_database_with_prices.csv"
    if tv_path.exists():
        tv = pd.read_csv(tv_path)
        tv["_product_category"] = "TV"
        frames.append(tv)

    # Load Monitor data
    mon_path = DATA_DIR / "monitor_database_with_prices.csv"
    if mon_path.exists():
        mon = pd.read_csv(mon_path)
        mon["_product_category"] = "Monitor"
        frames.append(mon)

    if not frames:
        print("ERROR: No database files found")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Filter to QD products only
    qd = combined[combined["color_architecture"].isin(QD_ARCHITECTURES)].copy()
    print(f"QD products: {len(qd)} ({dict(qd['color_architecture'].value_counts())})")

    if len(qd) == 0:
        return pd.DataFrame()

    # Build export columns
    export = pd.DataFrame()
    export["brand"] = qd["brand"]
    export["model"] = qd.apply(
        lambda r: extract_model(r["fullname"], r["brand"]), axis=1)
    export["product_category"] = qd["_product_category"]
    export["sizes_available"] = qd["sizes_available"]

    # panel_type: map panel_sub_type to simplified type
    export["panel_type"] = qd["panel_sub_type"].map(PANEL_TYPE_MAP).fillna(
        qd["panel_sub_type"])

    export["panel_sub_type"] = qd["panel_sub_type"]

    # backlight_type: map RTINGS values
    export["backlight_type"] = qd["backlight_type_rtings"].map(BACKLIGHT_MAP).fillna("")

    export["resolution"] = qd["resolution"]

    # native_refresh_rate: extract numeric Hz
    export["native_refresh_rate"] = pd.to_numeric(
        qd["native_refresh_rate"], errors="coerce")

    export["dimming_zone_count"] = pd.to_numeric(
        qd.get("dimming_zone_count", pd.Series(dtype=float)), errors="coerce")

    export["hdr_peak_10pct_nits"] = pd.to_numeric(
        qd["hdr_peak_10pct_nits"], errors="coerce")

    export["sdr_dci_p3_coverage_pct"] = pd.to_numeric(
        qd["sdr_dci_p3_coverage_pct"], errors="coerce")

    export["released_at"] = qd["released_at"]
    export["review_url"] = qd["review_url"]
    export["spd_image"] = qd["spd_image"]

    # QD classification flags
    export["qd_confirmed"] = qd["color_architecture"].isin({"QD-LCD", "QD-OLED"})
    export["qd_material"] = qd["qd_material"].fillna("unknown")
    export["is_pseudo_qd"] = qd["color_architecture"] == "Pseudo QD"

    # SPD classification summary
    export["spd_classification"] = qd.apply(build_spd_summary, axis=1)

    # Sort by product category then brand then model
    export = export.sort_values(
        ["product_category", "brand", "model"]).reset_index(drop=True)

    return export


def send_qd_email(csv_path: Path, export_df: pd.DataFrame) -> bool:
    """Send the QD export CSV via email."""
    sender = os.environ.get("GMAIL_ADDRESS")
    app_pw = os.environ.get("GMAIL_APP_PASSWORD")

    if not all([sender, app_pw]):
        print("  Email secrets not configured — skipping QD export email")
        return False

    # Build email
    msg = MIMEMultipart("mixed")
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = f"RTINGS Dashboard <{sender}>"
    msg["To"] = QD_RECIPIENT

    # Body: summary + CSV contents as fallback
    n_total = len(export_df)
    n_confirmed = export_df["qd_confirmed"].sum()
    n_pseudo = export_df["is_pseudo_qd"].sum()
    n_tv = (export_df["product_category"] == "TV").sum()
    n_mon = (export_df["product_category"] == "Monitor").sum()

    body = (
        f"RTINGS SPD Verified QD Displays\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Total QD products: {n_total}\n"
        f"  QD confirmed: {n_confirmed}\n"
        f"  Pseudo QD: {n_pseudo}\n"
        f"  TVs: {n_tv}\n"
        f"  Monitors: {n_mon}\n\n"
        f"--- CSV contents (fallback) ---\n\n"
        f"{export_df.to_csv(index=False)}"
    )
    msg.attach(MIMEText(body, "plain"))

    # Attach CSV file
    with open(csv_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={csv_path.name}",
        )
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_pw)
            server.sendmail(sender, QD_RECIPIENT, msg.as_string())
        print(f"  QD export emailed to {QD_RECIPIENT} ({n_total} products)")
        return True
    except Exception as e:
        print(f"  QD export email failed: {e}")
        return False


def main():
    print("=" * 70)
    print("QD SKU TRACKER EXPORT")
    print("=" * 70)

    csv_only = "--csv-only" in sys.argv

    # Build export
    export_df = build_qd_export()
    if len(export_df) == 0:
        print("No QD products found — nothing to export")
        return

    # Save CSV
    export_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} ({len(export_df)} products)")

    # Email
    if csv_only:
        print("--csv-only mode, skipping email")
    else:
        send_qd_email(OUTPUT_PATH, export_df)

    print("Done!")


if __name__ == "__main__":
    main()
