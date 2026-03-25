#!/usr/bin/env python3
"""
Weekly Data Update — Master Orchestrator
================================================
Runs all pipeline scripts in sequence, detects changes,
maintains a product registry and changelog, and sends notifications.

Supports multiple product silos (TV, Monitor) via silo_config.py.

Works both locally (macOS notification) and in GitHub Actions
(writes to $GITHUB_STEP_SUMMARY).

Usage:
    python weekly_update.py                 # TV only (default)
    python weekly_update.py --silo tv       # TV only (explicit)
    python weekly_update.py --silo monitor  # Monitor only
    python weekly_update.py --silo all      # All silos sequentially
    python weekly_update.py --dry           # Show what would run, don't execute
"""

import argparse
import os
import sys
import shutil
import subprocess
import platform
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

import pandas as pd

from silo_config import get_silo, SILOS, TV

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

IN_CI = bool(os.environ.get("GITHUB_ACTIONS"))
TODAY = datetime.now().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level}: {msg}", flush=True)


def run_script(name, abort_on_fail=True, extra_args=None):
    """Run a Python script in the project root. Returns True on success."""
    path = ROOT / name
    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd.extend(extra_args)
    log(f"Running {' '.join(cmd[1:])}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=False,
            timeout=1200,  # 20 minute timeout per script
        )
        if result.returncode != 0:
            log(f"{name} exited with code {result.returncode}", "ERROR")
            if abort_on_fail:
                raise RuntimeError(f"{name} failed (exit {result.returncode})")
            return False
        log(f"{name} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        log(f"{name} timed out after 20 minutes", "ERROR")
        if abort_on_fail:
            raise
        return False
    except Exception as e:
        log(f"{name} error: {e}", "ERROR")
        if abort_on_fail:
            raise
        return False


# ---------------------------------------------------------------------------
# Drop guard — prevent catastrophic data loss
# ---------------------------------------------------------------------------

def check_drop_guard(old_count: int, new_csv_path: Path):
    """Abort the pipeline if the scraper output looks catastrophically wrong.

    Raises RuntimeError if:
      - Scraper produced 0 TVs
      - TV count dropped by >10 TVs AND >20% (both must be exceeded)
    """
    if not new_csv_path.exists():
        raise RuntimeError(
            f"Drop guard FAILED: scraper output missing ({new_csv_path})"
        )

    new_df = pd.read_csv(new_csv_path)
    new_count = len(new_df)

    if new_count == 0:
        raise RuntimeError(
            "Drop guard FAILED: scraper returned 0 TVs — aborting to prevent data loss"
        )

    if old_count > 0:
        drop = old_count - new_count
        drop_pct = drop / old_count * 100
        if drop > 10 and drop_pct > 20:
            raise RuntimeError(
                f"Drop guard FAILED: TV count dropped from {old_count} to {new_count} "
                f"({drop} TVs, {drop_pct:.0f}%) — aborting to prevent data loss. "
                f"If this is expected, delete the old CSV and re-run."
            )

    log(f"Drop guard passed: {old_count} → {new_count} TVs")


# ---------------------------------------------------------------------------
# Stale score fallback — recover data when session cookie expires
# ---------------------------------------------------------------------------

def recover_blurred_scores(old_priced_path: Path, new_priced_path: Path):
    """Merge scores/measurements from the previous good CSV when the current
    run produced blurred (all-null) columns due to expired session cookie.

    Returns (n_columns_recovered, n_values_recovered) or (0, 0) if no recovery needed.
    """
    if not old_priced_path.exists() or not new_priced_path.exists():
        return 0, 0

    old_df = pd.read_csv(old_priced_path)
    new_df = pd.read_csv(new_priced_path)

    # Find columns that are all-null in new but had data in old
    # Skip identity/metadata columns and columns that are legitimately nullable
    skip = {"backlight_type_v2", "backlight_type_rtings", "dimming_zone_count",
            "price_per_mixed_use"}
    recover_cols = []
    for col in new_df.columns:
        if col in skip:
            continue
        if new_df[col].notna().sum() == 0 and col in old_df.columns and old_df[col].notna().sum() > 0:
            recover_cols.append(col)

    if not recover_cols:
        return 0, 0

    log(f"Recovering {len(recover_cols)} blurred columns from previous data")

    # Drop empty columns from new, merge from old
    new_df.drop(columns=recover_cols, inplace=True)
    old_subset = old_df[["product_id"] + recover_cols].copy()
    old_subset["product_id"] = old_subset["product_id"].astype(int)
    new_df["product_id"] = new_df["product_id"].astype(int)

    merged = new_df.merge(old_subset, on="product_id", how="left")

    # Recompute price_per_mixed_use if we recovered mixed_usage
    if "mixed_usage" in recover_cols and "price_best" in merged.columns:
        merged["price_per_mixed_use"] = merged.apply(
            lambda r: r["price_best"] / r["mixed_usage"]
            if pd.notna(r.get("price_best")) and pd.notna(r.get("mixed_usage")) and r["mixed_usage"] > 0
            else None,
            axis=1,
        )

    n_values = sum(merged[col].notna().sum() for col in recover_cols)
    merged.to_csv(new_priced_path, index=False)

    for col in recover_cols[:10]:
        log(f"  Recovered {col}: {merged[col].notna().sum()}/{len(merged)} values")
    if len(recover_cols) > 10:
        log(f"  ...and {len(recover_cols) - 10} more columns")

    return len(recover_cols), n_values


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _classification_alert(fullname: str, color_arch: str, marketing: str) -> str | None:
    """Return a reason string if a new TV's classification deserves manual review.

    Flags:
      - QNED models classified as anything other than KSF/WLED (2026+ expected zero QD)
      - Marketing says QLED/QNED but SPD says KSF or WLED
      - New OLEDs that aren't WOLED or QD-OLED
    """
    name_upper = fullname.upper()

    # QNED models classified as QD — high bar for 2026+
    if "QNED" in name_upper and color_arch in ("QD-LCD", "Pseudo QD"):
        return f"QNED classified as {color_arch} — verify QD content (2026+ expected zero QD)"

    # QLED marketing but classified as WLED (no QD or KSF detected)
    if marketing in ("QLED", "QNED", "ULED") and color_arch == "WLED":
        return f"Marketed as {marketing} but classified as WLED — check SPD"

    # OLED that isn't WOLED or QD-OLED
    if color_arch and "OLED" in name_upper and color_arch not in ("WOLED", "QD-OLED") and "LCD" not in color_arch:
        return f"OLED model classified as {color_arch} — expected WOLED or QD-OLED"

    return None


def detect_changes(old_db, new_db):
    """Compare old and new tv_database to detect changes."""
    changes = []

    old_ids = set(old_db["product_id"].astype(str))
    new_ids = set(new_db["product_id"].astype(str))

    # New TVs + classification review
    for pid in new_ids - old_ids:
        row = new_db[new_db["product_id"].astype(str) == pid].iloc[0]
        fullname = row.get("fullname", "")
        color_arch = row.get("color_architecture", "")
        marketing = row.get("marketing_label", "")

        changes.append({
            "date": TODAY,
            "product_id": pid,
            "fullname": fullname,
            "change_type": "new_tv",
            "field": "",
            "old_value": "",
            "new_value": color_arch,
        })

        # Flag classification surprises for manual review
        alert = _classification_alert(fullname, color_arch, marketing)
        if alert:
            changes.append({
                "date": TODAY,
                "product_id": pid,
                "fullname": fullname,
                "change_type": "classification_review",
                "field": "color_architecture",
                "old_value": alert,  # reason for flag
                "new_value": color_arch,
            })

    # Removed TVs
    for pid in old_ids - new_ids:
        row = old_db[old_db["product_id"].astype(str) == pid].iloc[0]
        changes.append({
            "date": TODAY,
            "product_id": pid,
            "fullname": row.get("fullname", ""),
            "change_type": "removed_tv",
            "field": "",
            "old_value": row.get("color_architecture", ""),
            "new_value": "",
        })

    # Score changes for existing TVs
    score_cols = ["mixed_usage", "home_theater", "gaming", "sports", "bright_room"]
    common_ids = old_ids & new_ids
    for pid in common_ids:
        old_row = old_db[old_db["product_id"].astype(str) == pid].iloc[0]
        new_row = new_db[new_db["product_id"].astype(str) == pid].iloc[0]

        for col in score_cols:
            if col not in old_row or col not in new_row:
                continue
            old_val = old_row.get(col)
            new_val = new_row.get(col)
            if pd.isna(old_val) or pd.isna(new_val):
                continue
            if abs(float(new_val) - float(old_val)) > 0.1:
                changes.append({
                    "date": TODAY,
                    "product_id": pid,
                    "fullname": new_row.get("fullname", ""),
                    "change_type": "score_change",
                    "field": col,
                    "old_value": f"{old_val:.1f}",
                    "new_value": f"{new_val:.1f}",
                })

        # Technology reclassification
        old_tech = old_row.get("color_architecture", "")
        new_tech = new_row.get("color_architecture", "")
        if old_tech != new_tech and old_tech and new_tech:
            changes.append({
                "date": TODAY,
                "product_id": pid,
                "fullname": new_row.get("fullname", ""),
                "change_type": "tech_change",
                "field": "color_architecture",
                "old_value": str(old_tech),
                "new_value": str(new_tech),
            })

        # QD material reclassification (CdSe ↔ InP ↔ Unknown)
        old_mat = old_row.get("qd_material", "")
        new_mat = new_row.get("qd_material", "")
        if old_mat != new_mat and old_mat and new_mat:
            changes.append({
                "date": TODAY,
                "product_id": pid,
                "fullname": new_row.get("fullname", ""),
                "change_type": "material_change",
                "field": "qd_material",
                "old_value": str(old_mat),
                "new_value": str(new_mat),
            })

        # Bench version change
        old_bench = str(old_row.get("test_bench_version", ""))
        new_bench = str(new_row.get("test_bench_version", ""))
        if old_bench and new_bench and old_bench != new_bench:
            changes.append({
                "date": TODAY,
                "product_id": pid,
                "fullname": new_row.get("fullname", ""),
                "change_type": "bench_change",
                "field": "test_bench_version",
                "old_value": old_bench,
                "new_value": new_bench,
            })

    return changes


def append_changelog(changes):
    """Append changes to the rolling changelog."""
    if not changes:
        return
    changelog_path = DATA / "changelog.csv"
    new_df = pd.DataFrame(changes)
    if changelog_path.exists():
        existing = pd.read_csv(changelog_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(changelog_path, index=False)
    else:
        new_df.to_csv(changelog_path, index=False)
    log(f"Changelog: {len(changes)} entries appended to {changelog_path}")


# ---------------------------------------------------------------------------
# TV Registry (state tracking)
# ---------------------------------------------------------------------------

def update_registry(new_db):
    """Update the TV registry with first_seen, last_seen, and status."""
    registry_path = DATA / "tv_registry.csv"

    new_ids = set(new_db["product_id"].astype(str))
    name_map = dict(zip(new_db["product_id"].astype(str), new_db["fullname"]))

    if registry_path.exists():
        reg = pd.read_csv(registry_path, dtype={"product_id": str})
    else:
        reg = pd.DataFrame(columns=["product_id", "fullname", "first_seen_date",
                                     "last_seen_date", "status"])

    existing_ids = set(reg["product_id"].astype(str))

    # Update existing entries
    for idx, row in reg.iterrows():
        pid = str(row["product_id"])
        if pid in new_ids:
            reg.at[idx, "last_seen_date"] = TODAY
            reg.at[idx, "status"] = "active"
            if pid in name_map:
                reg.at[idx, "fullname"] = name_map[pid]
        else:
            reg.at[idx, "status"] = "removed"

    # Add new entries
    for pid in new_ids - existing_ids:
        reg = pd.concat([reg, pd.DataFrame([{
            "product_id": pid,
            "fullname": name_map.get(pid, ""),
            "first_seen_date": TODAY,
            "last_seen_date": TODAY,
            "status": "active",
        }])], ignore_index=True)

    reg.to_csv(registry_path, index=False)
    n_active = (reg["status"] == "active").sum()
    n_removed = (reg["status"] == "removed").sum()
    log(f"Registry: {n_active} active, {n_removed} removed → {registry_path}")
    return reg


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def send_email(subject, body_html):
    """Send an email report via Gmail SMTP. Requires secrets in env."""
    sender = os.environ.get("GMAIL_ADDRESS")
    app_pw = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("NOTIFY_EMAIL")

    if not all([sender, app_pw, recipient]):
        log("Email secrets not configured — skipping email notification", "WARN")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"TV Dashboard <{sender}>"
    msg["To"] = recipient
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_pw)
            server.sendmail(sender, recipient, msg.as_string())
        log(f"Email sent to {recipient}")
        return True
    except Exception as e:
        log(f"Email failed: {e}", "ERROR")
        return False


def build_email_report(summary, changes, errors, old_count, new_count, n_priced):
    """Build an HTML email report."""
    status = "with warnings" if errors else "successfully"
    html = f"""
    <div style="font-family:Inter,Arial,sans-serif;max-width:600px;margin:0 auto;color:#e0e0e0;background:#1a1a2e;padding:24px;border-radius:8px">
      <h2 style="color:#fff;margin-top:0">TV Dashboard — Weekly Update</h2>
      <p style="color:#90bfff;font-size:14px">{TODAY} &middot; Completed {status}</p>
      <hr style="border:1px solid #333">

      <table style="width:100%;font-size:15px;border-collapse:collapse">
        <tr><td style="padding:6px 0;color:#999">Total TVs</td><td style="padding:6px 0;text-align:right;color:#fff"><b>{new_count}</b></td></tr>
        <tr><td style="padding:6px 0;color:#999">Previous count</td><td style="padding:6px 0;text-align:right;color:#fff">{old_count}</td></tr>
        <tr><td style="padding:6px 0;color:#999">TVs with pricing</td><td style="padding:6px 0;text-align:right;color:#fff">{n_priced}</td></tr>
        <tr><td style="padding:6px 0;color:#999">Total changes</td><td style="padding:6px 0;text-align:right;color:#fff">{len(changes)}</td></tr>
      </table>"""

    new_tvs = [c for c in changes if c["change_type"] == "new_tv"]
    removed_tvs = [c for c in changes if c["change_type"] == "removed_tv"]
    score_changes = [c for c in changes if c["change_type"] == "score_change"]
    bench_changes = [c for c in changes if c["change_type"] == "bench_change"]
    class_reviews = [c for c in changes if c["change_type"] == "classification_review"]

    if new_tvs:
        html += '<h3 style="color:#4ade80;margin-bottom:8px">New TVs</h3><ul style="margin:0;padding-left:20px">'
        for c in new_tvs:
            html += f'<li style="padding:2px 0">{c["fullname"]} ({c["new_value"]})</li>'
        html += '</ul>'

    if class_reviews:
        html += f'<h3 style="color:#f59e0b;margin-bottom:8px">⚠ Classification Review ({len(class_reviews)})</h3>'
        html += '<p style="color:#999;font-size:13px;margin:4px 0 8px">These new TVs have classifications that may need manual verification:</p>'
        html += '<ul style="margin:0;padding-left:20px">'
        for c in class_reviews:
            html += (f'<li style="padding:4px 0"><b>{c["fullname"]}</b> → {c["new_value"]}'
                     f'<br><span style="color:#f59e0b;font-size:13px">{c["old_value"]}</span></li>')
        html += '</ul>'

    if removed_tvs:
        html += '<h3 style="color:#f87171;margin-bottom:8px">Removed TVs</h3><ul style="margin:0;padding-left:20px">'
        for c in removed_tvs:
            html += f'<li style="padding:2px 0">{c["fullname"]}</li>'
        html += '</ul>'

    if bench_changes:
        html += f'<h3 style="color:#60a5fa;margin-bottom:8px">Bench Version Changes ({len(bench_changes)})</h3><ul style="margin:0;padding-left:20px">'
        for c in bench_changes[:20]:
            html += f'<li style="padding:2px 0">{c["fullname"]}: {c["old_value"]} → {c["new_value"]}</li>'
        if len(bench_changes) > 20:
            html += f'<li style="color:#999">...and {len(bench_changes) - 20} more</li>'
        html += '</ul>'

    if score_changes:
        html += f'<h3 style="color:#fbbf24;margin-bottom:8px">Score Changes ({len(score_changes)})</h3><ul style="margin:0;padding-left:20px">'
        for c in score_changes[:15]:
            html += f'<li style="padding:2px 0">{c["fullname"]}: {c["field"]} {c["old_value"]} → {c["new_value"]}</li>'
        if len(score_changes) > 15:
            html += f'<li style="color:#999">...and {len(score_changes) - 15} more</li>'
        html += '</ul>'

    if errors:
        html += '<h3 style="color:#f87171;margin-bottom:8px">Warnings</h3><ul style="margin:0;padding-left:20px">'
        for e in errors:
            html += f'<li style="padding:2px 0">{e}</li>'
        html += '</ul>'

    html += '<hr style="border:1px solid #333"><p style="color:#666;font-size:12px;margin-bottom:0">Automated report from TV Display Technology Dashboard</p></div>'
    return html


def notify(title, message):
    """Send a notification — macOS native or GitHub Actions summary."""
    log(f"NOTIFICATION: {title} — {message}")
    if IN_CI:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(f"## {title}\n{message}\n")
    elif platform.system() == "Darwin":
        try:
            subprocess.run([
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"'
            ], timeout=10)
        except Exception:
            pass  # Don't fail on notification errors


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_tv_pipeline():
    """Run the full TV pipeline — scrape, analyze, build, price, detect changes.

    Returns (summary_str, changes_list, errors_list) for reporting.
    """
    errors = []
    changes = []

    # --- Snapshot "before" state ---
    db_path = DATA / "tv_database.csv"
    priced_path = DATA / "tv_database_with_prices.csv"
    old_db = pd.read_csv(db_path) if db_path.exists() else pd.DataFrame()
    old_count = len(old_db)

    # Save a copy of the priced CSV for stale score fallback
    old_priced_bak = DATA / ".tv_database_with_prices_backup.csv"
    if priced_path.exists():
        shutil.copy2(priced_path, old_priced_bak)

    # --- Step 1: Scrape RTINGS data ---
    try:
        run_script("rtings_scraper.py", abort_on_fail=True,
                    extra_args=["--silo", "tv"])
    except Exception as e:
        errors.append(f"Scraper failed: {e}")
        notify("TV Data Update FAILED", f"Scraper error: {e}")
        return None, [], errors

    # --- Step 1b: Drop guard — abort if scraper output looks wrong ---
    scraper_csv = DATA / "rtings_tv_data.csv"
    try:
        check_drop_guard(old_count, scraper_csv)
    except RuntimeError as e:
        log(str(e), "ERROR")
        notify("TV Data Update ABORTED", str(e))
        send_email(
            f"TV Dashboard ABORTED — {TODAY}",
            f'<div style="font-family:Inter,Arial,sans-serif;max-width:600px;margin:0 auto;color:#e0e0e0;background:#1a1a2e;padding:24px;border-radius:8px">'
            f'<h2 style="color:#f87171;margin-top:0">Pipeline Aborted — Drop Guard</h2>'
            f'<p>{e}</p>'
            f'<p style="color:#999">Previous count: {old_count}</p>'
            f'</div>'
        )
        return None, [], [str(e)]

    # --- Step 1c: Check session cookie status ---
    session_flag = DATA / ".session_ok"
    if session_flag.exists() and session_flag.read_text().strip() == "0":
        log("RTINGS session cookie expired — scores will be blurred", "WARN")
        errors.append("RTINGS session cookie expired — scores/measurements are blurred")
        send_email(
            f"ACTION REQUIRED: RTINGS Session Expired — {TODAY}",
            '<div style="font-family:Inter,Arial,sans-serif;max-width:600px;margin:0 auto;color:#e0e0e0;background:#1a1a2e;padding:24px;border-radius:8px">'
            '<h2 style="color:#fbbf24;margin-top:0">RTINGS Session Cookie Expired</h2>'
            '<p>The weekly pipeline ran but all scores and measurements are <b>blurred</b> '
            'because the RTINGS session cookie has expired.</p>'
            '<h3 style="color:#fff">To fix:</h3>'
            '<ol style="line-height:1.8">'
            '<li>Log into <a href="https://www.rtings.com/login" style="color:#60a5fa">rtings.com</a> in Chrome</li>'
            '<li>Open DevTools: <code style="background:#333;padding:2px 6px;border-radius:3px">Cmd+Option+I</code></li>'
            '<li>Go to <b>Application</b> → <b>Cookies</b> → <code>https://www.rtings.com</code></li>'
            '<li>Copy the value of <code style="background:#333;padding:2px 6px;border-radius:3px">_rtings_session</code></li>'
            '<li>Update the GitHub secret: <a href="https://github.com/jy2k79/tv-tech-dashboard/settings/secrets/actions" style="color:#60a5fa">Settings → Secrets → RTINGS_SESSION</a></li>'
            '</ol>'
            '<p style="color:#999;font-size:13px">The cookie expires ~30 days after login. '
            'Pipeline will use stale scores until refreshed.</p>'
            '</div>'
        )

    # --- Step 2: SPD Analysis ---
    spd_ok = run_script("spd_analyzer.py", abort_on_fail=False)
    if not spd_ok:
        errors.append("SPD analyzer failed — using previous classifications")

    # --- Step 3: Build Schema ---
    try:
        run_script("build_schema.py", abort_on_fail=True)
    except Exception as e:
        errors.append(f"Schema build failed: {e}")
        notify("TV Data Update FAILED", f"Schema error: {e}")
        return None, [], errors

    # --- Step 4: Pricing Pipeline ---
    pricing_ok = run_script("pricing_pipeline.py", abort_on_fail=False)
    if not pricing_ok:
        errors.append("Pricing pipeline failed — using stale prices")

    # --- Step 4b: Stale score fallback ---
    session_flag = DATA / ".session_ok"
    session_expired = session_flag.exists() and session_flag.read_text().strip() == "0"
    if session_expired and old_priced_bak.exists():
        n_cols, n_vals = recover_blurred_scores(old_priced_bak, priced_path)
        if n_cols > 0:
            log(f"Recovered {n_cols} blurred columns ({n_vals} values) from previous data")
            errors.append(f"Session expired — recovered {n_cols} columns from previous data")

    # Clean up backup
    if old_priced_bak.exists():
        old_priced_bak.unlink()

    # --- Step 5: Change detection ---
    new_db = pd.read_csv(db_path) if db_path.exists() else pd.DataFrame()
    new_count = len(new_db)

    n_new = 0
    n_removed = 0
    n_changes = 0
    if len(old_db) > 0 and len(new_db) > 0:
        changes = detect_changes(old_db, new_db)
        append_changelog(changes)
        n_new = sum(1 for c in changes if c["change_type"] == "new_tv")
        n_removed = sum(1 for c in changes if c["change_type"] == "removed_tv")
        n_changes = len(changes)
    else:
        log("No previous database found — skipping change detection (first run)")

    # --- Step 6: Registry update ---
    if len(new_db) > 0:
        update_registry(new_db)

    # --- Build summary ---
    n_priced = 0
    if priced_path.exists():
        pdb = pd.read_csv(priced_path)
        n_priced = pdb["price_best"].notna().sum()

    summary_parts = [f"{new_count} TVs"]
    if n_new > 0:
        summary_parts.append(f"{n_new} new")
    if n_removed > 0:
        summary_parts.append(f"{n_removed} removed")
    if n_changes > 0:
        summary_parts.append(f"{n_changes} changes")
    summary_parts.append(f"{n_priced} priced")
    if errors:
        summary_parts.append(f"{len(errors)} warnings")

    return ", ".join(summary_parts), changes, errors


def run_monitor_pipeline():
    """Run the monitor pipeline — scrape + SPD analysis.

    Schema builder and pricing pipeline for monitors are not yet implemented
    (Phase 2). This runs the available stages and reports status.

    Returns (summary_str, changes_list, errors_list) for reporting.
    """
    errors = []
    silo_cfg = get_silo("monitor")
    paths = silo_cfg["paths"]

    # --- Step 1: Scrape RTINGS monitor data ---
    try:
        run_script("rtings_scraper.py", abort_on_fail=True,
                    extra_args=["--silo", "monitor"])
    except Exception as e:
        errors.append(f"Monitor scraper failed: {e}")
        return None, [], errors

    # --- Step 2: SPD Analysis ---
    spd_ok = run_script("spd_analyzer.py", abort_on_fail=False,
                        extra_args=["--silo", "monitor"])
    if not spd_ok:
        errors.append("Monitor SPD analyzer failed — using previous classifications")

    # --- Step 3: Build Schema ---
    schema_ok = run_script("build_monitor_schema.py", abort_on_fail=False)
    if not schema_ok:
        errors.append("Monitor schema build failed")

    # --- Step 4: Pricing Pipeline ---
    pricing_ok = run_script("monitor_pricing_pipeline.py", abort_on_fail=False)
    if not pricing_ok:
        errors.append("Monitor pricing pipeline failed — using stale prices")

    # Count products scraped
    scraped_csv = paths["scraped_csv"]
    n_products = 0
    if scraped_csv.exists():
        n_products = len(pd.read_csv(scraped_csv))

    summary = f"{n_products} monitors scraped (pipeline incomplete — Phase 2 pending)"
    return summary, [], errors


def main():
    parser = argparse.ArgumentParser(description="Weekly Data Update Orchestrator")
    parser.add_argument(
        "--silo", default="tv", choices=list(SILOS.keys()) + ["all"],
        help="Which product silo to update (default: tv)",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--qd-export-only", action="store_true",
        help="Skip pipeline, just build and send the QD export from existing data",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"WEEKLY DATA UPDATE — {args.silo.upper()}")
    print(f"Date: {TODAY}  |  Environment: {'GitHub Actions' if IN_CI else 'Local'}")
    print("=" * 70)

    if args.qd_export_only:
        log("QD export only — skipping pipeline")
        run_script("qd_export.py", abort_on_fail=False)
        sys.exit(0)

    if args.dry:
        log("DRY RUN — showing execution plan only")
        silos_to_run = list(SILOS.keys()) if args.silo == "all" else [args.silo]
        for silo_name in silos_to_run:
            log(f"\n--- {silo_name.upper()} pipeline ---")
            log(f"  1. rtings_scraper.py --silo {silo_name} (abort on fail)")
            log(f"  2. spd_analyzer.py (continue on fail)")
            if silo_name == "tv":
                log("  3. build_schema.py (abort on fail)")
                log("  4. pricing_pipeline.py (continue on fail)")
                log("  5. Change detection + changelog")
                log("  6. Registry update")
            else:
                log(f"  3. build_{silo_name}_schema.py (not yet implemented)")
                log(f"  4. {silo_name}_pricing_pipeline.py (not yet implemented)")
            log("  7. Notification")
        log("\n--- Post-pipeline ---")
        log("  8. qd_export.py — QD SKU Tracker CSV + email (continue on fail)")
        return

    # Determine which silos to run
    silos_to_run = list(SILOS.keys()) if args.silo == "all" else [args.silo]
    all_summaries = []
    all_changes = []
    all_errors = []
    any_critical_failure = False

    for silo_name in silos_to_run:
        print(f"\n{'=' * 70}")
        print(f"  {silo_name.upper()} PIPELINE")
        print(f"{'=' * 70}")

        if silo_name == "tv":
            summary, changes, errors = run_tv_pipeline()
        elif silo_name == "monitor":
            summary, changes, errors = run_monitor_pipeline()
        else:
            log(f"Unknown silo '{silo_name}' — skipping", "ERROR")
            continue

        if summary is None:
            # Critical failure in this silo
            any_critical_failure = True
            all_errors.extend(errors)
            all_summaries.append(f"{silo_name}: FAILED")
        else:
            all_summaries.append(f"{silo_name}: {summary}")
            all_changes.extend(changes)
            all_errors.extend(errors)

    # --- Notification ---
    combined_summary = " | ".join(all_summaries)
    notify("Dashboard Updated", combined_summary)

    # --- Email report (TV-only for now, monitor reporting in Phase 4) ---
    if "tv" in silos_to_run:
        # Find the TV summary/changes/errors from the run
        tv_changes = [c for c in all_changes]  # Currently all changes are from TV
        priced_path = DATA / "tv_database_with_prices.csv"
        db_path = DATA / "tv_database.csv"
        new_count = len(pd.read_csv(db_path)) if db_path.exists() else 0
        old_count = new_count  # approximation — actual old_count tracked inside run_tv_pipeline
        n_priced = 0
        if priced_path.exists():
            n_priced = pd.read_csv(priced_path)["price_best"].notna().sum()

        email_html = build_email_report(
            combined_summary, tv_changes,
            all_errors, old_count, new_count, n_priced)
        send_email(f"Dashboard Update — {TODAY}", email_html)

    # --- QD SKU Tracker export (runs after all silos complete) ---
    qd_ok = run_script("qd_export.py", abort_on_fail=False)
    if not qd_ok:
        all_errors.append("QD SKU Tracker export failed")

    print(f"\n{'=' * 70}")
    print(f"UPDATE COMPLETE: {combined_summary}")
    if all_errors:
        print(f"\nWarnings:")
        for e in all_errors:
            print(f"  - {e}")
    print("=" * 70)

    # Exit 1 only if a critical silo (TV) failed when running alone
    # When running --silo all, monitor failure shouldn't block TV
    if any_critical_failure and args.silo != "all":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
