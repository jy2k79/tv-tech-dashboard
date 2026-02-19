#!/usr/bin/env python3
"""
Weekly TV Data Update — Master Orchestrator
================================================
Runs all pipeline scripts in sequence, detects changes,
maintains a TV registry and changelog, and sends notifications.

Works both locally (macOS notification) and in GitHub Actions
(writes to $GITHUB_STEP_SUMMARY).

Usage:
    python weekly_update.py          # Full update
    python weekly_update.py --dry    # Show what would run, don't execute
"""

import os
import sys
import subprocess
import traceback
import platform
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

import pandas as pd

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


def run_script(name, abort_on_fail=True):
    """Run a Python script in the project root. Returns True on success."""
    path = ROOT / name
    log(f"Running {name}...")
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=False,
            timeout=600,  # 10 minute timeout per script
        )
        if result.returncode != 0:
            log(f"{name} exited with code {result.returncode}", "ERROR")
            if abort_on_fail:
                raise RuntimeError(f"{name} failed (exit {result.returncode})")
            return False
        log(f"{name} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        log(f"{name} timed out after 10 minutes", "ERROR")
        if abort_on_fail:
            raise
        return False
    except Exception as e:
        log(f"{name} error: {e}", "ERROR")
        if abort_on_fail:
            raise
        return False


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def detect_changes(old_db, new_db):
    """Compare old and new tv_database to detect changes."""
    changes = []

    old_ids = set(old_db["product_id"].astype(str))
    new_ids = set(new_db["product_id"].astype(str))

    # New TVs
    for pid in new_ids - old_ids:
        row = new_db[new_db["product_id"].astype(str) == pid].iloc[0]
        changes.append({
            "date": TODAY,
            "product_id": pid,
            "fullname": row.get("fullname", ""),
            "change_type": "new_tv",
            "field": "",
            "old_value": "",
            "new_value": row.get("color_architecture", ""),
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

    if new_tvs:
        html += '<h3 style="color:#4ade80;margin-bottom:8px">New TVs</h3><ul style="margin:0;padding-left:20px">'
        for c in new_tvs:
            html += f'<li style="padding:2px 0">{c["fullname"]} ({c["new_value"]})</li>'
        html += '</ul>'

    if removed_tvs:
        html += '<h3 style="color:#f87171;margin-bottom:8px">Removed TVs</h3><ul style="margin:0;padding-left:20px">'
        for c in removed_tvs:
            html += f'<li style="padding:2px 0">{c["fullname"]}</li>'
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

def main():
    print("=" * 70)
    print("WEEKLY TV DATA UPDATE")
    print(f"Date: {TODAY}  |  Environment: {'GitHub Actions' if IN_CI else 'Local'}")
    print("=" * 70)

    dry_run = "--dry" in sys.argv
    if dry_run:
        log("DRY RUN — showing execution plan only")
        log("  1. rtings_scraper.py (abort on fail)")
        log("  2. spd_analyzer.py (continue on fail)")
        log("  3. build_schema.py (abort on fail)")
        log("  4. pricing_pipeline.py (continue on fail)")
        log("  5. Change detection + changelog")
        log("  6. Registry update")
        log("  7. Notification")
        return

    errors = []
    n_new = 0
    n_removed = 0
    n_changes = 0

    # --- Snapshot "before" state ---
    db_path = DATA / "tv_database.csv"
    old_db = pd.read_csv(db_path) if db_path.exists() else pd.DataFrame()
    old_count = len(old_db)

    # --- Step 1: Scrape RTINGS data ---
    try:
        run_script("rtings_scraper.py", abort_on_fail=True)
    except Exception as e:
        errors.append(f"Scraper failed: {e}")
        notify("TV Data Update FAILED", f"Scraper error: {e}")
        sys.exit(1)

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
        sys.exit(1)

    # --- Step 4: Pricing Pipeline ---
    pricing_ok = run_script("pricing_pipeline.py", abort_on_fail=False)
    if not pricing_ok:
        errors.append("Pricing pipeline failed — using stale prices")

    # --- Step 5: Change detection ---
    new_db = pd.read_csv(db_path) if db_path.exists() else pd.DataFrame()
    new_count = len(new_db)

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

    # --- Step 7: Summary & notification ---
    priced_db = DATA / "tv_database_with_prices.csv"
    n_priced = 0
    if priced_db.exists():
        pdb = pd.read_csv(priced_db)
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

    summary = ", ".join(summary_parts)
    notify("TV Dashboard Updated", summary)

    # --- Step 8: Email report ---
    email_html = build_email_report(
        summary, changes if 'changes' in dir() else [],
        errors, old_count, new_count, n_priced)
    send_email(f"TV Dashboard Update — {TODAY}", email_html)

    print(f"\n{'=' * 70}")
    print(f"UPDATE COMPLETE: {summary}")
    if errors:
        print(f"\nWarnings:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)

    # Exit 0 even with warnings — only critical failures (scraper/schema)
    # cause early exit(1) above. Warnings shouldn't block the data commit.
    sys.exit(0)


if __name__ == "__main__":
    main()
