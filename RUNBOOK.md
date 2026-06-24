# RTINGS Dashboard — Operations Runbook

Low-maintenance operating guide. Target: a **~15-minute monthly check-in** keeps this healthy. Everything that needs a human is designed to fail *loudly* (email + red CI run), so if you get no alerts, it's probably fine.

---

## What runs automatically

| Job | Schedule | File | Output |
|-----|----------|------|--------|
| Weekly data update (TV + monitor scrape → SPD → schema → pricing) | Mondays 13:00 UTC | `.github/workflows/weekly-update.yml` → `weekly_update.py` | commits `data/*.csv` |
| Monthly intelligence report (PDF) | First Monday | `monthly_report.py` | `data/reports/display_intelligence_YYYY_MM.pdf` |
| Dashboard | always-on | Streamlit Cloud, deploys from `main` | reads `data/*_with_prices.csv` |

The dashboard auto-redeploys on every push to `main`.

---

## Monthly check-in (≈15 min)

1. **Inbox:** search for **"ACTION REQUIRED"** or **"ABORTED"** emails from the pipeline (sent to `NOTIFY_EMAIL`). If none → the weekly runs have been healthy.
2. **GitHub Actions:** open the [Actions tab](https://github.com/jy2k79/tv-tech-dashboard/actions). The last ~4 weekly runs should be green. A **red run = something needs you** (most likely the cookie — see below).
3. **Dashboard smoke test:** open the app, confirm all **7 technology classes** show data (WLED, KSF, Pseudo QD, QD-LCD, RGB MiniLED, WOLED, QD-OLED) and the sidebar version/changelog looks current. If only ~3 classes show → the cookie lapsed and the guard didn't commit (data is stale-but-safe) → refresh the cookie.
4. **Cookie age:** the `_rtings_session` cookie lasts **~30 days**. If your last refresh was ~3+ weeks ago, refresh it proactively (2 min) so the next weekly run doesn't abort.

---

## Alert reference — what each means & what to do

| Alert / symptom | Meaning | Action |
|-----------------|---------|--------|
| Email **"ACTION REQUIRED: RTINGS Session Expired"** | Cookie expired OR RTINGS served the placeholder SPD image. The run **aborted and committed nothing** (guard working as designed). | Refresh the cookie (below), then re-run the weekly workflow manually (Actions → Run workflow). |
| Email **"TV/Monitor Data Update ABORTED"** | A guard fired: drop guard (row count collapsed), SPD collapse (>50% identical FWHM), or session expiry. No data committed. | Read the email body for which guard. Usually = cookie. Refresh + re-run. |
| **Red weekly Actions run** | Pipeline exited non-zero (fail-closed). No commit happened. | Open the run logs; the failing step names the cause. |
| Dashboard shows fewer classes than expected / pricing looks stale | Almost always upstream data, not the app. | Check Actions + cookie. The app faithfully renders whatever was last committed. |

---

## Refreshing the RTINGS session cookie (the one recurring chore)

The cookie expires ~30 days after login. Full procedure is in `CLAUDE.md` → "Refreshing the RTINGS session cookie". Short version (Safari):

1. Log in at [rtings.com/login](https://www.rtings.com/login) — confirm your username shows top-right.
2. Right-click → **Inspect Element** → **Storage** tab → **Cookies** → `https://www.rtings.com` → copy the **`_rtings_session`** value (~1100 chars, ends in `--…==`). *(It's HttpOnly — `document.cookie` won't show it; use the Storage panel.)*
3. Paste it into **both**:
   - GitHub secret **`RTINGS_SESSION`** → [repo secrets](https://github.com/jy2k79/tv-tech-dashboard/settings/secrets/actions) (for weekly CI)
   - local `.env` line `RTINGS_SESSION=…` (only needed for local runs)
4. Verify: `python rtings_scraper.py --silo tv` → expect `Session OK: N/N ratings unblurred, N/N distinct SPD images`.

> **Future zero-touch option:** to eliminate this chore, capture the login network request once (DevTools → Network → log in → find the auth POST) and a developer can wire an auto-login into `rtings_scraper.py`. Not done yet because RTINGS's login is JS/API-rendered and a blind implementation would be fragile.

---

## Secrets & key rotation

All CI secrets live in **GitHub → Settings → Secrets → Actions**. Local runs read `.env` (gitignored). Streamlit login reads Streamlit Cloud secrets.

| Secret | Used by | Renewal cadence | How to renew |
|--------|---------|-----------------|--------------|
| `RTINGS_SESSION` | scraper | **~30 days** | Re-copy cookie (above) |
| `KEEPA_API_KEY` | pricing | rarely (account-based) | keepa.com account |
| `BESTBUY_API_KEY` | pricing | rarely | developer.bestbuy.com |
| `ANTHROPIC_API_KEY` | monthly report | when it lapses / runs out of credit | console.anthropic.com |
| `GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD` | alert emails | app password can be revoked | Google account → App passwords |
| `NOTIFY_EMAIL` | alert recipient | — | should point at a mailbox that outlives any one person |
| `app_password` (Streamlit secret) | dashboard login | as desired | Streamlit Cloud → app → Settings → Secrets |

---

## Emergency: roll back a bad data commit

If a weekly run committed visibly wrong data (shouldn't happen now — guards are fail-closed):
```bash
git revert <bad-commit-sha>     # or: git revert HEAD
git push origin main            # dashboard redeploys with restored data
```

---

## Credentials & handoff inventory  *(fill in the TODOs)*

> Ownership of these is **undecided** as of 2026-06. The biggest long-term risk is not code — it's these accounts being deprovisioned. Fill in the owner/survival columns and migrate anything that won't survive a departure to a **shared org account**.

| Asset | Where it lives | Owner | Survives departure? | Migration action |
|-------|----------------|-------|---------------------|------------------|
| GitHub repo `jy2k79/tv-tech-dashboard` | github.com | TODO | TODO | Transfer to Nanosys/Shoei org if `jy2k79` is personal |
| GitHub Actions secrets (all above) | repo settings | TODO | TODO | Re-add under org repo after transfer |
| Streamlit Cloud deployment | streamlit.io account | TODO | TODO | Move app to a team Streamlit account |
| `NOTIFY_EMAIL` / Gmail alert account | Google | TODO | TODO | Point alerts at a team distribution list |
| RTINGS member account (for the cookie) | rtings.com | TODO | TODO | Move to a team-owned RTINGS login |
| API keys (Keepa, Best Buy, Anthropic) | provider accounts | TODO | TODO | Re-issue under team-owned/billed accounts |
| GCP service account (shared w/ SKU Tracker) | Google Cloud | TODO | TODO | Confirm GCP project ownership |

**Before you lose access:** transfer the repo, re-create the secrets under the new owner, move the Streamlit deploy, and repoint `NOTIFY_EMAIL` to a durable inbox. Once `NOTIFY_EMAIL` is a team inbox, the monthly check-in can be done by anyone.
