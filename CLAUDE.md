# TV Tech Dashboard

## Project Overview
Automated TV display technology analysis pipeline + interactive Streamlit dashboard. Scrapes RTINGS.com API for TV review data, analyzes SPD (spectral power distribution) images to classify display technologies (WLED, KSF, QD-LCD, WOLED, QD-OLED), tracks pricing via Keepa/Best Buy APIs, and presents everything in a password-gated dashboard.

## Architecture

### Pipeline (runs weekly via GitHub Actions)
```
rtings_scraper.py → spd_analyzer.py → build_schema.py → pricing_pipeline.py
     ↓                    ↓                  ↓                   ↓
 rtings_tv_data.csv   spd_analysis_     tv_database.csv    tv_database_with_
                      results.csv                          prices.csv
```
Orchestrated by `weekly_update.py`, triggered by `.github/workflows/weekly-update.yml`.

### Dashboard
- `dashboard.py` — Streamlit app, reads `data/tv_database_with_prices.csv`
- Deployed on Streamlit Cloud, auto-deploys from `main` branch

### Key Files
| File | Purpose |
|------|---------|
| `rtings_scraper.py` | Scrapes RTINGS API for TV specs, scores, SPD images |
| `spd_analyzer.py` | Peak detection, FWHM measurement, tech classification |
| `build_schema.py` | Merges scraped + SPD data into unified schema |
| `pricing_pipeline.py` | Keepa + Best Buy price history |
| `weekly_update.py` | CI orchestrator |
| `monthly_report.py` | Monthly PDF intelligence report |
| `dashboard.py` | Streamlit dashboard |

### Data Directory (`data/`)
- `tv_database_with_prices.csv` — **primary dashboard input** (NOT `tv_database.csv`)
- `spd_analysis_results.csv` — per-TV FWHM values and classifications
- `tv_registry.csv` — TV tracking registry with first_seen dates
- `changelog.csv` — data change log
- `price_history.csv`, `keepa_prices.csv`, `bestbuy_prices.csv` — pricing data

## Development Workflow

### Issue Tracking
- **GitHub Project board**: [TV Tech Dashboard](https://github.com/users/jy2k79/projects/3)
- Kanban columns: Todo → In Progress → Done
- Priority field: P0 Critical, P1 High, P2 Medium, P3 Low
- Labels: `bug`, `enhancement`, `pipeline`, `dashboard`, `spd-analysis`, `pricing`, `data-quality`, `infra`

### Working on Issues — Kanban Workflow
When starting work on an issue, **always update the GitHub project board**:

1. **Start:** Move the issue to "In Progress" on the board (`gh project item-edit`)
2. **Branch:** Create a branch for non-trivial changes: `fix/issue-N-description` or `feat/issue-N-description`
3. **Reference:** Include the issue number in commits (`Fixes #N` or `Relates to #N`)
4. **Test:** Test pipeline changes locally before pushing: `python weekly_update.py`
5. **Regenerate:** After schema/FWHM changes, always regenerate `tv_database_with_prices.csv`
6. **Verify:** Verify dashboard loads correctly: `streamlit run dashboard.py`
7. **Complete:** Move the issue to "Done" on the board after merge/push, and close it

**Board commands reference:**
```bash
# List project items and their IDs
gh project item-list 3 --owner jy2k79 --format json

# Move an item to a new status column
gh project item-edit --project-id PVT_kwHODNnfCs4BSCMM --id <ITEM_ID> \
  --field-id PVTSSF_lAHODNnfCs4BSCMMzg_r7Eo \
  --single-select-option-id <STATUS_OPTION_ID>

# Project ID:  PVT_kwHODNnfCs4BSCMM
# Status field: PVTSSF_lAHODNnfCs4BSCMMzg_r7Eo
#   Todo:        f75ad846
#   In Progress: 47fc9ee4
#   Done:        98236657
# Priority field: PVTSSF_lAHODNnfCs4BSCMMzg_r7QM
#   P0 Critical: 2faf899a
#   P1 High:     1cfd2858
#   P2 Medium:   a949b272
#   P3 Low:      5f4b40a1
```

### Commit Conventions
- Prefix with context: `[pipeline]`, `[dashboard]`, `[spd]`, `[pricing]`, `[ci]`
- Reference issues: `Fixes #N` or `Relates to #N`

## Technology Classification Order
Least to most premium: `WLED > KSF > Pseudo QD > QD-LCD > WOLED > QD-OLED`

## Testing Locally
```bash
pip install -r requirements.txt

# Run full pipeline
python weekly_update.py

# Run individual stages
python rtings_scraper.py
python spd_analyzer.py
python build_schema.py
python pricing_pipeline.py  # needs KEEPA_API_KEY, BESTBUY_API_KEY

# Run dashboard
streamlit run dashboard.py
```

## Environment Variables / Secrets
| Variable | Where | Purpose |
|----------|-------|---------|
| `KEEPA_API_KEY` | GitHub Secrets | Keepa pricing API |
| `BESTBUY_API_KEY` | GitHub Secrets | Best Buy pricing API |
| `GMAIL_ADDRESS` | GitHub Secrets | Email notifications |
| `GMAIL_APP_PASSWORD` | GitHub Secrets | Gmail SMTP auth |
| `NOTIFY_EMAIL` | GitHub Secrets | Notification recipient |
| `ANTHROPIC_API_KEY` | GitHub Secrets | Monthly report generation |
| `RTINGS_SESSION` | GitHub Secrets + `.env` | RTINGS member cookie (`_rtings_session`), ~30 day expiry |
| `app_password` | Streamlit Secrets | Dashboard login (`nanosys2026`) |

## Common Pitfalls
- Dashboard reads `tv_database_with_prices.csv`, NOT `tv_database.csv` — must regenerate after schema changes
- SPD images are cached in CI — stale cache can mask classification changes
- FWHM is measured from absolute zero baseline, not scipy prominence
- WLED red uses HWHM mirroring (one side doesn't cross half-max)
- QNED is a grab-bag of technologies — don't assume QD
