# TV Dashboard Roadmap

## Backlog

### 1. SPD image cache staleness detection
**Priority: Medium | Effort: Low**

Basic `actions/cache` for `spd_images/` is implemented. Scraper skips existing files. Remaining work:

**Plan:**
- Add HTTP `HEAD` requests to compare `Content-Length` with cached file size
- Only re-download images whose size changed (catches re-published measurements)

**Future:** Check RTINGS review changelogs for SPD-related corrections; track `spd_last_verified` per TV in registry.

---

### 2. Extract panel_sub_type from RTINGS scraper
**Priority: High | Effort: Medium**

The scraper doesn't extract `panel_sub_type` (QD-OLED vs WOLED). All OLEDs fall back to SPD-based classification (medium confidence) instead of metadata-based (high confidence).

**Plan:**
- Investigate RTINGS API for panel sub-type field (different test ID or product metadata)
- Add extraction to `build_product_records()` in `rtings_scraper.py`
- Fallback: derive from product name heuristics (Samsung S90/S95 = QD-OLED, LG C/G/B = WOLED)

**Impact:** High-confidence OLED classification on CI.

---

### 3. WOLED FWHM deconvolution / color gamut cross-validation
**Priority: Medium | Effort: High**

WOLED uses RGBW subpixels. The W subpixel passes full unfiltered spectrum, broadening measured green/red FWHM beyond true emitter widths. Current values are "composite."

**Ideas:**
- **Color gamut cross-validation:** Back-calculate implied FWHM from RTINGS DCI-P3/Rec.2020 coverage data
- **YAG phosphor modeling:** Model unfiltered W component and subtract
- **Minimum:** Flag WOLED FWHM as "composite" in dashboard charts

**Context:** WOLED color filters are much more aggressive than WLED. No simple solution -- known display metrology challenge.

---

### 4. Monthly Display Technology Intelligence Report
**Priority: High | Effort: High**

Automated monthly analyst report delivered via email on the first Monday of each month. Three sections with data-driven insights, charts, and editorial analysis. Written in an engaging Wired-style voice -- factual but compelling.

#### Schedule & Integration
- Runs on first Monday of each month, after the weekly data pull completes
- Add a second GitHub Actions workflow (`monthly-report.yml`) triggered by `workflow_dispatch` + cron
- Cron runs every Monday; script checks `if today.day <= 7` to only execute on first Monday
- Uses latest data from the weekly pipeline (tv_database, price_history, changelog, registry)

#### Section 1: Display Technology Overview
- **New devices**: TVs added since last report (from changelog + registry `first_seen_date`)
- **Removed devices**: TVs dropped from RTINGS coverage
- **Performance trends**: Score changes (mixed_usage, gaming, home_theater) by technology
- **Technology shifts**: Are QD gamuts getting wider? Is WOLED closing the gap? FWHM trends over time
- **Model year comparisons**: 2026 vs 2025 models on key metrics as new reviews arrive
- **Charts**: Score distributions by tech (box plots), FWHM trend lines, new model scorecards

#### Section 2: Pricing Intelligence
- **Price trends by technology**: Month-over-month, quarter-over-quarter price movement
- **New vs. old model pricing**: Are 2026 models launching higher/lower than 2025 equivalents?
- **Technology price gaps**: QD-LCD vs WOLED vs QD-OLED price/performance spread
- **Value frontier shifts**: Has the efficient frontier moved? New best-value picks?
- **Seasonal patterns**: As data accumulates, identify sale cycles (Black Friday, Prime Day, etc.)
- **Charts**: Price trend lines by tech, price/performance scatter evolution, value frontier overlay

#### Section 3: Macro & Industry Context
- **Display industry news**: Panel factory output, supply chain developments, new product announcements
- **Macro economics**: Consumer spending trends, tariff impacts, currency effects on pricing
- **Technology roadmap**: Upcoming display technologies (microLED, tandem OLED, perovskite QD)
- **Competitive landscape**: Brand strategy shifts, market share indicators
- **Sources**: Web search for recent industry reporting (Display Supply Chain Consultants, Omdia, Display Daily, LEDinside, etc.)
- **Editorial voice**: Synthesize data + context into "so what" takeaways for the team

#### Implementation Plan

**New files:**
- `monthly_report.py` -- Report generator orchestrator
- `report_templates/` -- HTML/CSS templates for PDF rendering
- `.github/workflows/monthly-report.yml` -- Monthly workflow

**Technical approach:**
1. **Data collection** (`monthly_report.py`):
   - Load tv_database, changelog, registry, price_history CSVs
   - Compute all metrics: deltas, trends, rankings, comparisons
   - Web search for macro/industry context (via API or scraping)

2. **Analysis & narrative** (Claude API):
   - Pass structured data + metrics to Claude API
   - Prompt for each section with tone/style guidelines
   - Claude generates narrative text with specific data citations
   - Requires `ANTHROPIC_API_KEY` as a GitHub Secret

3. **Visualization** (matplotlib/plotly):
   - Generate charts as PNG/SVG embedded in report
   - Consistent styling matching dashboard aesthetic (dark theme)
   - Product images pulled from RTINGS (cached)

4. **PDF generation** (weasyprint or reportlab):
   - HTML template with CSS styling → PDF
   - Nanosys/company branding (colors, logo if provided)
   - Professional layout: cover page, table of contents, sections, charts inline

5. **Delivery**:
   - Email PDF attachment to jeff@jeffyurek.com via existing Gmail SMTP
   - Also save PDF to `data/reports/` (gitignored, large binaries)
   - Summary excerpt in email body for quick scanning

**Dependencies to add:**
- `anthropic` (Claude API for narrative generation)
- `weasyprint` (HTML → PDF) or `fpdf2` (lighter weight)
- Possibly `jinja2` (HTML templating, already installed via streamlit)

**Secrets to add:**
- `ANTHROPIC_API_KEY` for Claude API calls

**Cost estimate:**
- Claude API: ~$0.50-2.00 per report (Sonnet for analysis, ~10k input + 5k output tokens per section)
- Runs once per month, so ~$6-24/year

**Report length target:** 4-8 pages PDF, ~2000-3000 words + 6-10 charts

---

### 5. Panasonic W95A classification
**Priority: Low | Effort: Low**

SPD analysis shows CdSe-like FWHM (green 23.7nm, red 17.6nm) despite Panasonic's no-Cadmium policy. Ground truth says "Cd-Free" but spectral signature is clearly CdSe. Could be a legacy "Hyperion" hybrid material (green InP + red CdSe) but both peaks are narrow, suggesting pure CdSe.

**Action:** Flag as anomaly in dashboard; revisit if Panasonic confirms material choice.

---

### 6. WLED red FWHM still displaying narrow on dashboard
**Priority: Medium | Effort: Medium**

Samsung U8000F and other WLEDs still show red FWHM under 25nm on the dashboard despite the HWHM mirroring fix. The correction works at the `spd_analyzer.py` level (76-86nm) but the narrow values persist in the displayed data. Needs investigation into whether:
- The corrected values aren't propagating through `build_schema.py` to `tv_database.csv`
- The dashboard is reading a stale CSV
- The HWHM mirroring threshold needs further tuning for certain WLED spectra

**Action:** Trace the data flow from `spd_analysis_results.csv` → `tv_database.csv` → dashboard display. Verify the corrected FWHM values are present at each stage.

---

## Completed

### FWHM overlap correction (2026-02-19)
Added HWHM mirroring for overlapping peaks in `spd_analyzer.py`. WLED red FWHM corrected from 24-34nm (artificially narrow due to prominence-based measurement) to 76-86nm (physically correct). QD-LCD measurements unchanged.

### Pipeline code review fixes (2026-02-19)
- Fixed timeout log message (10 → 20 min)
- Removed unused import
- Guarded pricing pipeline against empty snapshot IndexError

### Security + deployment (prior session)
- Password gate on dashboard
- Removed RTINGS branding
- Email notifications via Gmail SMTP
- GitHub Actions weekly automation
