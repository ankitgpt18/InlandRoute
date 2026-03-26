# Presentation Content — 8 Slides
## Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Machine Learning

**Presented by:** Dev Yadav · Chakshu Vashisth · Ankit Gupta
**Affiliation:** Gati Shakti Vishwavidyalaya, Vadodara

---

## Slide 1 — Title Slide

**Title:** Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Machine Learning

**Subtitle:** A Data-Driven Approach to Unlock India's Inland Waterway Potential

**Presented by:**
- Dev Yadav
- Chakshu Vashisth
- Ankit Gupta

Gati Shakti Vishwavidyalaya, Vadodara, India

---

## Slide 2 — The Problem: Why This Matters

**India's inland waterways are vastly underutilised.**

- India has ~14,500 km of navigable rivers and canals — yet only **2%** carries meaningful cargo.
- Compare: **EU uses 44%** of its waterways for freight, the **US uses 25%**.
- Government declared **111 National Waterways** in 2016, but most remain unsurveyed.

**The bottleneck? Navigability assessment.**

- Vessels need a **minimum 2.5–3.0 m depth** to operate safely.
- Current method: **Monthly echo-sounder surveys by IWAI** — slow, expensive, covers tiny segments.
- Result: planners and logistics operators **lack reliable, timely channel data**.

> *Without knowing whether a river stretch is navigable, water transport cannot grow.*

---

## Slide 3 — Our Solution: What We Propose

**A satellite-powered, machine learning framework to predict navigability — continuously, affordably, at scale.**

**Core Idea:**
- Use **free Sentinel-2 satellite imagery** (10 m resolution, every 5 days) to extract water depth, width, and turbidity signals.
- Combine with **CWC gauge data** and **ERA5 climate data** as ground truth.
- Train **ML models (Random Forest, XGBoost, LightGBM)** to estimate depth and classify navigability.

**Outputs:**
1. **Navigability Maps** — colour-coded segments (Navigable / Conditional / Non-Navigable)
2. **Seasonal Navigability Calendar** — month-by-month usability outlook
3. **Risk Alerts** — early warning for segments at risk of depth drop

**Study Areas:** NW-1 (Ganga: Varanasi–Haldia) and NW-2 (Brahmaputra: Dhubri–Sadiya)

---

## Slide 4 — How It Works: Methodology Overview

**Five-stage pipeline:**

| Stage | What We Do | Key Detail |
|-------|-----------|------------|
| **1. Study Area Selection** | Choose NW-1 (Ganga) and NW-2 (Brahmaputra) | River segmented into 5 km analysis units |
| **2. Data Acquisition** | Gather satellite imagery, gauge data, climate data | Sentinel-2, Landsat, CWC, ERA5, SRTM DEM |
| **3. Feature Engineering** | Extract spectral indices, depth proxies, width, turbidity | MNDWI, NDWI, Stumpf band ratio, channel width |
| **4. Model Development** | Train depth estimation & navigability classifiers | RF, XGBoost, LightGBM with spatial cross-validation |
| **5. Output Generation** | Produce maps, calendars, and risk alerts | Interactive GeoJSON maps via Folium/Plotly |

**Key Innovation:** We don't just estimate depth — we classify each river segment into **three actionable navigability classes** every month.

---

## Slide 5 — Technical Approach: Models & Features

**Task A — Depth Estimation (Regression)**
- Models: Random Forest (500 trees), XGBoost (1000 estimators), LightGBM
- Training labels: CWC gauge depth readings matched with Sentinel-2 composites
- Validation: 5-fold spatial block cross-validation to prevent data leakage
- Target: **R² > 0.85, RMSE < 2.0 m**

**Task B — Navigability Classification**
- Three classes based on depth, width, and obstruction:
  - 🟢 **Navigable:** Depth ≥ 3.0 m, Width ≥ 50 m
  - 🟡 **Conditional:** Depth 2.0–3.0 m or Width 30–50 m
  - 🔴 **Non-Navigable:** Depth < 2.0 m or Width < 30 m

**Feature Set:** Spectral reflectance (B2–B8), water indices (MNDWI, NDWI), Stumpf depth ratio, turbidity proxy, surface width, seasonal variability, rainfall, discharge, channel sinuosity

---

## Slide 6 — What Makes This Project Unique

**Three key contributions:**

**1. Cost & Scale Advantage**
- A physical echo-sounder survey of 200 km costs **₹8–12 lakh and 7–10 days**.
- Our satellite-based approach: **near-zero marginal cost, results within hours**, covering thousands of kilometres.

**2. Filling a Critical Research Gap**
- International studies focus on **clear coastal waters** — our work addresses **turbid, monsoon-fed Indian rivers** with heavy sediment loads.
- No existing study integrates depth estimation + water extent monitoring + navigability classification into **one unified pipeline** for inland corridors.

**3. Direct Policy Relevance**
- Supports the **National Logistics Policy** goal: reduce logistics cost from 13% to 8% of GDP.
- Helps IWAI and logistics operators make **data-driven route planning decisions**.
- Backs the **Maritime India Vision 2030** target of 200 MTPA via waterways.

---

## Slide 7 — Project Timeline & Milestones

**Duration: 18 Months**

| Phase | Activity | Duration | Milestone |
|-------|----------|----------|-----------|
| WP1 | Literature Survey & Study Area Finalisation | Months 1–3 | ✅ Study area maps ready |
| WP2 | Data Acquisition & Preprocessing Pipeline | Months 3–6 | ✅ Clean Sentinel-2 archive on GEE |
| WP3 | Feature Engineering & EDA | Months 5–8 | ✅ Feature matrix prepared |
| WP4 | Model Training & Benchmarking | Months 7–12 | ✅ Depth model with R² > 0.85 |
| WP5 | Navigability Classification & Mapping | Months 10–14 | ✅ Interactive maps & calendars |
| WP6 | Validation, Documentation & Paper | Months 13–18 | ✅ Final report & manuscript |

**Validation at three levels:**
1. Point-level — predicted depth vs. held-out CWC gauge readings
2. Segment-level — navigability class vs. IWAI LAD bulletins
3. Temporal — model trained on 2019–23, tested on 2024 data

---

## Slide 8 — Expected Impact & Conclusion

**What success looks like:**

- A **working prototype** that generates monthly navigability maps for NW-1 and NW-2 from freely available satellite data.
- **Seasonal navigability calendars** that tell operators exactly when and where they can sail.
- **Validated benchmarks** for ML-based depth estimation on Indian rivers — a first of its kind.

**Broader impact:**

- Scalable to all **111 National Waterways** with minimal adaptation.
- Can reduce IWAI's survey costs by **up to 60–70%** while increasing coverage.
- Supports India's vision of making **inland waterways a viable freight corridor**, reducing road congestion, emissions, and logistics cost.

**In one line:** *We're turning satellite pixels into navigable pathways — making India's rivers work for its economy.*

---

**Thank You**
*Questions & Discussion*

Dev Yadav · Chakshu Vashisth · Ankit Gupta
Gati Shakti Vishwavidyalaya, Vadodara
