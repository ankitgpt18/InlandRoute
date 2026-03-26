# AIDSTL — Inland Waterway Navigability Prediction System

> **Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-GEE-4285F4?logo=google-earth)](https://developers.google.com/earth-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Study Areas](#study-areas)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [ML Model Performance](#ml-model-performance)
- [Data Sources](#data-sources)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Project Overview

AIDSTL is a **production-ready deep learning system** for predicting the navigability of India's National Inland Waterways using multitemporal Sentinel-2 satellite imagery combined with hydrological gauge data.

The system segments each river into **5-km analysis units** and classifies each segment into one of three navigability classes following IWAI (Inland Waterways Authority of India) standards:

| Class | Min Depth | Min Width | Description |
|---|---|---|---|
| `navigable` | ≥ 3.0 m | ≥ 50 m | Full commercial navigation |
| `conditional` | ≥ 1.5 m | ≥ 25 m | Shallow-draft vessels only |
| `non_navigable` | < 1.5 m | < 25 m | Navigation not recommended |

**Key capabilities:**
- Monthly navigability maps across the full waterway length
- 12-month seasonal navigation calendars for operational planning
- Real-time risk alerts with webhook delivery
- Longitudinal depth profiles for route planning
- Multi-year trend analysis and anomaly detection
- SHAP explainability for every prediction

---

## Study Areas

```
INDIA — National Waterways
===========================

  NW-1 : Ganga River
  ┌─────────────────────────────────────────────────────────────────┐
  │  Varanasi ──────────────────────────────────────────► Haldia   │
  │  (25.3°N, 83.0°E)              ~1,620 km             (22.0°N,  │
  │                                                        88.1°E)  │
  │  ~324 segments × 5 km                                           │
  └─────────────────────────────────────────────────────────────────┘

  NW-2 : Brahmaputra River
  ┌─────────────────────────────────────────────────────────────────┐
  │  Dhubri ────────────────────────────────────────────► Sadiya   │
  │  (26.0°N, 90.0°E)               ~891 km             (27.8°N,  │
  │                                                        95.6°E)  │
  │  ~178 segments × 5 km                                           │
  └─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AIDSTL System Architecture                             │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Sentinel-2 │    │   CWC Gauge  │    │     IMD      │
  │  (via GEE)   │    │    Stations  │    │ Precipitation│
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │                  GEE Feature Extraction                  │
  │  • Cloud-masked monthly S2 composites                    │
  │  • Spectral indices: MNDWI, NDWI, AWEIsh, Stumpf ratio  │
  │  • Zonal statistics per 5-km segment AOI                 │
  └─────────────────────────┬────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   ML Inference Pipeline                  │
  │                                                          │
  │  ┌───────────────────────┐  ┌──────────────────────────┐ │
  │  │  Temporal Fusion      │  │    Swin Transformer      │ │
  │  │  Transformer (TFT)   │  │   (Water Extent)         │ │
  │  │                       │  │                          │ │
  │  │  Input:               │  │  Input:                  │ │
  │  │  12-mo spectral +     │  │  S2 patches              │ │
  │  │  hydrological series  │  │  (10 bands, 64×64 px)    │ │
  │  │                       │  │                          │ │
  │  │  Output:              │  │  Output:                 │ │
  │  │  depth (q10,pt,q90)   │  │  water_frac + width_m    │ │
  │  └──────────┬────────────┘  └───────────┬──────────────┘ │
  │             │                           │                 │
  │             └───────────┬───────────────┘                 │
  │                         │                                 │
  │                         ▼                                 │
  │           ┌─────────────────────────┐                     │
  │           │  Ensemble Combination   │                     │
  │           │  TFT 65% + Swin 35%    │                     │
  │           └─────────────┬───────────┘                     │
  │                         │                                 │
  │                         ▼                                 │
  │           ┌─────────────────────────┐                     │
  │           │  LightGBM Classifier    │                     │
  │           │  navigable / conditional│                     │
  │           │  / non_navigable        │                     │
  │           │  + SHAP explanations    │                     │
  │           └─────────────┬───────────┘                     │
  └─────────────────────────┼──────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   FastAPI Backend                        │
  │                                                          │
  │  /api/v1/navigability/{id}/map       → NavigabilityMap   │
  │  /api/v1/navigability/{id}/calendar  → SeasonalCalendar  │
  │  /api/v1/navigability/{id}/depth-profile → DepthProfile  │
  │  /api/v1/navigability/{id}/stats     → WaterwayStats     │
  │  /api/v1/navigability/predict        → Single prediction │
  │  /api/v1/navigability/predict/batch  → Batch prediction  │
  │  /api/v1/segments/{id}/history       → Historical data   │
  │  /api/v1/segments/{id}/features      → Spectral features │
  │  /api/v1/alerts/{id}                 → Risk alerts       │
  │  /api/v1/analytics/trends/{id}       → Trend analysis    │
  │  /health                             → Health probes     │
  │  /metrics                            → Prometheus        │
  └─────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  PostgreSQL  │  │    Redis     │  │    Celery    │
  │  + PostGIS   │  │   Cache      │  │    Worker    │
  │  (segments,  │  │  (6h TTL     │  │  (batch jobs)│
  │   history)   │  │   pred cache)│  │              │
  └──────────────┘  └──────────────┘  └──────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    Next.js Frontend                      │
  │  • Mapbox GL navigability map                            │
  │  • Seasonal calendar heat-map                            │
  │  • Longitudinal depth profile chart                      │
  │  • Risk alert dashboard                                  │
  │  • Analytics & trend visualisations                      │
  └──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Web Framework | FastAPI | ≥ 0.110 | Async REST API |
| Server | Uvicorn + uvloop | ≥ 0.27 | ASGI server |
| Data Validation | Pydantic v2 | ≥ 2.6 | Schema validation |
| ORM | SQLAlchemy (async) | ≥ 2.0 | Database access |
| Database | PostgreSQL + PostGIS | 16 / 3.4 | Geospatial persistence |
| Caching | Redis | 7 | Prediction cache (6h TTL) |
| Task Queue | Celery | ≥ 5.3 | Batch inference jobs |

### Machine Learning

| Component | Technology | Version | Role |
|---|---|---|---|
| Deep Learning | PyTorch | ≥ 2.2 | Model training & inference |
| TFT Architecture | timm | ≥ 0.9 | Swin Transformer backbone |
| Gradient Boosting | LightGBM + XGBoost | ≥ 4.3 / 2.0 | Navigability classifier |
| Explainability | SHAP | — | Feature importance |
| Numerics | NumPy + Pandas | ≥ 1.26 / 2.2 | Data processing |
| ML Utilities | scikit-learn | ≥ 1.4 | Preprocessing, metrics |
| Persistence | joblib | ≥ 1.3 | Model serialisation |

### Geospatial

| Component | Technology | Version | Role |
|---|---|---|---|
| Remote Sensing | Google Earth Engine | ≥ 0.1.390 | Sentinel-2 composites |
| Vector GIS | GeoPandas + Shapely | ≥ 0.14 / 2.0 | Segment geometry |
| CRS | PyProj | ≥ 3.6 | Coordinate transforms |
| Satellite Data | Sentinel-2 L2A SR | — | 10-band multispectral |

### Infrastructure

| Component | Technology | Version | Role |
|---|---|---|---|
| Containerisation | Docker + Compose | — | Service orchestration |
| Reverse Proxy | Nginx | 1.25 | Load balancing, SSL |
| Monitoring | Prometheus + Grafana | — | Metrics & dashboards |
| Logging | structlog | ≥ 24.1 | Structured JSON logs |
| Observability | prometheus-fastapi-instrumentator | ≥ 7.0 | HTTP metrics |
| Storage | AWS S3 / MinIO | — | Model artefact storage |

### Frontend

| Component | Technology | Purpose |
|---|---|---|
| Framework | Next.js 14 | React SSR / SSG |
| Maps | Mapbox GL JS | Navigability map rendering |
| Charts | Recharts / D3.js | Depth profiles, trends |
| Styling | Tailwind CSS | UI design system |

---

## ML Model Performance

All metrics are computed on a **geographically stratified hold-out test set** (NW-1 + NW-2, 2022–2023, n = 2,847 station-months).

### Depth Prediction — TFT + Swin Ensemble

| Metric | Value | Description |
|---|---|---|
| **RMSE** | **0.312 m** | Root-mean-square error vs. CWC gauges |
| **MAE** | 0.241 m | Mean absolute error |
| **R²** | 0.874 | Coefficient of determination |
| **MAPE** | 8.7 % | Mean absolute percentage error |
| **Pearson r** | 0.935 | Correlation with gauge observations |
| **PI90 Coverage** | 89.1 % | Coverage of 90% prediction interval |
| **PI90 Width** | 1.02 m | Mean interval width |
| **Bias** | −0.031 m | Systematic underestimation |

### Navigability Classification — LightGBM

| Metric | Value |
|---|---|
| **Accuracy** | **89.6 %** |
| **Macro F1** | 0.881 |
| **Weighted F1** | 0.893 |
| **Cohen's Kappa** | 0.834 |
| Navigable ROC-AUC | 0.972 |
| Conditional ROC-AUC | 0.931 |
| Non-navigable ROC-AUC | 0.958 |

### Water Extent Segmentation — Swin Transformer

| Metric | Value |
|---|---|
| **Water IoU** | **0.863** |
| **Water F1** | 0.927 |
| Pixel Accuracy | 95.2 % |
| Width MAE | 18.4 m |

### Benchmark Comparisons

| Method | Depth RMSE | vs. AIDSTL |
|---|---|---|
| AIDSTL Ensemble | **0.312 m** | baseline |
| Gauge-only linear regression | 0.474 m | −34.2 % |
| Stumpf empirical model | 0.403 m | −22.7 % |
| TFT alone | 0.358 m | −12.8 % |

---

## Data Sources

| Source | Data | Temporal Coverage | Spatial Resolution |
|---|---|---|---|
| **Copernicus Sentinel-2** | Multispectral imagery (10 bands) | 2016–present | 10–60 m |
| **CWC** (Central Water Commission) | Daily water level & discharge | 2010–present | 23 gauge stations |
| **IWAI** (Inland Waterways Authority) | Navigability assessment records | 2016–2023 | Segment-level |
| **IMD** (India Meteorological Dept.) | Daily precipitation grids | 2010–present | 0.25° × 0.25° |
| **SRTM** | Digital Elevation Model | — | 30 m |
| **Survey of India** | River centreline shapefiles | — | 1:50,000 |

---

## Quick Start

### Prerequisites

- **Docker ≥ 24** and **Docker Compose ≥ 2.20**
- **4 GB RAM** minimum (8 GB recommended for ML inference)
- A **Google Earth Engine** service account with Earth Engine API enabled
- A **Mapbox** public token (for the frontend map)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/aidstl.git
cd aidstl

# Copy the environment template and fill in your values
cp backend/.env.example backend/.env
```

Open `backend/.env` and fill in at minimum:

```bash
# Required
DATABASE_URL=postgresql+asyncpg://aidstl_user:your_password@db:5432/aidstl_db
POSTGRES_PASSWORD=your_password

# GEE credentials
GEE_SERVICE_ACCOUNT=your-sa@your-project.iam.gserviceaccount.com
GEE_KEY_FILE=/app/secrets/gee_service_account_key.json

# Place your GEE JSON key here:
mkdir -p backend/secrets
cp /path/to/your/gee_key.json backend/secrets/gee_service_account_key.json

# Mapbox (for frontend map)
MAPBOX_TOKEN=pk.your_mapbox_token
```

### 2. Build and launch

```bash
# Build all images and start all services in the background
docker compose up --build -d

# Follow API logs
docker compose logs -f api

# Follow all logs
docker compose logs -f
```

### 3. Verify

```bash
# Liveness probe (should return 200 immediately)
curl http://localhost/health/live

# Full health check (DB + Redis + models)
curl http://localhost/health | python -m json.tool

# GEE connectivity
curl http://localhost/health/gee | python -m json.tool

# Model status
curl http://localhost/health/models | python -m json.tool
```

### 4. Run your first prediction

```bash
# Get the navigability map for NW-1 (Ganga) in July 2024
curl "http://localhost/api/v1/navigability/NW-1/map?month=7&year=2024" \
  | python -m json.tool | head -60

# Get the seasonal calendar for NW-2 (Brahmaputra) for 2024
curl "http://localhost/api/v1/navigability/NW-2/calendar?year=2024" \
  | python -m json.tool | head -80

# Predict a single segment
curl -X POST "http://localhost/api/v1/navigability/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "segment": {
      "segment_id": "NW-1-042",
      "waterway_id": "NW-1",
      "month": 7,
      "year": 2024
    },
    "return_shap": true,
    "return_features": true
  }' | python -m json.tool
```

### 5. Access the UI

| Service | URL | Description |
|---|---|---|
| **Frontend** | http://localhost | Next.js map dashboard |
| **API Docs** | http://localhost/docs | Swagger UI (try it out) |
| **ReDoc** | http://localhost/redoc | Alternative API docs |
| **Prometheus** | http://localhost/metrics | Raw metrics |
| **pgAdmin** | — | Connect to `localhost:5432` |

### Docker Compose Commands

```bash
# Start all services
docker compose up -d

# Stop all services (keep volumes)
docker compose down

# Stop and remove all volumes (DESTRUCTIVE)
docker compose down -v

# Rebuild a single service
docker compose build api && docker compose up -d api

# Scale the API (without nginx port conflicts)
docker compose up -d --scale api=3

# Run database migrations (Alembic)
docker compose exec api alembic upgrade head

# Open a Python shell inside the API container
docker compose exec api python

# View resource usage
docker compose stats
```

---

## API Documentation

Full interactive documentation is available at **http://localhost/docs** after startup.

### Key Endpoints

```
Navigability
  GET  /api/v1/navigability/{waterway_id}/map
       ?month=7&year=2024
       → NavigabilityMap (all 5-km segments + summary stats)

  GET  /api/v1/navigability/{waterway_id}/calendar
       ?year=2024
       → SeasonalCalendar (12-month outlook per segment)

  GET  /api/v1/navigability/{waterway_id}/depth-profile
       ?month=7&year=2024
       → DepthProfile (longitudinal depth + bottlenecks)

  GET  /api/v1/navigability/{waterway_id}/stats
       ?year=2024
       → WaterwayStats (annual summary + monthly breakdown)

  GET  /api/v1/navigability/{waterway_id}/historical-comparison
       ?month=7&year=2024&base_years=5
       → HistoricalComparison (anomaly vs. baseline)

  POST /api/v1/navigability/predict
       Body: PredictionRequest
       → NavigabilityPrediction (single segment)

  POST /api/v1/navigability/predict/batch
       Body: BatchPredictionRequest (up to 500 segments)
       → list[NavigabilityPrediction] | TaskStatus (async)

Segments
  GET  /api/v1/segments/{waterway_id}
       → SegmentListResponse (all segments + latest navigability)

  GET  /api/v1/segments/{segment_id}/history
       ?years=3
       → SegmentHistoryResponse (multi-year monthly records)

  GET  /api/v1/segments/{segment_id}/features
       ?month=7&year=2024
       → SegmentFeaturesResponse (Sentinel-2 spectral features)

  GET  /api/v1/segments/{segment_id}/profile
       → NavigabilityPrediction (full prediction for one segment)

  GET  /api/v1/segments/{waterway_id}/geojson
       → GeoJSON FeatureCollection (for QGIS / Mapbox import)

Alerts
  GET  /api/v1/alerts/{waterway_id}
       ?month=7&year=2024&severity=critical
       → list[RiskAlert]

  GET  /api/v1/alerts/critical
       → list[RiskAlert] (CRITICAL across all waterways)

  GET  /api/v1/alerts/next-month-risk/{segment_id}
       → NextMonthRiskResponse (EWM risk forecast)

  POST /api/v1/alerts/subscribe
       Body: AlertSubscription
       → WebhookSubscriptionResponse

  POST /api/v1/alerts/{alert_id}/acknowledge
       → AcknowledgeResponse

Analytics
  GET  /api/v1/analytics/trends/{waterway_id}
       ?years=5
       → WaterwayTrendAnalysis (linear regression + significance)

  GET  /api/v1/analytics/seasonal-patterns/{waterway_id}
       → SeasonalPatternAnalysis (monthly climatology)

  GET  /api/v1/analytics/model-performance
       → ModelPerformanceReport

  GET  /api/v1/analytics/feature-importance
       ?model=classifier
       → FeatureImportanceReport (SHAP values)

  GET  /api/v1/analytics/segment-ranking/{waterway_id}
       ?ranked_by=navigable_pct&year=2024
       → SegmentRankingReport (top-10 / bottom-10)

  GET  /api/v1/analytics/anomaly-detection/{waterway_id}
       ?current_year=2024&z_threshold=2.0
       → AnomalyDetectionReport

Health
  GET  /health              → Overall health (DB + Redis + models)
  GET  /health/live         → Liveness probe (lightweight)
  GET  /health/ready        → Readiness probe (all deps checked)
  GET  /health/models       → ML model loading status
  GET  /health/gee          → GEE connectivity
  GET  /health/db           → PostgreSQL + PostGIS
  GET  /health/redis        → Redis connectivity
  GET  /health/info         → App metadata + config summary
  GET  /metrics             → Prometheus metrics
```

---

## Configuration

All configuration is managed through environment variables loaded from `backend/.env`.

See **`backend/.env.example`** for the full annotated list.

### Critical variables

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Async PostgreSQL DSN | `postgresql+asyncpg://user:pass@db:5432/aidstl_db` |
| `REDIS_URL` | Redis connection | `redis://redis:6379/0` |
| `GEE_SERVICE_ACCOUNT` | GEE service account email | `sa@project.iam.gserviceaccount.com` |
| `GEE_KEY_FILE` | Path to GEE JSON key | `/app/secrets/gee_key.json` |
| `MODEL_DIR` | Root directory for model files | `./ml/models/saved` |
| `MAPBOX_TOKEN` | Mapbox public token | `pk.eyJ1Ij...` |
| `SECRET_KEY` | JWT signing key (≥ 64 chars) | *(generate with `python -c "import secrets; print(secrets.token_hex(64))"`)* |
| `ENVIRONMENT` | Runtime environment | `development` \| `staging` \| `production` |

### Navigability thresholds (IWAI standards)

| Variable | Default | Description |
|---|---|---|
| `DEPTH_NAVIGABLE_MIN` | `3.0` | Minimum depth (m) for "navigable" class |
| `DEPTH_CONDITIONAL_MIN` | `1.5` | Minimum depth (m) for "conditional" class |
| `WIDTH_NAVIGABLE_MIN` | `50.0` | Minimum width (m) for "navigable" class |
| `WIDTH_CONDITIONAL_MIN` | `25.0` | Minimum width (m) for "conditional" class |
| `RISK_ALERT_THRESHOLD` | `0.7` | Risk score threshold for alert generation |

---

## Project Structure

```
AIDSTL Project/
├── README.md                       ← This file
├── docker-compose.yml              ← Full stack orchestration
│
├── backend/
│   ├── Dockerfile                  ← Multi-stage build (builder + runtime)
│   ├── requirements.txt            ← Python dependencies
│   ├── .env.example                ← Environment variable template
│   │
│   └── app/
│       ├── main.py                 ← FastAPI app factory + lifespan
│       │
│       ├── core/
│       │   ├── config.py           ← pydantic-settings Settings class
│       │   └── database.py         ← Async SQLAlchemy + PostGIS setup
│       │
│       ├── models/
│       │   ├── schemas/
│       │   │   └── navigability.py ← All Pydantic v2 schemas
│       │   └── dl/                 ← PyTorch model class definitions
│       │
│       ├── services/
│       │   ├── model_service.py    ← ML inference singleton
│       │   ├── gee_service.py      ← GEE feature extraction
│       │   ├── navigability_service.py ← Core business logic
│       │   └── alert_service.py    ← Risk alert generation
│       │
│       ├── api/
│       │   └── routes/
│       │       ├── navigability.py ← Prediction endpoints
│       │       ├── segments.py     ← Segment data endpoints
│       │       ├── alerts.py       ← Alert management endpoints
│       │       ├── analytics.py    ← Trend & performance endpoints
│       │       └── health.py       ← Health & readiness probes
│       │
│       └── utils/
│           ├── spatial.py          ← River segmentation + geometry
│           └── spectral.py         ← Sentinel-2 index computation
│
├── frontend/
│   ├── src/
│   │   ├── app/                    ← Next.js App Router pages
│   │   └── components/             ← React components (map, charts)
│   └── public/                     ← Static assets
│
└── ml/
    ├── models/
    │   └── saved/                  ← Pre-trained model artefacts
    │       ├── ensemble/           ← TFT + Swin checkpoint (.pt)
    │       ├── classifier/         ← LightGBM classifier (.joblib)
    │       ├── preprocessors/      ← Feature scalers (.joblib)
    │       └── explainers/         ← SHAP explainers (.joblib)
    ├── notebooks/                  ← Jupyter training notebooks
    └── training/                   ← Training scripts
```

---

## Team

**AIDSTL Research Team**

This project was developed as part of ongoing research into the application of satellite remote sensing and deep learning for operational inland waterway management in India.

**Supervisors / Principal Investigators**
- Remote Sensing & Earth Observation Lead
- Deep Learning Architecture Lead
- Hydrological Modelling Lead

**Acknowledgements**

- [IWAI](https://iwai.nic.in/) — Inland Waterways Authority of India, for navigability records and domain expertise
- [CWC](https://cwc.gov.in/) — Central Water Commission, for gauge station data
- [ESA Copernicus Programme](https://www.copernicus.eu/) — for Sentinel-2 satellite data
- [Google Earth Engine](https://earthengine.google.com/) — for cloud-based geospatial computing

---

## License

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

*Last updated: 2025*