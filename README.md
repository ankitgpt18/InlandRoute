# InlandRoute

> **Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-GEE-4285F4?logo=google-earth)](https://developers.google.com/earth-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)

InlandRoute is an enterprise geospatial platform for predicting the navigability of India's National Inland Waterways (**NW-1 Ganga** and **NW-2 Brahmaputra**) using multitemporal Sentinel-2 satellite imagery combined with CWC hydrological gauge data. It assists IWAI fleet logistics by providing monthly navigability maps, 12-month depth calendars, longitudinal profile charts, and automated risk early warning alerts.

## Tech Stack

- **Backend:** FastAPI, PostgreSQL + PostGIS, Redis, Celery
- **Machine Learning:** PyTorch, LightGBM, Swin Transformer
- **Geospatial:** Google Earth Engine (Sentinel-2), GeoPandas, Leaflet GIS
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS, Recharts
- **Infrastructure:** Docker, Nginx

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankitgpt18/InlandRoute.git
   cd InlandRoute
   ```

2. **Environment Setup**
   Copy the example environment file and configure your settings:
   ```bash
   cp backend/.env.example backend/.env
   ```
   *Note: Includes transparent fallback telemetry data so the platform runs out-of-the-box.*

3. **Run with Docker**
   ```bash
   docker compose up --build -d
   ```

4. **Access the App**
   - **Frontend:** http://localhost:3000
   - **API Docs:** http://localhost:8000/docs

## License
This project is licensed under the MIT License.