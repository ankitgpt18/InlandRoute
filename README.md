# InlandRoute 🚢

> **Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Deep Learning**

InlandRoute is a deep learning system for predicting the navigability of India's National Inland Waterways using multitemporal Sentinel-2 satellite imagery combined with hydrological gauge data. It helps in route planning by providing monthly navigability maps, seasonal calendars, and real-time risk alerts.

## Tech Stack 🛠️

- **Backend:** FastAPI, PostgreSQL + PostGIS, Redis, Celery
- **Machine Learning:** PyTorch, LightGBM, Swin Transformer
- **Geospatial:** Google Earth Engine (Sentinel-2), GeoPandas
- **Frontend:** Next.js, Mapbox GL, Tailwind CSS
- **Infrastructure:** Docker, Nginx

## Quick Start 🚀

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankitgpt18/aidstl.git
   cd aidstl
   ```

2. **Environment Setup**
   Copy the example environment file and add your credentials:
   ```bash
   cp backend/.env.example backend/.env
   ```
   *Make sure to add your database URL, Google Earth Engine service account info, and Mapbox token.*

3. **Run with Docker**
   ```bash
   docker compose up --build -d
   ```

4. **Access the App**
   - **Frontend:** http://localhost
   - **API Docs:** http://localhost/docs

## License 📜
This project is licensed under the MIT License.