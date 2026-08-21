"""
AIDSTL Project — API Package
==============================
Assembles all route modules into a single APIRouter that is mounted
in main.py under the /api/v1 prefix.
"""

from app.api.routes import alerts, analytics, health, navigability, segments
from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Top-level v1 router
# ---------------------------------------------------------------------------

api_router = APIRouter()

# Health & observability (no version prefix — mounted directly at root)
api_router.include_router(health.router)

# Domain routes
api_router.include_router(navigability.router)
api_router.include_router(segments.router)
api_router.include_router(alerts.router)
api_router.include_router(analytics.router)


__all__ = ["api_router"]
