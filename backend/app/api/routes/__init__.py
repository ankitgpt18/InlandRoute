"""
AIDSTL Project — API Routes Package
=====================================
Aggregates all route sub-modules and exposes a single ``api_router``
that is mounted in ``main.py`` under the ``/api/v1`` prefix.

Route prefixes (relative to /api/v1)
--------------------------------------
  /navigability   — navigability prediction endpoints
  /segments       — river segment data & history
  /alerts         — risk alert management
  /analytics      — trend analysis & model performance
"""

from app.api.routes.alerts import router as alerts_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.navigability import router as navigability_router
from app.api.routes.segments import router as segments_router
from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Top-level v1 router
# ---------------------------------------------------------------------------

api_router = APIRouter()

api_router.include_router(navigability_router)
api_router.include_router(segments_router)
api_router.include_router(alerts_router)
api_router.include_router(analytics_router)

__all__ = ["api_router"]
