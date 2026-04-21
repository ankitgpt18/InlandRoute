from __future__ import annotations

from typing import Any

from app.core.database import Base, TimestampMixin, StrPKMixin
from sqlalchemy import Float, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

class Segment(Base, TimestampMixin, StrPKMixin):
    """
    SQLAlchemy model for river segments with PostGIS support.
    
    The 'id' (from StrPKMixin) stores the segment_id (e.g., 'NW-1-042').
    """
    __tablename__ = "segments"

    waterway_id: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chainage_start_km: Mapped[float] = mapped_column(Float, nullable=False)
    chainage_end_km: Mapped[float] = mapped_column(Float, nullable=False)
    length_km: Mapped[float] = mapped_column(Float, nullable=False)
    sinuosity: Mapped[float] = mapped_column(Float, server_default="1.0")
    
    # PostGIS geometry column (LineString in EPSG:4326)
    geom: Mapped[Any] = mapped_column(
        Geometry(geometry_type="LINESTRING", srid=4326, spatial_index=True),
        nullable=False,
    )
    
    centroid_lon: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=False)
    
    bed_material: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gauge_station_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    # Store arbitrary metadata (e.g., district, state, bank info)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, server_default="{}")

    def __repr__(self) -> str:
        return f"<Segment(id={self.id}, waterway={self.waterway_id}, km={self.chainage_start_km})>"
