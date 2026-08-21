import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import 'app'
sys.path.append(str(Path(__file__).resolve().parent))

import ee
import geopandas as gpd
from sqlalchemy.ext.asyncio import AsyncSession
from shapely.geometry import shape, LineString, mapping

from app.core.config import get_settings
from app.core.database import get_engine, get_session_factory, Base, init_db
from app.models.domain import Segment
from app.utils.spatial import _ensure_single_linestring, _to_utm, _to_wgs84, _split_line_at_distance, make_segment_id

settings = get_settings()

# GEE Assets
ASSETS = {
    "NW-1": f"projects/{settings.GEE_PROJECT_ID}/assets/nw1_centreline",
    "NW-2": f"projects/{settings.GEE_PROJECT_ID}/assets/nw2_centreline"
}

def init_gee():
    """Authenticate with GEE."""
    key_file = settings.gee_key_file_path
    if not key_file.exists():
        raise FileNotFoundError(f"GEE key file not found: {key_file}")
    
    credentials = ee.ServiceAccountCredentials(
        email=settings.GEE_SERVICE_ACCOUNT,
        key_file=str(key_file),
    )
    ee.Initialize(credentials=credentials, project=settings.GEE_PROJECT_ID)
    print(f"Authenticated with GEE as {settings.GEE_SERVICE_ACCOUNT}")

def fetch_gee_geometry(asset_id):
    """Fetch LineString geometry from GEE asset."""
    fc = ee.FeatureCollection(asset_id)
    # Union all features into a single geometry
    merged = fc.geometry()
    info = merged.getInfo()
    geom = shape(info)
    return _ensure_single_linestring(geom)

async def seed_waterway(waterway_id: str, session: AsyncSession):
    """Fetch, segment and seed a waterway."""
    asset_id = ASSETS.get(waterway_id)
    print(f"Processing {waterway_id} from {asset_id}...")
    
    try:
        # 1. Fetch from GEE
        centreline = fetch_gee_geometry(asset_id)
        
        # 2. Segment (Reusing spatial.py logic)
        centreline_utm = _to_utm(centreline)
        segment_length_m = settings.SEGMENT_LENGTH_KM * 1000.0
        
        segments_utm = _split_line_at_distance(centreline_utm, segment_length_m)
        print(f"Generated {len(segments_utm)} segments for {waterway_id}.")
        
        # 3. Insert into DB
        chainage = 0.0
        for idx, seg_utm in enumerate(segments_utm):
            seg_len_km = seg_utm.length / 1000.0
            seg_wgs84 = _to_wgs84(seg_utm)
            centroid = seg_wgs84.centroid
            
            segment_id = make_segment_id(waterway_id, idx)
            
            # Create Segment instance
            # Note: geom expects WKB or a string, but geoalchemy2 handles shapely objects
            db_seg = Segment(
                id=segment_id,
                waterway_id=waterway_id,
                segment_index=idx,
                chainage_start_km=round(chainage, 3),
                chainage_end_km=round(chainage + seg_len_km, 3),
                length_km=round(seg_len_km, 4),
                geom=f"SRID=4326;{seg_wgs84.wkt}",
                centroid_lon=round(centroid.x, 6),
                centroid_lat=round(centroid.y, 6),
                sinuosity=1.0, # Default for now
                meta={}
            )
            session.add(db_seg)
            chainage += seg_len_km
            
        await session.flush()
        print(f"Successfully staged {len(segments_utm)} segments for {waterway_id}.")
        
    except Exception as e:
        print(f"Error seeding {waterway_id}: {e}")
        raise

async def main():
    # Ensure tables are created
    print("Initialising database and extensions...")
    await init_db()
    
    # Authenticate GEE
    print("Initialising GEE...")
    init_gee()
    
    engine = get_engine()
    async with engine.begin() as conn:
        # Create table if not exists (development shortcut)
        # In production we'd use Alembic
        from sqlalchemy import text
        print("Ensuring 'segments' table exists...")
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Clear existing data for these waterways to avoid duplicates
        for wid in ASSETS.keys():
            from sqlalchemy import delete
            print(f"Cleaning existing segments for {wid}...")
            await session.execute(delete(Segment).where(Segment.waterway_id == wid))
            
            await seed_waterway(wid, session)
            
        print("Committing changes...")
        await session.commit()
    
    print("Seeding complete!")

if __name__ == "__main__":
    asyncio.run(main())
