import asyncio
import json
import os
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import 'app'
sys.path.append(str(Path(__file__).resolve().parent))

import geopandas as gpd
from sqlalchemy.ext.asyncio import AsyncSession
from shapely.geometry import shape, LineString, Point
from sqlalchemy import delete

from app.core.config import get_settings
from app.core.database import get_engine, get_session_factory, Base, init_db
from app.models.domain import Segment
from app.utils.spatial import _to_utm, _to_wgs84, make_segment_id

settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FILES = {
    "NW-1": PROJECT_ROOT / "ml/training/data/processed/nw1_features.geojson",
    "NW-2": PROJECT_ROOT / "ml/training/data/processed/nw2_features.geojson"
}

async def seed_from_local(waterway_id: str, session: AsyncSession):
    """Seed segments from local point GeoJSON."""
    file_path = Path(LOCAL_FILES[waterway_id])
    if not file_path.exists():
        print(f"Local file not found: {file_path}")
        return

    print(f"Processing {waterway_id} from {file_path}...")
    gdf = gpd.read_file(file_path)
    
    # Filter for a single time slice to get unique segments (the file has multiple years/months per segment)
    # Actually, let's just group by segment_id and take the first geometry
    if 'segment_id' in gdf.columns:
        seg_col = 'segment_id'
    elif 'segment_index' in gdf.columns:
        seg_col = 'segment_index'
    else:
        print("Could not find segment column in GeoJSON")
        return

    # Sort by chainage if possible
    sort_col = 'chainage_km' if 'chainage_km' in gdf.columns else seg_col
    gdf = gdf.sort_values(by=sort_col)

    # Get unique segments by position
    unique_segs = gdf.drop_duplicates(subset=[seg_col])
    
    print(f"Found {len(unique_segs)} unique segments for {waterway_id}.")

    # Collect points to build a LineString
    points = [row.geometry for _, row in unique_segs.iterrows()]
    if len(points) < 2:
        print(f"Not enough points to build a LineString for {waterway_id}")
        return

    full_line = LineString(points)
    
    # Project to UTM to generate 5km segments
    line_utm = _to_utm(full_line)
    seg_len_m = settings.SEGMENT_LENGTH_KM * 1000.0
    
    # For each point, we'll create a 5km segment centred on it (simplified approach)
    # OR we can just join the points. Let's join the points and split.
    
    from app.utils.spatial import _split_line_at_distance
    segments_utm = _split_line_at_distance(line_utm, seg_len_m)
    
    print(f"Generated {len(segments_utm)} LineString segments for {waterway_id}.")

    chainage = 0.0
    for idx, seg_utm in enumerate(segments_utm):
        seg_wgs84 = _to_wgs84(seg_utm)
        centroid = seg_wgs84.centroid
        seg_len_km = seg_utm.length / 1000.0
        
        segment_id = make_segment_id(waterway_id, idx)
        
        # Extract metadata from the closest point in unique_segs
        # (This is an approximation)
        dists = unique_segs.geometry.distance(centroid)
        closest_idx = dists.idxmin()
        closest_row = unique_segs.loc[closest_idx]
        
        meta = closest_row.to_dict()
        # Clean up meta for JSON
        if 'geometry' in meta: del meta['geometry']
        for k, v in meta.items():
            if hasattr(v, 'item'): meta[k] = v.item() # numpy to python

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
            sinuosity=1.0,
            meta=meta
        )
        session.add(db_seg)
        chainage += seg_len_km

    await session.flush()
    print(f"Successfully staged {len(segments_utm)} segments for {waterway_id}.")

async def main():
    print("Initialising database...")
    await init_db()
    
    engine = get_engine()
    async with engine.begin() as conn:
        print("Ensuring 'segments' table exists...")
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        for wid in LOCAL_FILES.keys():
            print(f"Cleaning existing segments for {wid}...")
            await session.execute(delete(Segment).where(Segment.waterway_id == wid))
            await seed_from_local(wid, session)
            
        print("Committing changes...")
        await session.commit()
    
    print("Seeding from local data complete!")

if __name__ == "__main__":
    asyncio.run(main())
