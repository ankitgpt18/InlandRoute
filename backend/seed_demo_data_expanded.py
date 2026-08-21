import asyncio
import json
import random
from sqlalchemy import text
from app.core.database import get_engine, get_session_factory
from app.models.domain import Segment
from shapely.geometry import Point, LineString, mapping
from geoalchemy2.shape import from_shape

async def seed_demo_data():
    engine = get_engine()
    session_factory = get_session_factory()
    print("Seeding expanded demo data for NW-1...")
    
    # Path coordinates (Lat, Lon)
    # Varanasi, Patna, Farakka, Haldia
    waypoints = [
        (25.3176, 83.0062),
        (25.5941, 85.1376),
        (24.8143, 87.8967),
        (22.0305, 88.1102)
    ]
    
    # Generate 100 segments total
    num_segments = 100
    points_per_leg = num_segments // (len(waypoints) - 1)
    
    all_segments_data = []
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i+1]
        
        for j in range(points_per_leg):
            t = j / points_per_leg
            lat = start[0] + (end[0] - start[0]) * t
            lon = start[1] + (end[1] - start[1]) * t
            
            next_t = (j + 1) / points_per_leg
            n_lat = start[0] + (end[0] - start[0]) * next_t
            n_lon = start[1] + (end[1] - start[1]) * next_t
            
            geom = LineString([(lon, lat), (n_lon, n_lat)])
            
            # Random features for meta
            meta = {
                "blue": random.uniform(0.01, 0.05),
                "green": random.uniform(0.05, 0.15),
                "red": random.uniform(0.02, 0.08),
                "nir": random.uniform(0.1, 0.3),
                "swir1": random.uniform(0.01, 0.1),
                "swir2": random.uniform(0.01, 0.05),
                "water_width_m": random.uniform(100, 800),
                "gauge_discharge_m3s": random.uniform(2000, 15000),
                "sinuosity": random.uniform(1.0, 1.3),
            }
            
            all_segments_data.append(Segment(
                id=f"NW1-SEG-{len(all_segments_data) + 1:03d}",
                waterway_id="NW-1",
                segment_index=len(all_segments_data) + 1,
                chainage_start_km=len(all_segments_data) * 14.0,
                chainage_end_km=(len(all_segments_data) + 1) * 14.0,
                length_km=14.0,
                geom=from_shape(geom, srid=4326),
                centroid_lon=(lon + n_lon) / 2,
                centroid_lat=(lat + n_lat) / 2,
                meta=meta
            ))

    async with session_factory() as session:
        # Clear existing
        await session.execute(text("DELETE FROM segments WHERE waterway_id = 'NW-1'"))
        
        session.add_all(all_segments_data)
        await session.commit()
            
    print(f"Successfully seeded {len(all_segments_data)} segments.")

if __name__ == "__main__":
    asyncio.run(seed_demo_data())
