
import asyncio
from app.core.database import get_db_session
from sqlalchemy import text

async def check_db():
    async for session in get_db_session():
        try:
            result = await session.execute(text("SELECT count(*) FROM segments"))
            count = result.scalar()
            print(f"Total segments in DB: {count}")
            
            # Check for specific waterways
            result = await session.execute(text("SELECT waterway_id, count(*) FROM segments GROUP BY waterway_id"))
            for row in result:
                print(f"Waterway {row[0]}: {row[1]} segments")
                
        except Exception as e:
            print(f"Error checking DB: {e}")
        break

if __name__ == "__main__":
    asyncio.run(check_db())
