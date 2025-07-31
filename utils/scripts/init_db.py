# init_db.py
import asyncio
import sys, os
from server.db import engine, Base  # 路径按你实际的文件夹来改

async def create_all():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created.")

if __name__ == "__main__":
    asyncio.run(create_all())