import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect()
        print("Connected!")
        await conn.close()
    except Exception as e:
        print("Connection failed:", e)

asyncio.run(test())
