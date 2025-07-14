import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect(user='racun_user', password='Retreo238!?', database='mydb', host='localhost')
        print("Connected!")
        await conn.close()
    except Exception as e:
        print("Connection failed:", e)

asyncio.run(test())
