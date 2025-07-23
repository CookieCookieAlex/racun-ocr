# test_conn.py
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
url = os.getenv("DATABASE_URL")
print(f"Connecting to: {url}")

engine = create_engine(url)
conn = engine.connect()
print("✅ Connected successfully")
conn.close()
