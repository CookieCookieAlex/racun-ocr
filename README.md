# racun-ocr (Troškomjer)


This project is an OCR-based tool for scanning grocery receipts.
It extracts items, prices, totals, store, and date, then saves everything into a database for analysis.

The goal: help people track their spending, compare prices between stores, and see trends over time.

Features

Extracts text from receipt images (OpenCV + Tesseract/EasyOCR)

Parses store name, date, items, and totals

Saves results into PostgreSQL

FastAPI backend for uploading and querying receipts

Future plans: price trends, budgeting, store comparisons

Getting Started
1. Requirements

Python 3.10+

PostgreSQL

Tesseract OCR installed on your system

2. Setup

Clone the repo:

git clone https://github.com/CookieCookieAlex/racun-ocr.git
cd racun-ocr


Install dependencies:

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Create a .env file:

DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME


Run migrations (if using Alembic):

alembic upgrade head

3. Run the API
uvicorn app.main:app --reload


Open your browser at:
👉 http://127.0.0.1:8000/docs

Roadmap

Smarter parsing for different stores

Track item price changes over time

Monthly/weekly spending summaries

Export data (CSV/JSON)
