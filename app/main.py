from fastapi import FastAPI
from app.api import endpoints

app = FastAPI(title="Receipt OCR")

app.include_router(endpoints.router)


@app.get("/")
async def root():
    return {"message": "Hello from racun-ocr API!"}