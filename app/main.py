from fastapi import FastAPI
from app.api.endpoints import router as api_router


app = FastAPI(title="Receipt OCR")

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Hello from racun-ocr API!"}
