from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.db import session as db_session
from app.db import crud
from app.ocr.ocr_service import run_ocr_with_fallback
from app.ocr.image_loader import load_image
from app.db.schemas import UserCreate
from app.db.receipt_db_writer import save_parsed_receipt_to_db

router = APIRouter()

@router.get("/")
def root():
    return {"message": "OCR Receipt API is live"}

@router.post("/test-user")
def create_test_user(db: Session = Depends(db_session.get_db)):
    user_data = UserCreate(email="test@test.com", username="tester")
    return crud.create_user(db, user_data)

@router.post("/ocr")
async def ocr_preview(file: UploadFile = File(...)):
    img = await load_image(file)
    result = run_ocr_with_fallback(img)
    return result

@router.post("/upload-receipt")
async def upload_and_parse_only(file: UploadFile = File(...)):
    from app.ocr.image_loader import load_image
    from app.ocr.ocr_service import run_ocr_with_fallback

    img = await load_image(file)
    result = run_ocr_with_fallback(img)
    return result
