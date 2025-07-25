from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.db.session import get_db
from app.db import schemas
from app.ocr import processor, parser
from app.ocr.image_loader import load_image
from app.ocr.ocr_service import run_ocr_with_fallback
from app.ocr.receipt_service import save_receipt

router = APIRouter()


@router.post("/test-user/", response_model=schemas.UserOut)
def create_test_user(db: Session = Depends(get_db)):
    """
    Creates a test user if one doesn't already exist.
    Useful for initial testing without full auth system.
    """
    from app.db import models
    from sqlalchemy.exc import IntegrityError

    test_user = models.User(
        username="test_user",
        email="test@example.com",
        hashed_password="notsecure",
        created_at=datetime.utcnow()
    )
    db.add(test_user)
    try:
        db.commit()
        db.refresh(test_user)
    except IntegrityError:
        db.rollback()
        existing = db.query(models.User).filter_by(username="test_user").first()
        if existing:
            return existing
        raise HTTPException(status_code=500, detail="Couldn't create or fetch test user.")

    return test_user


@router.post("/upload-receipt/")
async def upload_receipt_for_user(
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads an image file, performs OCR, parses the result,
    and saves the receipt data into the database.
    """
    try:
        image = await load_image(file)
        processed_img, status = processor.preprocess_for_ocr(image)

        if processed_img is None:
            raise HTTPException(400, f"Preprocessing failed: {status}")

        ocr_lines = run_ocr_with_fallback(processed_img, image)
        parsed = parser.parse_receipt(ocr_lines, log_debug=True)

        if not parsed.get("total"):
            raise HTTPException(422, "Total cost not found in receipt")

        receipt = save_receipt(db, user_id, parsed)

        return {
            "message": "Receipt saved",
            "store": parsed['store'],
            "items_saved": len(parsed['items'])
        }

    except Exception as e:
        logging.error(f"OCR receipt upload failed: {e}")
        return {"error": str(e)}


@router.get("/")
async def root():
    """Basic root health check."""
    return {"message": "Hello from racun-ocr API!"}


@router.post("/ocr/")
async def ocr_preview(file: UploadFile = File(...)):
    """
    Processes a receipt image and returns parsed fields without saving to DB.
    Used for preview/debugging.
    """
    try:
        from app.ocr.image_loader import load_image
        from app.ocr.ocr_service import run_ocr_with_fallback

        from app.ocr.processor import ReceiptProcessor, preprocess_for_ocr
        from app.ocr import parser

        image = await load_image(file)
        processor_instance = ReceiptProcessor(debug=True)
        processor_instance.save_debug_image(image, "original")

        processed, status = preprocess_for_ocr(image)
        if processed is None:
            raise HTTPException(400, f"Preprocessing failed: {status}")

        processor_instance.save_debug_image(processed, "processed")

        lines = run_ocr_with_fallback(processed, image)
        parsed = parser.parse_receipt(lines, log_debug=True)

        return {
            "status": "success",
            "store": parsed.get("store"),
            "date": parsed.get("date"),
            "total": parsed.get("total"),
            "items": parsed.get("items", []),
            "lines_raw": lines[:10],  # preview text
            "line_count": len(lines),
            "debug_images": ["original.png", "processed.png"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"OCR processing failed: {e}")
