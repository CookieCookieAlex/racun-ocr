from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
import numpy as np
import cv2
import pytesseract
import logging
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import crud, schemas
from datetime import datetime

from app.ocr import processor, parser

router = APIRouter()

@router.post("/test-user/", response_model=schemas.UserOut)
def create_test_user(db: Session = Depends(get_db)):
    """Create a simple user for testing."""
    from app.db.models import User
    from sqlalchemy.exc import IntegrityError

    test_user = User(
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
        existing = db.query(User).filter_by(username="test_user").first()
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
    try:
        import numpy as np
        import cv2
        from app.db import models
        from app.ocr import processor, parser
        import pytesseract
        from datetime import datetime

        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image could not be read")

        processed_img, status = processor.preprocess_for_ocr(img)

        if processed_img is None:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {status}")

        ocr_text = pytesseract.image_to_string(processed_img, lang='hrv')
        lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

        if not lines:
            raise HTTPException(status_code=400, detail="No text detected")

        result = parser.parse_receipt(lines, log_debug=True)

        # -----------------------
        # 🧠 Validate OCR result
        # -----------------------
        total_cost = result.get("total")
        if total_cost is None:
            raise HTTPException(status_code=422, detail="Could not extract total cost from receipt.")

        purchase_date = result.get("date")
        if purchase_date is None:
            purchase_date = datetime.utcnow()

        store_name = result.get("store") or "Unknown Store"

        # -----------------------
        # ✅ Save store
        # -----------------------
        store = db.query(models.Store).filter_by(name=store_name).first()
        if not store:
            store = models.Store(name=store_name)
            db.add(store)
            db.commit()
            db.refresh(store)

        # -----------------------
        # ✅ Save receipt
        # -----------------------
        receipt = models.Receipt(
            user_id=user_id,
            store_id=store.id,
            unique_key=result.get("receipt_id", f"R-{datetime.utcnow().isoformat()}"),
            purchase_date=purchase_date,
            total_cost=total_cost
        )
        db.add(receipt)
        db.flush()

        # -----------------------
        # ✅ Save items
        # -----------------------
        for item in result.get("items", []):
            db.add(models.ReceiptItem(
                receipt_id=receipt.id,
                name=item["name"],
                quantity=item["quantity"],
                price_per_item=item["price_per_item"],
                total_price=item["total"]
            ))

        db.commit()

        return {
            "message": "Receipt saved",
            "store": store.name,
            "items_saved": len(result.get("items", []))
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"OCR receipt upload failed: {e}")
        return {"error": str(e)}

@router.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...),
                        db: Session = Depends(get_db)  # ✅ add this
):
                       
    try:
        # Load uploaded image into OpenCV format
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to load image"}

        processor.save_debug_image(img, "original")

        # Use the full preprocessing pipeline: find, crop, enhance
        processed_img, status = processor.preprocess_for_ocr(img)

        if processed_img is None:
            return {"error": f"Preprocessing failed: {status}"}

        processor.save_debug_image(processed_img, "text_enhanced")

        # Run OCR on enhanced image
        ocr_text = pytesseract.image_to_string(processed_img, lang='hrv')
        print("[OCR RAW TEXT]", repr(ocr_text)) 
        lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

        # Fallback: try OCR on cropped (non-enhanced) image if result too weak
        if len(lines) < 5:
            logging.info("Fallback: OCR result weak, trying raw cropped image")
            contour = processor.find_receipt_contour(img)
            if contour is not None:
                cropped = processor.crop_by_corners(img, contour)
                ocr_raw = pytesseract.image_to_string(cropped, lang='hrv')
                lines_raw = [line.strip() for line in ocr_raw.split("\\n") if line.strip()]
                if len(lines_raw) > len(lines):
                    lines = lines_raw

        if not lines:
            return {"error": "No text detected in image"}

        # Parse text lines into structured receipt fields
        result = parser.parse_receipt(lines, log_debug=True)
        from app.db import models

        # -- Save store --
        store = db.query(models.Store).filter_by(name=result["store"]).first()
        if not store:
            store = models.Store(name=result["store"])
            db.add(store)
            db.commit()
            db.refresh(store)

        # -- Create receipt --
        receipt = models.Receipt(
            user_id=1,  # ✅ temp for testing; we’ll use real user later
            store_id=store.id,
            unique_key=result.get("receipt_id", f"R-{datetime.utcnow().isoformat()}"),
            purchase_date=result.get("date", datetime.utcnow()),
            total_cost=result["total"]
        )
        db.add(receipt)
        db.flush()  # get receipt.id without full commit

        # -- Add items --
        for item in result["items"]:
            db.add(models.ReceiptItem(
                receipt_id=receipt.id,
                name=item["name"],
                quantity=item["quantity"],
                price_per_item=item["price_per_item"],
                total_price=item["total"]
            ))

        db.commit()


        result["debug_info"] = {
            "raw_lines": lines[:10],
            "line_source": status,
            "line_count": len(lines)
        }

        return result

    except Exception as e:
        logging.error(f"OCR processing failed: {e}")
        return {"error": f"Processing failed: {str(e)}"}