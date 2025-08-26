from sqlalchemy.orm import Session
from datetime import datetime
from app.db import models

def save_parsed_receipt_to_db(db: Session, user_id: int, parsed_data: dict):
    """
    Saves parsed receipt data into the database including store, receipt, and items.
    """
    store_name = parsed_data.get("store") or "Unknown Store"
    total_cost = parsed_data.get("total")
    purchase_date = parsed_data.get("date") or datetime.utcnow()
    unique_key = parsed_data.get("receipt_id", f"R-{datetime.utcnow().isoformat()}")

    # Get or create store
    store = db.query(models.Store).filter_by(name=store_name).first()
    if not store:
        store = models.Store(name=store_name)
        db.add(store)
        db.commit()
        db.refresh(store)

    # Save receipt
    receipt = models.Receipt(
        user_id=user_id,
        store_id=store.id,
        unique_key=unique_key,
        purchase_date=purchase_date,
        total_cost=total_cost
    )
    db.add(receipt)
    db.flush()

    # Save items
    for item in parsed_data.get("items", []):
        db.add(models.ReceiptItem(
            receipt_id=receipt.id,
            name=item["name"],
            quantity=item["quantity"],
            price_per_item=item["price_per_item"],
            total_price=item["total"]
        ))

    db.commit()
    return receipt
