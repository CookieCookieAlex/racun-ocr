# app/db/crud.py
from sqlalchemy.orm import Session
from app.db import models, schemas

def get_or_create_store(db: Session, store_name: str):
    store = db.query(models.Store).filter_by(name=store_name).first()
    if store:
        return store
    store = models.Store(name=store_name)
    db.add(store)
    db.commit()
    db.refresh(store)
    return store

def create_receipt(db: Session, receipt_data: schemas.ReceiptCreate):
    store = get_or_create_store(db, receipt_data.store_name)

    receipt = models.Receipt(
        user_id=receipt_data.user_id,
        store_id=store.id,
        unique_key=receipt_data.unique_key,
        purchase_date=receipt_data.purchase_date,
        total_cost=receipt_data.total_cost,
    )
    db.add(receipt)
    db.commit()
    db.refresh(receipt)

    for item in receipt_data.items:
        db_item = models.ReceiptItem(
            receipt_id=receipt.id,
            name=item.name,
            quantity=item.quantity,
            price_per_item=item.price_per_item,
            total_price=item.total_price
        )
        db.add(db_item)

    db.commit()
    db.refresh(receipt)
    return receipt

def get_receipt(db: Session, receipt_id: int):
    return db.query(models.Receipt).filter(models.Receipt.id == receipt_id).first()

def list_user_receipts(db: Session, user_id: int, limit: int = 50):
    return db.query(models.Receipt).filter(models.Receipt.user_id == user_id).limit(limit).all()
