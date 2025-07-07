from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Float, Text, Boolean, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    receipts = relationship("Receipt", back_populates="user")

class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    # optional: store metadata

    receipts = relationship("Receipt", back_populates="store")

class Receipt(Base):
    __tablename__ = "receipts"
    __table_args__ = (UniqueConstraint('store_id', 'unique_key', name='unique_store_key'),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)

    unique_key = Column(String(255), nullable=False)  # unique per store (bill key)
    purchase_date = Column(DateTime, nullable=True)
    total_cost = Column(Float, nullable=False)

    user = relationship("User", back_populates="receipts")
    store = relationship("Store", back_populates="receipts")
    items = relationship("ReceiptItem", back_populates="receipt", cascade="all, delete-orphan")
    corrections = relationship("Correction", back_populates="receipt", cascade="all, delete-orphan")
    labels = relationship("Label", back_populates="receipt", cascade="all, delete-orphan")
    images = relationship("Image", back_populates="receipt", cascade="all, delete-orphan")

class ReceiptItem(Base):
    __tablename__ = "receipt_items"

    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    name = Column(String(255), nullable=False)
    quantity = Column(Float, default=1)
    price_per_item = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)

    receipt = relationship("Receipt", back_populates="items")

class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    receipt = relationship("Receipt", back_populates="labels")

class Correction(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    field_name = Column(String(100), nullable=False)  # e.g. 'total_cost', 'item_name'
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    receipt = relationship("Receipt", back_populates="corrections")

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    image_path = Column(String(255), nullable=False)  # path or URL to image storage
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    receipt = relationship("Receipt", back_populates="images")
