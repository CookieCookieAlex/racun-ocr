from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# ------------------------
# Receipt Item Schemas
# ------------------------

class ReceiptItemCreate(BaseModel):
    name: str
    quantity: float
    price_per_item: float
    total_price: float

class ReceiptItemOut(ReceiptItemCreate):
    id: int

    class Config:
        orm_mode = True

# ------------------------
# Receipt Schemas
# ------------------------

class ReceiptCreate(BaseModel):
    user_id: int
    store_name: str
    unique_key: str
    purchase_date: Optional[datetime]
    total_cost: float
    items: List[ReceiptItemCreate]

class ReceiptOut(BaseModel):
    id: int
    user_id: int
    store_id: int
    unique_key: str
    purchase_date: Optional[datetime]
    total_cost: float
    items: List[ReceiptItemOut]

    class Config:
        orm_mode = True

# ------------------------
# User Schemas
# ------------------------

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    username: str
    email: str