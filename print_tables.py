from app.db.models import Base
from app.db import models

print("[DEBUG] Table Names:", Base.metadata.tables.keys())
