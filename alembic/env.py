import os
import sys
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context
from dotenv import load_dotenv

# Add app to sys.path and load environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

# This file ensures Alembic picks up our app DB config + models


from app import config as app_config
from app.db.models import Base
from app.db import models  # force model loading

# Alembic configuration
config = context.config
config.set_main_option("sqlalchemy.url", str(app_config.DATABASE_URL))

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata from your models
target_metadata = Base.metadata

# Online-only migration execution
connectable = create_engine(str(app_config.DATABASE_URL),connect_args={"connect_timeout": 5}, poolclass=pool.NullPool)

with connectable.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()
