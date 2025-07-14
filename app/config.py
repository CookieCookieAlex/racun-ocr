from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into the environment

DATABASE_URL = os.getenv("DATABASE_URL")
