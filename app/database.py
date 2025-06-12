from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DATABASE_URL with fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://frauduser:fraudpass@db:5432/frauddb"
)

print(f"Connecting to database: {DATABASE_URL}")  # Debug line

try:
    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    print("Database engine created successfully")
except Exception as e:
    print(f"Database connection error: {e}")
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()