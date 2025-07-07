"""
FastAPI dependency injection
"""
from fastapi import HTTPException
from database import DatabaseManager

# Create a single instance to be shared
db_manager = DatabaseManager()

def get_db():
    """Dependency to get database cursor"""
    cursor = db_manager.get_cursor()
    if not cursor:
        raise HTTPException(status_code=500, detail="Database connection failed")
    return cursor