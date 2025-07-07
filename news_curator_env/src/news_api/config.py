"""
Configuration settings
"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_NAME: str = os.getenv('DB_NAME', 'news_curator')
    DB_USER: str = os.getenv('DB_USER', 'news_user')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '1234')
    
    # API
    NEWS_API_KEY: str = os.getenv('NEWS_API_KEY', '')
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    @property
    def db_config(self):
        return {
            'host': self.DB_HOST,
            'database': self.DB_NAME,
            'user': self.DB_USER,
            'password': self.DB_PASSWORD,
            'charset': 'utf8mb4',
            'use_unicode': True,
            'autocommit': False
        }

settings = Settings()