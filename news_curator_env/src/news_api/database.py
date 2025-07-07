"""
Database connection management
"""
import mysql.connector
from mysql.connector import Error
import logging
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try:
            self.connection = mysql.connector.connect(**settings.db_config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Database connection established")
            return True
        except Error as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")
    
    def get_cursor(self):
        if not self.connection or not self.connection.is_connected():
            self.connect()
        return self.cursor