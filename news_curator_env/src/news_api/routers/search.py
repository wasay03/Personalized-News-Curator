"""
Search-related endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from mysql.connector import Error
from datetime import datetime
import logging

from models.schemas import SearchResponse
from dependencies import get_db
from routers.articles import parse_article_data

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/search", response_model=SearchResponse)
async def search_articles(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    cursor = Depends(get_db)
):
    """Search articles using full-text search"""
    try:
        start_time = datetime.now()
        
        # Full-text search query
        search_query = """
        SELECT id, title, content, url, source, published_at, category, 
               summary, image_url, author, word_count, reading_time, 
               credibility_score, created_at, entities,
               MATCH(title, content) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
        FROM articles 
        WHERE MATCH(title, content) AGAINST(%s IN NATURAL LANGUAGE MODE)
        AND is_active = TRUE
        ORDER BY relevance DESC, created_at DESC
        LIMIT %s
        """
        
        cursor.execute(search_query, (q, q, limit))
        articles_data = cursor.fetchall()
        
        # Get total count for the search
        count_query = """
        SELECT COUNT(*) as total
        FROM articles 
        WHERE MATCH(title, content) AGAINST(%s IN NATURAL LANGUAGE MODE)
        AND is_active = TRUE
        """
        cursor.execute(count_query, (q,))
        total = cursor.fetchone()['total']
        
        # Convert to response models
        articles = [parse_article_data(article) for article in articles_data]
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            articles=articles,
            total=total,
            query=q,
            execution_time_ms=execution_time
        )
        
    except Error as e:
        logger.error(f"Database error in search_articles: {e}")
        raise HTTPException(status_code=500, detail="Database error")