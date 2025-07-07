# ============= routers/articles.py =============
"""
Article-related endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from mysql.connector import Error
from typing import Optional
from datetime import datetime, timedelta
import json
import logging

from models.schemas import ArticleResponse, ArticleListResponse
from dependencies import get_db
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

def parse_article_data(article_data: dict) -> ArticleResponse:
    """Helper function to parse article data from database"""
    entities = None
    if article_data['entities']:
        try:
            entities = json.loads(article_data['entities'])
        except json.JSONDecodeError:
            entities = None
    
    return ArticleResponse(
        id=article_data['id'],
        title=article_data['title'],
        content=article_data['content'],
        url=article_data['url'],
        source=article_data['source'],
        published_at=article_data['published_at'],
        category=article_data['category'],
        summary=article_data['summary'],
        image_url=article_data['image_url'],
        author=article_data['author'],
        word_count=article_data['word_count'],
        reading_time=article_data['reading_time'],
        credibility_score=article_data['credibility_score'],
        created_at=article_data['created_at'],
        entities=entities
    )

@router.get("/articles", response_model=ArticleListResponse)
async def get_articles(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    category: Optional[str] = Query(None, description="Filter by category"),
    source: Optional[str] = Query(None, description="Filter by source"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Articles from last N days"),
    cursor = Depends(get_db)
):
    """Get articles with pagination and filtering"""
    try:
        # Build WHERE clause
        where_conditions = ["is_active = TRUE"]
        params = []
        
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if source:
            where_conditions.append("source = %s")
            params.append(source)
        
        if days:
            where_conditions.append("created_at >= %s")
            params.append(datetime.now() - timedelta(days=days))
        
        where_clause = " AND ".join(where_conditions)
        
        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM articles WHERE {where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()['total']
        
        # Calculate pagination
        offset = (page - 1) * limit
        total_pages = (total + limit - 1) // limit
        
        # Get articles
        articles_query = f"""
        SELECT id, title, content, url, source, published_at, category, 
               summary, image_url, author, word_count, reading_time, 
               credibility_score, created_at, entities
        FROM articles 
        WHERE {where_clause}
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(articles_query, params + [limit, offset])
        articles_data = cursor.fetchall()
        
        # Convert to response models
        articles = [parse_article_data(article) for article in articles_data]
        
        return ArticleListResponse(
            articles=articles,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages
        )
        
    except Error as e:
        logger.error(f"Database error in get_articles: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: int, cursor = Depends(get_db)):
    """Get a specific article by ID"""
    try:
        query = """
        SELECT id, title, content, url, source, published_at, category, 
               summary, image_url, author, word_count, reading_time, 
               credibility_score, created_at, entities
        FROM articles 
        WHERE id = %s AND is_active = TRUE
        """
        
        cursor.execute(query, (article_id,))
        article = cursor.fetchone()
        
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return parse_article_data(article)
        
    except Error as e:
        logger.error(f"Database error in get_article: {e}")
        raise HTTPException(status_code=500, detail="Database error")