"""
Metadata endpoints - categories, sources, stats, trending
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from mysql.connector import Error
from typing import List
from datetime import datetime, timedelta
import json
import logging

from models.schemas import CategoryResponse, SourceResponse, StatsResponse
from dependencies import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/categories", response_model=List[CategoryResponse])
async def get_categories(cursor = Depends(get_db)):
    """Get all categories with article counts"""
    try:
        query = """
        SELECT category, COUNT(*) as article_count
        FROM articles 
        WHERE is_active = TRUE
        GROUP BY category
        ORDER BY article_count DESC
        """
        
        cursor.execute(query)
        categories = cursor.fetchall()
        
        return [
            CategoryResponse(
                category_name=cat['category'],
                article_count=cat['article_count']
            )
            for cat in categories
        ]
        
    except Error as e:
        logger.error(f"Database error in get_categories: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/sources", response_model=List[SourceResponse])
async def get_sources(cursor = Depends(get_db)):
    """Get all sources with article counts"""
    try:
        query = """
        SELECT a.source, COUNT(*) as article_count, AVG(a.credibility_score) as avg_credibility
        FROM articles a
        WHERE a.is_active = TRUE
        GROUP BY a.source
        ORDER BY article_count DESC
        """
        
        cursor.execute(query)
        sources = cursor.fetchall()
        
        return [
            SourceResponse(
                source_name=source['source'],
                article_count=source['article_count'],
                credibility_rating=source['avg_credibility']
            )
            for source in sources
        ]
        
    except Error as e:
        logger.error(f"Database error in get_sources: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/trending", response_model=List[str])
async def get_trending_topics(
    limit: int = Query(10, ge=1, le=50, description="Number of trending topics"),
    cursor = Depends(get_db)
):
    """Get trending topics from recent articles"""
    try:
        # Get entities from articles in the last 24 hours
        query = """
        SELECT entities
        FROM articles 
        WHERE created_at >= %s AND is_active = TRUE AND entities IS NOT NULL
        """
        
        cursor.execute(query, (datetime.now() - timedelta(days=1),))
        articles = cursor.fetchall()
        
        # Count entity mentions
        entity_counts = {}
        for article in articles:
            if article['entities']:
                try:
                    entities = json.loads(article['entities'])
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            if len(entity) > 2:  # Filter out very short entities
                                entity_counts[entity] = entity_counts.get(entity, 0) + 1
                except json.JSONDecodeError:
                    continue
        
        # Sort by count and return top entities
        trending = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return [entity for entity, count in trending[:limit]]
        
    except Error as e:
        logger.error(f"Database error in get_trending_topics: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/stats", response_model=StatsResponse)
async def get_stats(cursor = Depends(get_db)):
    """Get general statistics about the news database"""
    try:
        stats = {}
        
        # Total articles
        cursor.execute("SELECT COUNT(*) as total FROM articles WHERE is_active = TRUE")
        stats['total_articles'] = cursor.fetchone()['total']
        
        # Articles today
        cursor.execute("""
        SELECT COUNT(*) as today 
        FROM articles 
        WHERE DATE(created_at) = CURDATE() AND is_active = TRUE
        """)
        stats['articles_today'] = cursor.fetchone()['today']
        
        # Unique sources
        cursor.execute("SELECT COUNT(DISTINCT source) as sources FROM articles WHERE is_active = TRUE")
        stats['total_sources'] = cursor.fetchone()['sources']
        
        # Categories
        cursor.execute("SELECT COUNT(DISTINCT category) as categories FROM articles WHERE is_active = TRUE")
        stats['total_categories'] = cursor.fetchone()['categories']
        
        # Average credibility
        cursor.execute("""
        SELECT AVG(credibility_score) as avg_credibility 
        FROM articles 
        WHERE is_active = TRUE AND credibility_score IS NOT NULL
        """)
        result = cursor.fetchone()
        stats['average_credibility'] = float(result['avg_credibility']) if result['avg_credibility'] else 0.0
        
        return StatsResponse(**stats)
        
    except Error as e:
        logger.error(f"Database error in get_stats: {e}")
        raise HTTPException(status_code=500, detail="Database error")