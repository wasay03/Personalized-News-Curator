"""
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class ArticleResponse(BaseModel):
    id: int
    title: str
    content: Optional[str] = None
    url: str
    source: str
    published_at: Optional[datetime] = None
    category: str
    summary: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None
    word_count: Optional[int] = None
    reading_time: Optional[int] = None
    credibility_score: Optional[float] = None
    created_at: datetime
    entities: Optional[Dict] = None

class ArticleListResponse(BaseModel):
    articles: List[ArticleResponse]
    total: int
    page: int
    limit: int
    total_pages: int

class CategoryResponse(BaseModel):
    category_name: str
    article_count: int

class SourceResponse(BaseModel):
    source_name: str
    article_count: int
    credibility_rating: Optional[float] = None

class SearchResponse(BaseModel):
    articles: List[ArticleResponse]
    total: int
    query: str
    execution_time_ms: float

class StatsResponse(BaseModel):
    total_articles: int
    articles_today: int
    total_sources: int
    total_categories: int
    average_credibility: float