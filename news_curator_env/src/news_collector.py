#!/usr/bin/env python3
"""
NewsAPI News Collector for News Curator Database
Fetches news articles from NewsAPI and populates the MySQL database
"""

import mysql.connector
from mysql.connector import Error
import requests
from datetime import datetime, timezone
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import re
from urllib.parse import urljoin, urlparse
import hashlib
import json
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import ssl
import certifi
from dateutil import parser as date_parser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Try to download both possible versions
resources_to_try = [
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for resource in resources_to_try:
    try:
        nltk.download(resource)
        print(f"Successfully downloaded: {resource}")
    except Exception as e:
        print(f"Failed to download {resource}: {e}")

# Try both versions of the NE chunker
chunkers = ['maxent_ne_chunker', 'maxent_ne_chunker_tab']

for chunker in chunkers:
    try:
        nltk.download(chunker)
        print(f"Successfully downloaded: {chunker}")
    except Exception as e:
        print(f"Failed to download {chunker}: {e}")

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

@dataclass
class NewsSource:
    """Data class for news source information"""
    id: int
    source_name: str
    source_url: str
    newsapi_source_id: str  # Changed from rss_feed to newsapi_source_id
    credibility_rating: float
    source_type: str
    crawl_frequency_hours: int

@dataclass
class Article:
    """Data class for article information"""
    title: str
    content: str
    url: str
    source: str
    published_at: Optional[datetime]
    category: str
    summary: str
    entities: Dict
    image_url: Optional[str]
    author: Optional[str]
    word_count: int
    reading_time: int

class NewsAPICollector:
    """Main class for collecting news from NewsAPI"""
    
    def __init__(self, db_config: Dict[str, str], newsapi_key: str):
        """
        Initialize the NewsAPICollector
        
        Args:
            db_config: Database configuration dictionary
            newsapi_key: NewsAPI key
        """
        self.db_config = db_config
        self.newsapi_key = newsapi_key
        self.newsapi_base_url = "https://newsapi.org/v2"
        self.connection = None
        self.cursor = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('news_collector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup SSL context for HTTPS requests
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NewsAPICollector/1.0)',
            'X-API-Key': self.newsapi_key
        })
    
    def connect_to_database(self) -> bool:
        """
        Establish database connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor(dictionary=True)
            self.logger.info("Database connection established")
            return True
        except Error as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect_from_database(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("Database connection closed")
    
    def log_system_message(self, level: str, component: str, message: str, details: Optional[Dict] = None):
        """
        Log message to system_logs table
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            component: Component name
            message: Log message
            details: Optional additional details as JSON
        """
        if not self.cursor or not self.connection:
            self.logger.error("Database connection not established")
            return
            
        try:
            query = """
            INSERT INTO system_logs (log_level, component, message, details, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            values = (level, component, message, json.dumps(details) if details else None, datetime.now())
            self.cursor.execute(query, values)
            self.connection.commit()
        except Error as e:
            self.logger.error(f"Failed to log system message: {e}")
    
    def get_active_sources(self) -> List[NewsSource]:
        """
        Retrieve active news sources from database
        
        Returns:
            List of NewsSource objects
        """
        if not self.cursor:
            self.logger.error("Database connection not established")
            return []
            
        try:
            # First, let's check what columns exist in the table
            self.cursor.execute("DESCRIBE news_sources")
            columns = [row['Field'] for row in self.cursor.fetchall()]
            self.logger.info(f"Available columns in news_sources: {columns}")
            
            # Check if we have newsapi_source_id or rss_feed column
            newsapi_column = 'newsapi_source_id' if 'newsapi_source_id' in columns else 'rss_feed'
            
            query = f"""
            SELECT id, source_name, source_url, {newsapi_column} as newsapi_source_id, 
                   credibility_rating, source_type, crawl_frequency_hours
            FROM news_sources 
            WHERE is_active = TRUE AND source_type IN ('newsapi', 'api')
            ORDER BY credibility_rating DESC
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            sources = []
            for row in results:
                try:
                    sources.append(NewsSource(
                        id=int(row['id']) if row['id'] is not None else 0,
                        source_name=str(row['source_name']) if row['source_name'] is not None else '',
                        source_url=str(row['source_url']) if row['source_url'] is not None else '',
                        newsapi_source_id=str(row['newsapi_source_id']) if row['newsapi_source_id'] is not None else '',
                        credibility_rating=float(row['credibility_rating']) if row['credibility_rating'] is not None else 0.0,
                        source_type=str(row['source_type']) if row['source_type'] is not None else '',
                        crawl_frequency_hours=int(row['crawl_frequency_hours']) if row['crawl_frequency_hours'] is not None else 24
                    ))
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Failed to parse source row: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(sources)} active NewsAPI sources")
            return sources
            
        except Error as e:
            self.logger.error(f"Failed to retrieve news sources: {e}")
            return []
    
    def should_crawl_source(self, source: NewsSource) -> bool:
        """
        Check if source should be crawled based on last crawl time
        
        Args:
            source: NewsSource object
            
        Returns:
            bool: True if source should be crawled
        """
        if not self.cursor:
            self.logger.error("Database connection not established")
            return True
            
        try:
            query = "SELECT last_crawled FROM news_sources WHERE id = %s"
            self.cursor.execute(query, (source.id,))
            result = self.cursor.fetchone()
            
            if not result or not result['last_crawled']:
                return True
            
            last_crawled = result['last_crawled']
            if isinstance(last_crawled, datetime):
                hours_since_crawl = (datetime.now() - last_crawled).total_seconds() / 3600
            else:
                return True
            
            return hours_since_crawl >= source.crawl_frequency_hours
            
        except Error as e:
            self.logger.error(f"Failed to check crawl status for {source.source_name}: {e}")
            return True
    
    def update_source_crawl_time(self, source_id: int):
        """
        Update the last crawled time for a source
        
        Args:
            source_id: Source ID to update
        """
        if not self.cursor or not self.connection:
            self.logger.error("Database connection not established")
            return
            
        try:
            query = "UPDATE news_sources SET last_crawled = %s WHERE id = %s"
            self.cursor.execute(query, (datetime.now(), source_id))
            self.connection.commit()
        except Error as e:
            self.logger.error(f"Failed to update crawl time for source {source_id}: {e}")
    
    def fetch_newsapi_articles(self, source_id: str, page_size: int = 100) -> Optional[Dict]:
        """
        Fetch articles from NewsAPI for a specific source
        
        Args:
            source_id: NewsAPI source ID
            page_size: Number of articles to fetch (max 100)
            
        Returns:
            NewsAPI response data or None if failed
        """
        try:
            url = f"{self.newsapi_base_url}/top-headlines"
            params = {
                'sources': source_id,
                'pageSize': min(page_size, 100),  # NewsAPI max is 100
                'apiKey': self.newsapi_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return None
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch NewsAPI articles for {source_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse NewsAPI response: {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract named entities from text using NLTK
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted entities
        """
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            
            # Extract named entities
            entities = ne_chunk(tagged)
            
            # Process entities
            extracted_entities = {
                'PERSON': [],
                'ORGANIZATION': [],
                'GPE': [],  # Geopolitical entities
                'OTHER': []
            }
            
            for entity in entities:
                if isinstance(entity, Tree):
                    entity_name = ' '.join([token for token, pos in entity.leaves()])
                    entity_type = entity.label()
                    
                    if entity_type in extracted_entities:
                        extracted_entities[entity_type].append(entity_name)
                    else:
                        extracted_entities['OTHER'].append(entity_name)
            
            # Remove duplicates
            for key in extracted_entities:
                extracted_entities[key] = list(set(extracted_entities[key]))
            
            return extracted_entities
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {e}")
            return {}
    
    def categorize_article(self, title: str, content: str) -> str:
        """
        Simple rule-based article categorization
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Category name
        """
        text = (title + ' ' + content).lower()
        
        # Define category keywords
        categories = {
            'technology': ['tech', 'software', 'ai', 'artificial intelligence', 'computer', 'digital', 
                          'internet', 'app', 'startup', 'silicon valley', 'programming', 'data'],
            'politics': ['politics', 'government', 'election', 'vote', 'president', 'congress', 
                        'senate', 'policy', 'democracy', 'republican', 'democrat'],
            'business': ['business', 'economy', 'finance', 'market', 'stock', 'trade', 'company', 
                        'corporate', 'investment', 'revenue', 'profit', 'economic'],
            'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'treatment', 'medicine', 
                      'covid', 'vaccine', 'wellness', 'fitness'],
            'sports': ['sports', 'football', 'basketball', 'baseball', 'soccer', 'olympics', 
                      'athlete', 'team', 'game', 'championship', 'league'],
            'entertainment': ['entertainment', 'movie', 'film', 'music', 'celebrity', 'hollywood', 
                             'actor', 'actress', 'concert', 'album', 'show'],
            'science': ['science', 'research', 'study', 'scientist', 'discovery', 'experiment', 
                       'climate', 'space', 'astronomy', 'biology', 'physics']
        }
        
        # Count keyword matches
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores, key=lambda k: category_scores[k])
        return 'general'
    
    def generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """
        Generate a simple extractive summary
        
        Args:
            content: Article content
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Generated summary
        """
        try:
            if not content:
                return ""
            
            # Split into sentences
            sentences = sent_tokenize(content)
            
            if len(sentences) <= max_sentences:
                return content
            
            # Simple heuristic: take first few sentences
            summary_sentences = sentences[:max_sentences]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return content[:500] + "..." if len(content) > 500 else content
    
    def calculate_reading_time(self, word_count: int) -> int:
        """
        Calculate estimated reading time in minutes
        Average reading speed: 200 words per minute
        
        Args:
            word_count: Number of words in article
            
        Returns:
            Estimated reading time in minutes
        """
        return max(1, round(word_count / 200))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def article_exists(self, url: str) -> bool:
        """
        Check if article already exists in database
        
        Args:
            url: Article URL
            
        Returns:
            True if article exists, False otherwise
        """
        if not self.cursor:
            self.logger.error("Database connection not established")
            return False
            
        try:
            query = "SELECT id FROM articles WHERE url = %s"
            self.cursor.execute(query, (url,))
            return self.cursor.fetchone() is not None
        except Error as e:
            self.logger.error(f"Failed to check article existence: {e}")
            return False
    
    def parse_article_from_newsapi(self, article_data: Dict, source: NewsSource) -> Optional[Article]:
        """
        Parse article data from NewsAPI response
        
        Args:
            article_data: NewsAPI article data
            source: NewsSource object
            
        Returns:
            Article object or None if parsing failed
        """
        try:
            # Log the raw article data for debugging
            self.logger.debug(f"Raw article data: {json.dumps(article_data, indent=2)}")
            
            # Extract basic information
            title = self.clean_text(article_data.get('title', ''))
            content = self.clean_text(article_data.get('content', ''))
            description = self.clean_text(article_data.get('description', ''))
            
            # Use description if content is truncated or missing
            if not content or content.endswith('[+') or content.endswith('...'):
                content = description
            
            # Extract URL
            url = article_data.get('url', '')
            if not url:
                return None
            
            # Parse published date
            published_at = None
            if article_data.get('publishedAt'):
                try:
                    published_at = date_parser.parse(article_data['publishedAt'])
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except Exception as e:
                    self.logger.warning(f"Failed to parse date: {e}")
            
            # Extract author - handle various formats
            author = None
            author_raw = article_data.get('author')
            self.logger.debug(f"Raw author data: {author_raw}")
            
            if author_raw:
                if isinstance(author_raw, str):
                    author = author_raw.strip()
                    # Clean up common null-like values
                    if author.lower() in ['null', 'none', '', 'unknown', 'n/a']:
                        author = None
                elif isinstance(author_raw, dict):
                    # Some APIs return author as an object
                    author = author_raw.get('name', '').strip()
                elif isinstance(author_raw, list) and len(author_raw) > 0:
                    # Some APIs return author as an array
                    author = str(author_raw[0]).strip()
            
            # Extract image URL - handle various formats
            image_url = None
            image_raw = article_data.get('urlToImage')
            self.logger.debug(f"Raw image URL data: {image_raw}")
            
            if image_raw:
                if isinstance(image_raw, str):
                    image_url = image_raw.strip()
                    # Clean up common null-like values and validate URL format
                    if (image_url.lower() in ['null', 'none', '', 'n/a'] or 
                        not image_url.startswith(('http://', 'https://'))):
                        image_url = None
                elif isinstance(image_raw, dict):
                    # Some APIs return image as an object
                    image_url = image_raw.get('url', '').strip()
            
            # Additional logging for debugging
            self.logger.info(f"Extracted - Author: '{author}', Image URL: '{image_url}'")
            
            # Calculate word count
            word_count = len(word_tokenize(content)) if content else 0
            
            # Generate summary
            summary = self.generate_summary(content)
            
            # Extract entities
            entities = self.extract_entities(title + ' ' + content)
            
            # Categorize article
            category = self.categorize_article(title, content)
            
            # Calculate reading time
            reading_time = self.calculate_reading_time(word_count)
            
            return Article(
                title=title,
                content=content,
                url=url,
                source=source.source_name,
                published_at=published_at,
                category=category,
                summary=summary,
                entities=entities,
                image_url=image_url,
                author=author,
                word_count=word_count,
                reading_time=reading_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse article from NewsAPI data: {e}")
            return None
    
    def save_article(self, article: Article, source: NewsSource) -> bool:
        """
        Save article to database
        
        Args:
            article: Article object to save
            source: NewsSource object
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.cursor or not self.connection:
            self.logger.error("Database connection not established")
            return False
            
        try:
            # Check if article already exists
            if self.article_exists(article.url):
                self.logger.debug(f"Article already exists: {article.url}")
                return False
            
            query = """
            INSERT INTO articles (
                title, content, url, source, published_at, category, 
                sentiment_score, credibility_score, summary, entities, 
                image_url, author, word_count, reading_time, 
                is_processed, is_active, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            values = (
                article.title,
                article.content,
                article.url,
                article.source,
                article.published_at,
                article.category,
                None,  # sentiment_score (to be calculated later)
                source.credibility_rating,  # use source credibility as base
                article.summary,
                json.dumps(article.entities),
                article.image_url,
                article.author,
                article.word_count,
                article.reading_time,
                False,  # is_processed
                True,   # is_active
                datetime.now(),
                datetime.now()
            )
            
            self.cursor.execute(query, values)
            self.connection.commit()
            
            self.logger.info(f"Saved article: {article.title[:50]}...")
            return True
            
        except Error as e:
            self.logger.error(f"Failed to save article: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def process_source(self, source: NewsSource) -> Tuple[int, int]:
        """
        Process a single news source
        
        Args:
            source: NewsSource object to process
            
        Returns:
            Tuple of (articles_processed, articles_saved)
        """
        articles_processed = 0
        articles_saved = 0
        
        try:
            self.logger.info(f"Processing source: {source.source_name}")
            
            # Fetch articles from NewsAPI
            response_data = self.fetch_newsapi_articles(source.newsapi_source_id)
            if not response_data:
                self.log_system_message('ERROR', 'newsapi_collector', 
                                      f'Failed to fetch articles for {source.source_name}')
                return articles_processed, articles_saved
            
            # Process articles
            articles = response_data.get('articles', [])
            for article_data in articles:
                articles_processed += 1
                
                # Parse article
                article = self.parse_article_from_newsapi(article_data, source)
                if not article:
                    continue
                
                # Save article
                if self.save_article(article, source):
                    articles_saved += 1
                
                # Small delay to be respectful to the API
                time.sleep(0.1)
            
            # Update source crawl time
            self.update_source_crawl_time(source.id)
            
            self.logger.info(f"Processed {articles_processed} articles from {source.source_name}, saved {articles_saved}")
            
            self.log_system_message('INFO', 'newsapi_collector', 
                                  f'Processed source {source.source_name}',
                                  {'articles_processed': articles_processed, 'articles_saved': articles_saved})
            
        except Exception as e:
            self.logger.error(f"Error processing source {source.source_name}: {e}")
            self.log_system_message('ERROR', 'newsapi_collector', 
                                  f'Error processing source {source.source_name}: {str(e)}')
        
        return articles_processed, articles_saved
    
    def get_available_sources(self) -> List[Dict]:
        """
        Get available sources from NewsAPI
        
        Returns:
            List of available sources
        """
        try:
            url = f"{self.newsapi_base_url}/sources"
            params = {
                'apiKey': self.newsapi_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            return data.get('sources', [])
            
        except Exception as e:
            self.logger.error(f"Failed to fetch available sources: {e}")
            return []
    
    def test_newsapi_response(self, source_id: str = "bbc-news") -> None:
        """
        Test method to see what NewsAPI actually returns
        
        Args:
            source_id: NewsAPI source ID to test
        """
        try:
            self.logger.info(f"Testing NewsAPI response for source: {source_id}")
            response_data = self.fetch_newsapi_articles(source_id, page_size=3)
            
            if response_data and 'articles' in response_data:
                self.logger.info(f"Total articles returned: {response_data.get('totalResults', 0)}")
                
                for i, article in enumerate(response_data['articles'][:3]):
                    self.logger.info(f"\n--- Article {i+1} ---")
                    self.logger.info(f"Title: {article.get('title', 'N/A')}")
                    self.logger.info(f"Author: {article.get('author', 'N/A')} (Type: {type(article.get('author'))})")
                    self.logger.info(f"Image URL: {article.get('urlToImage', 'N/A')} (Type: {type(article.get('urlToImage'))})")
                    self.logger.info(f"Source: {article.get('source', {}).get('name', 'N/A')}")
                    self.logger.info(f"Published: {article.get('publishedAt', 'N/A')}")
                    self.logger.info(f"URL: {article.get('url', 'N/A')}")
                    
                    # Log the full article structure for debugging
                    self.logger.debug(f"Full article data: {json.dumps(article, indent=2)}")
            else:
                self.logger.error("No articles returned from NewsAPI")
                
        except Exception as e:
            self.logger.error(f"Error testing NewsAPI: {e}")
    
    def run_collection(self):
        """
        Main method to run the news collection process
        """
        self.logger.info("Starting NewsAPI collection process")
        
        if not self.connect_to_database():
            return
        
        try:
            # Test NewsAPI response first (uncomment for debugging)
            # self.test_newsapi_response()
            
            # Get active sources
            sources = self.get_active_sources()
            if not sources:
                self.logger.warning("No active NewsAPI sources found")
                return
            
            total_processed = 0
            total_saved = 0
            
            # Process each source
            for source in sources:
                if self.should_crawl_source(source):
                    processed, saved = self.process_source(source)
                    total_processed += processed
                    total_saved += saved
                else:
                    self.logger.info(f"Skipping {source.source_name} - not due for crawling")
            
            self.logger.info(f"Collection complete. Processed: {total_processed}, Saved: {total_saved}")
            
            self.log_system_message('INFO', 'newsapi_collector', 
                                  'NewsAPI collection completed',
                                  {'total_processed': total_processed, 'total_saved': total_saved})
            
        except Exception as e:
            self.logger.error(f"Error during collection process: {e}")
            self.log_system_message('ERROR', 'newsapi_collector', 
                                  f'Error during collection process: {str(e)}')
        
        finally:
            self.disconnect_from_database()

def main():
    """
    Main function to run the news collector
    """
    # Load API key from environment
    NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
    
    if not NEWSAPI_KEY:
        print("Error: NEWS_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with NEWS_API_KEY=your_api_key")
        return
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'news_curator',
        'user': 'news_user',
        'password': '1234',
        'charset': 'utf8mb4',
        'use_unicode': True,
        'autocommit': False
    }
    
    # Create and run collector
    collector = NewsAPICollector(db_config, NEWSAPI_KEY)
    collector.run_collection()

if __name__ == "__main__":
    main()