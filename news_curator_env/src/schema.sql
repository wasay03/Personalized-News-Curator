-- Complete MySQL Schema for News Curator with Optimized Indexes
-- FIXED VERSION - Resolves VARCHAR key length issues
-- Run this script after creating your database

-- Create database (run separately if needed)
-- CREATE DATABASE news_curator CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE news_curator;

-- =====================================================
-- DROP ALL TABLES (in correct order due to foreign keys)
-- =====================================================
DROP VIEW IF EXISTS recent_articles_with_stats;
DROP VIEW IF EXISTS user_engagement_summary;

DROP PROCEDURE IF EXISTS UpdateArticleStats;
DROP PROCEDURE IF EXISTS CleanOldData;

DROP TABLE IF EXISTS system_logs;
DROP TABLE IF EXISTS search_queries;
DROP TABLE IF EXISTS trending_topics;
DROP TABLE IF EXISTS article_categories;
DROP TABLE IF EXISTS user_preferences_summary;
DROP TABLE IF EXISTS user_interactions;
DROP TABLE IF EXISTS article_chunks;
DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS news_sources;

-- =====================================================
-- 1. NEWS SOURCES TABLE
-- =====================================================
CREATE TABLE news_sources (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255) UNIQUE NOT NULL,
    source_url VARCHAR(768), -- Reduced from 2048
    rss_feed VARCHAR(768), -- Reduced from 2048
    credibility_rating DECIMAL(3,2) DEFAULT 0.50,
    last_crawled TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    source_type ENUM('rss', 'api', 'scraper') DEFAULT 'rss',
    crawl_frequency_hours INT DEFAULT 6,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_source_name (source_name),
    INDEX idx_active (is_active),
    INDEX idx_last_crawled (last_crawled),
    INDEX idx_source_type (source_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 2. USERS TABLE
-- =====================================================
CREATE TABLE users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    preferences JSON,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_active (is_active),
    INDEX idx_last_login (last_login)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 3. ARTICLES TABLE (Main content table)
-- =====================================================
CREATE TABLE articles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title TEXT NOT NULL,
    content LONGTEXT,
    url VARCHAR(768) UNIQUE NOT NULL, -- Reduced from 2048
    source VARCHAR(255) NOT NULL,
    published_at DATETIME NULL,
    category VARCHAR(100) DEFAULT 'general',
    sentiment_score DECIMAL(3,2) NULL,
    credibility_score DECIMAL(3,2) NULL,
    summary TEXT,
    entities JSON,
    image_url VARCHAR(768), -- Reduced from 2048
    author VARCHAR(255),
    word_count INT,
    reading_time INT, -- estimated reading time in minutes
    is_processed BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Basic indexes
    INDEX idx_source (source),
    INDEX idx_category (category),
    INDEX idx_published (published_at),
    INDEX idx_created (created_at),
    INDEX idx_credibility (credibility_score),
    INDEX idx_processed (is_processed),
    INDEX idx_active (is_active),
    
    -- Composite indexes for common queries
    INDEX idx_category_created (category, created_at),
    INDEX idx_source_created (source, created_at),
    INDEX idx_credibility_created (credibility_score, created_at),
    INDEX idx_category_published (category, published_at),
    INDEX idx_active_created (is_active, created_at),
    INDEX idx_processed_created (is_processed, created_at),
    
    -- Full-text search indexes
    FULLTEXT idx_title_content (title, content),
    FULLTEXT idx_title_fulltext (title),
    FULLTEXT idx_content_fulltext (content),
    FULLTEXT idx_summary_fulltext (summary)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE news_sources ADD COLUMN newsapi_source_id VARCHAR(255);

-- =====================================================
-- 4. ARTICLE CHUNKS TABLE (For RAG system)
-- =====================================================
CREATE TABLE article_chunks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id BIGINT NOT NULL,
    chunk_text LONGTEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_tokens INT,
    vector_id VARCHAR(255),
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_article_id (article_id),
    INDEX idx_vector_id (vector_id),
    INDEX idx_chunk_index (chunk_index),
    INDEX idx_embedding_model (embedding_model),
    
    -- Composite indexes
    INDEX idx_article_chunk (article_id, chunk_index),
    
    -- Full-text search
    FULLTEXT idx_chunk_fulltext (chunk_text)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 5. USER INTERACTIONS TABLE
-- =====================================================
CREATE TABLE user_interactions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    article_id BIGINT NOT NULL,
    interaction_type ENUM('click', 'like', 'share', 'skip', 'dislike', 'save', 'read_time') NOT NULL,
    interaction_data JSON,
    session_id VARCHAR(128), -- Reduced from 255 for efficiency
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_user_id (user_id),
    INDEX idx_article_id (article_id),
    INDEX idx_interaction_type (interaction_type),
    INDEX idx_created_at (created_at),
    INDEX idx_session_id (session_id),
    
    -- Composite indexes for common queries
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_article_type (article_id, interaction_type),
    INDEX idx_user_type (user_id, interaction_type),
    INDEX idx_user_article (user_id, article_id),
    
    -- Unique constraint to prevent duplicate interactions
    UNIQUE KEY unique_user_article_interaction (user_id, article_id, interaction_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 6. USER PREFERENCES SUMMARY (Materialized view)
-- =====================================================
CREATE TABLE user_preferences_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    preferred_categories JSON,
    preferred_sources JSON,
    interaction_stats JSON,
    total_interactions INT DEFAULT 0,
    avg_reading_time DECIMAL(5,2),
    favorite_topics JSON,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- Indexes
    UNIQUE KEY unique_user_prefs (user_id),
    INDEX idx_last_updated (last_updated),
    INDEX idx_total_interactions (total_interactions)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 7. ARTICLE CATEGORIES TABLE (Reference table)
-- =====================================================
CREATE TABLE article_categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    category_description TEXT,
    parent_category_id INT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,
    
    -- Self-referencing foreign key for hierarchical categories
    FOREIGN KEY (parent_category_id) REFERENCES article_categories(id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_category_name (category_name),
    INDEX idx_parent_category (parent_category_id),
    INDEX idx_active (is_active),
    INDEX idx_sort_order (sort_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 8. TRENDING TOPICS TABLE (Cache for performance)
-- =====================================================
CREATE TABLE trending_topics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL,
    topic_type VARCHAR(50), -- PERSON, ORG, GPE, etc.
    mention_count INT DEFAULT 1,
    growth_rate DECIMAL(5,2),
    time_period ENUM('1h', '6h', '24h', '7d', '30d') DEFAULT '24h',
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_topic_name (topic_name),
    INDEX idx_topic_type (topic_type),
    INDEX idx_mention_count (mention_count),
    INDEX idx_time_period (time_period),
    INDEX idx_calculated_at (calculated_at),
    
    -- Composite indexes
    INDEX idx_period_mentions (time_period, mention_count DESC),
    INDEX idx_type_mentions (topic_type, mention_count DESC),
    
    -- Unique constraint per time period
    UNIQUE KEY unique_topic_period (topic_name, time_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 9. SEARCH QUERIES TABLE (Analytics)
-- =====================================================
CREATE TABLE search_queries (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NULL,
    query_text TEXT NOT NULL,
    results_count INT DEFAULT 0,
    query_type ENUM('search', 'rag_question', 'filter') DEFAULT 'search',
    filters_applied JSON,
    execution_time_ms INT,
    session_id VARCHAR(128), -- Reduced from 255 for efficiency
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_user_id (user_id),
    INDEX idx_query_type (query_type),
    INDEX idx_created_at (created_at),
    INDEX idx_session_id (session_id),
    
    -- Full-text search on queries for analytics
    FULLTEXT idx_query_text (query_text)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- 10. SYSTEM LOGS TABLE
-- =====================================================
CREATE TABLE system_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    log_level ENUM('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') DEFAULT 'INFO',
    component VARCHAR(100), -- 'news_collector', 'rag_system', 'api', etc.
    message TEXT NOT NULL,
    details JSON,
    user_id BIGINT NULL,
    session_id VARCHAR(128), -- Reduced from 255 for efficiency
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_log_level (log_level),
    INDEX idx_component (component),
    INDEX idx_created_at (created_at),
    INDEX idx_user_id (user_id),
    
    -- Composite index for filtering logs
    INDEX idx_level_component_time (log_level, component, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================
-- INSERT SAMPLE DATA
-- =====================================================

-- Insert default categories
INSERT INTO article_categories (category_name, category_description, sort_order) VALUES
('general', 'General News', 1),
('politics', 'Political News and Analysis', 2),
('technology', 'Technology and Innovation', 3),
('health', 'Health and Medical News', 4),
('business', 'Business and Economics', 5),
('sports', 'Sports News', 6),
('entertainment', 'Entertainment and Celebrity News', 7),
('science', 'Science and Research', 8),
('world', 'International News', 9),
('local', 'Local News', 10);

-- Insert sample news sources (with shortened URLs)
INSERT INTO news_sources (source_name, source_url, rss_feed, credibility_rating, source_type) VALUES
('BBC News', 'https://www.bbc.com/news', 'http://feeds.bbci.co.uk/news/rss.xml', 0.95, 'rss'),
('Reuters', 'https://www.reuters.com', 'http://feeds.reuters.com/reuters/topNews', 0.98, 'rss'),
('CNN', 'https://www.cnn.com', 'http://rss.cnn.com/rss/edition.rss', 0.85, 'rss'),
('Associated Press', 'https://apnews.com', 'https://feeds.apnews.com/apnews/topnews', 0.92, 'rss'),
('NPR', 'https://www.npr.org', 'https://feeds.npr.org/1001/rss.xml', 0.90, 'rss'),
('The Guardian', 'https://www.theguardian.com', 'https://www.theguardian.com/world/rss', 0.88, 'rss'),
('Al Jazeera', 'https://www.aljazeera.com', 'https://www.aljazeera.com/xml/rss/all.xml', 0.83, 'rss'),
('New York Times', 'https://www.nytimes.com', 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml', 0.94, 'rss'),
('Washington Post', 'https://www.washingtonpost.com', 'http://feeds.washingtonpost.com/rss/national', 0.91, 'rss'),
('Bloomberg', 'https://www.bloomberg.com', 'https://www.bloomberg.com/feed/podcast/etf-report.xml', 0.89, 'rss'),
('Financial Times', 'https://www.ft.com', 'https://www.ft.com/?format=rss', 0.90, 'rss'),
('Politico', 'https://www.politico.com', 'https://www.politico.com/rss/politics08.xml', 0.84, 'rss'),
('The Verge', 'https://www.theverge.com', 'https://www.theverge.com/rss/index.xml', 0.80, 'rss'),
('TechCrunch', 'https://techcrunch.com', 'http://feeds.feedburner.com/TechCrunch/', 0.79, 'rss'),
('Engadget', 'https://www.engadget.com', 'https://www.engadget.com/rss.xml', 0.77, 'rss'),
('CBS News', 'https://www.cbsnews.com', 'https://www.cbsnews.com/latest/rss/main', 0.82, 'rss'),
('ABC News', 'https://abcnews.go.com', 'https://abcnews.go.com/abcnews/topstories', 0.81, 'rss'),
('Fox News', 'https://www.foxnews.com', 'http://feeds.foxnews.com/foxnews/latest', 0.75, 'rss'),
('Time', 'https://time.com', 'https://time.com/feed/', 0.86, 'rss'),
('USA Today', 'https://www.usatoday.com', 'http://rssfeeds.usatoday.com/usatoday-NewsTopStories', 0.82, 'rss');


-- =====================================================
-- USEFUL VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for recent articles with user interactions
CREATE VIEW recent_articles_with_stats AS
SELECT 
    a.*,
    COUNT(ui.id) as interaction_count,
    COUNT(CASE WHEN ui.interaction_type = 'like' THEN 1 END) as like_count,
    COUNT(CASE WHEN ui.interaction_type = 'share' THEN 1 END) as share_count,
    COUNT(CASE WHEN ui.interaction_type = 'click' THEN 1 END) as click_count,
    AVG(a.credibility_score) as avg_credibility
FROM articles a
LEFT JOIN user_interactions ui ON a.id = ui.article_id
WHERE a.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
AND a.is_active = TRUE
GROUP BY a.id;

-- View for user engagement summary
CREATE VIEW user_engagement_summary AS
SELECT 
    u.id,
    u.username,
    COUNT(ui.id) as total_interactions,
    COUNT(DISTINCT ui.article_id) as unique_articles_interacted,
    COUNT(CASE WHEN ui.interaction_type = 'like' THEN 1 END) as total_likes,
    COUNT(CASE WHEN ui.interaction_type = 'share' THEN 1 END) as total_shares,
    MAX(ui.created_at) as last_interaction,
    DATEDIFF(NOW(), MAX(ui.created_at)) as days_since_last_interaction
FROM users u
LEFT JOIN user_interactions ui ON u.id = ui.user_id
WHERE u.is_active = TRUE
GROUP BY u.id, u.username;

-- =====================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- =====================================================

DELIMITER //

-- Procedure to update article statistics
CREATE PROCEDURE UpdateArticleStats(IN article_id BIGINT)
BEGIN
    DECLARE interaction_count INT DEFAULT 0;
    DECLARE avg_sentiment DECIMAL(3,2) DEFAULT 0;
    
    -- Count interactions
    SELECT COUNT(*) INTO interaction_count
    FROM user_interactions 
    WHERE article_id = article_id;
    
    -- Update article with computed stats
    UPDATE articles 
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = article_id;
END//

-- Procedure to clean old data
CREATE PROCEDURE CleanOldData()
BEGIN
    -- Delete old search queries (older than 90 days)
    DELETE FROM search_queries 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    -- Delete old system logs (older than 30 days, except errors)
    DELETE FROM system_logs 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 30 DAY)
    AND log_level NOT IN ('ERROR', 'CRITICAL');
    
    -- Archive old user interactions (older than 1 year)
    -- You might want to archive to another table instead of deleting
    DELETE FROM user_interactions 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 1 YEAR);
END//

DELIMITER ;

-- =====================================================
-- INDEXES OPTIMIZATION NOTES
-- =====================================================
/*
Key Changes Made to Fix VARCHAR Length Issues:

1. articles.url: VARCHAR(2048) → VARCHAR(768)
2. articles.image_url: VARCHAR(2048) → VARCHAR(768)
3. news_sources.source_url: VARCHAR(2048) → VARCHAR(768)
4. news_sources.rss_feed: VARCHAR(2048) → VARCHAR(768)
5. session_id fields: VARCHAR(255) → VARCHAR(128) (for efficiency)

VARCHAR Length Calculations for utf8mb4:
- VARCHAR(768) × 4 bytes = 3072 bytes (maximum key length)
- VARCHAR(255) × 4 bytes = 1020 bytes (safe for most cases)
- VARCHAR(128) × 4 bytes = 512 bytes (very safe)

Performance Tips:

1. FULLTEXT indexes are great for search but can slow down inserts
2. Composite indexes should have most selective columns first
3. JSON columns can be indexed using generated columns for specific paths
4. Consider partitioning for very large tables (by date for articles)
5. Use EXPLAIN to analyze query performance

Example of adding generated column index for JSON:
ALTER TABLE articles ADD COLUMN entity_count INT GENERATED ALWAYS AS (JSON_LENGTH(entities)) STORED;
ALTER TABLE articles ADD INDEX idx_entity_count (entity_count);

For very large datasets, consider partitioning:
ALTER TABLE articles PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026)
);

Alternative Solutions for Long URLs:
If you need URLs longer than 768 characters, consider:

1. Hash-based uniqueness:
ALTER TABLE articles ADD COLUMN url_hash CHAR(64) UNIQUE;
ALTER TABLE articles MODIFY COLUMN url TEXT;

2. Prefix indexing:
CREATE UNIQUE INDEX idx_url_prefix ON articles (url(768));

3. Separate URL table for normalization
*/