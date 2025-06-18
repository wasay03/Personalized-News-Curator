class OptimizedMySQLQueries:
    
    def get_personalized_articles_optimized(self, user_id: int, limit: int = 20):
        """Optimized query using MySQL-specific features"""
        query = """
        SELECT a.*, 
               COALESCE(ups.preferred_categories->'$.{}'.category, 0) as category_score,
               COALESCE(ups.preferred_sources->'$.{}'.source, 0) as source_score
        FROM articles a
        LEFT JOIN user_preferences_summary ups ON ups.user_id = %s
        WHERE a.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        ORDER BY 
            (category_score + source_score + a.credibility_score * 10) DESC,
            a.created_at DESC
        LIMIT %s
        """.format(category=category, source=source)  # Dynamic JSON path
        
    def search_with_ranking(self, query: str, user_id: int = None):
        """Advanced search with relevance ranking"""
        search_query = """
        SELECT a.*,
               MATCH(a.title) AGAINST(%s IN NATURAL LANGUAGE MODE) * 2 as title_score,
               MATCH(a.content) AGAINST(%s IN NATURAL LANGUAGE MODE) as content_score,
               (MATCH(a.title) AGAINST(%s IN NATURAL LANGUAGE MODE) * 2 + 
                MATCH(a.content) AGAINST(%s IN NATURAL LANGUAGE MODE) +
                a.credibility_score * 5) as total_score
        FROM articles a
        WHERE MATCH(a.title, a.content) AGAINST(%s IN NATURAL LANGUAGE MODE)
        ORDER BY total_score DESC
        LIMIT 20
        """
        
    def get_trending_with_growth(self, days: int = 7):
        """Get trending topics with growth calculation"""
        query = """
        WITH topic_counts AS (
            SELECT 
                JSON_UNQUOTE(JSON_EXTRACT(entity_item, '$.text')) as entity,
                DATE(created_at) as article_date,
                COUNT(*) as daily_mentions
            FROM articles,
            JSON_TABLE(entities, '$[*]' COLUMNS (entity_item JSON PATH '$')) as et
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            AND entities IS NOT NULL
            GROUP BY entity, DATE(created_at)
        )
        SELECT 
            entity,
            SUM(daily_mentions) as total_mentions,
            AVG(daily_mentions) as avg_daily,
            (SUM(CASE WHEN article_date >= DATE_SUB(NOW(), INTERVAL 3 DAY) 
                      THEN daily_mentions ELSE 0 END) / 3