import React, { useState, useEffect } from 'react';
import { Search, Clock, User, ExternalLink, TrendingUp, Filter, ChevronLeft, ChevronRight, Star, BookOpen } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api';

// Custom hook for API calls
const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = async (url, options = {}) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}${url}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  return { request, loading, error };
};

// Article Card Component
const ArticleCard = ({ article, onClick }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div 
      className="group bg-white rounded-2xl shadow-md hover:shadow-xl transition-all duration-300 cursor-pointer border border-gray-100 overflow-hidden transform hover:-translate-y-1"
      onClick={() => onClick(article)}
    >
      {article.image_url && (
        <div className="relative overflow-hidden">
          <img 
            src={article.image_url} 
            alt={article.title}
            className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.target.style.display = 'none';
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </div>
      )}
      
      <div className="p-6">
        <div className="flex items-center justify-between mb-3">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-blue-500 to-purple-600 text-white">
            {article.category}
          </span>
          {article.credibility_score && (
            <div className="flex items-center text-sm text-amber-600">
              <Star size={14} className="mr-1 fill-current" />
              <span className="font-medium">{(article.credibility_score * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
        
        <h3 className="font-bold text-lg mb-3 line-clamp-2 text-gray-900 leading-tight group-hover:text-blue-600 transition-colors">
          {article.title}
        </h3>
        
        {article.summary && (
          <p className="text-gray-600 text-sm mb-4 line-clamp-3 leading-relaxed">
            {article.summary}
          </p>
        )}
        
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center space-x-3">
            <span className="font-semibold text-blue-600 hover:text-blue-700 transition-colors">
              {article.source}
            </span>
            {article.author && (
              <div className="flex items-center">
                <User size={14} className="mr-1 text-gray-400" />
                <span className="truncate max-w-24">{article.author}</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            {article.reading_time && (
              <div className="flex items-center text-gray-500">
                <Clock size={14} className="mr-1" />
                <span>{article.reading_time}m</span>
              </div>
            )}
            <span className="text-xs">{formatDate(article.published_at)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Article Modal Component
const ArticleModal = ({ article, onClose }) => {
  if (!article) return null;

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-white rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl">
        <div className="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-100 px-8 py-6 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900">Article Details</h2>
          <button 
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-500 hover:text-gray-700 transition-all duration-200"
          >
            <span className="text-xl">×</span>
          </button>
        </div>
        
        <div className="overflow-auto max-h-[calc(90vh-88px)]">
          <div className="p-8">
            {article.image_url && (
              <img 
                src={article.image_url} 
                alt={article.title}
                className="w-full h-72 object-cover rounded-2xl mb-6 shadow-lg"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            )}
            
            <div className="flex items-center flex-wrap gap-3 mb-6">
              <span className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-medium">
                {article.category}
              </span>
              {article.credibility_score && (
                <span className="bg-gradient-to-r from-amber-400 to-orange-500 text-white px-4 py-2 rounded-full text-sm font-medium flex items-center">
                  <Star size={14} className="mr-1 fill-current" />
                  Trust: {(article.credibility_score * 100).toFixed(0)}%
                </span>
              )}
            </div>
            
            <h1 className="text-3xl font-bold mb-6 text-gray-900 leading-tight">{article.title}</h1>
            
            <div className="flex items-center flex-wrap gap-6 mb-8 text-sm text-gray-600">
              <span className="font-semibold text-blue-600">{article.source}</span>
              {article.author && (
                <div className="flex items-center">
                  <User size={16} className="mr-2 text-gray-400" />
                  <span>{article.author}</span>
                </div>
              )}
              <span className="font-medium">{formatDate(article.published_at)}</span>
              {article.reading_time && (
                <div className="flex items-center text-gray-500">
                  <BookOpen size={16} className="mr-2" />
                  <span>{article.reading_time} min read</span>
                </div>
              )}
            </div>
            
            {article.summary && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-2xl mb-8 border border-blue-100">
                <h3 className="font-bold text-lg mb-3 text-gray-900">Summary</h3>
                <p className="text-gray-700 leading-relaxed">{article.summary}</p>
              </div>
            )}
            
            <div className="prose max-w-none mb-8">
              <p className="text-gray-700 whitespace-pre-wrap leading-relaxed text-lg">{article.content}</p>
            </div>
            
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pt-6 border-t border-gray-200">
              <a 
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                <ExternalLink size={18} className="mr-2" />
                Read Original Article
              </a>
              
              {article.entities && Object.keys(article.entities).length > 0 && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Topics: </span>
                  <span className="text-gray-600">
                    {Object.values(article.entities).flat().slice(0, 5).join(', ')}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Filters Component
const FiltersPanel = ({ 
  categories, 
  sources, 
  selectedCategory, 
  selectedSource, 
  onCategoryChange, 
  onSourceChange,
  onReset 
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
          <Filter size={20} className="text-white" />
        </div>
        <h3 className="font-bold text-xl text-gray-900">Filters</h3>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Category</label>
          <select 
            value={selectedCategory || ''} 
            onChange={(e) => onCategoryChange(e.target.value || null)}
            className="w-full border-2 border-gray-200 rounded-xl px-4 py-3 text-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
          >
            <option value="">All Categories</option>
            {categories.map(cat => (
              <option key={cat.category_name} value={cat.category_name}>
                {cat.category_name} ({cat.article_count})
              </option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Source</label>
          <select 
            value={selectedSource || ''} 
            onChange={(e) => onSourceChange(e.target.value || null)}
            className="w-full border-2 border-gray-200 rounded-xl px-4 py-3 text-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all"
          >
            <option value="">All Sources</option>
            {sources.map(source => (
              <option key={source.source_name} value={source.source_name}>
                {source.source_name} ({source.article_count})
              </option>
            ))}
          </select>
        </div>
        
        <div className="flex items-end">
          <button 
            onClick={onReset}
            className="w-full bg-gradient-to-r from-gray-500 to-gray-600 text-white px-6 py-3 rounded-xl hover:from-gray-600 hover:to-gray-700 transition-all duration-200 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
          >
            Reset Filters
          </button>
        </div>
      </div>
    </div>
  );
};

// Pagination Component
const Pagination = ({ currentPage, totalPages, onPageChange }) => {
  return (
    <div className="flex items-center justify-center space-x-3 mt-12">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
        className="flex items-center px-4 py-2 border-2 border-gray-200 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 hover:border-gray-300 transition-all duration-200"
      >
        <ChevronLeft size={16} className="mr-1" />
        Previous
      </button>
      
      <span className="px-6 py-2 text-sm text-gray-600 bg-gray-50 rounded-xl font-medium">
        Page {currentPage} of {totalPages}
      </span>
      
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
        className="flex items-center px-4 py-2 border-2 border-gray-200 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 hover:border-gray-300 transition-all duration-200"
      >
        Next
        <ChevronRight size={16} className="ml-1" />
      </button>
    </div>
  );
};

// Main App Component
const NewsApp = () => {
  const [articles, setArticles] = useState([]);
  const [categories, setCategories] = useState([]);
  const [sources, setSources] = useState([]);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [selectedSource, setSelectedSource] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [stats, setStats] = useState({});
  
  const { request, loading, error } = useApi();

  // Fetch articles
  const fetchArticles = async (page = 1, category = null, source = null) => {
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: '12'
      });
      
      if (category) params.append('category', category);
      if (source) params.append('source', source);
      
      const data = await request(`/articles?${params}`);
      setArticles(data.articles);
      setCurrentPage(data.page);
      setTotalPages(data.total_pages);
    } catch (err) {
      console.error('Failed to fetch articles:', err);
    }
  };

  // Search articles
  const searchArticles = async (query) => {
    if (!query.trim()) {
      fetchArticles();
      return;
    }
    
    try {
      const data = await request(`/search?q=${encodeURIComponent(query)}&limit=20`);
      setArticles(data.articles);
      setCurrentPage(1);
      setTotalPages(1);
    } catch (err) {
      console.error('Failed to search articles:', err);
    }
  };

  // Fetch metadata
  const fetchMetadata = async () => {
    try {
      const [categoriesData, sourcesData, statsData] = await Promise.all([
        request('/categories'),
        request('/sources'),
        request('/stats')
      ]);
      
      setCategories(categoriesData);
      setSources(sourcesData);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to fetch metadata:', err);
    }
  };

  // Handle search
  const handleSearch = (e) => {
    if (e) e.preventDefault();
    searchArticles(searchQuery);
  };

  // Handle filter changes
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setCurrentPage(1);
    if (!searchQuery) {
      fetchArticles(1, category, selectedSource);
    }
  };

  const handleSourceChange = (source) => {
    setSelectedSource(source);
    setCurrentPage(1);
    if (!searchQuery) {
      fetchArticles(1, selectedCategory, source);
    }
  };

  const handleResetFilters = () => {
    setSelectedCategory(null);
    setSelectedSource(null);
    setSearchQuery('');
    setCurrentPage(1);
    fetchArticles();
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
    if (!searchQuery) {
      fetchArticles(page, selectedCategory, selectedSource);
    }
  };

  // Load initial data
  useEffect(() => {
    fetchArticles();
    fetchMetadata();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-gray-200/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mr-4">
                <TrendingUp className="h-7 w-7 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                News Curator
              </h1>
            </div>
            
            {/* Search Bar */}
            <div className="flex-1 max-w-2xl mx-8">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search articles..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch(e)}
                  className="w-full pl-12 pr-6 py-4 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition-all duration-200 text-lg placeholder-gray-400"
                />
                <Search className="absolute left-4 top-4 h-6 w-6 text-gray-400" />
              </div>
            </div>
            
            {/* Stats */}
            <div className="flex items-center space-x-6 text-sm">
              <div className="text-center">
                <div className="font-bold text-lg text-gray-900">{stats.total_articles || 0}</div>
                <div className="text-gray-500">Total Articles</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-lg text-blue-600">{stats.articles_today || 0}</div>
                <div className="text-gray-500">Today</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Filters */}
        <FiltersPanel
          categories={categories}
          sources={sources}
          selectedCategory={selectedCategory}
          selectedSource={selectedSource}
          onCategoryChange={handleCategoryChange}
          onSourceChange={handleSourceChange}
          onReset={handleResetFilters}
        />

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col justify-center items-center h-64">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-200 border-t-blue-600 mb-4"></div>
            <p className="text-gray-600 font-medium">Loading articles...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 mb-8 text-center">
            <p className="text-red-800 font-medium">Error: {error}</p>
          </div>
        )}

        {/* Search Results Info */}
        {searchQuery && !loading && (
          <div className="mb-8 text-center">
            <p className="text-gray-600 text-lg">
              Search results for <span className="font-semibold text-blue-600">"{searchQuery}"</span> 
              <span className="text-gray-500"> • {articles.length} articles found</span>
            </p>
          </div>
        )}

        {/* Articles Grid */}
        {!loading && articles.length > 0 && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {articles.map((article) => (
                <ArticleCard
                  key={article.id}
                  article={article}
                  onClick={setSelectedArticle}
                />
              ))}
            </div>

            {/* Pagination */}
            {!searchQuery && totalPages > 1 && (
              <Pagination
                currentPage={currentPage}
                totalPages={totalPages}
                onPageChange={handlePageChange}
              />
            )}
          </>
        )}

        {/* No Results */}
        {!loading && articles.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <Search className="h-10 w-10 text-gray-400" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              {searchQuery ? 'No articles found' : 'No articles available'}
            </h3>
            <p className="text-gray-600 text-lg max-w-md mx-auto">
              {searchQuery 
                ? 'Try adjusting your search terms or filters to find what you\'re looking for'
                : 'Check back later for new articles and updates'
              }
            </p>
          </div>
        )}
      </div>

      {/* Article Modal */}
      {selectedArticle && (
        <ArticleModal
          article={selectedArticle}
          onClose={() => setSelectedArticle(null)}
        />
      )}
    </div>
  );
};

export default NewsApp;