# RAG Integration Guide for News Curator

## Overview

Adding RAG (Retrieval-Augmented Generation) transforms your news curator from a passive recommendation system into an interactive knowledge base where users can ask questions about news articles and get AI-generated answers based on the actual content.

## Key Benefits

- **Interactive Q&A**: Users can ask specific questions about news events
- **Source Attribution**: Answers are grounded in actual articles with citations
- **Temporal Queries**: "What happened with X last week?"
- **Cross-Article Synthesis**: Combine information from multiple sources
- **Trending Analysis**: Identify emerging topics and themes

## Architecture Components

### 1. Vector Database Layer
```
News Articles → Text Chunks → Embeddings → Vector DB (ChromaDB/Pinecone)
```

### 2. Retrieval System
```
User Query → Query Embedding → Similarity Search → Relevant Chunks
```

### 3. Generation System
```
Query + Retrieved Context → LLM → Generated Answer + Sources
```

## Step-by-Step Integration

### Phase 1: Install Dependencies

```bash
pip install sentence-transformers chromadb faiss-cpu langchain tiktoken spacy
pip install transformers torch
python -m spacy download en_core_web_sm

# For production, consider:
pip install pinecone-client  # Alternative to ChromaDB
pip install openai          # For GPT-based generation
```

### Phase 2: Modify Your Existing News Collector

```python
# Enhanced news collector with RAG processing
from your_existing_collector import NewsCollector
from rag_system import NewsRAGSystem

class RAGEnabledNewsCollector(NewsCollector):
    def __init__(self):
        super().__init__()
        self.rag_system = NewsRAGSystem()
    
    def collect_and_process_articles(self):
        # Use your existing collection logic
        articles = self.collect_from_rss()
        
        # Process each article for RAG
        processed_count = 0
        for article in articles:
            try:
                # Add to RAG system
                article_id = self.rag_system.process_and_store_article(article)
                processed_count += 1
                print(f"Processed article {article_id}: {article['title'][:50]}...")
                
            except Exception as e:
                print(f"Error processing article: {e}")
        
        print(f"Successfully processed {processed_count} articles for RAG")
        return articles
```

### Phase 3: Frontend Integration

#### Enhanced HTML Interface
```html
<!-- Add to your existing index.html -->
<div class="rag-section">
    <h2>Ask Questions About the News</h2>
    
    <div class="question-input">
        <input type="text" id="question-input" placeholder="Ask me anything about recent news..." />
        <button onclick="askQuestion()">Ask</button>
    </div>
    
    <div class="filters">
        <select id="category-filter">
            <option value="">All Categories</option>
            <option value="politics">Politics</option>
            <option value="technology">Technology</option>
            <option value="health">Health</option>
        </select>
        
        <select id="timeframe-filter">
            <option value="">All Time</option>
            <option value="1">Last 24 hours</option>
            <option value="7">Last week</option>
            <option value="30">Last month</option>
        </select>
    </div>
    
    <div id="suggested-questions">
        <h3>Suggested Questions</h3>
        <div id="questions-list"></div>
    </div>
    
    <div id="answer-section" style="display: none;">
        <h3>Answer</h3>
        <div id="answer-text"></div>
        <div id="sources">
            <h4>Sources</h4>
            <ul id="sources-list"></ul>
        </div>
        <div id="confidence">
            <span>Confidence: <span id="confidence-score"></span>%</span>
        </div>
    </div>
</div>

<div class="trending-topics">
    <h3>Trending Topics</h3>
    <div id="trending-list"></div>
</div>
```

#### Enhanced JavaScript
```javascript
// Add to your existing script.js
class RAGInterface {
    constructor() {
        this.apiBase = 'http://localhost:5001/api';
        this.loadSuggestedQuestions();
        this.loadTrendingTopics();
    }
    
    async askQuestion() {
        const question = document.getElementById('question-input').value;
        const category = document.getElementById('category-filter').value;
        const timeframe = document.getElementById('timeframe-filter').value;
        
        if (!question.trim()) return;
        
        // Show loading state
        this.showLoading();
        
        try {
            const response = await fetch(`${this.apiBase}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    filters: {
                        category: category || undefined,
                        days: timeframe ? parseInt(timeframe) : undefined
                    }
                })
            });
            
            const result = await response.json();
            this.displayAnswer(result);
            
        } catch (error) {
            console.error('Error asking question:', error);
            this.showError('Failed to get answer. Please try again.');
        }
    }
    
    displayAnswer(result) {
        document.getElementById('answer-text').textContent = result.answer;
        
        // Display sources
        const sourcesList = document.getElementById('sources-list');
        sourcesList.innerHTML = '';
        
        result.sources.forEach(source => {
            const li = document.createElement('li');
            li.innerHTML = `
                <a href="${source.url}" target="_blank">${source.title}</a>
                <span class="source-info">${source.source} (${Math.round(source.relevance_score * 100)}% relevant)</span>
            `;
            sourcesList.appendChild(li);
        });
        
        // Display confidence
        document.getElementById('confidence-score').textContent = 
            Math.round(result.confidence * 100);
        
        // Show answer section
        document.getElementById('answer-section').style.display = 'block';
    }
    
    async loadSuggestedQuestions(category = null) {
        try {
            const url = `${this.apiBase}/suggest-questions${category ? `?category=${category}` : ''}`;
            const response = await fetch(url);
            const data = await response.json();
            
            const questionsList = document.getElementById('questions-list');
            questionsList.innerHTML = '';
            
            data.suggested_questions.forEach(question => {
                const button = document.createElement('button');
                button.className = 'suggested-question';
                button.textContent = question;
                button.onclick = () => {
                    document.getElementById('question-input').value = question;
                    this.askQuestion();
                };
                questionsList.appendChild(button);
            });
            
        } catch (error) {
            console.error('Error loading suggested questions:', error);
        }
    }
    
    async loadTrendingTopics() {
        try {
            const response = await fetch(`${this.apiBase}/trending-topics`);
            const data = await response.json();
            
            const trendingList = document.getElementById('trending-list');
            trendingList.innerHTML = '';
            
            data.trending_topics.slice(0, 10).forEach(topic => {
                const div = document.createElement('div');
                div.className = 'trending-topic';
                div.innerHTML = `
                    <span class="topic-name">${topic.entity}</span>
                    <span class="topic-type">${topic.type}</span>
                    <span class="topic-mentions">${topic.mentions} mentions</span>
                `;
                div.onclick = () => {
                    const question = `Tell me about ${topic.entity}`;
                    document.getElementById('question-input').value = question;
                    this.askQuestion();
                };
                trendingList.appendChild(div);
            });
            
        } catch (error) {
            console.error('Error loading trending topics:', error);
        }
    }
    
    showLoading() {
        document.getElementById('answer-text').innerHTML = '<div class="loading">Searching for answer...</div>';
        document.getElementById('answer-section').style.display = 'block';
    }
    
    showError(message) {
        document.getElementById('answer-text').innerHTML = `<div class="error">${message}</div>`;
    }
}

// Initialize RAG interface
const ragInterface = new RAGInterface();

// Global function for button onclick
function askQuestion() {
    ragInterface.askQuestion();
}

// Update suggested questions when category changes
document.getElementById('category-filter').addEventListener('change', function() {
    ragInterface.loadSuggestedQuestions(this.value);
});
```

### Phase 4: Enhanced CSS
```css
/* Add to your existing styles.css */
.rag-section {
    margin: 20px 0;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #f9f9f9;
}

.question-input {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}

.question-input input {
    flex: 1;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
}

.question-input button {
    padding: 10px 20px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.filters {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
}

.filters select {
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

.suggested-question {
    display: block;
    width: 100%;
    margin: 5px 0;
    padding: 8px;
    background-color: #e9ecef;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    cursor: pointer;
    text-align: left;
}

.suggested-question:hover {
    background-color: #dee2e6;
}

#answer-section {
    margin-top: 20px;
    padding: 15px;
    background-color: white;
    border-radius: 4px;
    border: 1px solid #dee2e6;
}

#sources-list li {
    margin: 10px 0;
    padding: 8px;
    background-color: #f8f9fa;
    border-radius: 4px;
}

.source-info {
    display: block;
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}

.trending-topics {
    margin: 20px 0;
}

.trending-topic {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    margin: 5px 0;
    background-color: #f8f9fa;
    border-radius: 4px;
    cursor: pointer;
}

.trending-topic:hover {
    background-color: #e9ecef;
}

.topic-name {
    font-weight: bold;
}

.topic-type {
    font-size: 12px;
    color: #666;
    background-color: #dee2e6;
    padding: 2px 6px;
    border-radius: 12px;
}

.topic-mentions {
    font-size: 12px;
    color: #007bff;
}

.loading {
    text-align: center;
    color: #666;
    font-style: italic;
}

.error {
    color: #dc3545;
    background-color: #f8d7da;
    padding: 10px;
    border-radius: 4px;
}

#confidence {
    margin-top: 10px;
    font-size: 12px;
    color: #666;
}
```

## Advanced Features

### 1. Real-time Question Suggestions
```python
def get_contextual_questions(self, current_article_id: int) -> List[str]:
    """Generate questions based on current article context"""
    # Get article entities and content
    # Use LLM to generate relevant questions
    pass
```

### 2. Multi-modal RAG
```python
# Support for images, videos in news articles
def process_multimodal_article(self, article_with_media):
    # Extract text from images using OCR
    # Process video transcripts
    # Combine with text content
    pass
```

### 3. Conversation Memory
```python
class ConversationalRAG:
    def __init__(self):
        self.conversation_