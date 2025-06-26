# RAG-Enhanced News Curator with Question Answering
import os
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

# Core RAG dependencies
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings

# Text processing
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class NewsRAGSystem:
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2",
                 llm_model_name="microsoft/DialoGPT-medium",
                 chunk_size=500,
                 chunk_overlap=50):
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LLM for generation
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        # Initialize vector database
        self.setup_vector_db()
        
        # Text splitter for chunking articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Initialize NLP for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        
        # Token counter for context management
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def setup_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("news_articles")
        except:
            self.collection = self.chroma_client.create_collection(
                name="news_articles",
                metadata={"hnsw:space": "cosine"}
            )
    
    def setup_database(self):
        """Setup SQLite database for article metadata"""
        conn = sqlite3.connect('news_rag.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                url TEXT UNIQUE,
                source TEXT,
                published TEXT,
                category TEXT,
                summary TEXT,
                entities TEXT,
                collected_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS article_chunks (
                id INTEGER PRIMARY KEY,
                article_id INTEGER,
                chunk_text TEXT,
                chunk_index INTEGER,
                vector_id TEXT,
                FOREIGN KEY (article_id) REFERENCES articles (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_and_store_article(self, article: Dict):
        """Process article and store in both SQL and vector DB"""
        # Extract entities and generate summary
        entities = self.extract_entities(article['content'])
        summary = self.generate_summary(article['content'])
        
        # Store in SQL database
        conn = sqlite3.connect('news_rag.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO articles 
            (title, content, url, source, published, category, summary, entities, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['title'],
            article['content'],
            article['url'],
            article['source'],
            article.get('published', ''),
            article.get('category', 'general'),
            summary,
            json.dumps(entities),
            datetime.now().isoformat()
        ))
        
        article_id = cursor.lastrowid
        conn.commit()
        
        # Chunk the article content
        full_text = f"Title: {article['title']}\n\nContent: {article['content']}"
        chunks = self.text_splitter.split_text(full_text)
        
        # Generate embeddings and store in vector DB
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            vector_id = f"{article_id}_{i}"
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{
                    "article_id": article_id,
                    "chunk_index": i,
                    "title": article['title'],
                    "source": article['source'],
                    "url": article['url'],
                    "category": article.get('category', 'general'),
                    "published": article.get('published', '')
                }],
                ids=[vector_id]
            )
            
            # Store chunk metadata in SQL
            cursor.execute('''
                INSERT INTO article_chunks 
                (article_id, chunk_text, chunk_index, vector_id)
                VALUES (?, ?, ?, ?)
            ''', (article_id, chunk, i, vector_id))
        
        conn.commit()
        conn.close()
        
        return article_id
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        doc = self.nlp(text[:1000000])  # Limit text length for processing
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_)
            })
        
        return entities
    
    def generate_summary(self, text: str, max_length: 150) -> str:
        """Generate article summary"""
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Truncate text if too long
        if len(text) > 1024:
            text = text[:1024]
        
        try:
            summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except:
            # Fallback to first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5, 
                               filters: Dict = None) -> List[Dict]:
        """Retrieve most relevant article chunks for a query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Prepare where filter if provided
        where_filter = {}
        if filters:
            if 'category' in filters:
                where_filter['category'] = filters['category']
            if 'source' in filters:
                where_filter['source'] = filters['source']
            if 'date_from' in filters:
                where_filter['published'] = {"$gte": filters['date_from']}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return retrieved_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       max_context_tokens: int = 2000) -> str:
        """Generate answer using retrieved context"""
        
        # Build context from retrieved chunks
        context_parts = []
        token_count = 0
        
        for chunk in context_chunks:
            chunk_text = f"Source: {chunk['metadata']['source']}\n"
            chunk_text += f"Title: {chunk['metadata']['title']}\n"
            chunk_text += f"Content: {chunk['content']}\n\n"
            
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            if token_count + chunk_tokens > max_context_tokens:
                break
                
            context_parts.append(chunk_text)
            token_count += chunk_tokens
        
        context = "".join(context_parts)
        
        # Create prompt for answer generation
        prompt = f"""Based on the following news articles, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer: """
        
        # Generate answer
        try:
            response = self.generator(
                f prompt,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract just the answer part
            full_response = response[0]['generated_text']
            answer = full_response.split("Answer: ")[-1].strip()
            
            return answer
            
        except Exception as e:
            return f"I apologize, but I encountered an error generating the answer: {str(e)}"
    
    def ask_question(self, query: str, filters: Dict = None, 
                    n_chunks: int = 5) -> Dict:
        """Main method for question answering"""
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(
            query, n_results=n_chunks, filters=filters
        )
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        
        # Extract unique sources
        sources = []
        seen_urls = set()
        
        for chunk in relevant_chunks:
            url = chunk['metadata']['url']
            if url not in seen_urls:
                sources.append({
                    'title': chunk['metadata']['title'],
                    'source': chunk['metadata']['source'],
                    'url': url,
                    'relevance_score': 1 - (chunk['distance'] or 0)
                })
                seen_urls.add(url)
        
        # Calculate confidence based on retrieval scores
        avg_distance = np.mean([chunk['distance'] or 0 for chunk in relevant_chunks])
        confidence = max(0, 1 - avg_distance)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "retrieved_chunks": len(relevant_chunks)
        }
    
    def get_trending_topics(self, days: int = 7) -> List[Dict]:
        """Get trending topics from recent articles"""
        conn = sqlite3.connect('news_rag.db')
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT entities FROM articles 
            WHERE collected_at > ? AND entities IS NOT NULL
        ''', (cutoff_date,))
        
        all_entities = []
        for row in cursor.fetchall():
            try:
                entities = json.loads(row[0])
                all_entities.extend(entities)
            except:
                continue
        
        # Count entity frequency
        entity_counts = {}
        for entity in all_entities:
            key = (entity['text'], entity['label'])
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        # Sort by frequency
        trending = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        trending_topics = []
        for (text, label), count in trending:
            trending_topics.append({
                'entity': text,
                'type': label,
                'mentions': count
            })
        
        conn.close()
        return trending_topics
    
    def suggest_questions(self, article_id: int = None, category: str = None) -> List[str]:
        """Suggest relevant questions based on content"""
        
        if article_id:
            # Get specific article
            conn = sqlite3.connect('news_rag.db')
            cursor = conn.cursor()
            cursor.execute('SELECT title, content, entities FROM articles WHERE id = ?', (article_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                title, content, entities_json = result
                entities = json.loads(entities_json) if entities_json else []
                
                questions = [
                    f"What is the main point of the article about {title}?",
                    "Who are the key people mentioned in this article?",
                    "What are the implications of this news?",
                ]
                
                # Add entity-specific questions
                for entity in entities[:3]:  # Top 3 entities
                    if entity['label'] in ['PERSON', 'ORG']:
                        questions.append(f"What role does {entity['text']} play in this story?")
                    elif entity['label'] in ['GPE', 'LOC']:
                        questions.append(f"How does this news affect {entity['text']}?")
                
                return questions
        
        # Generic questions for category
        generic_questions = {
            'politics': [
                "What are the latest political developments?",
                "Who are the key political figures in recent news?",
                "What policies are being discussed?"
            ],
            'technology': [
                "What are the latest tech innovations?",
                "Which companies are making headlines in tech?",
                "What are the implications of recent tech developments?"
            ],
            'health': [
                "What are the latest health and medical news?",
                "What health recommendations are experts making?",
                "What medical breakthroughs have been reported?"
            ]
        }
        
        return generic_questions.get(category, [
            "What are the most important news stories today?",
            "What trends are emerging in the news?",
            "What should I know about current events?"
        ])

# Example usage and API integration
class NewsRAGAPI:
    def __init__(self):
        self.rag_system = NewsRAGSystem()
        self.rag_system.setup_database()
    
    def add_article(self, article: Dict) -> int:
        """Add new article to the system"""
        return self.rag_system.process_and_store_article(article)
    
    def ask_question(self, query: str, filters: Dict = None) -> Dict:
        """Ask a question about the news"""
        return self.rag_system.ask_question(query, filters)
    
    def get_suggested_questions(self, article_id: int = None, 
                              category: str = None) -> List[str]:
        """Get suggested questions"""
        return self.rag_system.suggest_questions(article_id, category)
    
    def get_trending_topics(self, days: int = 7) -> List[Dict]:
        """Get trending topics"""
        return self.rag_system.get_trending_topics(days)

# Flask API endpoints for RAG functionality
from flask import Flask, request, jsonify

app = Flask(__name__)
rag_api = NewsRAGAPI()

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('question', '')
    filters = data.get('filters', {})
    
    if not query:
        return jsonify({'error': 'Question is required'}), 400
    
    result = rag_api.ask_question(query, filters)
    return jsonify(result)

@app.route('/api/suggest-questions')
def suggest_questions():
    article_id = request.args.get('article_id', type=int)
    category = request.args.get('category')
    
    questions = rag_api.get_suggested_questions(article_id, category)
    return jsonify({'suggested_questions': questions})

@app.route('/api/trending-topics')
def trending_topics():
    days = request.args.get('days', 7, type=int)
    topics = rag_api.get_trending_topics(days)
    return jsonify({'trending_topics': topics})

@app.route('/api/add-article', methods=['POST'])
def add_article():
    article = request.json
    
    required_fields = ['title', 'content', 'url', 'source']
    if not all(field in article for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    article_id = rag_api.add_article(article)
    return jsonify({'article_id': article_id, 'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True, port=5001)