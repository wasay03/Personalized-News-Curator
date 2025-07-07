"""
News Curator API - Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from dependencies import db_manager
from routers import articles, search, metadata
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db_manager.connect()
    yield
    # Shutdown
    db_manager.disconnect()

# Initialize FastAPI app
app = FastAPI(
    title="News Curator API",
    description="API for accessing curated news articles",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(articles.router, prefix="/api", tags=["articles"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(metadata.router, prefix="/api", tags=["metadata"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "News Curator API is running", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)