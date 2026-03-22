import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Ollama API settings
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1"
    OLLAMA_MODEL: str = "qwen2.5"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Web search settings
    WEB_SEARCH_MAX_RESULTS: int = 5   # Max results per live web search
    WEB_SEARCH_ENABLED: bool = True   # Toggle web search without code changes

    # News RAG settings
    NEWS_RSS_URL: str = "https://feeds.bbci.co.uk/news/rss.xml"  # BBC News top stories feed
    NEWS_MAX_ARTICLES: int = 20          # Max articles to fetch per refresh
    NEWS_RETENTION_HOURS: int = 72       # Delete articles older than this many hours
    NEWS_REFRESH_INTERVAL_MINUTES: int = 60  # Auto-refresh interval (informational)

config = Config()


