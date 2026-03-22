"""
NewsRefreshManager — orchestrates the scrape → chunk → embed → store cycle.

Responsibilities:
1. Call NewsArticleScraper to fetch fresh articles
2. Skip articles already indexed (URL-based deduplication)
3. Chunk article text using the same sentence-chunker as DocumentProcessor
4. Store NewsChunk objects in VectorStore's news_content collection
5. Delete articles older than retention_hours after each refresh
"""

from typing import List, Tuple
from datetime import datetime, timezone

from models import NewsArticle, NewsChunk
from news_scraper import NewsArticleScraper
from document_processor import DocumentProcessor
from vector_store import VectorStore


class NewsRefreshManager:
    """Orchestrates scrape → chunk → embed → store for news articles."""

    def __init__(
        self,
        vector_store: VectorStore,
        rss_url: str,
        max_articles: int = 20,
        retention_hours: int = 72,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.vector_store = vector_store
        self.retention_hours = retention_hours
        self.scraper = NewsArticleScraper(rss_url, max_articles=max_articles)
        # Reuse DocumentProcessor's sentence-chunker
        self._chunker = DocumentProcessor(chunk_size, chunk_overlap)

    def refresh(self) -> Tuple[int, int]:
        """
        Fetch latest news, skip duplicates, store new chunks, prune old ones.

        Returns:
            (articles_added, chunks_added)
        """
        # 1. Fetch articles from RSS
        articles = self.scraper.fetch()

        # 2. Find which URLs are already indexed
        existing_urls = set(self.vector_store.get_existing_news_urls())

        # 3. Chunk and store new articles
        articles_added = 0
        chunks_added = 0
        for article in articles:
            if article.url in existing_urls:
                continue

            chunks = self._chunk_article(article)
            if chunks:
                self.vector_store.add_news_chunks(chunks)
                articles_added += 1
                chunks_added += len(chunks)

        # 4. Prune stale news
        self.vector_store.clear_old_news(self.retention_hours)

        print(f"[NewsRefreshManager] +{articles_added} articles, +{chunks_added} chunks "
              f"(pruned articles older than {self.retention_hours}h)")
        return articles_added, chunks_added

    def _chunk_article(self, article: NewsArticle) -> List[NewsChunk]:
        """Split article text into NewsChunk objects."""
        text = article.summary  # summary holds the full body (or RSS summary as fallback)
        if not text or not text.strip():
            return []

        raw_chunks = self._chunker.chunk_text(text)
        return [
            NewsChunk(
                content=chunk,
                url=article.url,
                title=article.title,
                source=article.source,
                section=article.section,
                published_at=article.published_at,
                fetched_at=article.fetched_at,
                chunk_index=i,
            )
            for i, chunk in enumerate(raw_chunks)
        ]
