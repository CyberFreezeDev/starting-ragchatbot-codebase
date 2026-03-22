"""
NewsArticleScraper — fetches articles from an RSS feed using feedparser + httpx.

Design decisions:
- feedparser handles RSS parsing (no fragile HTML scraping needed for feed metadata)
- httpx fetches the full article body for richer content
- Falls back to RSS summary if full-body fetch fails
- URL is used as the unique article ID to prevent duplicate indexing
"""

import httpx
import feedparser
from datetime import datetime, timezone
from typing import List, Optional
from models import NewsArticle


def _parse_rss_datetime(struct_time) -> Optional[str]:
    """Convert feedparser's time.struct_time to ISO-8601 string."""
    if not struct_time:
        return None
    try:
        dt = datetime(*struct_time[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def _fetch_article_text(url: str, timeout: int = 8) -> Optional[str]:
    """
    Fetch full article text from a URL.
    Returns the <article> or <main> text if available, else None.
    Falls back gracefully on network errors.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        # Simple extraction: find article/main tag contents without BeautifulSoup
        # (avoids adding bs4 dependency; BBC articles have predictable structure)
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._in_target = False
                self._depth = 0
                self.text_parts = []

            def handle_starttag(self, tag, attrs):
                if tag in ("article", "main") and not self._in_target:
                    self._in_target = True
                    self._depth = 1
                elif self._in_target:
                    self._depth += 1

            def handle_endtag(self, tag):
                if self._in_target:
                    self._depth -= 1
                    if self._depth <= 0:
                        self._in_target = False

            def handle_data(self, data):
                if self._in_target:
                    stripped = data.strip()
                    if stripped:
                        self.text_parts.append(stripped)

        extractor = _TextExtractor()
        extractor.feed(response.text)
        text = " ".join(extractor.text_parts)
        return text if len(text) > 100 else None
    except Exception:
        return None


class NewsArticleScraper:
    """Fetches news articles from an RSS feed and optionally enriches with full text."""

    def __init__(self, rss_url: str, max_articles: int = 20, fetch_full_text: bool = True):
        self.rss_url = rss_url
        self.max_articles = max_articles
        self.fetch_full_text = fetch_full_text

    def fetch(self) -> List[NewsArticle]:
        """
        Parse the RSS feed and return up to max_articles NewsArticle objects.
        Each article's content = full body if fetchable, else RSS summary.
        """
        feed = feedparser.parse(self.rss_url)
        fetched_at = datetime.now(timezone.utc).isoformat()
        articles: List[NewsArticle] = []

        for entry in feed.entries[: self.max_articles]:
            url = getattr(entry, "link", None)
            if not url:
                continue

            title = getattr(entry, "title", "Untitled")
            summary = getattr(entry, "summary", "") or ""

            # Try to get a richer section label from tags/categories
            tags = getattr(entry, "tags", [])
            section = tags[0].get("term", "general").lower() if tags else "general"

            published_at = _parse_rss_datetime(getattr(entry, "published_parsed", None))

            # Use full article text if available, else fall back to RSS summary
            body = None
            if self.fetch_full_text:
                body = _fetch_article_text(url)
            content = body if body else summary

            articles.append(NewsArticle(
                url=url,
                title=title,
                summary=content,
                published_at=published_at,
                fetched_at=fetched_at,
                source="BBC News",
                section=section,
            ))

        return articles
