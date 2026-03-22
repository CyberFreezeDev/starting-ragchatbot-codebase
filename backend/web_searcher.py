"""
WebSearcher — searches the web using a headless Chromium browser via Playwright.

Fallback chain: Google → Bing → DuckDuckGo
- Google tried first (best results, but frequently blocks bots)
- Bing tried second (good results, more lenient than Google)
- DuckDuckGo always works (HTML version, no JS, no bot detection)

Results are ephemeral — returned inline to the LLM, never stored in ChromaDB.
"""

from typing import List, Dict, Optional
import urllib.parse


# Realistic user-agent to reduce bot detection
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class WebSearcher:
    """Searches the web with a Google→Bing→DuckDuckGo fallback chain."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict]:
        """
        Run a web search with automatic engine fallback.

        Returns a list of dicts: [{"title", "url", "snippet", "engine"}, ...]
        Returns [] if all engines fail.
        """
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=_USER_AGENT)
            page = context.new_page()

            for engine_fn in [
                self._search_google,
                self._search_bing,
                self._search_duckduckgo,
            ]:
                try:
                    results = engine_fn(page, query)
                    if results:
                        browser.close()
                        return results
                except Exception as e:
                    print(f"[WebSearcher] {engine_fn.__name__} failed: {e}")
                    continue

            browser.close()
        return []

    # ------------------------------------------------------------------ #
    #  Engine implementations                                             #
    # ------------------------------------------------------------------ #

    def _search_google(self, page, query: str) -> List[Dict]:
        """Try Google. Returns [] if blocked (CAPTCHA / no results)."""
        encoded = urllib.parse.quote_plus(query)
        page.goto(f"https://www.google.com/search?q={encoded}&hl=en", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        # Detect CAPTCHA or consent gate
        title = page.title().lower()
        if "captcha" in title or "consent" in title or "before you continue" in title:
            return []

        results = []
        # Google result blocks: each organic result is inside a <div class="g"> or similar
        # Use a broad locator that works across Google's frequently-changing DOM
        blocks = page.locator("div.g").all()[:self.max_results * 2]  # grab extra, filter below

        for block in blocks:
            try:
                title_el = block.locator("h3").first
                link_el = block.locator("a").first
                snippet_el = block.locator("[data-sncf], [style*='webkit-line-clamp'], .VwiC3b").first

                title_text = title_el.inner_text(timeout=2000).strip()
                url = link_el.get_attribute("href", timeout=2000) or ""
                snippet = ""
                try:
                    snippet = snippet_el.inner_text(timeout=2000).strip()
                except Exception:
                    pass

                if title_text and url.startswith("http"):
                    results.append({"title": title_text, "url": url, "snippet": snippet, "engine": "Google"})
                    if len(results) >= self.max_results:
                        break
            except Exception:
                continue

        return results

    def _search_bing(self, page, query: str) -> List[Dict]:
        """Try Bing. Returns [] if blocked or no results."""
        encoded = urllib.parse.quote_plus(query)
        page.goto(f"https://www.bing.com/search?q={encoded}", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        title = page.title().lower()
        if "captcha" in title or "blocked" in title:
            return []

        results = []
        # Bing organic results are in <li class="b_algo">
        blocks = page.locator("li.b_algo").all()[:self.max_results * 2]

        for block in blocks:
            try:
                title_el = block.locator("h2 a").first
                snippet_el = block.locator("p, .b_caption p").first

                title_text = title_el.inner_text(timeout=2000).strip()
                url = title_el.get_attribute("href", timeout=2000) or ""
                snippet = ""
                try:
                    snippet = snippet_el.inner_text(timeout=2000).strip()
                except Exception:
                    pass

                if title_text and url.startswith("http"):
                    results.append({"title": title_text, "url": url, "snippet": snippet, "engine": "Bing"})
                    if len(results) >= self.max_results:
                        break
            except Exception:
                continue

        return results

    def _search_duckduckgo(self, page, query: str) -> List[Dict]:
        """
        DuckDuckGo HTML version — no JS, no CAPTCHA, always works.
        Uses the lite endpoint: html.duckduckgo.com/html/
        """
        encoded = urllib.parse.quote_plus(query)
        page.goto(f"https://html.duckduckgo.com/html/?q={encoded}", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        results = []
        # DDG HTML results: each result is a <div class="result">
        blocks = page.locator("div.result").all()[:self.max_results * 2]

        for block in blocks:
            try:
                title_el = block.locator("a.result__a").first
                snippet_el = block.locator("a.result__snippet").first

                title_text = title_el.inner_text(timeout=2000).strip()
                url = title_el.get_attribute("href", timeout=2000) or ""
                snippet = ""
                try:
                    snippet = snippet_el.inner_text(timeout=2000).strip()
                except Exception:
                    pass

                # DDG HTML links go through a redirect — grab the actual URL from data-href or href
                if not url.startswith("http"):
                    url = block.locator("a.result__url").first.inner_text(timeout=2000).strip()
                    if not url.startswith("http"):
                        url = "https://" + url

                if title_text and url:
                    results.append({"title": title_text, "url": url, "snippet": snippet, "engine": "DuckDuckGo"})
                    if len(results) >= self.max_results:
                        break
            except Exception:
                continue

        return results
