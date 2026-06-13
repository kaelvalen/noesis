"""Web ingestion wrapper.

Uses Firecrawl to fetch pages and convert them to clean markdown/text.
Falls back to simple urllib-based fetching if Firecrawl is unavailable.
"""

import re
import warnings


def _fetch_with_firecrawl(url):
    try:
        from firecrawl import FirecrawlApp

        app = FirecrawlApp()
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        return result.get("markdown", "") or result.get("html", "")
    except Exception as exc:
        warnings.warn(f"Firecrawl failed for {url}: {exc}")
        return None


def _fetch_with_urllib(url):
    try:
        from urllib.request import Request, urlopen

        req = Request(url, headers={"User-Agent": "NOESIS/0.1"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="ignore")
    except Exception as exc:
        warnings.warn(f"urllib fetch failed for {url}: {exc}")
        return ""


def _clean_html(html):
    """Very basic HTML-to-text cleaner."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ingest_web(url, use_firecrawl=True):
    """Fetch a single URL and return plain text."""
    text = None
    if use_firecrawl:
        text = _fetch_with_firecrawl(url)
    if text is None:
        text = _fetch_with_urllib(url)
    # If it looks like HTML, strip tags.
    if "<html" in text.lower() or "<!doctype" in text.lower():
        text = _clean_html(text)
    return text
