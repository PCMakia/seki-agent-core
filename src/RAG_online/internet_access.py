"""
Internet search, scrape, and summarization.

Uses httpx + DuckDuckGo Lite for search URLs, Playwright to load pages,
BeautifulSoup4 for content extraction.

Summarization modes:
- ``summarize_scrape_extractive`` — extractive bullets (Qwen3-Embed when available,
  else simple sentence trim); no extra LLM call.
- ``search_and_summarize`` — abstractive summary via the project LLM (Ollama).

Flow: query -> search -> top N URLs -> scrape -> combine text -> summarize.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

from bs4 import BeautifulSoup

from src.LLM_handler.llm_client import LLMClient

# Search and scrape limits
TOP_N_URLS = 15
MAX_CHARS_PER_PAGE = 6000  # Truncate to avoid context overflow

# DuckDuckGo Lite (html.duckduckgo.com returns "Error getting results" for bots)
DDG_LITE_SEARCH = "https://lite.duckduckgo.com/lite/"

# User-Agent for DuckDuckGo (403 without it)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _extract_main_text(soup: BeautifulSoup) -> str:
    """Extract main article/content text, removing nav, footer, scripts.

    Based on Stack Overflow / boilerplate removal patterns:
    - Remove script, style, nav, footer, header
    - Prefer article, main, [role='main'], .content, .article
    - Fallback to body text
    """
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Try common main-content selectors first
    for selector in (
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".article",
        ".post-content",
        ".article-body",
        ".entry-content",
        "[itemprop='articleBody']",
        "#content",
        ".main",
    ):
        el = soup.select_one(selector)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 100:  # Likely real content
                return _normalize_text(text)

    # Fallback: body
    body = soup.find("body")
    if body:
        return _normalize_text(body.get_text(separator=" ", strip=True))
    return _normalize_text(soup.get_text(separator=" ", strip=True))


def _normalize_text(text: str) -> str:
    """Collapse whitespace and trim."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_ddg_url(href: str) -> str:
    """Resolve DuckDuckGo redirect URLs (duckduckgo.com/l/?uddg=...) to final URL."""
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://html.duckduckgo.com" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.netloc or "") and "/l/" in parsed.path:
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [None])[0]
        if uddg:
            return uddg
    return href


def _search_urls_playwright(query: str, top_n: int = TOP_N_URLS) -> list[str]:
    """Use Playwright to load DuckDuckGo HTML search and extract result URLs."""
    import httpx

    urls: list[str] = []
    # DuckDuckGo blocks Playwright/headless; use httpx for search (Lite is static HTML)
    search_url = f"{DDG_LITE_SEARCH}?{urlencode({'q': query})}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(search_url, headers=headers)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    # DuckDuckGo Lite: links with uddg= in href (redirect to real URL)
    # Fallback: html.duckduckgo.com used .result .result__a
    link_els = soup.select('a[href*="uddg="]')
    if not link_els:
        link_els = [a for r in soup.select(".result") for a in (r.select_one(".result__a"),) if a]

    seen: set[str] = set()
    for a in link_els:
        if len(urls) >= top_n:
            break
        raw = a.get("href") if a else None
        if not raw:
            continue
        url = _resolve_ddg_url(raw)
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.netloc and "duckduckgo.com" in parsed.netloc:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return urls[:top_n]


def _scrape_page_playwright(url: str) -> str:
    """Use Playwright to load a page and return extracted main text."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=10000)
            html = page.content()
        except Exception:
            return ""
        finally:
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    text = _extract_main_text(soup)
    if len(text) > MAX_CHARS_PER_PAGE:
        text = text[:MAX_CHARS_PER_PAGE] + "..."
    return text.strip()


def fetch_scraped_corpus(
    query: str,
    *,
    top_n: int = 5,
    max_chars_per_page: int | None = None,
) -> str:
    """Search and scrape pages; return combined raw text (no LLM).

    Used for extractive summarization and memory node enrichment.
    """
    cap = max_chars_per_page if max_chars_per_page is not None else MAX_CHARS_PER_PAGE
    try:
        urls = _search_urls_playwright(query, top_n=top_n)
    except Exception:
        return ""

    if not urls:
        return ""

    contents: list[str] = []
    for i, url in enumerate(urls):
        try:
            text = _scrape_page_playwright(url)
            if text:
                if len(text) > cap:
                    text = text[:cap] + "..."
                contents.append(f"[Source {i + 1}: {url}]\n{text}")
        except Exception:
            continue

    if not contents:
        return ""
    return "\n\n---\n\n".join(contents)


def _simple_extractive_fallback(corpus: str, *, max_sentences: int = 4, max_chars: int = 900) -> str:
    """Lightweight extractive summary when embedding-based extractive is unavailable."""
    if not corpus.strip():
        return ""
    parts = re.split(r"(?<=[.!?])\s+", corpus)
    sents = [p.strip() for p in parts if len(p.strip()) > 20]
    if not sents:
        sents = [corpus.strip()[:max_chars]]
    out = " ".join(sents[:max_sentences]).strip()
    return out[:max_chars]


def summarize_scrape_extractive(
    corpus: str,
    *,
    max_bullets: int = 4,
    max_words_per_bullet: int = 22,
) -> str:
    """Turn scraped corpus into a short extractive-style gloss (bullets or sentences)."""
    corpus = (corpus or "").strip()
    if not corpus:
        return ""
    try:
        from src.RAG_online.summarizer import bullet_summary

        out = bullet_summary(
            corpus,
            max_bullets=max_bullets,
            mode="extractive",
            max_words_per_bullet=max_words_per_bullet,
            ratio=0.25,
        )
        if (out or "").strip():
            return out.strip()
    except Exception:
        pass
    return _simple_extractive_fallback(corpus)


def web_gloss_for_topic(topic: str, *, top_n: int = 3, max_bullets: int = 4) -> str:
    """Search the web for ``topic`` and return a short extractive gloss."""
    q = (topic or "").strip()
    if not q:
        return ""
    search_q = f"{q} what is definition overview"
    corpus = fetch_scraped_corpus(search_q, top_n=top_n)
    if not corpus:
        return ""
    return summarize_scrape_extractive(corpus, max_bullets=max_bullets)


def _abstractive_summary(
    query: str,
    combined_content: str,
    llm: LLMClient,
) -> str:
    """Produce abstractive summary of scraped content answering the query."""
    prompt = f"""You are a research assistant. The user asked: "{query}"

Below is raw text scraped from multiple web pages. Synthesize the information into a clear, concise summary that directly answers the user's question. Use bullet points where helpful. Include only factual, relevant information. If the content does not sufficiently address the question, say so.

Scraped content:
{combined_content}

Summary:"""

    result = llm.generate(
        prompt,
        system_prompt="You summarize web search results accurately and concisely.",
        history=[],
    )
    return (result.get("completion") or "").strip()


def search_and_summarize(
    query: str,
    *,
    top_n: int = TOP_N_URLS,
    llm: Optional[LLMClient] = None,
) -> str:
    """Search the internet, scrape top N pages, and return an abstractive summary.

    Uses Playwright for search and page loading, BeautifulSoup4 for content
    extraction, and the project LLM (Ollama) for abstractive summarization.

    Args:
        query: Search query string.
        top_n: Number of top search results to scrape (default 15).
        llm: LLMClient for summarization (default: new instance).

    Returns:
        Abstractive summary synthesizing the scraped content, or an error message
        if search/scrape fails.
    """
    llm = llm or LLMClient()

    try:
        combined = fetch_scraped_corpus(query, top_n=top_n)
    except Exception as e:
        return f"Search failed: {e}"

    if not combined:
        return "No search results found or could not extract content from search results."

    try:
        return _abstractive_summary(query, combined, llm)
    except Exception as e:
        return f"Summarization failed: {e}"
