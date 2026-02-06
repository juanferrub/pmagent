"""
Market research & web tools for the PM Agent.

Provides: web_search, browse_page, search_reddit, get_reddit_comments.
Uses Tavily for search, httpx+BeautifulSoup for browsing, PRAW for Reddit.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_web_circuit = CircuitBreaker(threshold=5, reset_timeout=60)
_reddit_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API for market research, competitor analysis, or general information.

    Args:
        query: Search query string.
        max_results: Number of results to return (1-10).

    Returns:
        JSON array of results with title, url, content snippet.
    """
    if _web_circuit.is_open:
        return '{"error": "Web search circuit breaker is open"}'
    try:
        from tavily import TavilyClient

        settings = get_settings()
        if not settings.tavily_api_key:
            return '{"error": "TAVILY_API_KEY not configured"}'
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=min(max_results, 10))
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
                "score": r.get("score", 0),
            })
        _web_circuit.record_success()
        logger.info("web_search", query=query, results=len(results))
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        _web_circuit.record_failure()
        logger.error("web_search_error", query=query, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def browse_page(url: str, instructions: str = "Extract the main content") -> str:
    """
    Browse a web page and extract content based on instructions.

    Args:
        url: Full URL to browse.
        instructions: What to extract from the page (e.g., "Extract pricing information").

    Returns:
        Extracted text content from the page.
    """
    if _web_circuit.is_open:
        return '{"error": "Web browse circuit breaker is open"}'
    try:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(url, timeout=15, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (PM-Agent/1.0)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style
        for s in soup(["script", "style", "nav", "footer", "header"]):
            s.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Truncate
        text = text[:5000]
        _web_circuit.record_success()
        logger.info("browse_page", url=url)
        return json.dumps({
            "url": url,
            "instructions": instructions,
            "content": text,
        }, ensure_ascii=False)
    except Exception as e:
        _web_circuit.record_failure()
        logger.error("browse_page_error", url=url, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def search_reddit(subreddit: str, query: str, limit: int = 10) -> str:
    """
    Search a subreddit for posts matching a query.

    Args:
        subreddit: Subreddit name (without r/, e.g., "product").
        query: Search query.
        limit: Max posts to return.

    Returns:
        JSON array of posts with title, url, score, comments count, selftext snippet.
    """
    if _reddit_circuit.is_open:
        return '{"error": "Reddit circuit breaker is open"}'
    try:
        import praw

        settings = get_settings()
        if not settings.reddit_client_id:
            return '{"error": "Reddit credentials not configured"}'
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        sub = reddit.subreddit(subreddit)
        posts = []
        for post in sub.search(query, limit=min(limit, 50)):
            posts.append({
                "title": post.title,
                "url": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "num_comments": post.num_comments,
                "selftext": (post.selftext or "")[:500],
                "created_utc": post.created_utc,
                "author": str(post.author) if post.author else "[deleted]",
            })
        _reddit_circuit.record_success()
        logger.info("search_reddit", subreddit=subreddit, query=query, count=len(posts))
        return json.dumps(posts, ensure_ascii=False)
    except Exception as e:
        _reddit_circuit.record_failure()
        logger.error("search_reddit_error", subreddit=subreddit, query=query, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def get_reddit_hot_posts(subreddit: str, limit: int = 10) -> str:
    """
    Get hot/trending posts from a subreddit.

    Args:
        subreddit: Subreddit name (without r/).
        limit: Max posts to return.

    Returns:
        JSON array of hot posts.
    """
    if _reddit_circuit.is_open:
        return '{"error": "Reddit circuit breaker is open"}'
    try:
        import praw

        settings = get_settings()
        if not settings.reddit_client_id:
            return '{"error": "Reddit credentials not configured"}'
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        sub = reddit.subreddit(subreddit)
        posts = []
        for post in sub.hot(limit=min(limit, 50)):
            posts.append({
                "title": post.title,
                "url": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "num_comments": post.num_comments,
                "selftext": (post.selftext or "")[:300],
                "author": str(post.author) if post.author else "[deleted]",
            })
        _reddit_circuit.record_success()
        logger.info("reddit_hot", subreddit=subreddit, count=len(posts))
        return json.dumps(posts, ensure_ascii=False)
    except Exception as e:
        _reddit_circuit.record_failure()
        logger.error("reddit_hot_error", subreddit=subreddit, error=str(e))
        return f'{{"error": "{str(e)}"}}'


research_tools = [web_search, browse_page, search_reddit, get_reddit_hot_posts]
