"""
Competitor monitoring tools for the PM Agent.

Provides: check_github_releases, check_competitor_changelogs, get_competitor_summary.
Monitors LangSmith, Langfuse, Arize Phoenix, W&B Weave, and other LLM observability tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)

# Competitor GitHub repositories to monitor
COMPETITOR_REPOS = {
    "langsmith": "langchain-ai/langsmith-sdk",
    "langfuse": "langfuse/langfuse", 
    "arize_phoenix": "Arize-ai/phoenix",
    "weave": "wandb/weave",
    "helicone": "Helicone/helicone",
    "braintrust": "braintrustdata/braintrust-sdk",
}

# Competitor changelog/blog URLs
COMPETITOR_URLS = {
    "langsmith": {
        "changelog": "https://docs.smith.langchain.com/changelog",
        "blog": "https://blog.langchain.dev",
    },
    "langfuse": {
        "changelog": "https://langfuse.com/changelog",
        "blog": "https://langfuse.com/blog",
    },
    "arize_phoenix": {
        "docs": "https://docs.arize.com/phoenix",
    },
    "weave": {
        "docs": "https://weave-docs.wandb.ai",
    },
}


def _get_github_client():
    """Lazy-init GitHub client."""
    from github import Github
    settings = get_settings()
    if not settings.github_token:
        raise ValueError("GITHUB_TOKEN not configured")
    return Github(settings.github_token)


@tool
def check_github_releases(days_back: int = 7) -> str:
    """
    Check for new releases from competitor GitHub repositories in the last N days.
    
    Monitors: LangSmith SDK, Langfuse, Arize Phoenix, W&B Weave, Helicone, Braintrust.
    
    Args:
        days_back: Number of days to look back for releases (default 7).
    
    Returns:
        JSON with releases grouped by competitor, including version, date, and release notes summary.
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    
    try:
        client = _get_github_client()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        all_releases = {}
        
        for competitor, repo_name in COMPETITOR_REPOS.items():
            try:
                repo = client.get_repo(repo_name)
                releases = []
                
                for release in repo.get_releases()[:10]:  # Check last 10 releases
                    if release.published_at and release.published_at.replace(tzinfo=timezone.utc) > cutoff:
                        releases.append({
                            "tag": release.tag_name,
                            "name": release.title or release.tag_name,
                            "published": release.published_at.isoformat(),
                            "url": release.html_url,
                            "body": (release.body or "")[:1000],
                            "prerelease": release.prerelease,
                        })
                
                if releases:
                    all_releases[competitor] = {
                        "repo": repo_name,
                        "releases": releases,
                    }
                    
            except Exception as e:
                logger.warning("competitor_release_check_failed", competitor=competitor, error=str(e))
                continue
        
        _circuit.record_success()
        logger.info("competitor_releases_checked", competitors=len(COMPETITOR_REPOS), found=len(all_releases))
        
        return json.dumps({
            "period_days": days_back,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "competitors_with_releases": list(all_releases.keys()),
            "releases": all_releases,
        }, ensure_ascii=False)
        
    except Exception as e:
        _circuit.record_failure()
        logger.error("competitor_releases_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def check_competitor_changelogs(competitor: str = "all") -> str:
    """
    Fetch and summarize competitor changelog/documentation pages.
    
    Args:
        competitor: Specific competitor name (langsmith, langfuse, arize_phoenix, weave) or "all".
    
    Returns:
        JSON with extracted changelog content from competitor websites.
    """
    if _circuit.is_open:
        return '{"error": "Web circuit breaker is open"}'
    
    try:
        import httpx
        from bs4 import BeautifulSoup
        
        targets = COMPETITOR_URLS if competitor == "all" else {competitor: COMPETITOR_URLS.get(competitor, {})}
        
        results = {}
        
        for comp_name, urls in targets.items():
            if not urls:
                continue
                
            comp_results = {}
            
            for url_type, url in urls.items():
                try:
                    resp = httpx.get(url, timeout=15, follow_redirects=True, headers={
                        "User-Agent": "Mozilla/5.0 (PM-Agent/1.0)"
                    })
                    resp.raise_for_status()
                    
                    soup = BeautifulSoup(resp.text, "html.parser")
                    
                    # Remove non-content elements
                    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        s.decompose()
                    
                    # Try to find changelog-specific content
                    content = ""
                    
                    # Look for common changelog containers
                    for selector in ["main", "article", ".changelog", ".content", "#content"]:
                        elem = soup.select_one(selector)
                        if elem:
                            content = elem.get_text(separator="\n", strip=True)[:3000]
                            break
                    
                    if not content:
                        content = soup.get_text(separator="\n", strip=True)[:3000]
                    
                    comp_results[url_type] = {
                        "url": url,
                        "content": content,
                    }
                    
                except Exception as e:
                    comp_results[url_type] = {"url": url, "error": str(e)}
            
            if comp_results:
                results[comp_name] = comp_results
        
        _circuit.record_success()
        logger.info("competitor_changelogs_checked", competitors=len(results))
        
        return json.dumps({
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "changelogs": results,
        }, ensure_ascii=False)
        
    except Exception as e:
        _circuit.record_failure()
        logger.error("competitor_changelogs_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def get_competitor_github_activity(days_back: int = 7) -> str:
    """
    Get recent GitHub activity (commits, issues, PRs) from competitor repositories.
    
    Args:
        days_back: Number of days to look back (default 7).
    
    Returns:
        JSON with activity summary for each competitor repo.
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    
    try:
        client = _get_github_client()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        activity = {}
        
        for competitor, repo_name in COMPETITOR_REPOS.items():
            try:
                repo = client.get_repo(repo_name)
                
                # Get recent commits
                recent_commits = 0
                commit_authors = set()
                try:
                    for commit in repo.get_commits(since=cutoff)[:100]:
                        recent_commits += 1
                        if commit.author:
                            commit_authors.add(commit.author.login)
                except:
                    pass
                
                # Get open issues count
                open_issues = repo.open_issues_count
                
                # Get recent stars (approximate via stargazers)
                stars = repo.stargazers_count
                
                # Get open PRs
                open_prs = len(list(repo.get_pulls(state="open")[:50]))
                
                activity[competitor] = {
                    "repo": repo_name,
                    "stars": stars,
                    "open_issues": open_issues,
                    "open_prs": open_prs,
                    "recent_commits": recent_commits,
                    "unique_contributors": len(commit_authors),
                    "url": f"https://github.com/{repo_name}",
                }
                
            except Exception as e:
                logger.warning("competitor_activity_failed", competitor=competitor, error=str(e))
                activity[competitor] = {"error": str(e)}
        
        _circuit.record_success()
        logger.info("competitor_activity_checked", competitors=len(activity))
        
        return json.dumps({
            "period_days": days_back,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "activity": activity,
        }, ensure_ascii=False)
        
    except Exception as e:
        _circuit.record_failure()
        logger.error("competitor_activity_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool  
def compare_competitor_features() -> str:
    """
    Get a structured comparison of Opik vs competitors based on known feature sets.
    
    Returns:
        JSON with feature comparison matrix for LLM observability tools.
    """
    # Static feature comparison - updated periodically
    comparison = {
        "last_updated": "2026-02-06",
        "features": {
            "tracing": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
            "evaluation": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
            "datasets": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
            "prompt_management": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": False,
                "weave": False,
            },
            "open_source": {
                "opik": True,
                "langsmith": False,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
            "self_hosted": {
                "opik": True,
                "langsmith": False,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
            "langgraph_integration": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": False,
            },
            "cost_tracking": {
                "opik": True,
                "langsmith": True,
                "langfuse": True,
                "arize_phoenix": True,
                "weave": True,
            },
        },
        "pricing_models": {
            "opik": "Open-source + Cloud (usage-based)",
            "langsmith": "Free tier + usage-based",
            "langfuse": "Open-source + Cloud (usage-based)",
            "arize_phoenix": "Open-source + Enterprise",
            "weave": "Included with W&B subscription",
        },
    }
    
    logger.info("competitor_features_compared")
    return json.dumps(comparison, ensure_ascii=False)


competitor_tools = [
    check_github_releases,
    check_competitor_changelogs,
    get_competitor_github_activity,
    compare_competitor_features,
]
