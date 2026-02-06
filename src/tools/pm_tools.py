"""
Product Management workflow tools for the PM Agent.

Provides: aggregate_customer_voice, generate_status_update, analyze_feature_requests.
Helps with customer feedback aggregation, status reporting, and prioritization.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from collections import defaultdict

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_jira_circuit = CircuitBreaker(threshold=5, reset_timeout=60)
_github_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


@tool
def aggregate_customer_voice(days_back: int = 7) -> str:
    """
    Aggregate customer feedback from Jira tickets and GitHub issues.
    Groups by theme/component and identifies trending requests.
    
    Args:
        days_back: Number of days to look back (default 7).
    
    Returns:
        JSON with aggregated customer feedback grouped by theme.
    """
    try:
        from atlassian import Jira
        from github import Github
        
        settings = get_settings()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        all_requests = []
        themes = defaultdict(list)
        
        # Gather from Jira
        if settings.jira_url and settings.jira_api_token:
            try:
                jira = Jira(
                    url=settings.jira_url,
                    username=settings.jira_user_email,
                    password=settings.jira_api_token,
                    cloud=True,
                )
                
                project_filter = ""
                if settings.jira_project_keys:
                    projects = settings.jira_project_keys.split(",")
                    project_filter = f"project IN ({','.join(projects)}) AND "
                
                # Get feature requests and bugs from customers
                jql = f'{project_filter}(type IN ("Bug", "Story", "Task") OR labels IN ("customer-request", "feature-request", "feedback")) AND created >= -{days_back}d ORDER BY created DESC'
                
                results = jira.jql(jql, limit=50)
                
                for issue in results.get("issues", []):
                    fields = issue.get("fields", {})
                    labels = fields.get("labels", [])
                    components = [c.get("name") for c in fields.get("components", [])]
                    
                    # Determine theme from components or labels
                    theme = "General"
                    if components:
                        theme = components[0]
                    elif labels:
                        for label in labels:
                            if label not in ["customer-request", "feature-request", "feedback"]:
                                theme = label
                                break
                    
                    request = {
                        "source": "jira",
                        "id": issue.get("key"),
                        "title": fields.get("summary"),
                        "type": fields.get("issuetype", {}).get("name"),
                        "priority": fields.get("priority", {}).get("name"),
                        "created": fields.get("created"),
                        "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else "Unknown",
                        "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                        "theme": theme,
                        "labels": labels,
                    }
                    
                    all_requests.append(request)
                    themes[theme].append(request)
                    
                _jira_circuit.record_success()
                
            except Exception as e:
                logger.warning("customer_voice_jira_error", error=str(e))
        
        # Gather from GitHub
        if settings.github_token and settings.github_repos:
            try:
                client = Github(settings.github_token)
                
                for repo_name in settings.github_repos.split(","):
                    repo_name = repo_name.strip()
                    if not repo_name:
                        continue
                        
                    try:
                        repo = client.get_repo(repo_name)
                        
                        for issue in repo.get_issues(state="open", sort="created", direction="desc")[:30]:
                            if issue.pull_request:
                                continue
                            
                            if issue.created_at and issue.created_at.replace(tzinfo=timezone.utc) < cutoff:
                                continue
                            
                            # Determine theme from labels
                            labels = [l.name for l in issue.labels]
                            theme = "General"
                            for label in labels:
                                if label not in ["bug", "enhancement", "question", "help wanted"]:
                                    theme = label
                                    break
                            
                            # Get reaction count as priority signal
                            reactions = 0
                            try:
                                for reaction in issue.get_reactions():
                                    reactions += 1
                            except:
                                pass
                            
                            request = {
                                "source": "github",
                                "id": f"{repo_name}#{issue.number}",
                                "title": issue.title,
                                "type": "enhancement" if "enhancement" in labels else "bug" if "bug" in labels else "issue",
                                "priority": "High" if reactions > 5 else "Medium" if reactions > 2 else "Normal",
                                "created": issue.created_at.isoformat() if issue.created_at else "",
                                "reporter": issue.user.login if issue.user else "Unknown",
                                "url": issue.html_url,
                                "theme": theme,
                                "labels": labels,
                                "reactions": reactions,
                                "comments": issue.comments,
                            }
                            
                            all_requests.append(request)
                            themes[theme].append(request)
                            
                    except Exception as e:
                        logger.warning("customer_voice_repo_error", repo=repo_name, error=str(e))
                
                _github_circuit.record_success()
                
            except Exception as e:
                logger.warning("customer_voice_github_error", error=str(e))
        
        # Analyze themes
        theme_summary = []
        for theme, requests in sorted(themes.items(), key=lambda x: len(x[1]), reverse=True):
            theme_summary.append({
                "theme": theme,
                "count": len(requests),
                "sources": {
                    "jira": len([r for r in requests if r["source"] == "jira"]),
                    "github": len([r for r in requests if r["source"] == "github"]),
                },
                "sample_requests": [r["title"] for r in requests[:3]],
            })
        
        logger.info("customer_voice_aggregated", total=len(all_requests), themes=len(themes))
        
        return json.dumps({
            "period_days": days_back,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "total_requests": len(all_requests),
            "themes": theme_summary,
            "top_requests": sorted(all_requests, key=lambda x: x.get("reactions", 0) + (1 if x.get("priority") == "High" else 0), reverse=True)[:10],
            "all_requests": all_requests,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error("customer_voice_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def generate_status_update(days_back: int = 7) -> str:
    """
    Generate a weekly status update with shipped features, in-progress work, and blockers.
    
    Args:
        days_back: Number of days to cover (default 7).
    
    Returns:
        JSON with structured status update data ready for formatting.
    """
    try:
        from atlassian import Jira
        from github import Github
        
        settings = get_settings()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        status = {
            "shipped": [],
            "in_progress": [],
            "blocked": [],
            "metrics": {
                "prs_merged": 0,
                "tickets_closed": 0,
                "bugs_fixed": 0,
                "features_shipped": 0,
            },
            "next_week": [],
        }
        
        # Get merged PRs from GitHub
        if settings.github_token and settings.github_repos:
            try:
                client = Github(settings.github_token)
                
                for repo_name in settings.github_repos.split(","):
                    repo_name = repo_name.strip()
                    if not repo_name:
                        continue
                    
                    try:
                        repo = client.get_repo(repo_name)
                        
                        for pr in repo.get_pulls(state="closed", sort="updated", direction="desc")[:50]:
                            if not pr.merged:
                                continue
                            
                            if pr.merged_at and pr.merged_at.replace(tzinfo=timezone.utc) < cutoff:
                                continue
                            
                            status["shipped"].append({
                                "type": "pr",
                                "id": f"#{pr.number}",
                                "title": pr.title,
                                "author": pr.user.login if pr.user else "",
                                "merged_at": pr.merged_at.isoformat() if pr.merged_at else "",
                                "url": pr.html_url,
                                "additions": pr.additions,
                                "deletions": pr.deletions,
                            })
                            status["metrics"]["prs_merged"] += 1
                            
                            # Categorize
                            title_lower = pr.title.lower()
                            if "fix" in title_lower or "bug" in title_lower:
                                status["metrics"]["bugs_fixed"] += 1
                            elif "feat" in title_lower or "add" in title_lower:
                                status["metrics"]["features_shipped"] += 1
                                
                    except Exception as e:
                        logger.warning("status_repo_error", repo=repo_name, error=str(e))
                        
            except Exception as e:
                logger.warning("status_github_error", error=str(e))
        
        # Get Jira ticket status
        if settings.jira_url and settings.jira_api_token:
            try:
                jira = Jira(
                    url=settings.jira_url,
                    username=settings.jira_user_email,
                    password=settings.jira_api_token,
                    cloud=True,
                )
                
                project_filter = ""
                if settings.jira_project_keys:
                    projects = settings.jira_project_keys.split(",")
                    project_filter = f"project IN ({','.join(projects)}) AND "
                
                # Closed tickets
                jql_closed = f'{project_filter}status CHANGED TO "Done" DURING (-{days_back}d, now()) ORDER BY updated DESC'
                closed = jira.jql(jql_closed, limit=30)
                
                for issue in closed.get("issues", []):
                    fields = issue.get("fields", {})
                    status["metrics"]["tickets_closed"] += 1
                    
                    if fields.get("issuetype", {}).get("name") == "Bug":
                        status["metrics"]["bugs_fixed"] += 1
                
                # In progress tickets
                jql_progress = f'{project_filter}status IN ("In Progress", "In Review", "In Development") ORDER BY priority DESC'
                in_progress = jira.jql(jql_progress, limit=20)
                
                for issue in in_progress.get("issues", []):
                    fields = issue.get("fields", {})
                    status["in_progress"].append({
                        "id": issue.get("key"),
                        "title": fields.get("summary"),
                        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else "Unassigned",
                        "priority": fields.get("priority", {}).get("name"),
                        "status": fields.get("status", {}).get("name"),
                        "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                    })
                
                # Blocked tickets
                jql_blocked = f'{project_filter}(labels IN ("blocked", "blocker") OR status = "Blocked") ORDER BY priority DESC'
                blocked = jira.jql(jql_blocked, limit=10)
                
                for issue in blocked.get("issues", []):
                    fields = issue.get("fields", {})
                    status["blocked"].append({
                        "id": issue.get("key"),
                        "title": fields.get("summary"),
                        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else "Unassigned",
                        "priority": fields.get("priority", {}).get("name"),
                        "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                    })
                
                # Next sprint items (high priority, not started)
                jql_next = f'{project_filter}status IN ("To Do", "Backlog", "Open") AND priority IN ("Highest", "High") ORDER BY priority DESC, created ASC'
                next_items = jira.jql(jql_next, limit=10)
                
                for issue in next_items.get("issues", []):
                    fields = issue.get("fields", {})
                    status["next_week"].append({
                        "id": issue.get("key"),
                        "title": fields.get("summary"),
                        "priority": fields.get("priority", {}).get("name"),
                        "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                    })
                
            except Exception as e:
                logger.warning("status_jira_error", error=str(e))
        
        logger.info("status_update_generated", 
                   shipped=len(status["shipped"]), 
                   in_progress=len(status["in_progress"]),
                   blocked=len(status["blocked"]))
        
        return json.dumps({
            "period_days": days_back,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error("status_update_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def analyze_feature_requests(days_back: int = 30) -> str:
    """
    Analyze feature requests to identify patterns and prioritization signals.
    
    Args:
        days_back: Number of days to analyze (default 30).
    
    Returns:
        JSON with feature request analysis including frequency, themes, and recommendations.
    """
    try:
        from atlassian import Jira
        from github import Github
        
        settings = get_settings()
        
        requests = []
        keyword_counts = defaultdict(int)
        
        # Gather from Jira
        if settings.jira_url and settings.jira_api_token:
            try:
                jira = Jira(
                    url=settings.jira_url,
                    username=settings.jira_user_email,
                    password=settings.jira_api_token,
                    cloud=True,
                )
                
                project_filter = ""
                if settings.jira_project_keys:
                    projects = settings.jira_project_keys.split(",")
                    project_filter = f"project IN ({','.join(projects)}) AND "
                
                jql = f'{project_filter}type IN ("Story", "New Feature") AND created >= -{days_back}d ORDER BY created DESC'
                results = jira.jql(jql, limit=100)
                
                for issue in results.get("issues", []):
                    fields = issue.get("fields", {})
                    title = fields.get("summary", "").lower()
                    
                    requests.append({
                        "source": "jira",
                        "id": issue.get("key"),
                        "title": fields.get("summary"),
                        "priority": fields.get("priority", {}).get("name"),
                    })
                    
                    # Extract keywords
                    for word in ["integration", "api", "export", "import", "dashboard", "report", 
                                "evaluation", "tracing", "prompt", "dataset", "comparison"]:
                        if word in title:
                            keyword_counts[word] += 1
                            
            except Exception as e:
                logger.warning("feature_analysis_jira_error", error=str(e))
        
        # Gather from GitHub
        if settings.github_token and settings.github_repos:
            try:
                client = Github(settings.github_token)
                
                for repo_name in settings.github_repos.split(","):
                    repo_name = repo_name.strip()
                    if not repo_name:
                        continue
                    
                    try:
                        repo = client.get_repo(repo_name)
                        
                        for issue in repo.get_issues(state="open", labels=["enhancement"])[:50]:
                            if issue.pull_request:
                                continue
                            
                            title = issue.title.lower()
                            
                            # Count reactions as priority signal
                            reactions = 0
                            try:
                                reactions = issue.reactions.get("+1", 0) if isinstance(issue.reactions, dict) else 0
                            except:
                                pass
                            
                            requests.append({
                                "source": "github",
                                "id": f"{repo_name}#{issue.number}",
                                "title": issue.title,
                                "priority": "High" if reactions > 5 else "Normal",
                                "reactions": reactions,
                            })
                            
                            for word in ["integration", "api", "export", "import", "dashboard", "report",
                                        "evaluation", "tracing", "prompt", "dataset", "comparison"]:
                                if word in title:
                                    keyword_counts[word] += 1
                                    
                    except Exception as e:
                        logger.warning("feature_analysis_repo_error", repo=repo_name, error=str(e))
                        
            except Exception as e:
                logger.warning("feature_analysis_github_error", error=str(e))
        
        # Sort keywords by frequency
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info("feature_requests_analyzed", total=len(requests), keywords=len(keyword_counts))
        
        return json.dumps({
            "period_days": days_back,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "total_requests": len(requests),
            "top_themes": [{"keyword": k, "count": v} for k, v in top_keywords],
            "high_priority_requests": [r for r in requests if r.get("priority") == "High"][:10],
            "recommendations": [
                f"Focus on '{top_keywords[0][0]}' - mentioned {top_keywords[0][1]} times" if top_keywords else "No clear pattern",
                f"Consider '{top_keywords[1][0]}' - mentioned {top_keywords[1][1]} times" if len(top_keywords) > 1 else "",
            ],
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error("feature_analysis_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


pm_tools = [
    aggregate_customer_voice,
    generate_status_update,
    analyze_feature_requests,
]
