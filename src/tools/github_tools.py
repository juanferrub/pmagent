"""
GitHub integration tools for the PM Agent.

Provides: list_github_issues, get_github_pr, get_github_file_contents, list_github_prs.
Uses PyGithub under the hood.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


def _get_github_client():
    """Lazy-init GitHub client."""
    from github import Github

    settings = get_settings()
    if not settings.github_token:
        raise ValueError("GITHUB_TOKEN not configured")
    return Github(settings.github_token)


@tool
def list_github_issues(repo_name: str, state: str = "open", labels: str = "", limit: int = 30) -> str:
    """
    List issues from a GitHub repository.

    Args:
        repo_name: Full repo name (e.g., "org/repo").
        state: Filter by state: "open", "closed", "all".
        labels: Comma-separated label filter (optional).
        limit: Max number of issues to return.

    Returns:
        JSON array of issues with number, title, state, labels, author, created, updated.
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    try:
        client = _get_github_client()
        repo = client.get_repo(repo_name)
        kwargs = {"state": state}
        if labels:
            kwargs["labels"] = [repo.get_label(l.strip()) for l in labels.split(",")]
        issues = []
        for issue in repo.get_issues(**kwargs)[:limit]:
            if issue.pull_request:
                continue  # skip PRs from issues list
            issues.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "labels": [l.name for l in issue.labels],
                "author": issue.user.login if issue.user else "",
                "created": issue.created_at.isoformat() if issue.created_at else "",
                "updated": issue.updated_at.isoformat() if issue.updated_at else "",
                "comments": issue.comments,
                "body": (issue.body or "")[:500],
            })
        _circuit.record_success()
        logger.info("github_list_issues", repo=repo_name, count=len(issues))
        return json.dumps(issues, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("github_list_issues_error", repo=repo_name, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def list_github_prs(repo_name: str, state: str = "open", limit: int = 20) -> str:
    """
    List pull requests from a GitHub repository.

    Args:
        repo_name: Full repo name (e.g., "org/repo").
        state: Filter by state: "open", "closed", "all".
        limit: Max number of PRs to return.

    Returns:
        JSON array of PRs with number, title, state, author, changed_files, additions, deletions, merged.
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    try:
        client = _get_github_client()
        repo = client.get_repo(repo_name)
        prs = []
        for pr in repo.get_pulls(state=state, sort="updated")[:limit]:
            prs.append({
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "author": pr.user.login if pr.user else "",
                "created": pr.created_at.isoformat() if pr.created_at else "",
                "updated": pr.updated_at.isoformat() if pr.updated_at else "",
                "changed_files": pr.changed_files,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "merged": pr.merged,
                "body": (pr.body or "")[:500],
            })
        _circuit.record_success()
        logger.info("github_list_prs", repo=repo_name, count=len(prs))
        return json.dumps(prs, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("github_list_prs_error", repo=repo_name, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def get_github_pr(repo_name: str, pr_number: int) -> str:
    """
    Get details of a specific pull request including diff stats.

    Args:
        repo_name: Full repo name (e.g., "org/repo").
        pr_number: PR number.

    Returns:
        JSON with PR details, files changed, and review comments.
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    try:
        client = _get_github_client()
        repo = client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        files = []
        for f in pr.get_files()[:50]:
            files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch": (f.patch or "")[:1000],
            })
        reviews = []
        for r in pr.get_reviews()[:20]:
            reviews.append({
                "user": r.user.login if r.user else "",
                "state": r.state,
                "body": (r.body or "")[:500],
            })
        result = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body or "",
            "state": pr.state,
            "merged": pr.merged,
            "author": pr.user.login if pr.user else "",
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "files": files,
            "reviews": reviews,
        }
        _circuit.record_success()
        logger.info("github_get_pr", repo=repo_name, pr=pr_number)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("github_get_pr_error", repo=repo_name, pr=pr_number, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def get_github_file_contents(repo_name: str, file_path: str, ref: str = "main") -> str:
    """
    Read a file's contents from a GitHub repository.

    Args:
        repo_name: Full repo name (e.g., "org/repo").
        file_path: Path to file in repo (e.g., "src/main.py").
        ref: Branch or commit ref (default "main").

    Returns:
        The file content as a string (truncated to 10000 chars).
    """
    if _circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    try:
        client = _get_github_client()
        repo = client.get_repo(repo_name)
        content = repo.get_contents(file_path, ref=ref)
        if isinstance(content, list):
            return json.dumps({"error": "Path is a directory, not a file"})
        decoded = content.decoded_content.decode("utf-8", errors="replace")
        _circuit.record_success()
        logger.info("github_get_file", repo=repo_name, path=file_path)
        return decoded[:10000]
    except Exception as e:
        _circuit.record_failure()
        logger.error("github_get_file_error", repo=repo_name, path=file_path, error=str(e))
        return f'{{"error": "{str(e)}"}}'


github_tools = [list_github_issues, list_github_prs, get_github_pr, get_github_file_contents]
