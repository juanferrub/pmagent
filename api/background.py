"""
Background scheduler for proactive PM Agent operations.

Implements:
- Daily digest (configurable cron, default 6 AM CET)
- Weekly market scan (Monday mornings)
- Hourly quick checks
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import get_settings
from src.utils import logger

_scheduler: Optional[AsyncIOScheduler] = None


async def run_daily_digest():
    """
    Daily digest workflow (AC-4.1).
    Aggregates Slack + Jira + GitHub + market data → Notion + Slack summary.
    """
    from src.graphs.main_graph import invoke_graph

    thread_id = f"daily-digest-{uuid.uuid4().hex[:8]}"
    query = (
        "Generate the daily product management digest. "
        "1) Check Slack channels for important discussions and action items from the last 24 hours. "
        "2) Review Jira for new/updated high-priority tickets. "
        "3) Check GitHub for new PRs, issues, and merged changes. "
        "4) Do a quick market scan for competitor news. "
        "5) Create a comprehensive summary report in Notion. "
        "6) Post a highlight summary to the product Slack channel."
    )
    logger.info("scheduled_daily_digest_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "daily_digest"},
        )
        logger.info("scheduled_daily_digest_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_daily_digest_error", error=str(e))


async def run_weekly_market_scan():
    """
    Weekly market scan workflow (AC-4.4).
    Competitor analysis + Reddit trends → Notion report + Slack.
    """
    from src.graphs.main_graph import invoke_graph

    thread_id = f"weekly-market-{uuid.uuid4().hex[:8]}"
    query = (
        "Perform the weekly market intelligence scan. "
        "1) Search for competitor news, feature launches, and pricing changes. "
        "2) Scan relevant Reddit subreddits for product discussions and user feedback. "
        "3) Identify trending topics in our industry. "
        "4) Analyze sentiment trends from community discussions. "
        "5) Generate a comprehensive market intelligence report in Notion. "
        "6) Post key insights and alerts to Slack."
    )
    logger.info("scheduled_weekly_market_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "weekly_market_scan"},
        )
        logger.info("scheduled_weekly_market_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_weekly_market_error", error=str(e))


async def run_hourly_check():
    """
    Lightweight hourly check (AC-3.2).
    Quick scan for high-priority items.
    """
    from src.graphs.main_graph import invoke_graph

    thread_id = f"hourly-check-{uuid.uuid4().hex[:8]}"
    query = (
        "Quick check: "
        "1) Are there any new P0/P1 Jira tickets in the last hour? "
        "2) Any urgent Slack messages mentioning 'blocker', 'urgent', or 'critical'? "
        "3) Any new GitHub issues labeled 'bug' or 'critical'? "
        "If you find anything urgent, create an alert summary."
    )
    logger.info("scheduled_hourly_check_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "hourly_check"},
        )
        logger.info("scheduled_hourly_check_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_hourly_check_error", error=str(e))


def _parse_cron(cron_expr: str) -> dict:
    """Parse a cron expression into APScheduler CronTrigger kwargs."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return {"hour": 6, "minute": 0}
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


def start_scheduler():
    """Start the APScheduler with all scheduled jobs."""
    global _scheduler
    settings = get_settings()

    _scheduler = AsyncIOScheduler(timezone=settings.timezone)

    # Daily digest
    daily_cron = _parse_cron(settings.daily_digest_cron)
    _scheduler.add_job(
        run_daily_digest,
        CronTrigger(**daily_cron, timezone=settings.timezone),
        id="daily_digest",
        name="Daily PM Digest",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="daily_digest", cron=settings.daily_digest_cron)

    # Weekly market scan
    weekly_cron = _parse_cron(settings.weekly_scan_cron)
    _scheduler.add_job(
        run_weekly_market_scan,
        CronTrigger(**weekly_cron, timezone=settings.timezone),
        id="weekly_market_scan",
        name="Weekly Market Scan",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="weekly_market_scan", cron=settings.weekly_scan_cron)

    # Hourly quick check
    _scheduler.add_job(
        run_hourly_check,
        CronTrigger(minute=0, timezone=settings.timezone),  # Every hour on the hour
        id="hourly_check",
        name="Hourly Quick Check",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="hourly_check", cron="0 * * * *")

    _scheduler.start()
    logger.info("scheduler_started", timezone=settings.timezone)


def stop_scheduler():
    """Gracefully stop the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")
    _scheduler = None


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Get the current scheduler instance."""
    return _scheduler
