"""
Background scheduler for proactive PM Agent operations.

Implements:
- Daily digest (configurable cron, default 8 AM Mon-Fri)
- Weekly market scan (Monday mornings)
- Hourly quick checks
- Critical alert monitoring (every 15 minutes)
- Weekly customer voice report (Friday afternoons)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_settings
from src.utils import logger

_scheduler: Optional[AsyncIOScheduler] = None


async def run_daily_digest():
    """
    Daily digest workflow - Comprehensive PM briefing with all intelligence sources.
    Runs Mon-Fri at configured time (default 8 AM).
    """
    from src.graphs.main_graph import invoke_graph
    from src.config import get_settings

    settings = get_settings()
    recipient = settings.email_recipient or "juanf@comet.com"

    thread_id = f"daily-digest-{uuid.uuid4().hex[:8]}"
    query = f"""Generate a comprehensive PM briefing report and email it to {recipient}.

**SECTION 1: INTERNAL INSIGHTS**

1. **Jira Analysis (OPIK project):**
   - Use check_critical_jira_tickets to find any P0/P1 tickets
   - Use check_blocked_tickets to find stale/blocked work
   - Top 10 highest priority issues with status and blockers
   - Any critical bugs or customer-reported issues

2. **GitHub Activity (comet-ml/opik):**
   - Recent merged PRs (last 24 hours) - what shipped?
   - Open PRs awaiting review
   - Use check_github_trending_issues to find popular feature requests
   - New issues opened - any patterns?

3. **Customer Voice Summary:**
   - Use aggregate_customer_voice to get feedback themes
   - Top requested features from Jira + GitHub
   - Trending community requests

**SECTION 2: COMPETITOR INTELLIGENCE**

4. **Competitor Releases:**
   - Use check_github_releases to find new versions from competitors
   - LangSmith, Langfuse, Arize Phoenix, W&B Weave updates
   - Use get_competitor_github_activity to see their development pace

5. **Competitor Changelogs:**
   - Use check_competitor_changelogs to scan for new features
   - Any pricing changes or announcements

**SECTION 3: MARKET INTELLIGENCE**

6. **LLM Provider News:**
   - Search for OpenAI, Anthropic, Google, Meta, Mistral updates
   - New models or API changes that affect our users

7. **Market Trends:**
   - Reddit/HN discussions about LLM observability
   - Agent framework ecosystem updates (LangChain, LlamaIndex, CrewAI)

**SECTION 4: ACTION ITEMS**

8. **Recommendations:**
   - Based on all the above, what should I focus on today?
   - Any urgent items requiring immediate attention?
   - Opportunities to capitalize on competitor gaps

**FORMAT:**
Create a well-structured HTML email with:
- Executive summary (5-7 key takeaways with priority indicators)
- Sections for each area with clear headers
- Action items table with priority and owner
- Links to all sources
- Quick stats dashboard (PRs merged, tickets closed, competitor releases)

Subject: [PM Briefing] Opik Daily Intelligence Report - {{date}}"""
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
    Lightweight hourly check - Quick scan for high-priority items.
    """
    from src.graphs.main_graph import invoke_graph

    thread_id = f"hourly-check-{uuid.uuid4().hex[:8]}"
    query = (
        "Quick check: "
        "1) Use check_critical_jira_tickets with hours_back=1 to find new P0/P1 tickets. "
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


async def run_critical_alert_check():
    """
    Critical alert check - Runs every 15 minutes to catch urgent issues.
    Sends immediate email alert if P0/critical issues are found.
    """
    from src.graphs.main_graph import invoke_graph
    from src.config import get_settings

    settings = get_settings()
    recipient = settings.email_recipient or "juanf@comet.com"

    thread_id = f"alert-check-{uuid.uuid4().hex[:8]}"
    query = f"""URGENT ALERT CHECK - Run silently unless issues found.

1. Use check_critical_jira_tickets with hours_back=1 to find new P0/P1 tickets.
2. Check for any GitHub issues with "critical" or "urgent" labels in the last hour.

IF any critical issues are found:
- Use send_urgent_alert to immediately notify {recipient}
- Include ticket/issue details, links, and recommended immediate actions
- Priority should be "critical" for P0/Blocker, "high" for P1/Critical

IF no critical issues found:
- Do NOT send any email
- Just log that the check completed with no alerts"""
    
    logger.info("scheduled_alert_check_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "critical_alert_check"},
        )
        logger.info("scheduled_alert_check_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_alert_check_error", error=str(e))


async def run_weekly_customer_voice():
    """
    Weekly customer voice report - Aggregates all customer feedback.
    Runs Friday afternoon to prepare for next week planning.
    """
    from src.graphs.main_graph import invoke_graph
    from src.config import get_settings

    settings = get_settings()
    recipient = settings.email_recipient or "juanf@comet.com"

    thread_id = f"customer-voice-{uuid.uuid4().hex[:8]}"
    query = f"""Generate a Weekly Customer Voice Report and email it to {recipient}.

1. Use aggregate_customer_voice with days_back=7 to gather all customer feedback.
2. Use analyze_feature_requests with days_back=7 to identify patterns.
3. Use check_github_trending_issues to find popular community requests.

Create a comprehensive report with:

**CUSTOMER FEEDBACK SUMMARY:**
- Total requests received this week (Jira + GitHub)
- Breakdown by source and type
- Top 5 themes/categories with request counts

**TRENDING REQUESTS:**
- Most upvoted GitHub issues
- Most mentioned feature areas
- Customer pain points identified

**COMPETITIVE CONTEXT:**
- Features customers are asking for that competitors have
- Gaps we can capitalize on

**RECOMMENDATIONS:**
- Top 3 features to prioritize based on customer demand
- Quick wins that could be addressed this sprint
- Items to add to roadmap discussion

**FORMAT:**
- Well-structured HTML email
- Charts/tables for data visualization
- Links to all source tickets/issues
- Clear action items for next week

Subject: [Customer Voice] Weekly Feedback Report - Week of {{date}}"""
    
    logger.info("scheduled_customer_voice_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "weekly_customer_voice"},
        )
        logger.info("scheduled_customer_voice_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_customer_voice_error", error=str(e))


async def run_weekly_status_update():
    """
    Weekly status update generator - Auto-generates team status report.
    Runs Friday afternoon for EOW reporting.
    """
    from src.graphs.main_graph import invoke_graph
    from src.config import get_settings

    settings = get_settings()
    recipient = settings.email_recipient or "juanf@comet.com"

    thread_id = f"status-update-{uuid.uuid4().hex[:8]}"
    query = f"""Generate a Weekly Status Update and email it to {recipient}.

1. Use generate_status_update with days_back=7 to gather all activity.
2. Get merged PRs from GitHub for comet-ml/opik.
3. Get completed Jira tickets.

Create a stakeholder-ready status report with:

**EXECUTIVE SUMMARY:**
- Key accomplishments this week (2-3 bullet points)
- Overall progress assessment

**SHIPPED THIS WEEK:**
- List of merged PRs with brief descriptions
- Completed Jira tickets
- Features/fixes that went to production

**IN PROGRESS:**
- Current sprint items and their status
- Expected completions for next week

**BLOCKERS & RISKS:**
- Any blocked items and what's needed to unblock
- Risks that need attention

**METRICS:**
- PRs merged: X
- Tickets closed: X
- Bugs fixed: X
- Features shipped: X

**NEXT WEEK PRIORITIES:**
- Top 3-5 items planned for next week

**FORMAT:**
- Professional HTML email suitable for stakeholders
- Clear sections with headers
- Metrics dashboard at top
- Links to relevant tickets/PRs

Subject: [Status Update] Opik Weekly Report - Week of {{date}}"""
    
    logger.info("scheduled_status_update_start", thread_id=thread_id)
    try:
        result = await invoke_graph(
            query=query,
            thread_id=thread_id,
            trigger_type="scheduled",
            metadata={"workflow": "weekly_status_update"},
        )
        logger.info("scheduled_status_update_complete", thread_id=thread_id)
        return result
    except Exception as e:
        logger.error("scheduled_status_update_error", error=str(e))


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

    # Daily digest - Mon-Fri at configured time (default 8 AM)
    daily_cron = _parse_cron(settings.daily_digest_cron)
    _scheduler.add_job(
        run_daily_digest,
        CronTrigger(**daily_cron, timezone=settings.timezone),
        id="daily_digest",
        name="Daily PM Digest",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="daily_digest", cron=settings.daily_digest_cron)

    # Weekly market scan - Monday mornings
    weekly_cron = _parse_cron(settings.weekly_scan_cron)
    _scheduler.add_job(
        run_weekly_market_scan,
        CronTrigger(**weekly_cron, timezone=settings.timezone),
        id="weekly_market_scan",
        name="Weekly Market Scan",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="weekly_market_scan", cron=settings.weekly_scan_cron)

    # Hourly quick check - every hour on the hour
    _scheduler.add_job(
        run_hourly_check,
        CronTrigger(minute=0, timezone=settings.timezone),
        id="hourly_check",
        name="Hourly Quick Check",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="hourly_check", cron="0 * * * *")

    # Critical alert check - every 15 minutes during business hours (8 AM - 8 PM Mon-Fri)
    _scheduler.add_job(
        run_critical_alert_check,
        CronTrigger(minute="*/15", hour="8-20", day_of_week="mon-fri", timezone=settings.timezone),
        id="critical_alert_check",
        name="Critical Alert Check",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="critical_alert_check", cron="*/15 8-20 * * 1-5")

    # Weekly customer voice report - Friday 4 PM
    _scheduler.add_job(
        run_weekly_customer_voice,
        CronTrigger(hour=16, minute=0, day_of_week="fri", timezone=settings.timezone),
        id="weekly_customer_voice",
        name="Weekly Customer Voice Report",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="weekly_customer_voice", cron="0 16 * * 5")

    # Weekly status update - Friday 5 PM
    _scheduler.add_job(
        run_weekly_status_update,
        CronTrigger(hour=17, minute=0, day_of_week="fri", timezone=settings.timezone),
        id="weekly_status_update",
        name="Weekly Status Update",
        replace_existing=True,
    )
    logger.info("scheduler_job_added", job="weekly_status_update", cron="0 17 * * 5")

    _scheduler.start()
    logger.info("scheduler_started", timezone=settings.timezone, jobs=6)


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
