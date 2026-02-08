"""
Configuration module for PM Agent.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings for validation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── LLM ──
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_llm_provider: str = "anthropic"  # "anthropic" or "openai"
    default_model: str = "claude-sonnet-4-20250514"

    # ── LangSmith ──
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "pm-agent"

    # ── Opik Observability ──
    opik_api_key: str = ""
    opik_workspace: str = "default"
    opik_project_name: str = "pm-agent"
    opik_url_override: str = ""  # Set to local URL like http://localhost:5175 for self-hosted

    # ── Slack ──
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_app_token: str = ""
    slack_alert_channel: str = "#pm-alerts"
    slack_summary_channel: str = "#product"
    slack_bot_user: str = "@pm-agent"  # Bot handle for invite instructions

    # ── Jira ──
    jira_url: str = ""
    jira_user_email: str = ""
    jira_api_token: str = ""
    jira_project_keys: str = "PROD,SUPPORT"  # comma-separated
    jira_customer_field_id: str = ""  # Custom field ID for customer/account (e.g., "customfield_10050")

    # ── Notion ──
    notion_api_key: str = ""
    notion_roadmap_db_id: str = ""
    notion_reports_page_id: str = ""

    # ── GitHub ──
    github_token: str = ""
    github_repos: str = ""  # comma-separated: org/repo1,org/repo2

    # ── Tavily ──
    tavily_api_key: str = ""

    # ── WhatsApp (Meta Cloud API) ──
    whatsapp_phone_number_id: str = ""
    whatsapp_access_token: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_recipient_phone: str = ""

    # ── Email (SMTP) ──
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    email_recipient: str = ""

    # ── Reddit ──
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "pm-agent/1.0"

    # ── Database ──
    postgres_uri: str = ""
    sqlite_path: str = "./checkpoints.db"

    # ── Scheduler ──
    daily_digest_cron: str = "0 6 * * *"
    weekly_scan_cron: str = "0 8 * * 1"
    timezone: str = "Europe/Madrid"

    # ── Server ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_auth_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Derived helpers ──
    @property
    def jira_projects(self) -> List[str]:
        return [p.strip() for p in self.jira_project_keys.split(",") if p.strip()]

    @property
    def github_repo_list(self) -> List[str]:
        return [r.strip() for r in self.github_repos.split(",") if r.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = True,
):
    """
    Factory to create the configured LLM instance.
    """
    settings = get_settings()
    provider = provider or settings.default_llm_provider
    model = model or settings.default_model

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=temperature,
            streaming=streaming,
            max_tokens=4096,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            openai_api_key=settings.openai_api_key,
            temperature=temperature,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
