"""
FastAPI application entry point for the PM Agent.

Provides:
- Health check
- POST /invoke  (sync invocation)
- POST /stream  (SSE streaming)
- POST /chat    (conversational)
- Webhook endpoints for Slack, Jira, GitHub
- Scheduler startup/shutdown
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import router
from api.background import start_scheduler, stop_scheduler
from src.utils import logger

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("pm_agent_starting")
    start_scheduler()
    logger.info("pm_agent_started")
    yield
    stop_scheduler()
    # Flush Opik traces before shutdown to ensure nothing is lost
    try:
        from opik.integrations.langchain import OpikTracer
        OpikTracer.flush()
    except Exception:
        pass
    logger.info("pm_agent_stopped")


app = FastAPI(
    title="PM Agent - Product Management AI Agent",
    description="AI-powered Product Management assistant using LangGraph multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for web dashboard / PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router)

# Serve static files (CSS, JS, images)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint (AC-11.2)."""
    return {
        "status": "healthy",
        "service": "pm-agent",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Serve the chat UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "PM Agent API. Visit /docs for API documentation."}
