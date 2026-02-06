# PM Agent - Product Management AI Agent

An AI-powered Product Management assistant built with **LangGraph** multi-agent architecture. The agent proactively monitors Slack, Jira, Notion, GitHub, and web sources to provide actionable insights, daily digests, market intelligence, and automated workflows.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Supervisor Agent                       │
│         (Intent Classification & Delegation)             │
└───────┬──────┬──────┬──────┬──────┬────────────────────┘
        │      │      │      │      │
   ┌────▼──┐ ┌▼────┐ ┌▼────┐ ┌▼────┐ ┌▼──────┐
   │ Slack │ │Jira │ │Git  │ │Mkt  │ │Notion │
   │ Agent │ │Agent│ │Agent│ │Rsch │ │Agent  │
   └───────┘ └─────┘ └─────┘ └─────┘ └───────┘
```

- **Supervisor**: Routes queries to specialist agents, aggregates results
- **Slack Agent**: Channel monitoring, message search, posting alerts
- **Jira Agent**: Issue tracking, priority management, ticket creation
- **GitHub Agent**: PR/issue monitoring, code analysis, community tracking
- **Market Research Agent**: Web search, competitor monitoring, Reddit analysis
- **Notion Agent**: Report generation, documentation, roadmap maintenance

## Quick Start

### 1. Prerequisites

- Python 3.12+
- An Anthropic API key (Claude) or OpenAI API key

### 2. Setup

```bash
# Clone and enter directory
cd pm-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Start the API server
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

### 4. Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/invoke` | Synchronous graph invocation |
| POST | `/stream` | SSE streaming responses |
| POST | `/chat` | Conversational interface |
| POST | `/webhooks/slack` | Slack event webhook |
| POST | `/webhooks/jira` | Jira event webhook |
| POST | `/webhooks/github` | GitHub event webhook |

### Example: Invoke

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "Summarize the latest Jira tickets and GitHub PRs"}'
```

### Example: Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top blockers this week?", "thread_id": "my-session"}'
```

## Proactive Schedules

| Schedule | Default | Description |
|----------|---------|-------------|
| Daily Digest | 6:00 AM CET | Slack + Jira + GitHub + Market summary |
| Weekly Market Scan | Monday 8:00 AM CET | Competitor + Reddit intelligence |
| Hourly Check | Every hour | P0/P1 ticket detection, urgent alerts |

## Environment Variables

See `.env.example` for the full list. Key variables:

- `ANTHROPIC_API_KEY` - Claude API key (required)
- `SLACK_BOT_TOKEN` - Slack bot token
- `JIRA_URL` / `JIRA_API_TOKEN` - Jira credentials
- `NOTION_API_KEY` - Notion integration token
- `GITHUB_TOKEN` - GitHub personal access token
- `TAVILY_API_KEY` - Tavily web search API key
- `POSTGRES_URI` - PostgreSQL connection string (for production checkpointing)

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v --tb=short

# Run with coverage
PYTHONPATH=. pytest tests/ -v --cov=src --cov=api --cov-report=term-missing
```

## Project Structure

```
pm-agent/
├── src/
│   ├── agents/           # Specialist agents (supervisor, slack, jira, etc.)
│   ├── tools/            # Integration tools (slack, jira, github, notion, research)
│   ├── graphs/           # LangGraph definitions
│   ├── state.py          # Agent state schema
│   ├── config.py         # Settings & LLM factory
│   └── utils.py          # Logging, retry, circuit breaker
├── api/
│   ├── main.py           # FastAPI application
│   ├── routes.py         # API routes & webhooks
│   └── background.py     # Scheduled jobs (APScheduler)
├── tests/                # Comprehensive test suite
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Docker Compose with PostgreSQL
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```
